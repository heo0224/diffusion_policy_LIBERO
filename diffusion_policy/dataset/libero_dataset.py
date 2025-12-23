from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
import copy
import zarr
from datasets import load_dataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

# Try to import transformers, but don't fail if not available
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError):
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not properly installed. Language encoding will be disabled.")


class LiberoDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_cache=False,
            max_episodes=None,
            task_suite=None
        ):
        """
        LIBERO dataset for diffusion policy with language conditioning.

        Args:
            shape_meta: Dictionary describing observation and action shapes
            horizon: Sequence length for training
            pad_before: Padding before sequence
            pad_after: Padding after sequence
            n_obs_steps: Number of observation steps
            seed: Random seed
            val_ratio: Validation split ratio
            max_train_episodes: Maximum number of training episodes
            use_cache: Whether to use cached dataset
            max_episodes: Maximum number of episodes to load from HuggingFace
            task_suite: Task suite to filter by. Options: 'libero_spatial', 'libero_object',
                       'libero_goal', 'libero_10', 'libero_90'. If None, loads all tasks.
        """

        # Initialize language encoder
        self.use_language_encoder = TRANSFORMERS_AVAILABLE
        if self.use_language_encoder:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.language_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.language_encoder.eval()
            for param in self.language_encoder.parameters():
                param.requires_grad = False
        else:
            self.tokenizer = None
            self.language_encoder = None

        # Load dataset from HuggingFace
        if task_suite is not None:
            print(f'Loading LIBERO dataset from HuggingFace (task suite: {task_suite})...')
        else:
            print('Loading LIBERO dataset from HuggingFace (all task suites)...')

        if max_episodes is not None:
            split_str = f'train[:{max_episodes}]'
        else:
            split_str = 'train'
        hf_dataset = load_dataset('physical-intelligence/libero', split=split_str)

        # Filter by task suite if specified
        if task_suite is not None:
            print(f'Filtering dataset for task suite: {task_suite}')
            # The dataset should have a 'task_suite' or similar field
            # We'll filter based on the task name or suite identifier
            filtered_indices = []
            for idx in range(len(hf_dataset)):
                episode = hf_dataset[idx]
                # Check if episode belongs to the requested task suite
                # This assumes the dataset has a 'task_suite' field or we can infer from task name
                episode_suite = episode.get('task_suite', None)
                if episode_suite is None:
                    # Fallback: try to infer from task name or other fields
                    # Some datasets might store this in metadata
                    task_name = episode.get('task_name', episode.get('task', ''))
                    # Map task names to suites (this is a heuristic)
                    episode_suite = self._infer_task_suite(task_name)

                if episode_suite == task_suite:
                    filtered_indices.append(idx)

            if len(filtered_indices) == 0:
                print(f'Warning: No episodes found for task suite {task_suite}. Loading all episodes.')
            else:
                hf_dataset = hf_dataset.select(filtered_indices)
                print(f'Filtered to {len(hf_dataset)} episodes for task suite {task_suite}')

        # Convert HuggingFace dataset to ReplayBuffer
        replay_buffer = self._convert_libero_to_replay(
            hf_dataset=hf_dataset,
            shape_meta=shape_meta
        )

        # Parse shape_meta to get rgb and lowdim keys
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type_str = attr.get('type', 'low_dim')
            if type_str == 'rgb':
                rgb_keys.append(key)
            elif type_str == 'low_dim':
                lowdim_keys.append(key)

        # Setup validation split
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Setup key_first_k for observation steps
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Create sequence sampler
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _infer_task_suite(self, task_name):
        """
        Infer task suite from task name or other identifiers.
        This is a heuristic that may need adjustment based on actual dataset structure.
        """
        task_name_lower = str(task_name).lower()

        # Check for explicit suite names in task name
        if 'spatial' in task_name_lower or 'libero_spatial' in task_name_lower:
            return 'libero_spatial'
        elif 'object' in task_name_lower or 'libero_object' in task_name_lower:
            return 'libero_object'
        elif 'goal' in task_name_lower or 'libero_goal' in task_name_lower:
            return 'libero_goal'
        elif 'libero_10' in task_name_lower:
            return 'libero_10'
        elif 'libero_90' in task_name_lower:
            return 'libero_90'

        # If no match, return None
        return None

    def _convert_libero_to_replay(self, hf_dataset, shape_meta):
        """
        Convert HuggingFace LIBERO dataset to ReplayBuffer format.

        Expected LIBERO dataset structure per episode:
        {
            "observation/state": np.array([T, 8]),
            "observation/image": np.array([T, 224, 224, 3]),
            "observation/wrist_image": np.array([T, 224, 224, 3]),
            "action": np.array([T, action_dim]),
            "prompt": str
        }
        """
        store = zarr.MemoryStore()
        root = zarr.group(store)
        data_group = root.require_group('data', overwrite=True)
        meta_group = root.require_group('meta', overwrite=True)

        # Count episodes and total steps
        n_episodes = len(hf_dataset)
        episode_ends = []
        prev_end = 0

        # First pass: count steps and collect episode info
        print('Counting episodes and steps...')
        for episode_idx in tqdm(range(n_episodes), desc="Processing episodes"):
            episode = hf_dataset[episode_idx]
            episode_length = len(episode['action'])
            episode_end = prev_end + episode_length
            episode_ends.append(episode_end)
            prev_end = episode_end

        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]

        # Save episode metadata
        _ = meta_group.array('episode_ends', episode_ends,
            dtype=np.int64, compressor=None, overwrite=True)

        # Initialize arrays for all data
        print('Initializing data arrays...')

        # Action array
        action_dim = shape_meta['action']['shape'][0]
        action_data = np.zeros((n_steps, action_dim), dtype=np.float32)

        # State array (observation/state)
        if 'agent_pos' in shape_meta['obs']:
            state_dim = shape_meta['obs']['agent_pos']['shape'][0]
            state_data = np.zeros((n_steps, state_dim), dtype=np.float32)

        # Image arrays
        if 'image' in shape_meta['obs']:
            img_shape = shape_meta['obs']['image']['shape']  # [C, H, W]
            c, h, w = img_shape
            image_data = np.zeros((n_steps, h, w, c), dtype=np.uint8)

        if 'wrist_image' in shape_meta['obs']:
            wrist_img_shape = shape_meta['obs']['wrist_image']['shape']  # [C, H, W]
            c, h, w = wrist_img_shape
            wrist_image_data = np.zeros((n_steps, h, w, c), dtype=np.uint8)

        # Language embedding array (will store per-episode embeddings)
        language_embeddings = []
        language_episode_ids = []  # Track which episode each step belongs to

        # Second pass: fill in data
        print('Loading episode data...')
        for episode_idx in tqdm(range(n_episodes), desc="Loading episodes"):
            episode = hf_dataset[episode_idx]
            start_idx = episode_starts[episode_idx]
            end_idx = episode_ends[episode_idx]
            episode_length = end_idx - start_idx

            # Load actions
            actions = np.array(episode['action'], dtype=np.float32)
            action_data[start_idx:end_idx] = actions[:episode_length]

            # Load state observations
            if 'agent_pos' in shape_meta['obs']:
                states = np.array(episode['observation/state'], dtype=np.float32)
                state_data[start_idx:end_idx] = states[:episode_length]

            # Load images
            if 'image' in shape_meta['obs']:
                images = np.array(episode['observation/image'], dtype=np.uint8)
                # Ensure shape is [T, H, W, C]
                if images.shape[-1] != c:
                    images = np.moveaxis(images, 1, -1)  # [T, C, H, W] -> [T, H, W, C]
                image_data[start_idx:end_idx] = images[:episode_length]

            if 'wrist_image' in shape_meta['obs']:
                wrist_images = np.array(episode['observation/wrist_image'], dtype=np.uint8)
                # Ensure shape is [T, H, W, C]
                if wrist_images.shape[-1] != c:
                    wrist_images = np.moveaxis(wrist_images, 1, -1)  # [T, C, H, W] -> [T, H, W, C]
                wrist_image_data[start_idx:end_idx] = wrist_images[:episode_length]

            # Encode language prompt
            prompt = episode['prompt']
            if self.use_language_encoder:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors='pt',
                                           padding=True, truncation=True, max_length=512)
                    outputs = self.language_encoder(**inputs)
                    # Use [CLS] token embedding
                    lang_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            else:
                # Fallback: use a hash-based embedding
                import hashlib
                prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:16], 16)
                np.random.seed(prompt_hash % (2**32))
                lang_embedding = np.random.randn(768).astype(np.float32)
                # Normalize it
                lang_embedding = lang_embedding / (np.linalg.norm(lang_embedding) + 1e-8)
            language_embeddings.append(lang_embedding)

            # Track episode ID for each step
            for _ in range(episode_length):
                language_episode_ids.append(episode_idx)

        # Save data to zarr
        print('Saving data to ReplayBuffer...')

        # Save action
        _ = data_group.array(
            name='action',
            data=action_data,
            shape=action_data.shape,
            chunks=action_data.shape,
            compressor=None,
            dtype=action_data.dtype
        )

        # Save state
        if 'agent_pos' in shape_meta['obs']:
            _ = data_group.array(
                name='agent_pos',
                data=state_data,
                shape=state_data.shape,
                chunks=state_data.shape,
                compressor=None,
                dtype=state_data.dtype
            )

        # Save images
        if 'image' in shape_meta['obs']:
            _ = data_group.array(
                name='image',
                data=image_data,
                shape=image_data.shape,
                chunks=(1,) + image_data.shape[1:],
                compressor=None,
                dtype=image_data.dtype
            )

        if 'wrist_image' in shape_meta['obs']:
            _ = data_group.array(
                name='wrist_image',
                data=wrist_image_data,
                shape=wrist_image_data.shape,
                chunks=(1,) + wrist_image_data.shape[1:],
                compressor=None,
                dtype=wrist_image_data.dtype
            )

        # Save language embeddings (per episode)
        language_embeddings = np.array(language_embeddings, dtype=np.float32)
        _ = meta_group.array(
            name='language_embeddings',
            data=language_embeddings,
            shape=language_embeddings.shape,
            chunks=language_embeddings.shape,
            compressor=None,
            dtype=language_embeddings.dtype
        )

        # Save episode IDs for each step
        language_episode_ids = np.array(language_episode_ids, dtype=np.int64)
        _ = meta_group.array(
            name='language_episode_ids',
            data=language_episode_ids,
            shape=language_episode_ids.shape,
            chunks=language_episode_ids.shape,
            compressor=None,
            dtype=language_episode_ids.dtype
        )

        replay_buffer = ReplayBuffer(root)
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action normalizer
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_range_normalizer_from_stat(stat)

        # Observation normalizers
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        # Image normalizers
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # Language embedding normalizer (identity, already normalized)
        meta_group = self.replay_buffer.root['meta']
        lang_embeddings = meta_group['language_embeddings'][:]
        stat = {
            'min': lang_embeddings.min(axis=0),
            'max': lang_embeddings.max(axis=0),
            'mean': lang_embeddings.mean(axis=0),
            'std': lang_embeddings.std(axis=0)
        }
        normalizer['language_embedding'] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # Get episode index for this sample
        # The sampler returns the actual indices, we need to find which episode it belongs to
        sample_info = self.sampler.indices[idx]
        episode_idx = sample_info[0]  # episode index

        # Get language embedding for this episode
        meta_group = self.replay_buffer.root['meta']
        lang_embedding = meta_group['language_embeddings'][episode_idx]

        # Prepare observation dict
        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()

        # Process RGB observations
        for key in self.rgb_keys:
            # Convert from [T, H, W, C] uint8 to [T, C, H, W] float32
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]

        # Process lowdim observations
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Add language embedding to observations
        obs_dict['language_embedding'] = lang_embedding.astype(np.float32)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }

        return torch_data
