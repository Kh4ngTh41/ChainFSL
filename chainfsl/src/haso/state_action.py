"""HASO State and Action Space Definition."""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box

STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

STATE_NAMES = [
    'cpu_util', 'ram_util', 'gpu_util', 'bandwidth',
    'current_loss', 'loss_std', 'neighbor_avail',
    'compute_queue', 'fusion_candidates', 'energy_ratio', 'shard_available'
]

CUT_LAYERS = [1, 2, 3, 4]
BATCH_SIZES = [8, 16, 32, 64]
H_CHOICES = [1, 2, 3, 5]

N_CUT = len(CUT_LAYERS)
N_BATCH = len(BATCH_SIZES)
N_H = len(H_CHOICES)
N_TARGETS = 8


class StateBuilder:
    """Builds normalized state vector from node profile and metrics."""

    def __init__(self):
        self.state_low = STATE_LOW
        self.state_high = STATE_HIGH
        self.state_names = STATE_NAMES

    def build_state(
        self,
        profile,
        loss_ema,
        loss_std,
        neighbor_avail,
        compute_queue,
        fusion_candidates,
        energy_ratio,
        shard_available
    ):
        """
        Build 11-dim normalized state vector.

        Args:
            profile: dict with cpu_util, ram_util, gpu_util, bandwidth
            loss_ema: exponential moving average of loss
            loss_std: standard deviation of loss
            neighbor_avail: fraction of available neighbors [0, 1]
            compute_queue: normalized queue length [0, 1]
            fusion_candidates: number of fusion candidates [0, 1]
            energy_ratio: energy consumption ratio [0, 1]
            shard_available: fraction of available shards [0, 1]

        Returns:
            np.array: 11-dim normalized state vector
        """
        state = np.array([
            profile.get('cpu_util', 0.0),
            profile.get('ram_util', 0.0),
            profile.get('gpu_util', 0.0),
            profile.get('bandwidth', 0.0),
            loss_ema,
            loss_std,
            neighbor_avail,
            compute_queue,
            fusion_candidates,
            energy_ratio,
            shard_available
        ], dtype=np.float32)

        normalized = np.clip(
            (state - self.state_low) / (self.state_high - self.state_low),
            0.0, 1.0
        )
        return normalized

    @property
    def state_space(self):
        """Return the Box state space."""
        return Box(low=STATE_LOW, high=STATE_HIGH, dtype=np.float32)


class ActionBuilder:
    """Converts action indices to structured actions and validates them."""

    def __init__(self):
        self.cut_layers = CUT_LAYERS
        self.batch_sizes = BATCH_SIZES
        self.h_choices = H_CHOICES
        self.n_targets = N_TARGETS

    def action_to_dict(self, action_idx):
        """
        Convert flat action index to human-readable dict.

        Args:
            action_idx: int in range(n_actions)

        Returns:
            dict with cut_layer, batch_size, H, target_node
        """
        n_cut = len(self.cut_layers)
        n_batch = len(self.batch_sizes)
        n_h = len(self.h_choices)

        cut_idx = action_idx % n_cut
        rest = action_idx // n_cut
        batch_idx = rest % n_batch
        rest = rest // n_batch
        h_idx = rest % n_h
        target_node = rest // n_h

        return {
            'cut_layer': self.cut_layers[cut_idx],
            'batch_size': self.batch_sizes[batch_idx],
            'H': self.h_choices[h_idx],
            'target_node': target_node
        }

    def get_valid_actions_mask(self, profile, memory_map):
        """
        Get boolean mask for valid actions given current profile and memory.

        Args:
            profile: dict with resource info (gpu_available, min_batch_size, etc.)
            memory_map: dict mapping node_id -> available memory

        Returns:
            np.array: boolean mask where True = valid action
        """
        n_cut = len(self.cut_layers)
        n_batch = len(self.batch_sizes)
        n_h = len(self.h_choices)
        n_actions = n_cut * n_batch * n_h * self.n_targets

        mask = np.ones(n_actions, dtype=bool)

        gpu_available = profile.get('gpu_available', True)
        min_batch = profile.get('min_batch_size', 8)
        max_memory_mb = profile.get('max_memory_mb', 4096)

        for i in range(n_actions):
            action = self.action_to_dict(i)

            if not gpu_available and action['cut_layer'] > 1:
                mask[i] = False

            if action['batch_size'] < min_batch:
                mask[i] = False

            est_memory = action['batch_size'] * action['cut_layer'] * 50
            if est_memory > max_memory_mb:
                mask[i] = False

            target = action['target_node']
            if target != self.n_targets and target not in memory_map:
                mask[i] = False

        return mask

    @property
    def action_space(self):
        """Return the MultiDiscrete action space."""
        return MultiDiscrete([N_CUT, N_BATCH, N_H, N_TARGETS])

    @property
    def state_space(self):
        """Return the Box state space."""
        return Box(low=STATE_LOW, high=STATE_HIGH, dtype=np.float32)

    def total_actions(self):
        """Return total number of possible actions."""
        return N_CUT * N_BATCH * N_H * N_TARGETS


def test():
    """Test state and action space definitions."""
    print("=== State/Action Space Tests ===")

    state_builder = StateBuilder()
    action_builder = ActionBuilder()

    print(f"State dimensions: {len(STATE_NAMES)}")
    assert len(STATE_NAMES) == 11, f"Expected 11 state dims, got {len(STATE_NAMES)}"

    print(f"State space: {state_builder.state_space}")

    action_space = action_builder.action_space
    print(f"Action space: {action_space}")
    print(f"Total actions: {action_builder.total_actions()}")

    for i in [0, 1, 10, 50, action_builder.total_actions() - 1]:
        action_dict = action_builder.action_to_dict(i)
        print(f"  action[{i}] -> {action_dict}")

    profile = {
        'cpu_util': 0.5,
        'ram_util': 0.3,
        'gpu_util': 0.8,
        'bandwidth': 0.6,
        'gpu_available': True,
        'min_batch_size': 8,
        'max_memory_mb': 4096
    }
    memory_map = {0: 1024, 1: 2048, 2: 512}

    mask = action_builder.get_valid_actions_mask(profile, memory_map)
    valid_count = mask.sum()
    print(f"\nValid actions: {valid_count}/{len(mask)}")

    state = state_builder.build_state(
        profile=profile,
        loss_ema=2.5,
        loss_std=0.5,
        neighbor_avail=0.7,
        compute_queue=0.3,
        fusion_candidates=0.5,
        energy_ratio=0.6,
        shard_available=0.9
    )
    print(f"\nState vector ({len(state)} dims):")
    for name, val in zip(STATE_NAMES, state):
        print(f"  {name}: {val:.4f}")

    assert len(state) == 11, f"Expected 11-dim state, got {len(state)}"
    assert np.all(state >= 0.0) and np.all(state <= 1.0), "State values out of [0,1] range"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test()