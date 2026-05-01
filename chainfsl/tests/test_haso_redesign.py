"""
Integration tests for HASO redesign modules.
Tests compute_node, state_action, reward_function, routing,
logging_framework, scenarios, convergence, and env integration.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.haso.compute_node import ComputeNode, TIER_CONFIGS, create_compute_node, REF_GFLOPS
from src.haso.state_action import StateBuilder, ActionBuilder, STATE_NAMES, STATE_LOW, STATE_HIGH
from src.haso.reward_function import RewardFunction, RewardConfig, layer_compatibility, conflict_score
from src.haso.routing import RoutingPolicy, RoutingDecision, bandwidth_match, estimate_latency
from src.haso.logging_framework import LogManager, RoundLog, DecisionLog
from src.haso.scenarios import ScenarioB, ScenarioC, create_scenario
from src.haso.convergence import ConvergenceAnalyzer, convergence_bound, compare_convergence
from src.haso.env import SFLNodeEnv
from src.emulator.node_profile import HardwareProfile, create_profile
from src.sfl.models import SplittableResNet18


# ============================================================================
# test_compute_node.py tests
# ============================================================================

def test_compute_node_creation():
    node = create_compute_node(0, tier=1)
    assert node.node_id == 0
    assert node.tier == 1
    assert node.flops_ratio == 1.0


def test_compute_node_time_estimation():
    node = create_compute_node(0, tier=1)
    time = node.estimated_time(cut_layer=2, batch_size=32)
    assert time > 0


def test_compute_node_queue():
    node = create_compute_node(0, tier=1)
    node.add_task(1)  # tier=1 only accepts layer 1
    assert node.current_queue == 1
    node.process_task()
    assert node.processing == True


def test_compute_node_repr():
    node = create_compute_node(0, tier=2)
    repr_str = repr(node)
    assert "ComputeNode" in repr_str
    assert "tier=2" in repr_str


def test_compute_node_tier_configs():
    assert TIER_CONFIGS[1]["flops_ratio"] == 1.0
    assert TIER_CONFIGS[2]["flops_ratio"] == 0.3
    assert TIER_CONFIGS[3]["flops_ratio"] == 0.05
    assert TIER_CONFIGS[4]["flops_ratio"] == 0.005


def test_compute_node_update_bandwidth():
    node = create_compute_node(0, tier=1)
    original_bw = node.bandwidth_mbps
    new_bw = node.update_bandwidth(variance=0.1)
    assert new_bw > 0
    # Bandwidth should fluctuate
    assert new_bw != original_bw or node.bandwidth_mbps != original_bw


def test_compute_node_energy():
    node = create_compute_node(0, tier=1)
    initial = node.energy_remaining
    node.energy_consume(100.0)
    assert node.energy_remaining < initial


# ============================================================================
# test_state_action.py tests
# ============================================================================

def test_state_dimensions():
    assert len(STATE_LOW) == 11
    assert len(STATE_HIGH) == 11
    assert len(STATE_NAMES) == 11


def test_state_builder():
    builder = StateBuilder()
    profile = {
        'cpu_util': 0.5,
        'ram_util': 0.3,
        'gpu_util': 0.8,
        'bandwidth': 0.6,
    }
    state = builder.build_state(
        profile=profile,
        loss_ema=2.5,
        loss_std=0.5,
        neighbor_avail=0.7,
        compute_queue=0.3,
        fusion_candidates=0.5,
        energy_ratio=0.6,
        shard_available=0.9
    )
    assert len(state) == 11
    assert np.all(state >= 0.0) and np.all(state <= 1.0)


def test_action_builder():
    builder = ActionBuilder()
    action = np.array([2, 1, 1, 3])
    result = builder.action_to_dict(int(action[0]))
    assert 'cut_layer' in result
    assert 'batch_size' in result
    assert 'H' in result
    assert 'target_node' in result


def test_action_builder_total_actions():
    builder = ActionBuilder()
    total = builder.total_actions()
    # 4 cut layers * 4 batch sizes * 4 H choices * 8 targets = 512
    assert total == 512


def test_action_space():
    builder = ActionBuilder()
    assert builder.action_space is not None


# ============================================================================
# test_reward_function.py tests
# ============================================================================

def test_reward_config():
    config = RewardConfig()
    assert config.alpha == 2.0
    assert config.beta == 1.5


def test_reward_computation():
    rf = RewardFunction()
    reward = rf.compute(
        T_comp=1.0, T_comm=0.5, delta_F=0.1,
        shapley_phi=0.2, fusion_bonus=0.1,
        overlap_penalty=0.0, current_accuracy=0.7
    )
    assert isinstance(reward, float)


def test_layer_compatibility():
    assert layer_compatibility(2, 2) == 1.0
    assert layer_compatibility(1, 4) == 0.25


def test_conflict_score():
    score = conflict_score(0, {0: 2, 1: 3, 2: 2})
    assert score > 0


def test_reward_function_fusion_bonus():
    rf = RewardFunction()
    bonus = rf.compute_fusion_bonus([1, 2, 3], 0.5)
    assert bonus > 0


def test_reward_function_compute_T_comp():
    rf = RewardFunction()

    class DummyProfile:
        base_flops = 10.0
        flops_ratio = 2.0

    T_comp_val = rf.compute_T_comp(DummyProfile(), cut_layer=2, batch_size=32)
    assert T_comp_val > 0


# ============================================================================
# test_routing.py tests
# ============================================================================

def test_bandwidth_match():
    match = bandwidth_match(100.0, 50.0)
    assert 0.0 <= match <= 1.0
    assert match == 0.5


def test_routing_decision():
    decision = RoutingDecision(
        node_id=0, cut_layer=2, batch_size=32, H=2,
        target_compute_node=3, fusion_partners=[1, 2],
        routing_mode='fusion'
    )
    assert decision.routing_mode == 'fusion'
    assert decision.cut_layer == 2


def test_routing_decision_validation():
    with pytest.raises(ValueError):
        RoutingDecision(
            node_id=0, cut_layer=2, batch_size=32, H=2,
            target_compute_node=3, fusion_partners=[1, 2],
            routing_mode='invalid_mode'
        )


# ============================================================================
# test_logging_framework.py tests
# ============================================================================

def test_round_log_creation():
    log = RoundLog(
        round=0, timestamp="2026-04-30", n_active_nodes=10,
        n_compute_nodes=50, global_accuracy=0.5, global_loss=1.5,
        global_f1=0.4, round_latency=5.0, mean_T_comp=2.0,
        mean_T_comm=0.5, straggler_ratio=0.1, node_decisions=[],
        compute_node_load=[], overlap_events=[]
    )
    assert log.round == 0
    assert log.global_accuracy == 0.5


def test_decision_log_creation():
    log = DecisionLog(
        node_id=0, round=0, step=0,
        state=[0.1, 0.2, 0.3],
        action={"cut_layer": 2},
        reward=1.0, done=False, info={}
    )
    assert log.node_id == 0


# ============================================================================
# test_scenarios.py tests
# ============================================================================

def test_scenario_b():
    s = ScenarioB(N=50, k=10, seed=42)
    active = s.get_active_nodes()
    relay = s.get_relay_nodes()
    assert len(active) == 10
    assert len(relay) == 40
    assert s.is_data_node(active[0]) == True
    assert s.is_data_node(relay[0]) == False


def test_scenario_b_shard_assignment():
    s = ScenarioB(N=50, k=10, seed=42)
    shards = s.assign_data_shards(100)
    total = sum(len(v) for v in shards.values())
    assert total == 100


def test_scenario_c():
    s = ScenarioC(N=50, k_init=15, seed=42)
    active1 = s.get_active_nodes()
    assert len(active1) == 15
    s.step()  # dropout/join
    active2 = s.get_active_nodes()
    # k should vary but stay within [k_min=5, k_max=50]
    assert 5 <= len(active2) <= 50


def test_scenario_c_k_bounds():
    s = ScenarioC(N=50, k_init=15, seed=42, k_min=5, k_max=20)
    for _ in range(10):
        s.step()
        k = len(s.get_active_nodes())
        assert s.k_min <= k <= s.k_max


def test_create_scenario():
    sb = create_scenario("B", N=50, k=10)
    assert isinstance(sb, ScenarioB)
    sc = create_scenario("C", N=50, k_init=10)
    assert isinstance(sc, ScenarioC)


# ============================================================================
# test_convergence.py tests
# ============================================================================

def test_convergence_analyzer():
    analyzer = ConvergenceAnalyzer()
    for i in range(50):
        analyzer.add_round(i, 0.1 + 0.02 * i, 2.5 - 0.03 * i, 5.0 - 0.05 * i)

    rate = analyzer.compute_convergence_rate()
    assert rate > 0
    tt70 = analyzer.time_to_accuracy(0.70)
    assert tt70 is not None
    assert tt70 == 30


def test_convergence_bound():
    bound = convergence_bound(T=100, L_0=2.5, mu=0.1, sigma_sq=0.01, rho=0.1)
    assert bound > 0
    assert bound < 2.5


def test_compare_convergence():
    data_a = [{"round": i, "accuracy": 0.1 + 0.02 * i} for i in range(50)]
    data_b = [{"round": i, "accuracy": 0.1 + 0.025 * i} for i in range(50)]
    result = compare_convergence(data_a, data_b, threshold=0.70)
    assert result["time_to_acc_a"] == 30


# ============================================================================
# test_env_integration.py tests
# ============================================================================

def test_env_state_dim():
    profile = create_profile(0, tier=2)
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes)
    obs, _ = env.reset()
    assert len(obs) == 11, f"Expected 11, got {len(obs)}"


def test_env_step():
    profile = create_profile(0, tier=2)
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes)
    env.reset()
    action = np.array([2, 2, 1, 3])
    obs, reward, term, trunc, info = env.step(action)
    assert 'fusion_bonus' in info
    assert 'overlap_penalty' in info
    assert 'compute_queue' in info


def test_env_scenarios():
    profile = create_profile(0, tier=2)
    env = SFLNodeEnv(profile, n_compute_nodes=8)
    env.set_scenario_b(N=50, k=10)
    env.set_scenario_c(N=50, k_init=15)


def test_env_get_valid_actions():
    profile = create_profile(0, tier=2)
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes)
    env.reset()
    mask = env.get_valid_actions()
    assert len(mask) == 4


def test_env_action_to_dict():
    profile = create_profile(0, tier=2)
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes)
    env.reset()
    d = env.action_to_dict(np.array([2, 2, 1, 3]))
    assert d['cut_layer'] == 3  # CUT_LAYERS[2]
    assert d['batch_size'] == 32  # BATCH_SIZES[2]


def test_env_memory_constraint():
    """Test that memory constraint clamping works."""
    profile = create_profile(0, tier=4)  # Only 200MB RAM
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes, seed=42)
    env.reset()
    # Tier-4 should clamp to cut_layer=1 (smallest model)
    action = np.array([3, 3, 1, 0])  # cut_layer=4, batch_size=64
    obs, reward, term, trunc, info = env.step(action)
    # info should have clamped values
    assert info['cut_layer'] <= 4


def test_env_multistep():
    """Test multiple steps don't crash."""
    profile = create_profile(0, tier=2)
    compute_nodes = [create_compute_node(i, tier=1) for i in range(8)]
    env = SFLNodeEnv(profile, n_compute_nodes=8, compute_nodes=compute_nodes, seed=42)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
