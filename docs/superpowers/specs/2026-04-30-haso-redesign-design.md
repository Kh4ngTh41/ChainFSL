# HASO Redesign - Design Document
**Date:** 2026-04-30
**Version:** 1.0
**Status:** Draft

---

## 1. Overview

**HASO** (Hierarchical Accelerated Split Learning Optimization) là module DRL orchestration trong ChainFSL, điều phối việc chọn cut_layer và routing decision cho từng node trong mạng split federated learning.

### Key Improvements

1. **Flexible Routing (A+B)**: Parallel + Cooperative Fusion routing
2. **Latency-Dominated Reward**: Primary focus on reducing round latency
3. **Compute Node Awareness**: State augmented với compute node queue/load
4. **Full Logging**: Detailed logs cho experiment reproducibility
5. **Converge Analysis**: Formal bounds + empirical validation

---

## 2. Architecture

### 2.1 Routing Model

```
┌──────────────────────────────────────────────────────────────────┐
│                      HASO Routing Graph                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Data Nodes (k nodes có data)                                  │
│          │                                                       │
│          ├── Parallel (A) ──► Compute Node j (direct)           │
│          │                   T = T_comp + T_comm               │
│          │                                                       │
│          └── Fusion (B) ──► Same Compute Node k                  │
│                              Node_a ─┐                          │
│                              Node_b ─┼─► merged activations     │
│                              Node_c ─┘                          │
│                                                                  │
│   Non-Data Nodes (N-k nodes): relay only, không train            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Routing Decision

Mỗi data node i chọn:
1. **cut_layer_i**: Layer để split (1-4)
2. **target_compute_node_j**: Node để forward smashed data
3. **fusion_mode**: Single target vs shared compute node
4. **H_i**: Local epochs

### 2.3 Layer Overlap Handling

Khi nhiều nodes gửi đến same compute node với overlapping layers:
- **Beneficial**: Share computation → reward bonus
- **Harmful**: Queue congestion → penalty

```
overlap_score = Σ nodes_sent_to_same_compute × layer_similarity
if overlap_score > threshold:
    penalty = -μ · overlap_score
else:
    bonus = +λ · overlap_score
```

---

## 3. State Space

**11-dimensional normalized [0,1]:**

| Index | Feature | Description | Source |
|-------|---------|-------------|--------|
| 0 | cpu_util | CPU utilization ratio | HardwareProfile |
| 1 | ram_util | RAM usage ratio | HardwareProfile |
| 2 | gpu_util | GPU utilization ratio | HardwareProfile |
| 3 | bandwidth | Normalized bandwidth | HardwareProfile.bandwidth / 100 |
| 4 | current_loss | Normalized loss | EMA, /10.0 |
| 5 | loss_std | Loss variance | EMA, /5.0 |
| 6 | neighbor_avail | Gossip neighbor availability | HASOGossip |
| 7 | compute_queue | Queue length at target compute node | Dynamic |
| 8 | fusion_candidates | Nodes có thể fusion với node hiện tại | Dynamic |
| 9 | energy_ratio | Energy remaining ratio | HardwareProfile |
| 10 | shard_available | Shard of data available at node | Fixed per node |

```python
STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
```

---

## 4. Action Space

**4-dimensional MultiDiscrete:**

| Index | Action | Choices | Values |
|-------|--------|---------|--------|
| 0 | cut_layer_idx | 4 | [1, 2, 3, 4] |
| 1 | batch_size_idx | 4 | [8, 16, 32, 64] |
| 2 | H_idx | 4 | [1, 2, 3, 5] |
| 3 | target_compute_node_idx | N_compute | [0, 1, ..., N_compute-1] |

---

## 5. Reward Function

**Latency-Dominated with Fusion Bonus:**

```
R_t = -α · T_comp(node_i, cut_layer)          # Computation time
    - β · T_comm(smashed_size, bw_src, bw_dst) # Communication time
    + γ · φ · ΔF                               # Shapley-weighted accuracy gain
    + λ · fusion_bonus                         # Fusion efficiency bonus
    - μ · overlap_penalty                      # Overlap congestion penalty
```

### Parameters

```python
α = 2.0   # Computation time weight (dominate)
β = 1.5   # Communication time weight (dominate)
γ = 0.5   # Shapley contribution weight (secondary)
λ = 0.3   # Fusion bonus weight
μ = 0.4   # Overlap penalty weight
```

### Time Calculations

```python
T_comp(node_i, cut_layer) = base_flops(cut_layer) / (node_i.flops_ratio * REF_GFLOPS)

T_comm(smashed_size, bw_src, bw_dst) =
    smashed_size / min(bw_src, bw_dst) + propagation_delay
```

### Fusion Bonus

```python
fusion_bonus = Σ (nodes_sharing_same_compute_node) × layer_compatibility
# layer_compatibility: cao nếu layers không overlap nhiều
# = 1.0 nếu perfect overlap (beneficial)
# = 0.0 nếu no overlap
```

### Overlap Penalty

```python
overlap_penalty = Σ (conflict_score for each compute_node)
conflict_score = Σ nodes_sent_to_same × layer_similarity
# layer_similarity: cao nếu layers gần nhau → more congestion
```

### Accuracy Guardrail

```python
MIN_ACC_THRESHOLD = 0.60
if accuracy < MIN_ACC_THRESHOLD:
    R_t -= 10.0  # Heavy penalty
```

---

## 6. Compute Node Model

### 6.1 Compute Node Architecture

```python
@dataclass
class ComputeNode:
    node_id: int
    tier: int                           # Hardware tier
    flops_ratio: float                   # Compute power
    max_memory_mb: int                   # Memory limit
    bandwidth_mbps: float                 # Network bandwidth
    current_queue: int                   # Current tasks in queue
    processing: bool                     # Is processing
    accepted_layers: List[int]          # Which layers it can run

    def estimated_time(self, cut_layer, batch_size):
        """Estimate time to process this task"""
        base_flops = self._flops_for_layer(cut_layer)
        queue_time = self.current_queue * avg_task_time
        return queue_time + base_flops / (self.flops_ratio * REF_GFLOPS)
```

### 6.2 Compute Node Selection Policy

PPO agent chọn target_compute_node dựa trên:
1. **Queue Length**: Node đang bận → penalize
2. **Bandwidth Match**: src/dst bandwidth compatibility
3. **Layer Compatibility**: Node có thể run requested cut_layer
4. **Historical Performance**: Past latency với node

```python
score = -w1 * queue_length - w2 * latency_estimate + w3 * bandwidth_match
```

---

## 7. Scenarios

### 7.1 Scenario B: Fixed Sparse

```
N = 50 nodes total
k = 10 nodes có data (20%)
N - k = 40 nodes chỉ relay (compute only)

Round t:
  - 10 data nodes chọn cut_layer và target_compute_node
  - 40 relay nodes forward smashed data
  - Compute nodes process và return gradients
  - Aggregation: weighted average by Shapley contributions
```

**Data Distribution:**
- k nodes được assigned specific data shards
- Non-k nodes: no training data, chỉ participate as compute relay

**Use Case:** Realistic IoT network - only subset of devices collect data

### 7.2 Scenario C: Dynamic Dropout

```
N = 50 nodes total
k(t) thay đổi theo round t

k(0) = 15 nodes with data
k(1) = 12 nodes (2 dropped out)
k(2) = 14 nodes (2 joined)
k(3) = 10 nodes (4 dropped - network issue)
...

Dropout model:
  P(dropout) = 0.1 + 0.05 * (tier == 4)  # Tier-4 more likely to drop
  P(join) = 0.02 per round
```

**Dynamic Assignment:**
- Active nodes: participate in training
- Inactive nodes: can be compute relay
- k_min = 5, k_max = N

**Use Case:** Mobile networks with unreliable connectivity

---

## 8. Logging Framework

### 8.1 Log Structure

```
logs/
├── e1_haso_effectiveness/
│   ├── run_2026-04-30_143022/
│   │   ├── config.json
│   │   ├── rounds/
│   │   │   ├── round_000.json
│   │   │   ├── round_001.json
│   │   │   └── ...
│   │   ├── decisions/
│   │   │   ├── node_00/
│   │   │   │   ├── round_000.json
│   │   │   │   └── ...
│   │   ├── compute_nodes/
│   │   │   └── ...
│   │   └── metrics/
│   │       ├── accuracy.csv
│   │       ├── latency.csv
│   │       └── reward.csv
│   └── ...
```

### 8.2 Per-Round Log

```json
{
  "round": 5,
  "timestamp": "2026-04-30T14:30:22",
  "n_active_nodes": 10,
  "n_compute_nodes": 50,
  "global_accuracy": 0.623,
  "global_loss": 1.234,
  "global_f1": 0.589,
  "round_latency": 4.23,
  "mean_T_comp": 1.82,
  "mean_T_comm": 0.45,
  "straggler_ratio": 0.08,
  "node_decisions": [
    {
      "node_id": 0,
      "cut_layer": 4,
      "batch_size": 32,
      "H": 2,
      "target_compute_node": 5,
      "fusion_partners": [1, 2],
      "T_comp": 1.42,
      "T_comm": 0.23,
      "reward": 12.5,
      "shapley_phi": 0.15
    }
  ],
  "compute_node_load": [
    {"node_id": 5, "queue_length": 3, "processing_layers": [4, 3, 4]}
  ],
  "overlap_events": [
    {"compute_node": 5, "conflict_score": 0.6, "penalty": -0.24}
  ]
}
```

### 8.3 Per-Decision Log

```json
{
  "node_id": 0,
  "round": 5,
  "step": 0,
  "state": [0.5, 0.3, 0.8, 0.9, 0.12, 0.05, 0.7, 0.2, 0.4, 0.9, 0.3],
  "action": {
    "cut_layer_idx": 3,
    "batch_size_idx": 2,
    "H_idx": 1,
    "target_compute_node_idx": 5
  },
  "reward": 12.5,
  "done": false,
  "info": {
    "cut_layer": 4,
    "batch_size": 32,
    "H": 2,
    "target_node": 5,
    "T_comp": 1.42,
    "T_comm": 0.23,
    "delta_F": 0.08,
    "shapley_ema": 0.15,
    "fusion_bonus": 0.1,
    "overlap_penalty": -0.05,
    "compute_queue": 2
  }
}
```

### 8.4 Metrics CSV

**accuracy.csv:**
```csv
round,accuracy,f1_score,precision,recall,loss_test
0,0.123,0.089,0.102,0.085,2.456
1,0.187,0.145,0.156,0.138,2.123
...
```

**latency.csv:**
```csv
round,mean_latency,median_latency,max_latency,min_latency,straggler_ratio
0,5.234,4.891,12.345,2.123,0.15
1,4.892,4.567,10.234,1.987,0.12
...
```

**reward.csv:**
```csv
round,node_id,reward,T_comp,T_comm,delta_F,shapley_phi,fusion_bonus,overlap_penalty
0,0,12.5,1.42,0.23,0.08,0.15,0.1,-0.05
0,1,11.8,1.38,0.21,0.07,0.12,0.08,-0.03
...
```

---

## 9. Converge Analysis

### 9.1 Empirical Validation

**Metrics per round:**
- Accuracy curve: accuracy vs rounds
- Time-to-accuracy: rounds to reach threshold (60%, 70%, 80%)
- Loss curve: loss vs rounds

**Convergence Rate:**
```
convergence_rate = Δaccuracy / Δround
avg_convergence = Σ (accuracy[t+1] - accuracy[t]) / T
```

### 9.2 Formal Bound

**Theorem:** Với các assumptions:
1. Loss function L is μ-smooth
2. Gradient variance bounded by σ²
3. Compression ratio ρ (smashed data / raw data)

**Convergence Rate:**
```
E[L_T] ≤ L_0 / (μT) + σ²/(μT) + ρ·L_0/T

Thì HASO converges at rate O(1/T) với:
- ρ → 0: smashed data compression helps
- Better routing → smaller T_comm → faster convergence
```

---

## 10. RQ Mapping

| RQ | Question | Experiment | Metrics |
|----|----------|------------|---------|
| RQ1 | How does HASO affect training latency? | E1 | Mean latency, straggler ratio, per-round latency |
| RQ2 | Does adaptive routing improve convergence speed? | E1, E2 | Time-to-accuracy, accuracy vs rounds |
| RQ3 | How does HASO utilize heterogeneous resources? | E1, E3 | GPU/CPU utilization, bandwidth consumption |
| RQ4 | How secure is HASO under attacks? | E4 | Attack success rate, model accuracy degradation |
| RQ5 | Does Shapley incentive maintain participation? | E5 | Participation rate, reward distribution (Jain's) |
| RQ6 | What is the contribution of each module? | E6 | Ablation study metrics |

---

## 11. Baseline Comparison

| Baseline | Description | What HASO Beats |
|----------|-------------|-----------------|
| FedAvg | Standard FL, no split | Latency (by avoiding straggler) |
| SplitFed-V1/V2 | SFL with uniform cut | Latency + flexibility |
| AdaptSFL | Adaptive cut + aggregation | Latency + convergence |
| DFL | Dynamic per-client split | Latency + resource efficiency |
| DISNET | Micro-split | Latency + overhead |
| ESFL | Efficient SFL | Latency + convergence |
| ChainFSL-NoHASO | Ablation | Shows HASO value |
| ChainFSL-NoTVE | Ablation | Shows TVE value |
| ChainFSL-NoGTM | Ablation | Shows GTM value |

---

## 12. Implementation Checklist

- [ ] State space augmentation (11-dim)
- [ ] Action space với compute node selection
- [ ] Reward function với fusion bonus + overlap penalty
- [ ] ComputeNode model với queue tracking
- [ ] Logging framework (per-round, per-decision)
- [ ] Scenario B: Fixed sparse (k nodes có data)
- [ ] Scenario C: Dynamic dropout
- [ ] Converge analysis utilities
- [ ] Metrics CSV exporters
- [ ] Integration với existing ChainFSL protocol

---

## 13. File Structure

```
src/haso/
├── env.py                    # SFLNodeEnv (enhanced)
├── agent.py                 # HASO agents
├── reward.py                # Reward function
├── compute_node.py          # NEW: Compute node model
├── routing.py               # NEW: Routing decision logic
├── logging.py               # NEW: Logging framework
├── convergence.py           # NEW: Converge analysis
├── scenarios.py             # NEW: B and C scenarios
├── __init__.py
```

---

**End of Design Document**