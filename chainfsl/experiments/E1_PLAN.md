# E1: HASO Effectiveness — Straggler Analysis
## Detailed Experimental Plan

---

## 1. Problem Statement

### 1.1 The Straggler Problem in Federated Split Learning

In Federated Split Learning (FSL), all nodes typically use the same static cut layer to split the neural network between client and server. This approach creates a **straggler problem**:

- **Static cut_layer=2** requires ~900MB RAM with Adam optimizer
- **Tier-4 devices** (200MB RAM, e.g., ESP32, Raspberry Pi Zero) cannot fit this
- Tier-3 devices (512MB RAM) can barely fit it with reduced batch size → slower
- Tier-1/2 devices (2-8GB RAM) complete quickly but must wait for stragglers

**Result**: The round latency is dominated by the slowest (weakest) node, wasting resources on powerful devices.

### 1.2 Our Solution: HASO

**HASO** (Hierarchical Accelerated Split Learning Optimization) uses Multi-Agent Deep Reinforcement Learning to assign optimal cut layers per-node based on:
- Hardware tier (memory constraints)
- Network bandwidth
- Historical contribution (Shapley feedback)
- Neighbor availability (gossip protocol)

**Key insight**: Assigning shallow cuts to weak devices and deep cuts to strong devices balances the round, reducing mean latency.

---

## 2. Experiment Design

### 2.1 Research Question

**Does HASO reduce training latency by avoiding stragglers compared to static cut baseline?**

### 2.2 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Model | ResNet-18 (splittable) |
| n_nodes | 20 |
| global_rounds | 50 |
| batch_size | 32 (default) |
| local_epochs (H) | 1 |
| seed | 42 |

### 2.3 Tier Distribution Scenarios

We test 4 hardware configurations to validate HASO across diverse IoT environments:

| Scenario | Tier-1 (8GB) | Tier-2 (4GB) | Tier-3 (512MB) | Tier-4 (200MB) | Description |
|----------|--------------|-------------|----------------|----------------|-------------|
| **iot_heavy** | 5% | 10% | 35% | 50% | Most challenging — many weak devices |
| **balanced** | 10% | 30% | 40% | 20% | Default — realistic mix |
| **gpu_heavy** | 30% | 35% | 25% | 10% | Easiest — mostly powerful devices |
| **uniform** | 25% | 25% | 25% | 25% | Equal representation |

### 2.4 Two Methods Per Scenario

#### Method 1: Static Cut (Baseline)
- All nodes use **cut_layer=2** regardless of tier
- Tier-4/Tier-3 nodes must reduce batch_size or be excluded → stragglers
- **Exposes** the straggler problem

#### Method 2: HASO (Ours)
- Each node runs PPO agent to decide optimal cut_layer
- **Tier-4**: cut_layer=1 (shallow, fits 200MB RAM)
- **Tier-3**: cut_layer=2 (medium, fits 512MB with optimization)
- **Tier-2**: cut_layer=3 (deeper, fits 4GB RAM)
- **Tier-1**: cut_layer=4 (deepest, fits 8GB RAM)
- Network latency simulation included for realism

---

## 3. Metrics

### 3.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Mean Round Latency** | Average time per round (seconds) | HASO < Static |
| **Straggler Ratio** | % nodes with latency > 1.5× mean | HASO < Static |
| **Final Accuracy** | Test accuracy at round 50 | HASO ≈ Static |
| **Time-to-Accuracy** | Rounds to reach 60% accuracy | HASO < Static |

### 3.2 Secondary Metrics

| Metric | Description |
|--------|-------------|
| F1-Score (macro) | Classification quality |
| Precision (macro) | Positive predictive value |
| Recall (macro) | Sensitivity |
| Fairness Index | Jain's fairness of reward distribution |
| Shapley Variance | Inequality in contributions |

### 3.3 Per-Node Metrics (detailed logging)

For each node, we log:
- `N{id}(T{tier}:L{cut_layer}:{time}s:{loss}){reason}`
- Example: `N3(T4:L1:0.82s:1.234)[SHALLOW:95%RAM]`

**Reason codes**:
- `[SHALLOW:{mem}%]` — Cut layer constrained by memory
- `[SAFE:tier{tier}]` — Memory-safe tier assignment
- `[OPTIMAL:tier{tier}]` — Best fit for tier
- `[EXCLUDED:no cut fits RAM]` — Node excluded due to OOM

---

## 4. Memory Model

### 4.1 ResNet-18 Split Points (with Adam optimizer)

| Cut Layer | Client Memory | Server Memory | Notes |
|----------|---------------|--------------|-------|
| L1 | ~200MB | ~2100MB | Minimum client compute |
| L2 | ~450MB | ~1500MB | Balanced split |
| L3 | ~900MB | ~900MB | Equal split |
| L4 | ~2100MB | ~450MB | Minimum server compute |

### 4.2 Tier Memory Limits

| Tier | Device Examples | RAM | Max Cut Layer |
|------|----------------|-----|--------------|
| Tier-1 | Jetson Orin, Desktop GPU | 8GB | L4 (2100MB) |
| Tier-2 | Raspberry Pi 4B, Edge GPU | 4GB | L3 (900MB) |
| Tier-3 | Raspberry Pi Zero 2W | 512MB | L2 (450MB) |
| Tier-4 | ESP32-S3, ESP32 | 200MB | L1 (200MB) |

---

## 5. Network Latency Simulation

### 5.1 Bandwidth by Tier

| Tier | Bandwidth (Mbps) | Latency (ms) |
|------|------------------|--------------|
| Tier-1 | 1000 | 1 |
| Tier-2 | 100 | 5 |
| Tier-3 | 10 | 20 |
| Tier-4 | 1 | 50 |

### 5.2 Transmission Time Formula

```
transmission_time = data_size_bytes / (min(bw_src, bw_dst) × 1e6 / 8)
propagation_latency = latency_ms / 1000
total_comm_time = transmission_time + propagation_latency
```

### 5.3 Smashed Data Sizes

| Cut Layer | Activations (batch=32) | Gradient |
|----------|------------------------|----------|
| L1 | ~2MB | ~0.5MB |
| L2 | ~4MB | ~1MB |
| L3 | ~8MB | ~2MB |
| L4 | ~16MB | ~4MB |

---

## 6. Expected Results

### 6.1 Hypothesis

| Scenario | Static Cut Problem | HASO Solution |
|----------|-------------------|---------------|
| iot_heavy | 50% Tier-4 → excluded/straggler | All nodes participate with appropriate cuts |
| balanced | 20% Tier-4 struggling | Tier-4 gets L1, completes faster |
| gpu_heavy | Few stragglers | Marginal improvement |
| uniform | Moderate stragglers | Balanced assignment |

### 6.2 Expected Outcomes

#### IoT-Heavy Scenario
```
Static Cut:
  Mean Latency: 12.5s
  Straggler Ratio: 45%
  Participants: 10/20 (50%)

HASO:
  Mean Latency: 4.2s (↓67%)
  Straggler Ratio: 8%
  Participants: 20/20 (100%)
```

#### Balanced Scenario
```
Static Cut:
  Mean Latency: 6.8s
  Straggler Ratio: 25%
  Participants: 16/20 (80%)

HASO:
  Mean Latency: 3.1s (↓54%)
  Straggler Ratio: 5%
  Participants: 20/20 (100%)
```

---

## 7. Output Format

### 7.1 Console Output (per round)

```
Round 5/50 | Loss: 1.234 | Acc: 45.2% | F1: 0.412 | P: 0.438 | R: 0.398 | Fairness: 0.847 | Valid: 18/20 | Latency: 4.2s | Reward: 124.5
  Round 5 HASO: N0:L4->T0 | N1:L3->T0 | N2:L2->T0 | N3:L1->T0 | N4:L2->T0 | ...
  Round 5 Train: N0(T1:L4:1.42s:1.303)[OPTIMAL:tier1] | N1(T2:L3:1.87s:1.287)[OPTIMAL:tier2] | ...
    Mean: 1.82s | Stragglers (2): [N3, N7]
  [Round 5] EVAL: acc=45.23%, loss=1.234, F1=0.412, P=0.438, R=0.398, valid=18/20
```

### 7.2 CSV Output Files

| File | Contents |
|------|----------|
| `e1_iot_heavy_static.csv` | All rounds × static method |
| `e1_iot_heavy_haso.csv` | All rounds × HASO method |
| `e1_balanced_static.csv` | ... |
| `e1_balanced_haso.csv` | ... |
| `e1_iot_heavy_comparison.csv` | Combined for analysis |

### 7.3 Comparison Table

```
================================================================================
E1 COMPARISON TABLE - iot_heavy
================================================================================
Method          Final Acc    Mean Latency   Straggler%    Fairness
--------------------------------------------------------------------------------
static_cut        62.3%       12.45s          45.2%        0.712
haso              63.1%        4.21s           7.8%        0.891
================================================================================

📊 HASO vs Static Cut Improvement:
   Latency: -66.2% (12.45s → 4.21s)
   Straggler Ratio: -82.7% (45.2% → 7.8%)
```

---

## 8. Running the Experiment

### 8.1 Single Scenario
```bash
python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50 --tier_dist iot_heavy
```

### 8.2 All Scenarios (4 × 2 = 8 runs)
```bash
python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50
```

### 8.3 With Pretrained Agents
```bash
python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50 --pretrain_rounds 200 --pretrain_dir pretrainppo
```

### 8.4 Skip Baselines (HASO only)
```bash
python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50 --skip_baselines
```

---

## 9. Verification Checklist

- [ ] All 4 tier distributions tested
- [ ] Static cut baseline clearly exposes straggler problem
- [ ] HASO assigns appropriate cut_layer per tier
- [ ] F1, precision, recall metrics computed each round
- [ ] Per-node logging shows cut decision + reason
- [ ] Straggler analysis printed (Mean, nodes > 1.5× mean)
- [ ] CSV files saved for all scenarios
- [ ] Comparison table shows improvement percentages

---

## 10. Key Insight

**HASO's advantage is not speed — it's balance.**

Static cut is fast for Tier-1/2 nodes but creates stragglers. HASO trades a slightly deeper cut (faster per node) for better round-level synchronization. The result: all nodes complete closer together, reducing round latency variance.

This is especially important for IoT-heavy scenarios where 50%+ of devices are constrained.