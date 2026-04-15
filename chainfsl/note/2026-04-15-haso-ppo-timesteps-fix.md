# Note: MA-HASO PPO Training Issue & Fix

**Date:** 2026-04-15
**Issue:** PPO agent receives too few timesteps per round to learn meaningful policy
**Severity:** High — RL agent essentially does not learn

---

## Problem: PPO Timesteps Far Too Low

### Before Fix

```python
# chainfsl.py:_phase_haso_update()
self.agent_pool.learn_all(total_timesteps=64)  # WAS: 64
```

**Total RL experience collected across entire training:**

| global_rounds | timesteps/round | Total timesteps | Sufficient for PPO? |
|---------------|-----------------|----------------|---------------------|
| 30 | 64 | 1,920 | **NO** (needs 50k+) |
| 100 | 64 | 6,400 | **NO** |
| 200 | 64 | 12,800 | **NO** |

Stable-Baselines3 PPO typically needs **50,000–500,000 timesteps** to learn a meaningful policy. 64 timesteps/round means the agent's policy buffer is discarded and reset every round — **no learning occurs**.

---

## Fix Applied

### 1. Updated `_phase_haso_update()` in `src/protocol/chainfsl.py`

```python
def _phase_haso_update(self, shapley_vals: Dict[int, float]) -> None:
    """Phase 8: Update PPO policies with Shapley-based reward shaping."""
    if not self.haso_enabled or self.agent_pool is None:
        return

    self.agent_pool.update_shapley_all(shapley_vals)
    # PPO needs sufficient timesteps to learn. 64 is far too little.
    # Use configurable ppo_update_timesteps from config (default 256)
    update_ts = self.cfg.get("ppo_update_timesteps", 256)
    self.agent_pool.learn_all(total_timesteps=update_ts)
```

### 2. Added `ppo_update_timesteps` to configs

**`config/default.yaml`:**
```yaml
ppo_n_steps: 512
ppo_batch_size: 64
ppo_n_epochs: 10
ppo_update_timesteps: 256  # NEW: timesteps per RL update per round
```

**`experiments/utils.py:build_config()`:**
```python
"ppo_update_timesteps": 256,  # timesteps per RL update per round (was 64, too little)
```

### 3. After Fix — Expected RL Experience

| global_rounds | timesteps/round | Total timesteps | Sufficient? |
|---------------|-----------------|----------------|-------------|
| 30 | 256 | 7,680 | Still low but workable |
| 100 | 256 | 25,600 | Minimum viable |
| 200 | 256 | 51,200 | **YES** — good for basic learning |
| 200 | 512 | 102,400 | **YES** — better convergence |

---

## MA-HASO Training Loop (Full Explanation)

### 8 Phases per Round

```
Round t = 1..global_rounds
│
├─ Phase 1: HASO DECISIONS
│   ├─ obs_list = [env._get_obs() for each node]
│   │    └─ State (7-dim): [cpu, mem, energy, bw, loss, loss_std, neighbor_avail]
│   └─ actions = agent_pool.decide_all(obs_list, deterministic=False)
│        └─ PPO inference → [cut_layer, batch_size, H, target_node]
│
├─ Phase 2: Client Training (H epochs per node)
│   └─ node.train(H=actions[node].H) for H in {1,2,3,5}
│
├─ Phase 3: TVE Verification
│   └─ Committee verifies proofs, detects lazy/malicious
│
├─ Phase 4: Server Aggregation
│   └─ FedAvg of client-side model updates
│
├─ Phase 5: GTM Shapley Calculation
│   ├─ TMCSShapley.compute_shapley(node_ids, value_fn)
│   └─ → shapley_vals = {node_id: φ_i}
│
├─ Phase 6: Blockchain Commit
│   └─ ledger.record(rewards, shapley_vals)
│
├─ Phase 7: **HASO POLICY UPDATE** ← RL LEARNS HERE
│   ├─ agent_pool.update_shapley_all(shapley_vals)
│   │    └─ env._shapley_ema = 0.9*old + 0.1*φ_i
│   └─ agent_pool.learn_all(total_timesteps=256)
│        └─ 256 timesteps → 10 PPO epochs update
│
└─ Phase 8: Metrics Collection
     └─ RoundMetrics saved
```

### Reward Function (Eq. 7 from paper)

```python
# env.py:step()
T_comp = compute_time_comp(cut_layer, batch_size)    # seconds
T_comm = compute_time_comm(cut_layer, batch_size)    # seconds
delta_F = max(0.0, self._loss_ema - performance_gain)  # loss improvement

reward = -α * T_comp - β * T_comm + γ * shapley_ema * delta_F
#       = -1.0*T_comp - 0.5*T_comm + 0.1*shapley_ema*delta_F
```

**Components:**
| Term | Meaning | Effect |
|------|---------|--------|
| `-α·T_comp` | Computation time penalty | Prefers shallow cut_layer, small batch |
| `-β·T_comm` | Communication time penalty | Prefers good bandwidth, balanced cut |
| `+γ·φ·ΔF` | Accuracy improvement bonus | Prefers config that improves model most |

### Shapley Value Flow

```
GTM Phase 5
    ↓ shapley_vals = {node_id: φ_i}
    ↓
update_shapley_all()
    ↓ → env._shapley_ema = 0.9*old + 0.1*φ_i
    ↓
learn_all(total_timesteps=256)  → PPO uses new reward shaping
```

### Action Space

```python
action = [cut_layer_idx, batch_size_idx, H_idx, target_node_idx]
# cut_layer_idx ∈ {0,1,2,3} → CUT_LAYERS = [1, 2, 3, 4]
# batch_size_idx ∈ {0,1,2,3} → BATCH_SIZES = [8, 16, 32, 64]
# H_idx ∈ {0,1,2,3} → H_CHOICES = [1, 2, 3, 5]
# target_node_idx ∈ {0..n_compute_nodes-1}
```

### Memory Constraint Validation

```python
# env.py:_apply_memory_constraint()
if not profile.can_fit_cut_layer(cut_layer, memory_map):
    # Fall back to shallowest cut that fits
    cut_layer = max(cl for cl in CUT_LAYERS if profile.can_fit_cut_layer(cl))
```

---

## Known Limitations

### 1. Asynchronous Reward Shaping
Shapley value from round t only affects PPO update in round t. The training that produced that Shapley value was already done with the OLD reward shaping. This is a **1-round delay** in the RL feedback loop.

**Possible fix:** Use value function (Critic) to predict Shapley, enable credit assignment over multiple rounds.

### 2. Simulated Performance (not real model training)
`_simulate_performance()` in env.py uses a simplified model:
```python
improvement = (0.3 + 0.2*neighbor_avail) * cut_factor * batch_factor * H_factor
noise = rng.normal(0, 0.05)
simulated_loss = max(0.1, self._loss_ema - improvement + noise)
```
This does NOT run the actual ResNet-18 model. Real performance gain depends on actual data quality and gradient dynamics.

**Possible fix:** Use actual validation accuracy from protocol's global model evaluation.

### 3. Still Too Few RL Timesteps for Best Learning
256 timesteps/round × 200 rounds = 51,200 total. This is the minimum viable. For production-quality RL:
- Recommend 512 timesteps/round × 300+ rounds = 153,600+
- Or: accumulate buffer across rounds, update every 5 rounds

**Config override for stronger RL:**
```bash
python experiments/run_experiment.py --exp e1 \
    --global_rounds 300 \
    --config chainfsl/config/default.yaml
# Then manually edit default.yaml: ppo_update_timesteps: 512
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/protocol/chainfsl.py` | `_phase_haso_update()` uses `ppo_update_timesteps` config |
| `config/default.yaml` | Added `ppo_update_timesteps: 256` |
| `experiments/utils.py` | Added `ppo_update_timesteps` to `build_config()` |

---

## Verification

To verify RL is actually learning:

```python
# In a test script
from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import build_config

config = build_config(n_nodes=10, global_rounds=200, haso_enabled=True)

protocol = ChainFSLProtocol(config=config, device=None, db_path="/tmp/test_haso.db")
metrics = protocol.run(total_rounds=200)

# Check if policy improved over time
early_decisions = [m.get('cut_layer', 2) for m in metrics[:20]]
late_decisions = [m.get('cut_layer', 2) for m in metrics[-20:]]
print(f"Early avg cut_layer: {sum(early_decisions)/len(early_decisions):.2f}")
print(f"Late avg cut_layer: {sum(late_decisions)/len(late_decisions):.2f}")
# If RL is learning, should see convergence to tier-appropriate cut_layer
```
