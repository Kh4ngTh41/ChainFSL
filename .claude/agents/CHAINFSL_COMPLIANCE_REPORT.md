# ChainFSL Full Compliance Report & Implementation Plan

## Executive Summary

**Total Issues Found: 23**
- P0 (Crashes): 5
- P1 (Functional gaps): 11
- P2 (Minor bugs): 7

**Estimated Total Effort:**
- Quick fixes (1-2h): 8 issues
- Medium changes (half-day): 10 issues
- Large refactors (1-2 days): 5 issues

**Critical Path (blocks experiments):**
```
ServerModel AttributeError (P0)
  → Gradient flow bugs (P0)
    → Gossip never broadcast (P0)
      → Shapley value_fn stub (P0)
```

---

## P0 — Immediate Crashes (Fix First)

### P0-1: ServerModel AttributeError
**File:** `src/sfl/split_model.py`, line 329
**Bug:** `self.model` should be `self.backbone`

```python
# Current (crashes):
output = self.model(smashed_input)

# Fix:
output = self.backbone(smashed_input)
```

**Effort:** 5 minutes
**Verify:** Run single training step

---

### P0-2: Gossip Never Broadcast
**File:** `src/protocol/chainfsl.py`
**Bug:** `GossipProtocol` instantiated but `broadcast()` never called. Table stays empty.

**Fix:** Add broadcast calls in `_phase_training()`:
```python
# After each node completes training, broadcast its LRH
self.gossip.broadcast(node.node_id, {
    "flops_ratio": node.flops_ratio,
    "ram_mb": node.ram_mb,
    "bandwidth_mbps": node.bandwidth_mbps,
    "reputation": node.reputation,
    "load": current_load,
    "round": self.current_round,
})
```

**Effort:** 30 minutes
**Verify:** Check gossip table not empty after first round

---

### P0-3: Shapley Value Function Stub
**File:** `src/protocol/chainfsl.py`, lines 657-667
**Bug:** Only uses data size as proxy. Spec requires: accuracy + verification quality + resource provision

**Fix:** Replace with multi-component value function:
```python
def value_fn(coalition: List[int]) -> float:
    if not coalition:
        return 0.0

    # Component 1: Data size
    total_data = sum(self.train_loaders[nid].dataset.__len__()
                     for nid in coalition if nid in self.train_loaders)
    data_component = total_data / 50000.0

    # Component 2: Verification quality (avg valid rate)
    verif_rates = [self.verification_rates.get(nid, 0.5) for nid in coalition]
    verif_component = np.mean(verif_rates) if verif_rates else 0.0

    # Component 3: Resource provision (normalized compute)
    compute_component = sum(
        self.nodes[nid].flops_ratio for nid in coalition
    ) / (len(self.nodes) * 1.0)

    # Weighted combination
    return 0.5 * data_component + 0.3 * verif_component + 0.2 * compute_component
```

**Effort:** 1 hour
**Verify:** Shapley values vary with verification quality

---

### P0-4: get_best_target Always Returns None
**File:** `src/haso/gossip.py`, line 77
**Bug:** Always returns `None`

**Fix:** Implement proper neighbor ranking:
```python
def get_best_target(self, node_id: int, exclude_self: bool = True, k: int = 5) -> Optional[int]:
    neighbors = self._protocol.get_neighbors(node_id, k=k)
    if not neighbors:
        return None

    # Track node_ids along with their info
    all_nodes = self._protocol._table  # {node_id: lrh}
    candidates = [(nid, info) for nid, info in all_nodes.items()
                   if nid != node_id or not exclude_self]

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x[1].get("reputation", 0.0))
    return best[0]
```

**Effort:** 30 minutes
**Verify:** `get_best_target()` returns valid node_id

---

### P0-5: Gradient Flow (from plan v2)
**File:** `src/sfl/models.py`
**Status:** Already partially fixed in plan v2. Verify `ClientModel.backward()` and `ServerModel.forward_backward()` are correct per the plan.

**Fix:** Apply Phase 2 and Phase 3 from `chainfsl-fix-plan-v2.md`

**Effort:** 2 hours
**Verify:** 30-round training without crash

---

## P1 — Functional Gaps (Fix Before Experiments)

### P1-1: Temporal State Features Missing
**File:** `src/haso/env.py`, lines 100-104, 222-243
**Bug:** No moving averages over last 5 rounds

**Fix:** Add rolling window tracking:
```python
# In SFLNodeEnv.__init__:
self._loss_history: deque = deque(maxlen=5)
self._shapley_history: deque = deque(maxlen=5)
self._T_comp_history: deque = deque(maxlen=5)
self._T_comm_history: deque = deque(maxlen=5)

# In step(), append to history
self._loss_history.append(performance_gain)
# ... etc

# In _get_obs(), include temporal features:
loss_trend = (self._loss_ema - (sum(self._loss_history)/len(self._loss_history))) if len(self._loss_history) > 1 else 0.0
shapley_trend = self._shapley_ema - (sum(self._shapley_history)/len(self._shapley_history)) if len(self._shapley_history) > 1 else 0.0
```

**Effort:** 2 hours
**Verify:** State vector has temporal features

---

### P1-2: Action Masking Not Applied to PPO
**File:** `src/haso/agent.py`, `src/haso/env.py`
**Bug:** `get_valid_actions()` exists but PPO doesn't use it

**Fix:** Use Stable-Baselines3's `Maskableppo` with `prepare_mask()`:
```python
# In agent.py:
from stable_baselines3.common.callbacks import BaseCallback

class MaskablePPOPolicy(nn.Module):
    # ... custom policy that applies action mask

# In env.py, return invalid action mask alongside observation
def step(self, action):
    mask = self._get_action_mask()  # [4,] bool mask
    return obs, reward, terminated, truncated, {**info, "action_mask": mask}
```

Or use `stable_baselines3.common.maskable.MaskablePPO` with `venv` wrapper.

**Effort:** 3 hours
**Verify:** PPO never selects invalid cut_layer for tier

---

### P1-3: TVE Tier 1 Missing Cosine Similarity
**File:** `src/tve/commitment.py`, lines 226-238
**Bug:** `VerifyGradientConsistency` (cosine similarity > 0.95) not implemented

**Fix:**
```python
@staticmethod
def verify_gradient_consistency(
    submitted_grad: torch.Tensor,
    recomputed_grad: torch.Tensor,
    threshold: float = 0.95,
) -> bool:
    """Verify submitted gradient matches recomputed via cosine similarity."""
    cos = nn.CosineSimilarity(dim=0)
    similarity = cos(submitted_grad.flatten(), recomputed_grad.flatten()).item()
    return similarity > threshold
```

**Effort:** 1 hour
**Verify:** Tier 1 verification rejects mismatched gradients

---

### P1-4: TVE Historical Norms Hardcoded
**File:** `src/tve/committee.py`, line 168
**Bug:** `historical_norms=(1.0, 0.5)` hardcoded, not tracked

**Fix:** Use `TieredVerificationEngine.update_historical_stats()` — call it from protocol:
```python
# In chainfsl.py _phase_verification():
for update in updates:
    self.tve.update_historical_stats(
        update["node_id"],
        update.get("gradient_norm", 1.0),
    )
```

**Effort:** 30 minutes
**Verify:** Gradient norm bounds adapt over rounds

---

### P1-5: VRF Reputation Formula Direction
**File:** `src/tve/vrf.py`, line 207
**Bug:** Threshold adjustment direction may be inverted

**Paper Eq. 11:** Higher reputation → lower threshold → more likely selected

**Current code:**
```python
rep_threshold = threshold / max(1e-9, 1 + omega * math.tanh(rep))
```

This DOES lower threshold for higher rep (dividing by >1). But the tanh direction might be wrong. Spec says `tanh(δ * s_i)` where s_i ∈ [-1, 1]. If reputation is [0, 1], tanh would be small.

**Fix:** Verify formula matches paper and adjust if needed:
```python
# Higher reputation should make threshold lower (more likely to be selected)
# threshold_i = (K/N) * max(ε, 1 - ω * tanh(delta * (rep - 0.5)))
rep_threshold = threshold * max(0.1, 1 - omega * math.tanh(delta * (rep - 0.5)))
```

**Effort:** 1 hour
**Verify:** Higher reputation nodes more likely in committee

---

### P1-6: TMCS Permutations Too Few
**File:** `src/gtm/shapley.py`, line 88
**Bug:** Uses `max(50, 3*n)` but spec says 1000 permutations

**Fix:**
```python
M_effective = max(1000, 3 * n)  # Spec: 1000 for Monte Carlo
```

**Effort:** 5 minutes
**Verify:** Shapley computation uses 1000+ permutations

---

### P1-7: Lazy Node Free-Riding Not Prevented
**File:** `src/gtm/tokenomics.py`, lines 188-197
**Bug:** Nodes with φᵢ < 0.01 × max(φ) still receive rewards

**Fix:**
```python
# In distribute():
min_shapley = 0.01 * max_phi
for node_id, phi in shapley_values.items():
    if phi < min_shapley:
        rewards[node_id] = 0.0  # Exclude from distribution
        continue
    # ... rest of distribution
```

**Effort:** 30 minutes
**Verify:** Low-Shapley nodes receive zero rewards

---

### P1-8: Gossip Broadcast Placement
**File:** `src/protocol/chainfsl.py`
**Bug:** Need to broadcast LRH after each node's training step

**Fix:** Add in `_phase_training()` after each node completes:
```python
# In train_node() lambda:
after training:
    lrh = {
        "flops_ratio": node.flops_ratio,
        "ram_mb": node.ram_mb,
        "bandwidth_mbps": node.bandwidth_mbps,
        "reputation": node.reputation,
        "load": comp_load,
        "round": self.current_round,
    }
    self.gossip.broadcast(node.node_id, lrh)
```

**Effort:** 30 minutes (already covered by P0-2)

---

### P1-9: Protocol Phase Count
**File:** `src/protocol/chainfsl.py`, lines 294-344
**Bug:** 8 phases vs spec's 9 (TVE proof generation should be separate phase)

**Fix:** Rename/structure comments to match spec:
1. HASO decisions
2. Training (forward to cut)
3. Server backward
4. Client update
5. TVE proof generation ← Make explicit
6. TVE verification
7. Aggregation
8. GTM rewards
9. Blockchain commit
10. HASO policy update

**Effort:** 1 hour (commentary only)
**Verify:** Phase comments match spec

---

### P1-10: TVE Tier 3 Deferred Audit Not Async
**File:** `src/tve/committee.py`
**Bug:** Tier 3 proofs marked valid immediately, not deferred to Tier 1 audit

**Fix:** Add audit queue and async processing:
```python
# In TieredVerificationEngine:
self._audit_queue: List[Proof] = []

# When verifying Tier 3 proof:
if tier == 3:
    self._audit_queue.append(proof)  # Queue for async audit
    return VerificationResult(is_valid=True, ...)  # Deferred

# Add audit method called periodically:
def process_audit_queue(self):
    # Tier 1 nodes audit Tier 3 proofs from queue
    for proof in self._audit_queue:
        # Verify with full Tier 1 logic
        pass
    self._audit_queue.clear()
```

**Effort:** 2 hours
**Verify:** Tier 3 proofs audited asynchronously

---

### P1-11: Shapley Value Function Components
**File:** `src/gtm/contribution.py`
**Bug:** `ContributionVector` and `VLIComputer` exist but unused

**Fix:** Integrate into Shapley value_fn:
```python
from ..gtm.contribution import VLIComputer

# In chainfsl.py _phase_gtm():
vli = VLIComputer()
for nid in node_ids:
    vli.accumulate(
        accuracy_delta=self.node_losses[nid],
        verif_rate=self.verification_rates.get(nid, 1.0),
        resourceProvision=self.nodes[nid].flops_ratio,
    )
vli_values = vli.compute()
```

**Effort:** 2 hours
**Verify:** Shapley values reflect all three components

---

## P2 — Minor Bugs (Fix After Core Works)

### P2-1: Fairness Penalty Not in Reward
**File:** `src/haso/reward.py`, `src/haso/env.py`
**Bug:** Gini coefficient exists but unused

**Fix:** Add to reward computation:
```python
# In env.py step():
all_rewards = [self.node_progress[nid].last_reward
               for nid in self.node_progress if nid in active_nodes]
fairness_bonus = FairnessPenalty.fairness_bonus(all_rewards, nu=0.2)
reward += fairness_bonus
```

**Effort:** 1 hour
**Verify:** Reward includes fairness term

---

### P2-2: Action Mask Logic Wrong
**File:** `src/haso/env.py`, lines 365-382
**Bug:** `mask[0] = False` for ALL cut layers if ANY fails, and breaks early

**Fix:**
```python
def get_valid_actions(self) -> np.ndarray:
    """Return [4,] bool mask: [cut_layer_valid, batch_valid, H_valid, target_valid]"""
    mask = np.array([True, True, True, True], dtype=bool)
    memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB

    # Cut layer: each choice individually valid/invalid
    for i, cl in enumerate(self.CUT_LAYERS):
        if not self.profile.can_fit_cut_layer(cl, memory_map):
            mask[0] = False  # This specific cut layer invalid

    # Batch size: check each
    for i, bs in enumerate(self.BATCH_SIZES):
        required = self._estimate_activation_mb(self.CUT_LAYERS[0], bs)
        if required > self.profile.ram_mb * 0.5:
            mask[1] = False

    return mask
```

**Effort:** 1 hour
**Verify:** Invalid actions masked correctly

---

### P2-3: Proof Type Hardcoded
**File:** `src/protocol/chainfsl.py`, line 714
**Bug:** Always `proof_type="zk_mock"`

**Fix:**
```python
# Use actual proof type from VerificationResult
proof_type = verif_result.proof_type if hasattr(verif_result, 'proof_type') else "unknown"
self.ledger.record_verification(
    epoch=epoch,
    node_id=node_id,
    is_valid=is_valid,
    penalty=penalty,
    proof_type=proof_type,  # Use actual type
)
```

**Effort:** 15 minutes
**Verify:** Proof types vary by tier

---

### P2-4: Gradient Accumulation in backward
**File:** `src/sfl/models.py`, lines 258-261
**Bug:** Uses `p.grad.add_()` which may cause double-gradient

**Fix:** Should be `p.grad = g` (replace, not accumulate):
```python
# ClientModel.backward() after autograd.backward:
for p in self.backbone.parameters():
    if p.grad is not None:
        p.grad.data.zero_()  # Clear before assign
    p.grad = g  # Assign (not add)
```

**Effort:** 30 minutes
**Verify:** Gradients correct, no doubling

---

### P2-5: cut_layer=0 Edge Case
**File:** `src/sfl/models.py`, lines 81-116
**Bug:** Behavior unclear for cut_layer=0

**Fix:** Document and handle explicitly:
```python
def get_client_model(self, cut_layer: int) -> nn.Sequential:
    if cut_layer == 0:
        # Full model stays on client
        return nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                           self.layer1, self.layer2, self.layer3, self.layer4,
                           self.avgpool, nn.Flatten(), self.fc)
```

**Effort:** 30 minutes
**Verify:** cut_layer=0 works without error

---

### P2-6: record_rewards_batch Unused
**File:** `src/blockchain/ledger.py`, `src/protocol/chainfsl.py`
**Bug:** Batch API defined but not used

**Fix:** Replace individual calls with batch:
```python
# In chainfsl.py _phase_blockchain():
self.ledger.record_rewards_batch(epoch, rewards, shapley_vals)
self.ledger.record_verifications_batch(epoch, verif_results)
```

**Effort:** 15 minutes
**Verify:** Batch calls work correctly

---

### P2-7: Tier 4 Delegation Not Implemented
**File:** `src/tve/`
**Bug:** No Tier 1 re-compute for Tier 4 updates

**Fix:** Add delegation mechanism:
```python
# In TieredVerificationEngine:
def delegate_tier4_to_tier1(self, proof: Proof, node_id: int):
    """Tier 1 re-computes Tier 4 update for audit."""
    # Fetch original input
    # Re-compute forward pass
    # Compare with submitted
    pass
```

**Effort:** 2 hours
**Verify:** Tier 4 updates audited by Tier 1

---

## Effort Summary

| Priority | Count | Total Effort |
|----------|-------|--------------|
| P0 | 5 | 4.5 hours |
| P1 | 11 | 15 hours |
| P2 | 7 | 7 hours |
| **Total** | **23** | **~26.5 hours** |

### Recommended Order

**Week 1 (Day 1-2):** P0 fixes — get experiments running
**Week 1 (Day 3-4):** P1 fixes — functional completeness
**Week 2 (Day 1):** P2 fixes — polish
**Week 2 (Day 2-3):** ExperimentAgent — metrics, tracking, crash-proofing

---

## Experiment Requirements

### Crash-Proof Requirements
- [ ] All P0 bugs fixed
- [ ] Timeout on each training step (30s max)
- [ ] OOM handling with graceful degradation
- [ ] Try/catch around each phase
- [ ] Graceful exit with state save on KeyboardInterrupt

### Progress Tracking
- [ ] Per-round: X/Total rounds (e.g., "Round 45/100 = 45%")
- [ ] Per-node: epochs_trained / total_epochs
- [ ] Per-epoch: batch_index / total_batches
- [ ] ETA based on rolling average round time

### Metrics (per round)
```json
{
  "round": 45,
  "progress_pct": 45.0,
  "train": {
    "loss": 1.234,
    "accuracy": 0.65,
    "precision": 0.63,
    "recall": 0.62,
    "f1": 0.625,
    "per_class": {"0": 0.5, "1": 0.7, ...}
  },
  "test": {
    "loss": 1.456,
    "accuracy": 0.58,
    "precision": 0.56,
    "recall": 0.55,
    "f1": 0.555,
    "per_class": {"0": 0.45, "1": 0.65, ...}
  },
  "per_node": {
    "0": {"loss": 1.2, "acc": 0.66, "progress_pct": 100.0},
    "1": {"loss": 1.3, "acc": 0.64, "progress_pct": 80.0},
    ...
  },
  "system": {
    "round_latency_s": 12.5,
    "verification_overhead_ms": 234.5,
    "ledger_size_kb": 456.7
  },
  "fairness": {
    "jains_index": 0.85,
    "gini": 0.15
  },
  "tve": {
    "attack_detection_rate": 0.95,
    "false_positive_rate": 0.05
  },
  "gtm": {
    "total_rewards": 500.0,
    "shapley_variance": 0.12
  }
}
```

### Checkpointing
- Save every 10 rounds: model state + metrics + node states
- Resume from checkpoint on crash
- Final save on completion
