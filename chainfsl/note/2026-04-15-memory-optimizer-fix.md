# Note: Memory Estimation Bug — Missing Optimizer State

**Date:** 2026-04-15
**Severity:** High
**Issue:** Memory estimates did NOT include Adam optimizer state, causing OOM on constrained devices
**Files affected:** `src/protocol/chainfsl.py`, `src/haso/env.py`

---

## Root Cause

### Two Different Memory Maps

```python
# sfl/models.py
# Memory WITHOUT optimizer — activations + gradients only
MEMORY_ESTIMATES_MB = {1: 150, 2: 300, 3: 500, 4: 700}  # MB, batch=32

# Memory WITH Adam optimizer state — activations + gradients + 2x gradients (momentum + variance)
MEMORY_WITH_ADAM_MB = {1: 450, 2: 900, 3: 1500, 4: 2100}  # MB, batch=32
```

**Critical gap:** Adam stores 2 copies of gradients (momentum + variance) = **3x gradient memory** vs. no-optimizer case.

### Which Devices Were Affected

| Tier | RAM | MEMORY_ESTIMATES (wrong) | MEMORY_WITH_ADAM (correct) | Status |
|------|-----|--------------------------|---------------------------|--------|
| Tier 1 | 8192 MB | 700 MB fits ✓ | 2100 MB fits ✓ | OK — all cuts |
| Tier 2 | 4096 MB | 700 MB fits ✓ | 2100 MB fits ✓ | OK — all cuts |
| Tier 3 | 512 MB | 300 MB fits ✓ | **450 MB fits ✓ (cut=1 only)** | cut=2+ OOM |
| Tier 4 | 200 MB | 150 MB fits ✓ | **450 MB — all cut layers OOM** | **Critical — excluded** |

**Problem:** `Tier 4` was believed to fit cut_layer=1 under the old estimate. In reality with Adam optimizer:
- Tier 4: deepest valid cut = **none** (450 MB > 200 MB available) → **permanently excluded**
- Tier 3: deepest valid cut = **only cut=1** (450 MB ≤ 512 MB) → can only use shallow cut

> **Note:** Tier 2 can actually fit ALL cut layers (2100 MB < 4096 MB). The original note had an error claiming Tier 2 could only do cut=2. Corrected values above.

---

## Old Behavior (Bug)

### chainfsl.py

```python
# BEFORE (line 351, 398)
memory_map = SplittableResNet18.MEMORY_ESTIMATES_MB  # NO optimizer state!

if not n.can_fit_cut_layer(cut_layer, memory_map):
    for cl in sorted(memory_map.keys(), reverse=True):
        if n.can_fit_cut_layer(cl, memory_map):
            cut_layer = cl
            break
    else:
        cut_layer = 1  # Fallback always to cut=1 — but this can still OOM!
```

**Tier 4 fallback chain:** Requested cut=4 → can't fit → fallback cut=1 → `can_fit_cut_layer(1, {1:150})` → TRUE → **runs at cut=1** → but actual MEMORY_WITH_ADAM is 450 MB → **OOM crash in practice**.

### haso/env.py

```python
# BEFORE — used node_profile.RESNET18_MEMORY_MAP (same as MEMORY_ESTIMATES_MB)
self.memory_map = memory_map or RESNET18_MEMORY_MAP  # wrong map!
```

RL agent learns policies assuming cut_layers fit in memory based on wrong estimates.

---

## Fix Applied

### 1. `src/protocol/chainfsl.py`

**Added helper method:**
```python
def _find_deepest_valid_cut_layer(
    self,
    node: HardwareProfile,
    memory_map: dict[int, float],
) -> Optional[int]:
    """Find deepest cut_layer that fits node RAM, including Adam optimizer state."""
    for cl in sorted(memory_map.keys(), reverse=True):
        if node.can_fit_cut_layer(cl, memory_map):
            return cl
    return None  # No valid cut layer — exclude node
```

**Updated `_phase_haso()` (line ~372):**
```python
# Enforce tier memory constraint (includes optimizer state for training)
memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB
valid_cut = self._find_deepest_valid_cut_layer(n, memory_map)

if valid_cut is None:
    # Node cannot fit any cut layer — exclude from this round
    configs[node_id] = None
    continue

cut_layer = valid_cut  # Clamp to deepest valid
```

**Updated `_phase_training()` (line ~420):**
```python
cfg = configs.get(node.node_id)
if cfg is None:
    return None  # Node was excluded (OOM)

# Final safety check against MEMORY_WITH_ADAM_MB
if not node.can_fit_cut_layer(cut_layer, SplittableResNet18.MEMORY_WITH_ADAM_MB):
    return None  # Skip this node — cannot train safely
```

**Return type changed:** `_phase_haso()` now returns `Dict[int, Optional[Dict[str, Any]]]` — `None` value means node excluded.

### 2. `src/haso/env.py`

**Updated `_apply_memory_constraint()`:**
```python
def _find_deepest_valid_cut_layer(self, memory_map: dict) -> Optional[int]:
    for cl in sorted(memory_map.keys(), reverse=True):
        if self.profile.can_fit_cut_layer(cl, memory_map):
            return cl
    return None

def _apply_memory_constraint(self, cut_layer: int, batch_size: int) -> Tuple[int, int]:
    memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB
    valid_cut = self._find_deepest_valid_cut_layer(memory_map)
    if valid_cut is None:
        return cut_layer, batch_size  # Let step() handle OOM case
    ...
```

**Added OOM handling in `step()`:**
```python
valid_cut = self._find_deepest_valid_cut_layer(memory_map)
if valid_cut is None:
    terminated = True
    reward = -100.0  # Large penalty for OOM condition
    info = {..., "oom": True, "error": "Node cannot fit any cut layer"}
    return self._get_obs(), float(reward), terminated, False, info
```

**Updated `get_valid_actions()`:** now uses `MEMORY_WITH_ADAM_MB`.

---

## New Tier Behavior After Fix

| Tier | RAM | Deepest Valid Cut | Notes |
|------|-----|-------------------|-------|
| Tier 1 (GPU) | 8192 MB | cut_layer=4 (2100 MB) | Full flexibility |
| Tier 2 (CPU) | 4096 MB | cut_layer=4 (2100 MB) | All cuts fit — can do deep splits |
| Tier 3 (IoT) | 512 MB | cut_layer=1 (450 MB) | Only shallow cut; cut=2+ OOM |
| Tier 4 (Minimal) | 200 MB | **None** | **Excluded from all training rounds** |

**Impact:** With default distribution `[0.1, 0.3, 0.4, 0.2]`:
- ~20% of nodes (Tier 4) are **permanently excluded** from training
- ~40% of nodes (Tier 3) are **restricted to cut=1 only**
- Only Tier 1 + Tier 2 have full flexibility

This means HASO's adaptive cut selection only matters for ~50% of nodes (Tier 1+2).

---

## Implications for Experiments

### E1 (HASO Effectiveness)
- With 50 nodes, default distribution: only ~20 nodes participate (Tier 1 + Tier 2)
- "HASO" advantage may disappear or reverse — fewer nodes means less diversity benefit

### E2 (Scalability)
- Effective n_nodes is much smaller than configured n_nodes
- Scalability curves shift significantly

### E3 (Non-IID)
- Same issue — only Tier 1+2 participate

### E4 (Security)
- Attack detection still works for participating nodes

### E6 (Ablation)
- Ablation results unchanged

### Possible Fixes for Tier 4 Exclusion:

**Option A: Increase Tier 4 RAM (recommended for realistic IoT)**
```yaml
# In config/default.yaml or TierFactory
Tier 4 ram_mb: 1024  # instead of 200 — realistic for edge gateway
```
With 1024 MB: cut=1 (450 MB) and cut=2 (900 MB) would both fit.

**Option B: Use SGD instead of Adam (reduces memory by ~33%)**
```python
# SGD: only stores gradients (1x), not momentum+variance
# Cut=1: ~300 MB, Cut=2: ~600 MB
```
Tier 4 with 200 MB still can't fit even cut=1 (300 MB > 200 MB).

**Option C: Reduce batch size for constrained tiers**
```python
# Tier 3-4 use batch_size=8 instead of 32
# Activation memory scales with batch_size
# With batch=8: cut=1 ~ 112 MB, cut=2 ~ 225 MB — Tier 4 fits cut=1!
```

**Option D (Current):** Accept Tier 4 exclusion. In real deployments, very constrained devices (200 MB RAM) cannot run any meaningful model split. They participate in TVE verification and GTM consensus only — not training. This is architecturally valid.

**Recommended:** Option C (per-tier batch size) + Option D (document exclusion).

---

## Files Modified

| File | Change |
|------|--------|
| `src/protocol/chainfsl.py` | `_find_deepest_valid_cut_layer()` helper; `_phase_haso()` returns None for OOM nodes; `_phase_training()` skips excluded nodes; uses `MEMORY_WITH_ADAM_MB` |
| `src/haso/env.py` | Imports `SplittableResNet18`; `_find_deepest_valid_cut_layer()`; `_apply_memory_constraint()` fixed; `step()` OOM penalty; `get_valid_actions()` uses `MEMORY_WITH_ADAM_MB` |

---

## Verification

```python
from src.emulator.node_profile import TIER_CONFIGS
from src.sfl.models import SplittableResNet18

for tier, cfg in TIER_CONFIGS.items():
    ram = cfg["ram_mb"]
    print(f"Tier {tier} ({ram} MB RAM):")
    for cl in sorted(SplittableResNet18.MEMORY_WITH_ADAM_MB.keys(), reverse=True):
        mem_needed = SplittableResNet18.MEMORY_WITH_ADAM_MB[cl]
        fits = "✓" if mem_needed <= ram else "✗ OOM"
        print(f"  cut_layer={cl}: {mem_needed} MB — {fits}")
    print()
```
