# ChainFSL Agent Team

## Purpose
Divide and conquer the ChainFSL implementation into specialized agents that can work in parallel without context overflow.

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Coordinator (you)                    │
│  - Plans, assigns tasks, reviews results, synthesizes       │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  GradientFix │   │  HASOTVEAgent │   │ ExperimentAgent│
│    Agent     │   │  (MA-HASO +   │   │ (metrics,     │
│              │   │   TVE + GTM)  │   │  tracking,    │
│ Fix gradient │   │               │   │  crash-proof)  │
│ flow bugs    │   │ Fix MA-HASO,  │   │               │
│              │   │ TVE Algo 2,   │   │ Setup progress │
│ Fix models.py│   │ GTM value_fn │   │ tracking,      │
│ Fix split_   │   │               │   │ metrics,       │
│ model.py     │   │ Fix gossip,   │   │ F1, accuracy,  │
│              │   │ action mask,  │   │ per-node/epoch │
└──────────────┘   │ VRF formula   │   │ progress       │
                   └───────────────┘   └───────────────┘
```

## Agent Definitions

### 1. GradientFix Agent
**Priority: P0 — Crashes on any training step**
**Files owned:** `src/sfl/models.py`, `src/sfl/split_model.py`

**Triggers:**
- `AttributeError: 'ServerModel' object has no attribute 'model'` (line 329 in split_model.py)
- Autograd errors, heap corruption
- Ghost gradient approach errors

**Fixes:**
1. `split_model.py:329`: `self.model` → `self.backbone`
2. `models.py`: Implement Phase 2 fix from chainfsl-fix-plan-v2.md
3. Verify gradient flow correctness

**Verify:** `python -m experiments.run_experiment --exp e1 --n_nodes 5 --global_rounds 30`

---

### 2. HASOTVE Agent (MA-HASO + TVE + GTM)
**Priority: P0/P1 — Core functionality gaps**
**Files owned:** `src/haso/`, `src/tve/`, `src/gtm/`, `src/protocol/chainfsl.py`

**Subtasks:**

#### MA-HASO fixes:
- P0: `chainfsl.py` never calls `gossip.broadcast()` — gossip table stays empty
- P1: `env.py` temporal features missing (no moving averages over last 5 rounds)
- P1: Action masking incomplete (`get_valid_actions` logic wrong, not applied to PPO)
- P2: Reward missing Gini fairness penalty

#### TVE fixes:
- P0: `gossip.py:77` `get_best_target()` always returns `None`
- P1: `committee.py:168` historical_norms hardcoded `(1.0, 0.5)`, not tracked
- P1: Algorithm 2 Tier 1: missing cosine similarity gradient consistency check
- P1: VRF `select_committee_reputation()` threshold adjustment formula direction

#### GTM fixes:
- P0: `chainfsl.py:657-667` value_fn uses data size only, not accuracy/verification/resources
- P1: `shapley.py:88` TMCS permutations = `max(50, 3*n)` but spec says 1000
- P1: `tokenomics.py` lazy nodes still get rewards (free-riding not prevented)

**Verify:** Each sub-task verified independently

---

### 3. ExperimentAgent
**Priority: P1 — Research requires reliable experiments**
**Files owned:** `src/protocol/chainfsl.py`, `experiments/`, new metrics files

**Goals:**
1. **Crash-proof experiments** — Add try/catch, timeouts, OOM handling
2. **Progress tracking** — Per-node, per-epoch, per-round progress
3. **Full metrics** — F1, precision, recall, accuracy, per-class metrics
4. **Checkpointing** — Save/load experiment state for recovery
5. **Real-time dashboard** — Print or log progress %

**Metrics to implement:**
```python
@dataclass
class ExperimentMetrics:
    # Per-round
    round: int
    round_progress_pct: float  # 0.0 to 1.0

    # Training metrics
    train_loss: float
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float

    # Test metrics
    test_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    per_class_accuracy: Dict[int, float]

    # Per-node metrics
    per_node_loss: Dict[int, float]
    per_node_accuracy: Dict[int, float]
    per_node_progress_pct: Dict[int, float]  # epochs done / total epochs

    # System metrics
    round_latency: float
    verification_overhead_ms: float
    ledger_size_kb: float

    # Fairness
    jains_fairness: float
    gini_coefficient: float

    # TVE metrics
    attack_detection_rate: float
    false_positive_rate: float

    # GTM metrics
    total_rewards_distributed: float
    shapley_variance: float
```

**Progress tracking formula:**
```
round_progress = current_step / total_steps
per_node_progress[node_id] = epochs_trained[node_id] / total_epochs_expected
```

**Verify:** Run 100-round experiment with no crashes, metrics logged to JSON

---

## Agent Communication Protocol

When the user assigns work:

1. **Main** identifies which agent owns the task
2. **Agent** reads its owned files, makes fixes, writes test plan
3. **Agent** reports completion with verification results
4. **Main** synthesizes and reports to user

## Shared Memory (for agents)

All agents read from these sources of truth:
- `/mnt/f/ChainFSL/.claude/skills/chainfsl-architect/SKILL.md` — architecture spec
- `/mnt/f/ChainFSL/.claude/agents/chainfsl-fix-plan-v2.md` — gradient fix plan
- `/mnt/f/ChainFSL/chainfsl/src/` — source code

## Task Queue

| Task | Agent | Priority | Status |
|------|-------|----------|--------|
| Fix ServerModel AttributeError | GradientFix | P0 | Pending |
| Fix gradient flow (plan v2) | GradientFix | P0 | Pending |
| Fix gossip never broadcast | HASOTVE | P0 | Pending |
| Fix value_fn (Shapley) | HASOTVE | P0 | Pending |
| Fix get_best_target returns None | HASOTVE | P0 | Pending |
| Add temporal state features | HASOTVE | P1 | Pending |
| Add action masking to PPO | HASOTVE | P1 | Pending |
| TVE cosine similarity (Algo 2) | HASOTVE | P1 | Pending |
| TVE historical norms tracking | HASOTVE | P1 | Pending |
| VRF reputation formula fix | HASOTVE | P1 | Pending |
| TMCS permutations (50→1000) | HASOTVE | P1 | Pending |
| Lazy node free-riding fix | HASOTVE | P1 | Pending |
| Fairness penalty in reward | HASOTVE | P2 | Pending |
| Tier-layer capability mapping | HASOTVE | P2 | Pending |
| Experiment crash-proofing | ExperimentAgent | P1 | Pending |
| Progress tracking system | ExperimentAgent | P1 | Pending |
| Full metrics (F1, etc.) | ExperimentAgent | P1 | Pending |
| Checkpointing system | ExperimentAgent | P2 | Pending |
