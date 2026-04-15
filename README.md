# ChainFSL

**ChainFSL: Blockchain-Enhanced Federated Split Learning**

A simulation framework for federated split learning in IoT environments, integrating tiered verification, game-theoretic incentive mechanisms, and a lightweight blockchain ledger.

---

## Overview

ChainFSL is a research framework for studying federated split learning (SFL) across heterogeneous IoT devices. It provides:

- **Split Federated Learning**: Neural network computation is split between clients and a server at configurable cut layers, reducing client-side compute overhead.
- **Heterogeneous Node Support**: Devices with varying hardware capabilities (CPU, RAM, energy, bandwidth) are modeled as tiers, with split point decisions adapted to each tier.
- **Tiered Verification**: A verification engine generates and validates proofs of correct computation, with tier-dependent proof complexity.
- **Incentive Mechanism**: Shapley value-based reward distribution quantifies each node's marginal contribution to the global model.
- **Blockchain Ledger**: A SQLite-based merkleized ledger records verification proofs and reward distributions for auditability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Coordinator                             │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌──────┐  ┌────────────┐      │
│  │  HASO   │  │ Trainer  │  │ TVE  │  │    GTM      │      │
│  │ (PPO)   │  │          │  │      │  │  Shapley    │      │
│  └────┬────┘  └────┬─────┘  └──┬───┘  └──────┬─────┘      │
│       │            │           │              │             │
└───────┼────────────┼───────────┼──────────────┼─────────────┘
        │            │           │              │
   ┌────▼────┐  ┌────▼─────┐  ┌─▼───┐  ┌──────▼─────┐
   │ Clients │  │  Server  │  │ Verif│  │  Blockchain │
   │(T1-T4)  │◄─┤          │◄─┤Committee│  │   Ledger    │
   └─────────┘  └──────────┘  └──────┘  └────────────┘
```

### Core Modules

| Module | Location | Description |
|--------|----------|-------------|
| **Protocol** | `src/protocol/chainfsl.py` | End-to-end orchestrator (Algorithm 2) |
| **SFL** | `src/sfl/` | Split model, data loader, aggregator, trainer |
| **HASO** | `src/haso/` | DRL-based split optimization (PPO agent + Gymnasium env) |
| **TVE** | `src/tve/` | Tiered verification engine (VRF, commitment, committee) |
| **GTM** | `src/gtm/` | Game-theoretic tokenomics (Shapley, reward distribution) |
| **Blockchain** | `src/blockchain/` | SQLite ledger with Merkle root commits |
| **Emulator** | `src/emulator/` | Hardware profiles, tier factory, network emulation |

---

## Installation

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn gymnasium stable-baselines3 pyyaml
```

Python 3.10+ recommended.

---

## Quick Start

```python
from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import build_config

# Build default configuration
config = build_config(n_nodes=10, global_rounds=30)

# Initialize protocol
protocol = ChainFSLProtocol(config=config, device=None, db_path="/tmp/chainfsl.db")

# Run training
metrics = protocol.run(total_rounds=30, eval_every=5)

# Access results
for m in metrics:
    print(f"Round {m.round}: acc={m.test_acc:.2f}%, latency={m.round_latency:.3f}s")
```

---

## Running Experiments

Experiments are organized in `experiments/`:

```bash
# Run a specific experiment
python -m experiments.run_experiment --exp e1 --n_nodes 10 --global_rounds 30

# Run all experiments
python -m experiments.run_experiment --exp all
```

| Exp | Description |
|-----|-------------|
| E1 | HASO effectiveness vs. baselines (SplitFed, FedAvg) |
| E2 | Scalability with varying node counts |
| E3 | Non-IID data robustness (Dirichlet partitions) |
| E4 | Security evaluation (lazy client attack detection) |
| E5 | Incentive mechanism (Nash equilibrium, Sybil profitability) |
| E6 | Ablation study (HASO, TVE, GTM contributions) |
| E7 | Blockchain ledger overhead measurement |

Results are saved as CSV files in `--log_dir` (default `./logs`).

### Generating Plots

```bash
python -m analysis.plot_results --exp e1 --log_dir ./logs
python -m analysis.plot_results --log_dir ./logs   # all plots
```

---

## Baseline Comparisons

Four baseline methods are implemented in `baselines/`:

| Baseline | Description |
|----------|-------------|
| **FedAvg** | Standard federated averaging, no split learning |
| **SplitFed** | Split learning with fixed uniform cut layer |
| **AdaptSFL** | Adaptive cut layer selection by node tier |
| **DFL** | Dynamic cut layer per client with energy/loss heuristics |

---

## Configuration

Key parameters (see `config/default.yaml`):

```yaml
# Network
n_nodes: 50
tier_distribution: [0.1, 0.3, 0.4, 0.2]  # Tier proportions

# Training
global_rounds: 100
batch_size_default: 32
dirichlet_alpha: 0.5

# HASO (RL agent)
haso_enabled: true
ppo_learning_rate: 3e-4
ppo_n_steps: 512
ppo_update_timesteps: 256

# TVE
tve_enabled: true
committee_size: 5

# GTM
gtm_enabled: true
shapley_M: 50
reward_total_init: 1000.0
```

---

## Project Structure

```
chainfsl/
├── src/
│   ├── protocol/       # Main ChainFSL protocol orchestrator
│   ├── sfl/            # Split federated learning (models, trainer, aggregator)
│   ├── haso/           # MA-HASO DRL environment and PPO agent
│   ├── tve/            # Tiered verification engine (VRF, committee)
│   ├── gtm/            # Game-theoretic module (Shapley, tokenomics)
│   ├── blockchain/     # Blockchain ledger (SQLite, Merkle root)
│   └── emulator/       # Node profiles, tier factory, network emulator
├── baselines/          # FedAvg, SplitFed, AdaptSFL, DFL implementations
├── experiments/       # E1–E7 experiment definitions and runner
├── analysis/          # Plotting tools for experiment results
├── config/            # YAML configuration files
│   ├── default.yaml
│   └── experiment_configs/
└── note/              # Development notes and issue tracking
```

---

## Limitations

This is a **simulation framework**. Several simplifications are made:

1. **Simulated training**: Client model training uses simplified performance models rather than full neural network forward-backward passes for speed.
2. **Single-machine execution**: Distributed communication is simulated via in-memory queues; no real network traffic.
3. **Mock VRF**: Verification randomness uses HMAC-SHA256 rather than cryptographic VRF signatures.
4. **SQLite ledger**: The blockchain is a local mock; no P2P consensus or distributed validation.
5. **Shapley approximation**: TMC-Shapley with limited permutations is used; exact Shapley values are not computed.

Performance numbers reported by this framework should be validated on real distributed deployments before publication.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@unpublished{chainfsl2026,
  title={ChainFSL: Blockchain-Enhanced Federated Split Learning for IoT},
  author={[Author Names]},
  year={2026},
  note={Technical Report / Source code: https://github.com/...}
}
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 [Authors]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
