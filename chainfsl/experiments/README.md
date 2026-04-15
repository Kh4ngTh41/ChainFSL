# ChainFSL Experiments

Experiment suite for ChainFSL federated split learning protocol.

## Requirements

Install dependencies (from project root):

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
```

## Running Experiments

### Run a single experiment

```bash
python experiments/run_experiment.py --exp e1
python experiments/run_experiment.py --exp e2
python experiments/run_experiment.py --exp e3
python experiments/run_experiment.py --exp e4
python experiments/run_experiment.py --exp e5
python experiments/run_experiment.py --exp e6
python experiments/run_experiment.py --exp e7
```

### Run all experiments

```bash
python experiments/run_experiment.py --exp all
```

### Override defaults

```bash
python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50 --alpha 0.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--exp` | required | Experiment ID (e1–e7 or all) |
| `--n_nodes` | 10 | Number of client nodes |
| `--global_rounds` | 30 | Number of federated rounds |
| `--alpha` | 0.5 | Dirichlet alpha for non-IID data (E3) |
| `--lazy_fraction` | 0.0 | Fraction of lazy/malicious clients (E4) |
| `--seed` | 42 | Random seed |
| `--log_dir` | `./logs` | Output directory for CSV files |

## Experiments Overview

### E1: HASO Effectiveness
Compares ChainFSL (full), ChainFSL-noHASO, SplitFed, and FedAvg.

```
Hypothesis: HASO improves accuracy by >5% over baselines.
```
**Metrics:** test_acc, round_latency, fairness_index

### E2: Scalability
Tests scalability with varying node counts: 5, 10, 20, 50.

```
Hypothesis: Latency grows sub-linearly with number of nodes.
```
**Metrics:** round_latency, test_acc, throughput (nodes/s)

### E3: Non-IID Robustness
Tests robustness across Dirichlet non-IID data partitions.

```
Hypothesis: ChainFSL outperforms baselines under high data heterogeneity.
```
**Metrics:** test_acc, fairness_index per alpha (0.1, 0.5, 1.0, 10.0)

### E4: Security (Lazy Client Attack)
Evaluates attack detection under lazy client fractions: 0.0, 0.1, 0.2, 0.3.

```
Hypothesis: TVE detects >95% of lazy clients at <=20% fraction.
```
**Metrics:** attack_detection_rate, test_acc, round_latency

### E5: Incentive Mechanism
Validates Nash equilibrium and Sybil attack profitability.

```
Hypothesis: Honest participation is always more profitable than Sybil deviation.
```
**Metrics:** mean_reward, fairness, expected_profit per Sybil size

### E6: Ablation Study
Systematically ablates HASO, TVE, and GTM modules.

```
Hypothesis: Each module contributes positively to final accuracy.
```
**Variants:** full, no_haso, no_tve, no_gtm, no_haso_no_tve, no_haso_no_gtm

### E7: Blockchain Overhead
Measures blockchain ledger overhead vs total training time.

```
Hypothesis: Blockchain overhead is <5% of total training time.
```
**Metrics:** overhead_seconds, overhead_percent, ledger_size_kb

## Output Files

All results are saved to `--log_dir` (default `./logs`) as CSV files.

```
logs/
├── e1_chainfsl.csv
├── e1_chainfsl_nohaso.csv
├── e1_splitfed.csv
├── e1_fedavg.csv
├── e1_haso_comparison.csv
├── e2_n5_results.csv
├── e2_n10_results.csv
├── e2_n20_results.csv
├── e2_n50_results.csv
├── e2_scalability_summary.csv
├── e3_alpha_0.1_results.csv
├── e3_alpha_0.5_results.csv
├── e3_alpha_1.0_results.csv
├── e3_alpha_10.0_results.csv
├── e3_noniid_summary.csv
├── e4_lazy_f0.0_results.csv
├── e4_lazy_f0.1_results.csv
├── e4_lazy_f0.2_results.csv
├── e4_lazy_f0.3_results.csv
├── e4_security_summary.csv
├── e5_honest_results.csv
├── e5_incentive_summary.csv
├── e6_full_results.csv
├── e6_no_haso_results.csv
├── e6_no_tve_results.csv
├── e6_no_gtm_results.csv
├── e6_no_haso_no_tve_results.csv
├── e6_no_haso_no_gtm_results.csv
├── e6_ablation_summary.csv
├── e7_with_blockchain.csv
├── e7_without_blockchain.csv
└── e7_overhead_summary.csv
```

## Generating Plots

After running experiments, generate plots from CSV results:

```bash
python analysis/plot_results.py --exp e1 --log_dir ./logs
python analysis/plot_results.py --log_dir ./logs    # all experiments
```

Plot output files:

```
logs/
├── e1_accuracy_latency.png    # E1: accuracy + latency over rounds
├── e1_comparison.png           # E1: final accuracy vs fairness bar chart
├── e2_scalability.png          # E2: latency, accuracy, throughput vs N
├── e3_noniid.png              # E3: accuracy curves + final acc bar
├── e4_security.png            # E4: detection rate + accuracy under attack
├── e6_ablation.png            # E6: ablation bar charts (3 metrics)
└── e7_overhead.png            # E7: ledger size + latency comparison
```

## Experiment Configuration

Default configuration (from `experiments/utils.py`):

```python
{
    "n_nodes": 10,
    "global_rounds": 30,
    "local_epochs": 5,
    "batch_size": 32,
    "lr": 0.01,
    "alpha": 0.5,                   # Dirichlet non-IID parameter
    "lazy_client_fraction": 0.0,
    "haso_enabled": True,
    "tve_enabled": True,
    "gtm_enabled": True,
    "blockchain_enabled": True,
    "tier_distribution": [0.1, 0.3, 0.4, 0.2],
    "cut_layer": 2,                 # Default split layer for baselines
    "stake_min": 10.0,
    "reward_total_init": 1000.0,
    "log_dir": "./logs",
}
```
