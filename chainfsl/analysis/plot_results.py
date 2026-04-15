#!/usr/bin/env python3
"""
ChainFSL Experiment Plotting Tool.

Generates plots for all experiments from CSV result files.

Usage:
    python analysis/plot_results.py                    # Plot all experiments
    python analysis/plot_results.py --exp e1          # Plot specific experiment
    python analysis/plot_results.py --exp e3 --format png  # PNG output
    python analysis/plot_results.py --output_dir ./plots     # Custom output dir

Output files:
    - logs/e1_accuracy.png       (E1: accuracy over rounds)
    - logs/e1_latency.png        (E1: latency over rounds)
    - logs/e1_comparison.png     (E1: method comparison)
    - logs/e2_scaling.png        (E2: scalability curves)
    - logs/e3_noniid.png        (E3: non-IID robustness)
    - logs/e4_security.png       (E4: attack detection rates)
    - logs/e5_incentive.png     (E5: Nash equilibrium)
    - logs/e6_ablation.png       (E6: ablation bar charts)
    - logs/e7_overhead.png       (E7: blockchain overhead)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Try to import matplotlib (optional)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[plot_results] Warning: matplotlib not installed. Plots will not be generated.")

import numpy as np
import csv


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV file into a list of dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Try to convert numeric values
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
        return rows


def load_exp_results(exp_name: str, log_dir: str = "./logs") -> Dict[str, List[Dict]]:
    """Load all CSV files for an experiment."""
    log_path = Path(log_dir)
    results = {}

    # Find all matching CSV files
    for p in log_path.glob(f"{exp_name}*.csv"):
        key = p.stem  # filename without extension
        results[key] = load_csv(p)

    return results


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

if HAS_PLT:

    def _setup_style():
        """Configure matplotlib style."""
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 150,
        })

    def _get_rounds(metrics: List[Dict]) -> np.ndarray:
        """Extract round numbers."""
        return np.array([m.get("round", i + 1) for i, m in enumerate(metrics)])

    def _plot_accuracy(ax, metrics: List[Dict], label: str, color: str) -> None:
        """Plot test accuracy over rounds."""
        rounds = _get_rounds(metrics)
        acc = np.array([m.get("test_acc", 0) for m in metrics])
        ax.plot(rounds, acc, label=label, color=color, linewidth=2, marker="o", markersize=3)

    def _plot_loss(ax, metrics: List[Dict], label: str, color: str) -> None:
        """Plot training loss over rounds."""
        rounds = _get_rounds(metrics)
        loss = np.array([m.get("train_loss", 0) for m in metrics])
        ax.plot(rounds, loss, label=label, color=color, linewidth=2, marker="s", markersize=3)

    def _plot_latency(ax, metrics: List[Dict], label: str, color: str) -> None:
        """Plot round latency over rounds."""
        rounds = _get_rounds(metrics)
        lat = np.array([m.get("round_latency", 0) for m in metrics])
        ax.plot(rounds, lat, label=label, color=color, linewidth=2, alpha=0.7)

    # ---- E1: HASO Effectiveness ----

    def plot_e1_haso(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E1 results: HASO effectiveness."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e1", log_dir)
        if not results:
            print("[plot] E1: No data found.")
            return

        _setup_style()

        # Accuracy plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E1: HASO Effectiveness", fontweight="bold")

        colors = {"chainfsl": "#2196F3", "chainfsl_nohaso": "#FF9800",
                  "splitfed": "#4CAF50", "fedavg": "#F44336"}

        for name, metrics in results.items():
            if not metrics:
                continue
            label = name.replace("e1_", "").replace("_", " ").title()
            color = colors.get(name.replace("e1_", ""), "#9E9E9E")
            _plot_accuracy(axes[0], metrics, label, color)
            _plot_latency(axes[1], metrics, label, color)

        axes[0].set_title("Test Accuracy")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend()
        axes[0].set_ylim([0, 100])

        axes[1].set_title("Round Latency")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Latency (s)")
        axes[1].legend()

        plt.tight_layout()
        p = out / "e1_accuracy_latency.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle("E1: Method Comparison", fontweight="bold")

        methods = []
        final_accs = []
        mean_fairness = []

        for name in ["chainfsl", "chainfsl_nohaso", "splitfed", "fedavg"]:
            key = f"e1_{name}"
            metrics = results.get(key, [])
            if metrics:
                methods.append(name.replace("_", " ").title())
                final_accs.append(metrics[-1].get("test_acc", 0))
                mean_fairness.append(np.mean([m.get("fairness_index", 0) for m in metrics]))

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, final_accs, width, label="Final Acc (%)", color="#2196F3")
        ax.set_ylabel("Accuracy (%)", color="#2196F3")
        ax.set_ylim([0, 100])

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, mean_fairness, width, label="Fairness", color="#FF9800", alpha=0.8)
        ax2.set_ylabel("Fairness Index", color="#FF9800")
        ax2.set_ylim([0, 1])

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title("Final Accuracy vs Fairness")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        p = out / "e1_comparison.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- E2: Scalability ----

    def plot_e2_scalability(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E2 results: scalability."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e2", log_dir)
        if not results:
            print("[plot] E2: No data found.")
            return

        _setup_style()

        # Scaling plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("E2: Scalability", fontweight="bold")

        n_nodes_list = []
        mean_latencies = []
        final_accs = []

        for name, metrics in sorted(results.items()):
            if not metrics or "metrics" in name:
                continue
            n_nodes = int(name.split("_n")[1]) if "_n" in name else 0
            if n_nodes == 0:
                continue

            n_nodes_list.append(n_nodes)
            mean_latencies.append(np.mean([m.get("round_latency", 0) for m in metrics]))
            final_accs.append(metrics[-1].get("test_acc", 0) if metrics else 0)

        # Sort by n_nodes
        sorted_pairs = sorted(zip(n_nodes_list, mean_latencies, final_accs))
        n_nodes_list, mean_latencies, final_accs = zip(*sorted_pairs)
        n_nodes_list, mean_latencies, final_accs = list(n_nodes_list), list(mean_latencies), list(final_accs)

        axes[0].plot(n_nodes_list, mean_latencies, "o-", color="#2196F3", linewidth=2, markersize=8)
        axes[0].set_title("Mean Latency vs N")
        axes[0].set_xlabel("Number of Nodes")
        axes[0].set_ylabel("Mean Latency (s)")

        axes[1].plot(n_nodes_list, final_accs, "s-", color="#4CAF50", linewidth=2, markersize=8)
        axes[1].set_title("Final Accuracy vs N")
        axes[1].set_xlabel("Number of Nodes")
        axes[1].set_ylabel("Final Accuracy (%)")
        axes[1].set_ylim([0, 100])

        # Throughput (N / latency)
        throughput = [n / lat for n, lat in zip(n_nodes_list, mean_latencies)]
        axes[2].plot(n_nodes_list, throughput, "^-", color="#FF9800", linewidth=2, markersize=8)
        axes[2].set_title("Throughput vs N")
        axes[2].set_xlabel("Number of Nodes")
        axes[2].set_ylabel("Throughput (nodes/s)")

        plt.tight_layout()
        p = out / "e2_scalability.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- E3: Non-IID ----

    def plot_e3_noniid(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E3 results: Non-IID robustness."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e3", log_dir)
        if not results:
            print("[plot] E3: No data found.")
            return

        _setup_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E3: Non-IID Data Robustness", fontweight="bold")

        colors = {"0.1": "#F44336", "0.5": "#FF9800", "1.0": "#4CAF50", "10.0": "#2196F3"}

        for name, metrics in sorted(results.items()):
            if not metrics:
                continue
            alpha = name.split("_")[-1] if "_" in name else "?"
            color = colors.get(alpha, "#9E9E9E")
            label = f"alpha = {alpha}"
            _plot_accuracy(axes[0], metrics, label, color)

        axes[0].set_title("Accuracy vs Alpha")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend(title="Dirichlet Alpha")
        axes[0].set_ylim([0, 100])

        # Bar chart of final accuracy
        alphas = []
        final_accs = []
        for name, metrics in sorted(results.items()):
            if not metrics:
                continue
            alpha = name.split("_")[-1] if "_" in name else "?"
            alphas.append(float(alpha))
            final_accs.append(metrics[-1].get("test_acc", 0) if metrics else 0)

        sorted_pairs = sorted(zip(alphas, final_accs))
        alphas, final_accs = zip(*sorted_pairs)

        bars = axes[1].bar([str(a) for a in alphas], final_accs, color="#2196F3", alpha=0.8)
        axes[1].set_title("Final Accuracy by Alpha")
        axes[1].set_xlabel("Dirichlet Alpha")
        axes[1].set_ylabel("Final Accuracy (%)")
        axes[1].set_ylim([0, 100])
        for bar, val in zip(bars, final_accs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", fontsize=9)

        plt.tight_layout()
        p = out / "e3_noniid.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- E4: Security ----

    def plot_e4_security(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E4 results: attack detection."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e4", log_dir)
        if not results:
            print("[plot] E4: No data found.")
            return

        _setup_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E4: Security Evaluation", fontweight="bold")

        # Detection rate bar chart
        fractions = []
        detection_rates = []

        for name, metrics in sorted(results.items()):
            if not metrics:
                continue
            frac = name.split("_f")[-1] if "_f" in name else "0"
            try:
                frac_val = float(frac)
            except ValueError:
                continue
            fractions.append(frac_val)
            detection_rates.append(np.mean([m.get("attack_detection_rate", 1.0) for m in metrics]) * 100)

        sorted_pairs = sorted(zip(fractions, detection_rates))
        fractions, detection_rates = zip(*sorted_pairs)

        axes[0].bar([str(f) for f in fractions], detection_rates, color="#F44336", alpha=0.8)
        axes[0].axhline(y=95, color="green", linestyle="--", label="95% target")
        axes[0].set_title("Attack Detection Rate")
        axes[0].set_xlabel("Attack Fraction")
        axes[0].set_ylabel("Detection Rate (%)")
        axes[0].legend()
        axes[0].set_ylim([0, 110])

        # Accuracy under attack
        for name, metrics in sorted(results.items()):
            if not metrics:
                continue
            frac = name.split("_f")[-1] if "_f" in name else "0"
            label = f"frac={frac}"
            _plot_accuracy(axes[1], metrics, label, "#9E9E9E")

        axes[1].set_title("Accuracy Under Attack")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].set_ylim([0, 100])

        plt.tight_layout()
        p = out / "e4_security.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- E6: Ablation ----

    def plot_e6_ablation(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E6 results: ablation study."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e6", log_dir)
        if not results:
            print("[plot] E6: No data found.")
            return

        _setup_style()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("E6: Ablation Study", fontweight="bold")

        variants = []
        final_accs = []
        mean_fairness = []
        mean_latencies = []

        for variant in ["full", "no_haso", "no_tve", "no_gtm", "no_haso_no_tve", "no_haso_no_gtm"]:
            key = f"e6_{variant}"
            metrics = results.get(key, [])
            if metrics:
                variants.append(variant.replace("_", "\n"))
                final_accs.append(metrics[-1].get("test_acc", 0) if metrics else 0)
                mean_fairness.append(np.mean([m.get("fairness_index", 0) for m in metrics]))
                mean_latencies.append(np.mean([m.get("round_latency", 0) for m in metrics]))

        colors = ["#2196F3"] + ["#FF9800"] * (len(variants) - 1)

        bars1 = axes[0].bar(variants, final_accs, color=colors, alpha=0.8)
        axes[0].set_title("Final Accuracy")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim([0, 100])
        for bar, val in zip(bars1, final_accs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", fontsize=8)

        bars2 = axes[1].bar(variants, mean_fairness, color=colors, alpha=0.8)
        axes[1].set_title("Fairness Index")
        axes[1].set_ylabel("Jain's Fairness")
        axes[1].set_ylim([0, 1])
        for bar, val in zip(bars2, mean_fairness):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", fontsize=8)

        bars3 = axes[2].bar(variants, mean_latencies, color=colors, alpha=0.8)
        axes[2].set_title("Mean Latency")
        axes[2].set_ylabel("Latency (s)")

        plt.tight_layout()
        p = out / "e6_ablation.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- E7: Blockchain Overhead ----

    def plot_e7_overhead(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot E7 results: blockchain overhead."""
        out = Path(output_dir) if output_dir else Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = load_exp_results("e7", log_dir)
        if not results:
            print("[plot] E7: No data found.")
            return

        _setup_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E7: Blockchain Overhead", fontweight="bold")

        # Ledger size growth
        bc_metrics = results.get("e7_with_blockchain", [])
        if bc_metrics:
            rounds = _get_rounds(bc_metrics)
            ledger_kb = np.array([m.get("ledger_size_kb", 0) for m in bc_metrics])
            axes[0].plot(rounds, ledger_kb, "o-", color="#9C27B0", linewidth=2, markersize=4)
            axes[0].set_title("Ledger Size Growth")
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Ledger Size (KB)")

        # Latency comparison
        labels = ["With Blockchain", "Without Blockchain"]
        latencies = []
        for key in ["e7_with_blockchain", "e7_without_blockchain"]:
            metrics = results.get(key, [])
            if metrics:
                latencies.append(np.mean([m.get("round_latency", 0) for m in metrics]))
            else:
                latencies.append(0)

        bars = axes[1].bar(labels, latencies, color=["#9C27B0", "#4CAF50"], alpha=0.8)
        axes[1].set_title("Mean Latency Comparison")
        axes[1].set_ylabel("Mean Latency (s)")
        for bar, val in zip(bars, latencies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}s", ha="center", fontsize=10)

        plt.tight_layout()
        p = out / "e7_overhead.png"
        plt.savefig(p, bbox_inches="tight")
        print(f"[plot] Saved: {p}")
        plt.close()

    # ---- All experiments ----

    def plot_all(log_dir: str = "./logs", output_dir: Optional[str] = None) -> None:
        """Plot all experiments."""
        print("[plot] Generating all plots...")
        plot_e1_haso(log_dir, output_dir)
        plot_e2_scalability(log_dir, output_dir)
        plot_e3_noniid(log_dir, output_dir)
        plot_e4_security(log_dir, output_dir)
        plot_e6_ablation(log_dir, output_dir)
        plot_e7_overhead(log_dir, output_dir)
        print("[plot] All plots generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChainFSL Experiment Plotting Tool")
    parser.add_argument("--exp", default=None, help="Experiment to plot (e1-e7). Default: all.")
    parser.add_argument("--log_dir", default="./logs", help="Directory containing CSV results.")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots. Default: same as log_dir.")
    args = parser.parse_args()

    if not HAS_PLT:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    if args.exp is None:
        plot_all(args.log_dir, args.output_dir)
    else:
        exp_map = {
            "e1": plot_e1_haso,
            "e2": plot_e2_scalability,
            "e3": plot_e3_noniid,
            "e4": plot_e4_security,
            "e6": plot_e6_ablation,
            "e7": plot_e7_overhead,
        }
        fn = exp_map.get(args.exp)
        if fn:
            fn(args.log_dir, args.output_dir)
        else:
            print(f"Unknown experiment: {args.exp}")


if __name__ == "__main__":
    main()
