import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RoundLog:
    round: int
    timestamp: str
    n_active_nodes: int
    n_compute_nodes: int
    global_accuracy: float
    global_loss: float
    global_f1: float
    round_latency: float
    mean_T_comp: float
    mean_T_comm: float
    straggler_ratio: float
    node_decisions: List[Dict]
    compute_node_load: List[Dict]
    overlap_events: List[Dict]


@dataclass
class DecisionLog:
    node_id: int
    round: int
    step: int
    state: List[float]
    action: Dict
    reward: float
    done: bool
    info: Dict


class LogManager:
    def __init__(self, base_dir: str, experiment_name: str):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.run_dir: Optional[Path] = None
        self.run_id: Optional[str] = None

    def setup_run(self, config: Dict) -> str:
        """Create run directory with timestamp, save config.json"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.run_dir = self.base_dir / self.experiment_name / self.run_id

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "rounds").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "decisions").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "compute_nodes").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "metrics").mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return self.run_id

    def log_round(self, round_log: RoundLog) -> None:
        """Save round JSON to rounds/round_{round:03d}.json"""
        if self.run_dir is None:
            raise RuntimeError("Call setup_run() first")
        path = self.run_dir / "rounds" / f"round_{round_log.round:03d}.json"
        with open(path, "w") as f:
            json.dump(asdict(round_log), f, indent=2)

    def log_decision(self, node_id: int, decision: DecisionLog) -> None:
        """Save decision to decisions/node_{node_id}/round_{round}.json"""
        if self.run_dir is None:
            raise RuntimeError("Call setup_run() first")
        node_dir = self.run_dir / "decisions" / f"node_{node_id:02d}"
        node_dir.mkdir(parents=True, exist_ok=True)
        path = node_dir / f"round_{decision.round:03d}.json"
        with open(path, "w") as f:
            json.dump(asdict(decision), f, indent=2)

    def export_metrics_csv(self, rounds_data: List[Dict]) -> None:
        """Export accuracy.csv, latency.csv, reward.csv"""
        export_accuracy_csv(rounds_data, str(self.run_dir / "metrics" / "accuracy.csv"))
        export_latency_csv(rounds_data, str(self.run_dir / "metrics" / "latency.csv"))
        export_reward_csv(rounds_data, str(self.run_dir / "metrics" / "reward.csv"))

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save to metrics/metrics.json"""
        if self.run_dir is None:
            raise RuntimeError("Call setup_run() first")
        path = self.run_dir / "metrics" / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def get_log_path(self, run_id: str) -> Path:
        """Return path to run directory"""
        return self.base_dir / self.experiment_name / run_id


def export_accuracy_csv(rounds_data: List[RoundLog], output_path: str) -> None:
    """accuracy.csv: round,accuracy,f1_score,precision,recall,loss_test"""
    rows = []
    for r in rounds_data:
        rows.append({
            "round": r.round,
            "accuracy": r.global_accuracy,
            "f1_score": r.global_f1,
            "precision": r.global_f1,  # placeholder if precision not available
            "recall": r.global_f1,     # placeholder if recall not available
            "loss_test": r.global_loss,
        })
    _write_csv(output_path, rows)


def export_latency_csv(rounds_data: List[RoundLog], output_path: str) -> None:
    """latency.csv: round,mean_latency,median_latency,max_latency,min_latency,straggler_ratio"""
    rows = []
    for r in rounds_data:
        rows.append({
            "round": r.round,
            "mean_latency": r.round_latency,
            "median_latency": r.round_latency,  # placeholder
            "max_latency": r.round_latency,     # placeholder
            "min_latency": r.round_latency,     # placeholder
            "straggler_ratio": r.straggler_ratio,
        })
    _write_csv(output_path, rows)


def export_reward_csv(rounds_data: List[RoundLog], output_path: str) -> None:
    """reward.csv: round,node_id,reward,T_comp,T_comm,delta_F,shapley_phi,fusion_bonus,overlap_penalty"""
    rows = []
    for r in rounds_data:
        for nd in r.node_decisions:
            rows.append({
                "round": r.round,
                "node_id": nd.get("node_id", 0),
                "reward": nd.get("reward", 0.0),
                "T_comp": nd.get("T_comp", 0.0),
                "T_comm": nd.get("T_comm", 0.0),
                "delta_F": nd.get("delta_F", 0.0),
                "shapley_phi": nd.get("shapley_phi", 0.0),
                "fusion_bonus": nd.get("fusion_bonus", 0.0),
                "overlap_penalty": nd.get("overlap_penalty", 0.0),
            })
    _write_csv(output_path, rows)


def _write_csv(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Quick smoke test
    lm = LogManager("/tmp/chainfsl_logs", "e1_haso_effectiveness")
    run_id = lm.setup_run({"lr": 0.001, "n_rounds": 10})
    print(f"Created run: {run_id}")

    rl = RoundLog(
        round=0,
        timestamp=datetime.now().isoformat(),
        n_active_nodes=5,
        n_compute_nodes=3,
        global_accuracy=0.75,
        global_loss=0.25,
        global_f1=0.72,
        round_latency=1.2,
        mean_T_comp=0.8,
        mean_T_comm=0.4,
        straggler_ratio=0.1,
        node_decisions=[
            {"node_id": 0, "reward": 1.0, "T_comp": 0.5, "T_comm": 0.2, "delta_F": 0.1, "shapley_phi": 0.3, "fusion_bonus": 0.05, "overlap_penalty": 0.01},
            {"node_id": 1, "reward": 0.9, "T_comp": 0.6, "T_comm": 0.3, "delta_F": 0.08, "shapley_phi": 0.25, "fusion_bonus": 0.04, "overlap_penalty": 0.01},
        ],
        compute_node_load=[{"node_id": 0, "load": 0.6}, {"node_id": 1, "load": 0.4}],
        overlap_events=[],
    )
    lm.log_round(rl)
    print("Logged round 0")

    dl = DecisionLog(
        node_id=0,
        round=0,
        step=0,
        state=[0.1, 0.2, 0.3],
        action={"join": True, "layer": 2, "batch_size": 16},
        reward=1.0,
        done=False,
        info={"T_comp": 0.5, "T_comm": 0.2},
    )
    lm.log_decision(0, dl)
    print("Logged decision for node 0")

    lm.export_metrics_csv([rl])
    print("Exported CSV metrics")

    lm.save_metrics({"accuracy": 0.75, "f1": 0.72})
    print("Saved metrics.json")

    print(f"Log path: {lm.get_log_path(run_id)}")
    print("All tests passed!")
