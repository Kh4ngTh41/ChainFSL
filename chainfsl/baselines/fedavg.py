"""
FedAvg Baseline — Standard Federated Averaging.

This baseline implements vanilla FedAvg (McMahan et al., 2017) without
split learning. Each client trains the full model locally for H epochs,
then the server averages all client weights.

Used as a performance baseline for E1 (HASO effectiveness).
"""

import time
import copy
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import numpy as np

import sys as _sys
from pathlib import Path as _Path

# Add src to path for absolute imports when baselines is used standalone
_project_root = _Path(__file__).parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from src.emulator.node_profile import HardwareProfile
from src.emulator.tier_factory import create_nodes
from src.sfl.models import SplittableResNet18
from src.sfl.data_loader import get_dataloaders, create_test_loader


class FedAvgBaseline:
    """
    FedAvg baseline for comparison with ChainFSL.

    Each round:
    1. Sample a fraction of clients
    2. Each client trains full model locally for H epochs
    3. Server averages client weights (FedAvg)
    4. Repeat
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            config: Config dict with keys:
                - n_nodes, tier_distribution, n_classes
                - global_rounds, batch_size_default, dirichlet_alpha, local_epochs
                - sample_fraction (fraction of clients per round)
            device: Computation device.
        """
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create nodes
        from src.emulator.tier_factory import TierDistribution
        tier_dist_list = config.get("tier_distribution", [0.1, 0.3, 0.4, 0.2])
        tier_dist = TierDistribution(tiers=[1, 2, 3, 4], probabilities=tier_dist_list)
        self.nodes = create_nodes(config["n_nodes"], distribution=tier_dist)
        self.n_nodes = len(self.nodes)

        # Global model (full ResNet-18, no split)
        self.global_model = SplittableResNet18(
            n_classes=config.get("n_classes", 10),
            cut_layer=0,  # No split — full model on client
        ).to(self.device)

        # Data loaders
        self.train_loaders = get_dataloaders(
            dataset_name=config.get("dataset", "cifar10"),
            n_clients=self.n_nodes,
            alpha=config.get("dirichlet_alpha", 0.5),
            batch_size=config.get("batch_size_default", 32),
            data_dir="./data",
            download=True,
            seed=config.get("seed", 42),
        )

        # Hyperparameters
        self.lr = config.get("local_lr", 0.01)
        self.momentum = config.get("local_momentum", 0.9)
        self.local_epochs = config.get("local_epochs", 1)
        self.sample_fraction = config.get("sample_fraction", 1.0)
        self.global_rounds = config.get("global_rounds", 100)

        # Metrics
        self.metrics_history: List[Dict[str, float]] = []

    def run(self) -> List[Dict[str, float]]:
        """
        Run FedAvg for global_rounds.

        Returns:
            List of per-round metrics dicts.
        """
        for t in range(1, self.global_rounds + 1):
            round_start = time.perf_counter()

            # Sample clients
            n_sample = max(1, int(self.sample_fraction * self.n_nodes))
            sampled_nodes = self.nodes[:n_sample]

            # Local training
            client_states = []
            train_losses = {}

            def train_client(node: HardwareProfile) -> Optional[Dict]:
                model = copy.deepcopy(self.global_model)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                )
                criterion = nn.CrossEntropyLoss()
                loader = self.train_loaders[node.node_id]

                model.train()
                total_loss = 0.0
                n_batches = 0

                for epoch in range(self.local_epochs):
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        out = model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                return {
                    "node_id": node.node_id,
                    "state": {k: v.clone().detach().cpu() for k, v in model.state_dict().items()},
                    "data_size": len(loader.dataset),
                    "loss": avg_loss,
                }

            with ThreadPoolExecutor(max_workers=min(len(sampled_nodes), 16)) as executor:
                futures = {executor.submit(train_client, n): n for n in sampled_nodes}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        client_states.append(result)

            # FedAvg aggregation
            self._aggregate(client_states)

            elapsed = time.perf_counter() - round_start
            avg_loss = float(np.mean([c["loss"] for c in client_states]))

            metrics = {
                "round": t,
                "round_latency": elapsed,
                "train_loss": avg_loss,
                "n_participants": len(client_states),
            }
            self.metrics_history.append(metrics)

            if t % 10 == 0:
                test_acc = self._evaluate()
                metrics["test_acc"] = test_acc
                print(
                    f"[FedAvg] Round {t:3d}/{self.global_rounds} | "
                    f"Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}% | "
                    f"Time: {elapsed:.2f}s"
                )

        return self.metrics_history

    def _aggregate(self, client_states: List[Dict]) -> None:
        """FedAvg: weighted average of client model states."""
        total_data = sum(cs["data_size"] for cs in client_states)

        new_state = {}
        for key in self.global_model.state_dict().keys():
            weighted_sum = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
            for cs in client_states:
                weight = cs["data_size"] / total_data
                weighted_sum += weight * cs["state"][key].float()
            new_state[key] = weighted_sum.to(self.global_model.state_dict()[key].dtype)

        self.global_model.load_state_dict(new_state)

    def _evaluate(self) -> float:
        """Evaluate global model on test set."""
        self.global_model.eval()
        correct = 0
        total = 0

        try:
            test_loader = create_test_loader(
                dataset_name=self.cfg.get("dataset", "cifar10"),
                batch_size=64,
                data_dir="./data",
            )
        except Exception:
            return 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.global_model(x)
                _, predicted = out.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)

        if total == 0:
            return 0.0
        return 100.0 * correct / total
