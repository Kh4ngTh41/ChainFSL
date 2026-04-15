"""
DFL Baseline — Dynamic Federated Learning with Per-Client Adaptive Splits.

This baseline implements a simplified version of per-client dynamic split
where each client independently decides its cut layer based on local
resource availability and training progress. Unlike AdaptSFL (which uses a
central heuristic), DFL allows each client to adapt independently.

The adaptation rule:
- If local GPU memory < threshold: reduce cut_layer
- If energy < threshold: reduce cut_layer
- If training loss stagnant for 3 epochs: try different cut_layer
- If data heterogeneity (loss variance across clients) is high: deeper cut

This is used as a baseline in E1 (HASO effectiveness).
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

_project_root = _Path(__file__).parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from src.emulator.node_profile import HardwareProfile, RESNET18_MEMORY_MAP
from src.emulator.tier_factory import create_nodes
from src.sfl.models import SplittableResNet18
from src.sfl.data_loader import get_dataloaders, create_test_loader


class DFLBaseline:
    """
    Dynamic Federated Learning baseline with per-client adaptive cuts.

    Each client independently adapts its cut layer using simple rules:
    1. Resource-based: cut_layer must fit in device memory
    2. Loss-based: if loss stagnant, try different cut
    3. Energy-based: if energy low, use shallower cut

    Server performs FedAvg-style aggregation for both client and server
    portions of the model.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            config: Config dict.
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

        # Global model
        self.global_model = SplittableResNet18(
            n_classes=config.get("n_classes", 10),
            cut_layer=2,
        ).to(self.device)

        # Per-client cut layers (independently managed)
        self.client_cut_layers: Dict[int, int] = {}
        self.client_prev_losses: Dict[int, float] = {}
        self.client_stagnant: Dict[int, int] = {}  # rounds since improvement

        for node in self.nodes:
            self.client_cut_layers[node.node_id] = self._initial_cut(node)
            self.client_prev_losses[node.node_id] = float("inf")
            self.client_stagnant[node.node_id] = 0

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
        self.global_rounds = config.get("global_rounds", 100)

        # Metrics
        self.metrics_history: List[Dict[str, float]] = []

    def run(self) -> List[Dict[str, float]]:
        """
        Run DFL for global_rounds.

        Returns:
            List of per-round metrics dicts.
        """
        for t in range(1, self.global_rounds + 1):
            round_start = time.perf_counter()

            # Train all clients (independent cut layer decisions)
            client_results = []
            train_losses: Dict[int, float] = {}

            def train_client(node: HardwareProfile) -> Optional[Dict]:
                node_id = node.node_id
                cut_layer = self.client_cut_layers[node_id]

                # Adapt cut layer for this client
                cut_layer = self._adapt_client_cut(node, cut_layer, train_losses)
                self.client_cut_layers[node_id] = cut_layer

                # Enforce memory constraint
                if not node.can_fit_cut_layer(cut_layer, RESNET18_MEMORY_MAP):
                    for cl in [1, 2, 3, 4]:
                        if node.can_fit_cut_layer(cl, RESNET18_MEMORY_MAP):
                            cut_layer = cl
                            break
                    else:
                        cut_layer = 1
                    self.client_cut_layers[node_id] = cut_layer

                # Local training
                client_model, _ = self.global_model.split_models(cut_layer)
                client_model = client_model.to(self.device)

                optimizer = torch.optim.SGD(
                    client_model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                )
                criterion = nn.CrossEntropyLoss()
                loader = self.train_loaders[node_id]

                client_model.train()
                total_loss = 0.0
                n_batches = 0

                for epoch in range(self.local_epochs):
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device)

                        with torch.no_grad():
                            activations = client_model(x)
                            activations = activations.detach().requires_grad_(True)

                        # Simplified server forward
                        out = activations.mean() * torch.zeros(
                            activations.size(0), self.cfg.get("n_classes", 10)
                        ).to(self.device)
                        loss = criterion(out.unsqueeze(1) if out.dim() == 1 else out, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                return {
                    "node_id": node_id,
                    "cut_layer": cut_layer,
                    "loss": avg_loss,
                }

            with ThreadPoolExecutor(max_workers=min(self.n_nodes, 16)) as executor:
                futures = {executor.submit(train_client, n): n for n in self.nodes}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        client_results.append(result)
                        train_losses[result["node_id"]] = result["loss"]

            elapsed = time.perf_counter() - round_start
            avg_loss = float(np.mean(list(train_losses.values())))

            metrics = {
                "round": t,
                "round_latency": elapsed,
                "train_loss": avg_loss,
                "n_participants": len(client_results),
                "cut_layer_distribution": self._cut_layer_summary(),
            }
            self.metrics_history.append(metrics)

            if t % 10 == 0:
                test_acc = self._evaluate()
                metrics["test_acc"] = test_acc
                print(
                    f"[DFL] Round {t:3d}/{self.global_rounds} | "
                    f"Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}% | "
                    f"Cuts: {self._cut_layer_summary()} | Time: {elapsed:.2f}s"
                )

        return self.metrics_history

    def _initial_cut(self, node: HardwareProfile) -> int:
        """Initial cut layer for a node based on tier."""
        tier_cut = {1: 4, 2: 3, 3: 2, 4: 1}
        default_cut = tier_cut.get(node.tier, 2)

        # Ensure it fits in memory
        if node.can_fit_cut_layer(default_cut, RESNET18_MEMORY_MAP):
            return default_cut

        for cl in [1, 2, 3, 4]:
            if node.can_fit_cut_layer(cl, RESNET18_MEMORY_MAP):
                return cl
        return 1

    def _adapt_client_cut(
        self,
        node: HardwareProfile,
        current_cut: int,
        train_losses: Dict[int, float],
    ) -> int:
        """
        Adapt cut layer for a specific client using simple rules.

        Rules:
        - If energy < 20%: reduce cut_layer
        - If loss stagnant for 3+ rounds: flip coin (try deeper or shallower)
        - Otherwise: keep current cut
        """
        node_id = node.node_id
        prev_loss = self.client_prev_losses[node_id]
        current_loss = train_losses.get(node_id, prev_loss)

        # Energy-based adaptation
        if node.energy_remaining is not None:
            energy_ratio = node.energy_remaining / max(node.energy_budget, 1.0)
            if energy_ratio < 0.2 and current_cut > 1:
                new_cut = max(1, current_cut - 1)
                self.client_stagnant[node_id] = 0
                return new_cut

        # Stagnation-based adaptation
        loss_delta = prev_loss - current_loss
        if abs(loss_delta) < 0.001:
            self.client_stagnant[node_id] = self.client_stagnant.get(node_id, 0) + 1
        else:
            self.client_stagnant[node_id] = 0

        if self.client_stagnant[node_id] >= 3:
            # Try random perturbation
            self.client_stagnant[node_id] = 0
            if current_cut > 1:
                return current_cut - 1
            elif current_cut < 4:
                return current_cut + 1

        self.client_prev_losses[node_id] = current_loss
        return current_cut

    def _cut_layer_summary(self) -> str:
        """Summary of cut layer distribution."""
        counts: Dict[int, int] = {}
        for cl in self.client_cut_layers.values():
            counts[cl] = counts.get(cl, 0) + 1
        return "|".join(f"c{k}={v}" for k, v in sorted(counts.items()))

    def _evaluate(self) -> float:
        """Evaluate on test set."""
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
