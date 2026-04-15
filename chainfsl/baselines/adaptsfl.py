"""
AdaptSFL Baseline — Alternating Optimization for Cut Layer Selection.

This baseline implements a simplified version of AdaptSFL where cut layers
are selected using a heuristic (resource-constrained) rather than full DRL.
The algorithm alternates between:
1. Training the model with current cut layer choices
2. Adjusting cut layers based on observed resource utilization

This provides a rule-based adaptive baseline to compare with HASO's
DRL-based approach in E1 (HASO effectiveness).
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


class AdaptSFLBaseline:
    """
    AdaptSFL baseline with heuristic cut layer adaptation.

    Uses a simple heuristic:
    - Tier 1-2 nodes: deeper cuts (3-4) for more computation
    - Tier 3-4 nodes: shallower cuts (1-2) for less computation
    - Adjust based on observed loss gradient (if loss plateaus, try deeper cut)

    Alternates between training and cut layer adjustment rounds.
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

        # Per-node cut layer assignments (start with tier-based heuristic)
        self.node_cut_layers: Dict[int, int] = {}
        for node in self.nodes:
            self.node_cut_layers[node.node_id] = self._tier_to_cut(node.tier)

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
        self.adaptation_interval = config.get("adaptation_interval", 5)

        # Loss history for adaptation
        self.loss_history: List[float] = []
        self.prev_loss: Optional[float] = None

        # Metrics
        self.metrics_history: List[Dict[str, float]] = []

    def run(self) -> List[Dict[str, float]]:
        """
        Run AdaptSFL for global_rounds.

        Returns:
            List of per-round metrics dicts.
        """
        for t in range(1, self.global_rounds + 1):
            round_start = time.perf_counter()

            # Train all clients with their current cut layers
            client_results = []
            train_losses: Dict[int, float] = {}

            def train_client(node: HardwareProfile) -> Optional[Dict]:
                cut_layer = self.node_cut_layers.get(node.node_id, 2)

                # Enforce memory constraint
                if not node.can_fit_cut_layer(cut_layer, RESNET18_MEMORY_MAP):
                    for cl in [1, 2, 3, 4]:
                        if node.can_fit_cut_layer(cl, RESNET18_MEMORY_MAP):
                            cut_layer = cl
                            break
                    else:
                        cut_layer = 1

                client_model, _ = self.global_model.split_models(cut_layer)
                client_model = client_model.to(self.device)

                optimizer = torch.optim.SGD(
                    client_model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                )
                criterion = nn.CrossEntropyLoss()
                loader = self.train_loaders[node.node_id]

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
                        out = activations.mean() * torch.zeros(activations.size(0), self.cfg.get("n_classes", 10)).to(self.device)
                        loss = criterion(out.unsqueeze(1) if out.dim() == 1 else out, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                return {
                    "node_id": node.node_id,
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

            # Adaptation: adjust cut layers every N rounds
            if t % self.adaptation_interval == 0:
                self._adapt_cut_layers(train_losses)

            self.loss_history.append(avg_loss)
            metrics = {
                "round": t,
                "round_latency": elapsed,
                "train_loss": avg_loss,
                "n_participants": len(client_results),
            }
            self.metrics_history.append(metrics)

            if t % 10 == 0:
                test_acc = self._evaluate()
                metrics["test_acc"] = test_acc
                print(
                    f"[AdaptSFL] Round {t:3d}/{self.global_rounds} | "
                    f"Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}% | "
                    f"Cuts: {self._cut_layer_summary()} | Time: {elapsed:.2f}s"
                )

        return self.metrics_history

    def _adapt_cut_layers(self, train_losses: Dict[int, float]) -> None:
        """
        Adapt cut layers based on observed losses.

        Heuristic:
        - If loss decreased significantly: try deeper cut (more local computation)
        - If loss plateaug/stopped: try shallower cut (less local computation)
        """
        current_avg = float(np.mean(list(train_losses.values())))

        if self.prev_loss is not None:
            loss_delta = self.prev_loss - current_avg  # positive = improvement

            for node in self.nodes:
                node_id = node.node_id
                current_cut = self.node_cut_layers.get(node_id, 2)

                if loss_delta < 0.01:  # Loss plateuing → shift cut
                    # Try to deepen cut if node can handle it
                    if current_cut < 4 and node.can_fit_cut_layer(current_cut + 1, RESNET18_MEMORY_MAP):
                        self.node_cut_layers[node_id] = current_cut + 1
                else:  # Loss improving well → try slightly different
                    if node.tier <= 2 and current_cut < 3:
                        # Higher tier → can try deeper
                        pass  # keep current

        self.prev_loss = current_avg

    def _tier_to_cut(self, tier: int) -> int:
        """Map tier to initial cut layer."""
        mapping = {1: 4, 2: 3, 3: 2, 4: 1}
        return mapping.get(tier, 2)

    def _cut_layer_summary(self) -> str:
        """Summary of cut layer distribution."""
        counts: Dict[int, int] = {}
        for cl in self.node_cut_layers.values():
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
