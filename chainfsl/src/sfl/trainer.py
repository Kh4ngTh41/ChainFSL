"""
Trainer for Split Federated Learning (SFL).

Implements the SFL training loop for a single node with:
- Client-side forward to cut layer → smash data → send to server
- Server-side forward/backward → return gradient to client
- Local client-side weight update
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .models import SplittableResNet18, ClientModel, ServerModel


@dataclass
class TrainResult:
    """Result of one local training step."""

    node_id: int
    cut_layer: int
    batch_size: int
    loss: float
    T_comp: float   # Compute time (seconds)
    T_comm: float   # Communication time (seconds)
    grad_norm: float  # Norm of returned gradient


class SFLTrainer:
    """
    End-to-end SFL trainer for one node.

    Coordinates client-side forward/backward with server-side processing
    via smash data transmission.
    """

    def __init__(
        self,
        node_id: int,
        model: SplittableResNet18,
        cut_layer: int,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            node_id: Node identifier.
            model: Full SplittableResNet18 model.
            cut_layer: Split point (1-4).
            criterion: Loss function. Defaults to CrossEntropyLoss.
            device: Computation device.
        """
        self.node_id = node_id
        self.cut_layer = cut_layer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build client and server sub-models
        # Use deepcopy because split_models returns nn.Sequential sharing the SAME
        # layer objects from the original model. Without deepcopy, client and server
        # backbones share conv1/bn1/layerX objects, causing autograd version conflicts
        # when server forward_backward and client backward both access the same params.
        import copy
        client_backbone = copy.deepcopy(model.get_client_model(cut_layer)).to(self.device)
        server_backbone = copy.deepcopy(model.get_server_model(cut_layer)).to(self.device)

        self.client = ClientModel(client_backbone, cut_layer)
        self.server = ServerModel(server_backbone, criterion=criterion)

        self.model = model  # Keep reference to full model
        self._last_grad_norm: float = 0.0  # Track last gradient norm for TVE
        self._last_grad: Optional[torch.Tensor] = None  # Last gradient tensor for Tier 1 cosine sim
        self._last_smash_data: Optional[torch.Tensor] = None  # Last smash data for verification

    def local_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> TrainResult:
        """
        One SFL step: forward → send smash data → server backward → client update.

        Args:
            inputs: Input batch (N, C, H, W).
            labels: Ground truth labels.

        Returns:
            TrainResult with metrics.
        """
        import time

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # --- Phase 1: Client forward to cut layer ---
        t_comp_start = time.perf_counter()
        smash_data = self.client.forward(inputs)
        t_comp = time.perf_counter() - t_comp_start

        # --- Phase 2: Send smash data (simulated as gradient hook) ---
        t_comm_start = time.perf_counter()
        t_comm = t_comm_start - t_comp_start  # Would be actual comm in distributed

        # Store smash data for Tier 1 verification
        self._last_smash_data = smash_data.detach().clone()

        # --- Phase 3: Server forward-backward ---
        loss, grad = self.server.forward_backward(smash_data, labels)

        # --- Phase 4: Client backward ---
        self.client.backward(grad)

        # Gradient norm for monitoring and TVE
        self._last_grad = grad
        self._last_grad_norm = grad.norm().item() if grad is not None else 0.0
        grad_norm = self._last_grad_norm

        return TrainResult(
            node_id=self.node_id,
            cut_layer=self.cut_layer,
            batch_size=inputs.size(0),
            loss=loss,
            T_comp=t_comp,
            T_comm=t_comm,
            grad_norm=grad_norm,
        )

    def local_epochs(
        self,
        dataloader,
        H: int,
        verbose: bool = False,
        step_callback = None,
    ) -> Tuple[float, float]:
        """
        Run H local epochs on a dataloader.

        Args:
            dataloader: DataLoader for this node.
            H: Number of local epochs.
            verbose: Print progress.
            step_callback: Optional callable to invoke after each batch.

        Returns:
            (avg_loss, total_time) tuple.
        """
        self.client.backbone.train()
        total_loss = 0.0
        total_steps = 0

        for epoch in range(H):
            epoch_loss = 0.0
            steps = 0
            for inputs, labels in dataloader:
                result = self.local_step(inputs, labels)
                epoch_loss += result.loss
                steps += 1
                total_steps += 1
                if step_callback:
                    step_callback()

            avg_epoch_loss = epoch_loss / max(steps, 1)
            total_loss += epoch_loss
            if verbose:
                print(f"  Node {self.node_id} epoch {epoch+1}/{H}: loss={avg_epoch_loss:.4f}")

        return total_loss / max(total_steps, 1), 0.0  # time tracked per-step

    def get_client_state(self) -> Dict[str, torch.Tensor]:
        """Return client-side state dict for aggregation."""
        return {
            k: v.clone().detach()
            for k, v in self.client.backbone.state_dict().items()
        }

    def get_server_state(self) -> Dict[str, torch.Tensor]:
        """Return server-side state dict for aggregation."""
        return {
            k: v.clone().detach()
            for k, v in self.server.backbone.state_dict().items()
        }

    def get_last_grad_norm(self) -> float:
        """Return gradient norm from last local_step."""
        return self._last_grad_norm

    def get_last_grad(self) -> Optional[torch.Tensor]:
        """Return gradient tensor from last local_step (for Tier 1 verification)."""
        return self._last_grad

    def get_last_smash_data(self) -> Optional[torch.Tensor]:
        """Return smash data from last local_step (for Tier 1 verification)."""
        return self._last_smash_data

    def load_client_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load client-side state from dict."""
        self.client.backbone.load_state_dict(state_dict)

    def load_server_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load server-side state from dict."""
        self.server.backbone.load_state_dict(state_dict)

    def sync_from_global(self, global_state: Dict[str, torch.Tensor], cut_layer: int) -> None:
        """
        Sync client and server weights from global state dict.

        Args:
            global_state: Full model state dict.
            cut_layer: Current cut layer to determine split.
        """
        # Build layer name sets based on cut_layer
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        client_prefixes = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        server_prefixes = ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]

        client_state = {}
        server_state = {}

        for key, value in global_state.items():
            # Determine which side based on key prefix
            if any(key.startswith(p) for p in ["conv1", "bn1", "relu", "maxpool"]):
                client_state[key] = value
            elif any(key.startswith(p) for p in ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]):
                # Need to determine split based on layerX.Y pattern
                if key.startswith("layer"):
                    # Extract layer number from key like "layer1.0.conv1"
                    parts = key.split(".")
                    layer_num = int(parts[0].replace("layer", ""))
                    if layer_num <= cut_layer:
                        client_state[key] = value
                    else:
                        server_state[key] = value
                elif key.startswith(("avgpool", "fc")):
                    server_state[key] = value

        if client_state:
            self.client.backbone.load_state_dict(client_state, strict=False)
        if server_state:
            self.server.backbone.load_state_dict(server_state, strict=False)