"""
Split-Federated Learning models.

Provides ResNet-18 with cut-layer support, client/server model wrappers,
and memory profiling utilities.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


# ---------------------------------------------------------------------------
# Splittable ResNet-18
# ---------------------------------------------------------------------------

class SplittableResNet18(nn.Module):
    """
    ResNet-18 with splittable architecture for Split Federated Learning.

    Supports split at residual block boundaries: cut_layer 1-4.
    cut_layer=0 means no split (full model stays on client).

    Split points map to network layers:
      - cut_layer=1: conv1+bn1+relu+maxpool+layer1
      - cut_layer=2: +layer2
      - cut_layer=3: +layer3
      - cut_layer=4: +layer4 (near-full model on client, only avgpool+fc on server)

    Attributes:
        n_classes: Number of output classes.
        cut_layer: Current split layer (1-4), 0 = no split.
    """

    # Canonical split points (residual block boundaries)
    CUT_LAYER_BOUNDARIES = (1, 2, 3, 4)

    # Memory requirements per cut layer (MB, batch=32, input=224x224x3)
    # Includes: activations + gradients + model params (no optimizer state)
    MEMORY_ESTIMATES_MB: dict[int, float] = {
        1: 150.0,
        2: 300.0,
        3: 500.0,
        4: 700.0,
    }

    # Memory with optimizer state (Adam: 3x gradient memory)
    MEMORY_WITH_ADAM_MB: dict[int, float] = {
        1: 150.0 + 150.0 * 2,   # approx: activations + 2x gradients
        2: 300.0 + 300.0 * 2,
        3: 500.0 + 500.0 * 2,
        4: 700.0 + 700.0 * 2,
    }

    def __init__(self, n_classes: int = 10, cut_layer: int = 2):
        """
        Args:
            n_classes: Number of classification classes.
            cut_layer: Split point (1-4). 0 = no split.
        """
        super().__init__()
        self.n_classes = n_classes
        self.cut_layer = cut_layer

        # Build full ResNet-18
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(512, n_classes)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def get_client_model(self, cut_layer: int) -> nn.Sequential:
        """
        Extract client-side sub-network up to cut_layer.

        Args:
            cut_layer: Split point (1-4).

        Returns:
            nn.Sequential containing client layers.
        """
        layers = [self.conv1, self.bn1, self.relu, self.maxpool]
        block_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(cut_layer, 4)):
            layers.append(block_layers[i])
        return nn.Sequential(*layers)

    def get_server_model(self, cut_layer: int) -> nn.Sequential:
        """
        Extract server-side sub-network from cut_layer+1 to end.

        Args:
            cut_layer: Split point (1-4).

        Returns:
            nn.Sequential containing server layers (avgpool through fc).
        """
        if cut_layer >= 4:
            # Nothing left for server
            return nn.Sequential(nn.Identity())

        block_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        layers = []
        for i in range(cut_layer, 4):
            layers.append(block_layers[i])
        layers += [self.avgpool, nn.Flatten(start_dim=1), self.fc]
        return nn.Sequential(*layers)

    def split_models(self, cut_layer: int) -> tuple[nn.Sequential, nn.Sequential]:
        """
        Get both client and server models for a given cut layer.

        Args:
            cut_layer: Split point (1-4).

        Returns:
            (client_model, server_model) tuple.
        """
        return self.get_client_model(cut_layer), self.get_server_model(cut_layer)

    @staticmethod
    def memory_requirement_mb(cut_layer: int, include_optimizer: bool = False) -> float:
        """
        Estimated RAM to train client-side at cut_layer.

        Args:
            cut_layer: Split point (1-4).
            include_optimizer: If True, include Adam optimizer state (3x gradients).

        Returns:
            Memory requirement in MB.
        """
        base = SplittableResNet18.MEMORY_ESTIMATES_MB.get(cut_layer, 800.0)
        if include_optimizer:
            base = SplittableResNet18.MEMORY_WITH_ADAM_MB.get(cut_layer, base * 3)
        return base

    @staticmethod
    def smashed_data_size(cut_layer: int, batch_size: int = 32) -> int:
        """
        Estimate smashed data (activation) size in bytes.

        Args:
            cut_layer: Split point (1-4).
            batch_size: Batch size.

        Returns:
            Estimated size in bytes.
        """
        # Feature map dimensions at each residual block
        size_map = {
            1: 64 * 56 * 56 * 4,    # layer1: 64 ch, 56x56
            2: 128 * 28 * 28 * 4,   # layer2: 128 ch, 28x28
            3: 256 * 14 * 14 * 4,   # layer3: 256 ch, 14x14
            4: 512 * 7 * 7 * 4,     # layer4: 512 ch, 7x7
        }
        channels_h_w = size_map.get(cut_layer, 512 * 7 * 7 * 4)
        return channels_h_w * batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (no split)."""
        return super().forward(x)


# ---------------------------------------------------------------------------
# ClientModel / ServerModel Wrappers
# ---------------------------------------------------------------------------

class ClientModel:
    """
    Client-side model wrapper for split learning.

    Handles forward pass to cut layer and backward from server gradient.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cut_layer: int,
        optimizer_cls=torch.optim.SGD,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.backbone = backbone
        self.cut_layer = cut_layer
        self.optimizer = optimizer_cls(backbone.parameters(), lr=lr, momentum=momentum)
        self._last_activation: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to cut layer.

        Returns:
            Smashed data (activation tensor) detached for transmission.
        """
        with torch.enable_grad():
            a = self.backbone(x)
        self._last_activation = a
        return a.detach().requires_grad_(True)

    def backward(self, grad_a: torch.Tensor) -> None:
        """
        Backward pass using gradient from server.

        Args:
            grad_a: Gradient of loss w.r.t. smashed activation.
        """
        self.optimizer.zero_grad()
        if self._last_activation is not None and self._last_activation.grad is not None:
            self._last_activation.backward(grad_a)
        elif self._last_activation is not None:
            self._last_activation.backward(grad_a)
        self.optimizer.step()

    def update(self, gradients: torch.Tensor) -> None:
        """
        Apply gradients directly to parameters (alternative to backward).

        Args:
            gradients: Gradient tensor from server.
        """
        self.optimizer.zero_grad()
        if self._last_activation is not None:
            self._last_activation.backward(gradients)
        self.optimizer.step()

    @property
    def device(self) -> torch.device:
        """Device of the underlying model."""
        return next(self.backbone.parameters()).device


class ServerModel:
    """
    Server-side model wrapper for split learning.

    Handles forward pass from smashed data and backward to compute
    gradients for client.
    """

    def __init__(
        self,
        backbone: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer_cls=torch.optim.SGD,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.backbone = backbone
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(backbone.parameters(), lr=lr, momentum=momentum)

    def forward_backward(
        self, smashed: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """
        Complete forward + backward pass on server.

        Args:
            smashed: Activation tensor from client.
            labels: Ground truth labels.

        Returns:
            (loss_value, gradient_at_cut_layer) tuple.
        """
        self.optimizer.zero_grad()
        smashed = smashed.requires_grad_(True)
        output = self.model(smashed)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()

        grad = smashed.grad.detach() if smashed.grad is not None else torch.zeros_like(smashed)
        return loss.item(), grad

    @property
    def model(self) -> nn.Module:
        return self.backbone

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_layer_count(model: nn.Module) -> int:
    """Count learnable parameter layers in a model."""
    return sum(1 for _ in model.parameters())


def model_size_mb(model: nn.Module, dtype=torch.float32) -> float:
    """
    Compute model size in MB.

    Args:
        model: nn.Module.
        dtype: Data type for size calculation.

    Returns:
        Size in megabytes.
    """
    total = sum(p.numel() * dtype.itemsize for p in model.parameters())
    return total / (1024 ** 2)