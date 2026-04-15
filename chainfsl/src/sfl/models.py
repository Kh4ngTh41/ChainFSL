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
        if cut_layer > 4:
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
        """Standard forward (no split) through full model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
        self._saved_activation: Optional[torch.Tensor] = None
        self._saved_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to cut layer.

        Returns:
            Smashed data (activation tensor) — fully detached, no autograd graph.
        """
        with torch.no_grad():
            self._saved_input = x.detach().clone()
            a = self.backbone(x)
        # Clone and ensure contiguous — no requires_grad, no graph connection
        self._saved_activation = a.clone().contiguous()
        return self._saved_activation

    def backward(self, grad_a: torch.Tensor) -> None:
        """
        Apply gradient from server to update client parameters.

        In split learning, grad_a is dL/d(activation) from server.
        We need to compute dL/d(params) using chain rule through activation.

        Args:
            grad_a: Gradient of loss w.r.t. smashed activation (from server).
        """
        self.optimizer.zero_grad()

        if self._saved_input is None or self._saved_activation is None or grad_a is None:
            self.optimizer.step()
            self._saved_input = None
            self._saved_activation = None
            return

        # Rebuild computation graph: input → ... → activation
        # Clone saved input to avoid graph reuse issues
        x = self._saved_input.clone().detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            output = self.backbone(x)
            # Compute dL/d(params) via chain rule through actual forward graph
            # grad_outputs=grad_a connects server's upstream gradient at output
            params = list(self.backbone.parameters())
            grads = torch.autograd.grad(
                outputs=[output],
                inputs=params,
                grad_outputs=[grad_a.to(output.device)],
                retain_graph=False,
                allow_unused=False,
            )
            # Assign gradients (replace, not accumulate)
            for p, g in zip(params, grads):
                if g is not None:
                    if p.grad is not None:
                        p.grad.zero_()
                    p.grad = g

        self.optimizer.step()
        self._saved_input = None
        self._saved_activation = None

    def update(self, gradients: torch.Tensor) -> None:
        """Direct gradient application (unused in SFL flow)."""
        self.optimizer.zero_grad()
        if self._saved_activation is not None and gradients is not None:
            torch.autograd.backward(
                tensors=[self._saved_activation.detach()],
                grad_tensors=[gradients],
            )
        self.optimizer.step()
        self._saved_activation = None

    @property
    def device(self) -> torch.device:
        """Device of the underlying model."""
        params = list(self.backbone.parameters())
        return torch.device("cpu") if not params else params[0].device


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
        params = list(backbone.parameters())
        if not params:
            self.optimizer = None  # No parameters (e.g., Identity placeholder for cut_layer=4)
        else:
            self.optimizer = optimizer_cls(params, lr=lr, momentum=momentum)

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
        # All autograd computation inside one isolated context
        with torch.set_grad_enabled(True):
            smashed_input = smashed.detach().clone().requires_grad_(True)
            dev = self.device
            if smashed_input.device != dev:
                smashed_input = smashed_input.to(dev)

            output = self.model(smashed_input)
            loss = self.criterion(output, labels)

            # Compute d(loss)/d(smashed_input) correctly
            # Use outputs=[loss] and let backward compute proper gradients
            grad = torch.autograd.grad(
                outputs=[loss],
                inputs=[smashed_input],
                retain_graph=False,
            )[0]

        return loss.item(), grad.detach()

    @property
    def model(self) -> nn.Module:
        return self.backbone

    @property
    def device(self) -> torch.device:
        params = list(self.backbone.parameters())
        return torch.device("cpu") if not params else params[0].device


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