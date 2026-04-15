---
name: split-federated-learning-pytorch
description: >
  Implement split learning and split federated learning (SFL) architectures in PyTorch.
  Use this skill when implementing SplittableResNet, ClientModel, ServerModel, cut layer
  mechanisms, SmashData transmission, gradient detachment strategies, forward/backward pass
  coordination across client-server boundaries, Queue-based network simulation, or any
  distributed deep learning with model splitting. Also applies to federated split learning,
  privacy-preserving ML, edge AI with resource constraints, collaborative training without
  raw data sharing, or implementing cut layers in CNNs like ResNet-18/50.
---

# Split Federated Learning in PyTorch

## Core Concept

Split Learning divides a neural network at a **cut layer** into client-side and server-side sub-models. The client processes data up to the cut layer, sends activations ("smashed data") to the server, which completes forward propagation. Gradients flow back from server to client, enabling collaborative training without sharing raw data.

**Critical insight:** Detaching activations before transmission breaks the computational graph temporarily—but you must reattach gradients on the backward pass to preserve end-to-end training.

## SplittableResNet18 Architecture

ResNet-18 has a natural 4-stage structure after the initial conv/pooling:

```
conv1 + bn1 + maxpool  →  layer1 (64 ch)  →  layer2 (128 ch)  →  layer3 (256 ch)  →  layer4 (512 ch)  →  avgpool + fc
```

**Four natural cut points:**
- Cut 0: after layer1 (64 channels)
- Cut 1: after layer2 (128 channels)  
- Cut 2: after layer3 (256 channels)
- Cut 3: after layer4 (512 channels)

Earlier cuts reduce client computation but increase communication (larger activation maps). Later cuts preserve privacy better (more compact representations) but require more client-side compute.

### Implementation Pattern

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

class SplittableResNet18(nn.Module):
    def __init__(self, num_classes=10, cut_layer=2):
        super().__init__()
        base = resnet18(pretrained=False)
        self.cut_layer = cut_layer
        
        # Initial layers (always on client)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Four residual block groups
        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        self.layer3 = base.layer3  # 256 channels
        self.layer4 = base.layer4  # 512 channels
        
        # Classification head (always on server)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)
    
    def forward_client(self, x):
        """Run client-side layers up to cut point."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if self.cut_layer == 0:
            return x
        
        x = self.layer2(x)
        if self.cut_layer == 1:
            return x
        
        x = self.layer3(x)
        if self.cut_layer == 2:
            return x
        
        x = self.layer4(x)
        return x  # cut_layer == 3
    
    def forward_server(self, x):
        """Run server-side layers from cut point to output."""
        # Resume from cut point
        if self.cut_layer == 0:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.cut_layer == 1:
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.cut_layer == 2:
            x = self.layer4(x)
        # else cut_layer == 3, already have layer4 output
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

## ClientModel and ServerModel Wrappers

These wrappers handle network communication (simulated via Queue) and gradient flow.

### Key Challenge: Gradient Flow Across Network Boundary

**The problem:** When sending activations over the network, you must `.detach()` them (they're just data to transmit). But this breaks PyTorch's autograd chain.

**The solution:** Use a custom autograd function that:
1. Forward: detaches and sends activations
2. Backward: receives gradients from server and reattaches them to the computation graph

```python
import queue
import threading

class SmashData:
    """Container for activations sent across network."""
    def __init__(self, data, requires_grad=True):
        self.data = data.detach()  # Break graph for transmission
        self.requires_grad = requires_grad

class NetworkBoundary(torch.autograd.Function):
    """Custom autograd function to handle gradient flow across network."""
    
    @staticmethod
    def forward(ctx, x, grad_queue):
        ctx.grad_queue = grad_queue
        # Return detached tensor (simulates network transmission)
        return x.detach().requires_grad_(True)
    
    @staticmethod
    def backward(ctx, grad_output):
        # This grad_output comes from server's backward pass
        # Pass it back through the boundary
        return grad_output, None

class ClientModel(nn.Module):
    def __init__(self, model, cut_layer, to_server_queue, from_server_queue):
        super().__init__()
        self.model = model
        self.cut_layer = cut_layer
        self.to_server = to_server_queue
        self.from_server = from_server_queue
        self.smashed_data = None
    
    def forward(self, x):
        # Client-side forward pass
        activations = self.model.forward_client(x)
        
        # Apply network boundary (preserves gradient flow)
        activations = NetworkBoundary.apply(activations, self.from_server)
        
        # Send to server (detached copy)
        smash = SmashData(activations, requires_grad=True)
        self.to_server.put(smash)
        self.smashed_data = activations  # Keep for backward
        
        # Wait for server's output
        result = self.from_server.get()
        return result
    
    def backward_client(self, grad_from_server):
        """Receive gradients from server and continue backprop."""
        if self.smashed_data is not None:
            self.smashed_data.backward(grad_from_server)

class ServerModel(nn.Module):
    def __init__(self, model, cut_layer, from_client_queue, to_client_queue):
        super().__init__()
        self.model = model
        self.cut_layer = cut_layer
        self.from_client = from_client_queue
        self.to_client = to_client_queue
    
    def forward(self, labels=None):
        # Receive smashed data from client
        smash = self.from_client.get()
        
        # Reconstruct tensor with gradient tracking
        x = smash.data.requires_grad_(True)
        
        # Server-side forward pass
        output = self.model.forward_server(x)
        
        if labels is not None:
            # Compute loss and backward
            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            
            # Send gradients back to client
            grad_to_client = x.grad.detach()
            self.to_client.put(grad_to_client)
            
            return output, loss
        else:
            self.to_client.put(output)
            return output
```

## Training Loop Pattern

```python
def train_split_learning(client_model, server_model, train_loader, optimizer_client, optimizer_server):
    client_model.train()
    server_model.train()
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer_client.zero_grad()
        optimizer_server.zero_grad()
        
        # Client forward (sends smashed data)
        _ = client_model(data)
        
        # Server forward + backward (sends gradients back)
        output, loss = server_model(labels)
        
        # Client backward (receives gradients)
        grad_from_server = client_model.from_server.get()
        client_model.backward_client(grad_from_server)
        
        # Update both models
        optimizer_client.step()
        optimizer_server.step()
        
        if batch_idx % 10 == 0:
            print(f'Loss: {loss.item():.4f}')
```

## Simplified Alternative: Direct Gradient Passing

For simulation (not real network), you can skip Queue complexity:

```python
class SimpleClientModel(nn.Module):
    def __init__(self, model, cut_layer):
        super().__init__()
        self.model = model
        self.cut_layer = cut_layer
    
    def forward(self, x):
        return self.model.forward_client(x)

class SimpleServerModel(nn.Module):
    def __init__(self, model, cut_layer):
        super().__init__()
        self.model = model
        self.cut_layer = cut_layer
    
    def forward(self, x):
        return self.model.forward_server(x)

# Training
for data, labels in train_loader:
    optimizer.zero_grad()
    
    # Client forward
    smashed = client_model(data)
    smashed.retain_grad()  # Important!
    
    # Server forward
    output = server_model(smashed)
    loss = criterion(output, labels)
    
    # Backward (automatic through both models)
    loss.backward()
    
    optimizer.step()
```

**Key:** Use `.retain_grad()` on the cut layer activations so you can access `smashed.grad` if needed.

## Cut Layer Selection Guidelines

**Privacy vs. Efficiency Trade-off:**

| Cut Point | Client Compute | Communication | Privacy |
|-----------|----------------|---------------|----------|
| After layer1 | Low | High (large maps) | Lower (early features) |
| After layer2 | Medium | Medium | Medium |
| After layer3 | Medium-High | Low | Higher (semantic features) |
| After layer4 | High | Very Low | Highest (compact) |

**Recommendation:** Start with cut_layer=2 (after layer3) for a balanced trade-off. Earlier cuts suit resource-constrained clients; later cuts suit privacy-sensitive applications.

## Common Pitfalls

1. **Forgetting `.detach()` before transmission:** Leads to "trying to backward through the graph a second time" errors
2. **Not using `.requires_grad_(True)` on received data:** Server-side gradients won't flow
3. **Mismatched cut points:** Ensure client's output matches server's expected input shape
4. **Queue deadlocks:** In multi-threaded simulation, ensure gets/puts are properly sequenced
5. **Optimizer confusion:** Client and server need separate optimizers for their respective parameters

## Federated Split Learning Extension

For multiple clients with FedAvg-style aggregation:

```python
# After each round, aggregate client-side models
def federated_averaging(client_models):
    global_state = {}
    for key in client_models[0].state_dict():
        global_state[key] = torch.stack([
            client.state_dict()[key].float() for client in client_models
        ]).mean(0)
    return global_state

# Each client trains locally
for client_id, client_model in enumerate(client_models):
    train_split_learning(client_model, server_model, client_loaders[client_id], ...)

# Aggregate client-side parameters
global_client_state = federated_averaging(client_models)

# Distribute back to clients
for client_model in client_models:
    client_model.load_state_dict(global_client_state)
```

Server-side model can either be shared (SFL-V2) or per-client (SFL-V1). V2 is simpler and converges faster under heterogeneous data.

## Testing Your Implementation

```python
# Sanity check: split model should match full model
full_model = resnet18(num_classes=10)
split_model = SplittableResNet18(num_classes=10, cut_layer=2)

# Copy weights
split_model.load_state_dict(full_model.state_dict(), strict=False)

x = torch.randn(2, 3, 224, 224)

# Full forward
full_out = full_model(x)

# Split forward
smashed = split_model.forward_client(x)
split_out = split_model.forward_server(smashed)

assert torch.allclose(full_out, split_out, atol=1e-5)
print("✓ Split model matches full model")
```
