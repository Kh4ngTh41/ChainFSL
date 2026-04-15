#!/usr/bin/env python
"""Quick test to verify gradient fixes work."""
import sys
sys.path.insert(0, 'f:/ChainFSL/chainfsl/src')

import torch
import torch.nn as nn
from sfl.models import SplittableResNet18, ClientModel, ServerModel

print("Testing gradient flow fixes...")

# Test 1: SplittableResNet18 forward
print("\n=== Test 1: SplittableResNet18 forward ===")
model = SplittableResNet18(n_classes=10, cut_layer=2)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"Output shape: {out.shape}")

# Test 2: ClientModel forward stores input
print("\n=== Test 2: ClientModel forward stores input ===")
client_backbone, _ = model.split_models(2)
client = ClientModel(client_backbone, cut_layer=2)
smash = client.forward(x)
print(f"Smash shape: {smash.shape}")
print(f"_saved_input exists: {client._saved_input is not None}")
print(f"_saved_activation exists: {client._saved_activation is not None}")

# Test 3: ServerModel forward_backward
print("\n=== Test 3: ServerModel forward_backward ===")
_, server_backbone = model.split_models(2)
server = ServerModel(server_backbone, criterion=nn.CrossEntropyLoss())
labels = torch.randint(0, 10, (2,))
loss, grad = server.forward_backward(smash, labels)
print(f"Loss: {loss:.4f}")
print(f"Grad shape: {grad.shape}")
print(f"Grad device: {grad.device}")

# Test 4: ClientModel backward
print("\n=== Test 4: ClientModel backward ===")
# Use a fresh client to avoid _saved_input being None
client2 = ClientModel(client_backbone, cut_layer=2)
smash2 = client2.forward(x)
loss2, grad2 = server.forward_backward(smash2, labels)
print(f"Calling backward with grad shape: {grad2.shape}")
client2.backward(grad2)
print("Backward completed without error!")

print("\n=== All gradient tests PASSED ===")