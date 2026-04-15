# Gradient Flow in Split Learning

## The Core Challenge

PyTorch's autograd builds a computational graph during the forward pass. When you `.detach()` a tensor (necessary for network transmission), you sever the graph. The server's backward pass computes gradients w.r.t. the received activations, but those gradients can't automatically flow back to the client.

## Solution 1: Manual Gradient Passing

The simplest approach for simulation:

```python
# Client forward
activations = client_model(data)
activations.retain_grad()  # Keep gradients for this intermediate tensor

# Server forward + backward
output = server_model(activations)
loss = criterion(output, labels)
loss.backward()  # Computes activations.grad automatically

# Access gradient at cut layer
grad_at_cut = activations.grad

# If you need to continue client-side backprop manually:
client_output.backward(grad_at_cut)
```

**When to use:** Single-threaded simulation, educational purposes, debugging.

## Solution 2: Custom Autograd Function

For realistic network simulation with queues:

```python
class SplitBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, client_id):
        ctx.client_id = client_id
        # Detach and send over network
        return x.detach().requires_grad_(True)
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output received from server
        # Return it as-is to continue client backprop
        return grad_output, None

# Usage in client forward:
activations = client_layers(x)
activations = SplitBoundary.apply(activations, client_id)
```

**When to use:** Multi-threaded simulation, realistic network delay modeling, production-like code.

## Solution 3: Hooks (Advanced)

```python
class ClientModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.grad_from_server = None
        
        # Register hook on last layer
        self.layers[-1].register_full_backward_hook(self._backward_hook)
    
    def _backward_hook(self, module, grad_input, grad_output):
        # Replace grad_output with gradient from server
        if self.grad_from_server is not None:
            return (self.grad_from_server,)
        return grad_output
    
    def forward(self, x):
        return self.layers(x)
```

**When to use:** Complex architectures where autograd functions are cumbersome.

## Verification

Always verify gradient flow:

```python
# Check client parameters receive gradients
for name, param in client_model.named_parameters():
    assert param.grad is not None, f"{name} has no gradient"
    assert not torch.isnan(param.grad).any(), f"{name} has NaN gradients"

# Check gradient magnitudes are reasonable
grad_norm = torch.nn.utils.clip_grad_norm_(client_model.parameters(), float('inf'))
print(f"Client gradient norm: {grad_norm:.4f}")  # Should be > 0 and < 1000
```

## Common Errors

**"RuntimeError: element 0 of tensors does not require grad"**
- Fix: Add `.requires_grad_(True)` when reconstructing received tensors

**"RuntimeError: Trying to backward through the graph a second time"**
- Fix: Use `.detach()` before sending, or set `retain_graph=True`

**Gradients are None**
- Fix: Ensure `retain_grad()` on intermediate tensors, or use hooks

**Gradients don't flow to client**
- Fix: Check that server's backward pass actually computes gradients for the input tensor (not just parameters)
