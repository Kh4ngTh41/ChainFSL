# ChainFSL Fix Plan v2 — Correct Gradient Flow

## Problem Summary

The current implementation has critical errors in how gradients flow through the split learning pipeline:

1. **ClientModel.backward()** uses a "ghost gradient" approach that computes gradients for a synthetic loss `L' = Σ(activation × grad_a)` instead of the actual server loss `L_server = criterion(output, labels)`
2. **ServerModel.forward_backward()** uses incorrect `grad_outputs=[torch.ones_like(output)]` which assumes unit upstream gradient, but cross-entropy loss has different gradient structure

## Root Cause

```
Split Learning Gradient Flow (Correct):
  Client: x → backbone(x) = activation
  Server: activation → server_model(activation) → logits → loss
  Server backward: dL/d(activation) = grad_a (returned to client)
  Client backward: dL/d(params) = dL/d(activation) × d(activation)/d(params)

Current Ghost Gradient (Wrong):
  Client backward computes: dL'/d(params) where L' = Σ(activation_i × grad_a_i)
  This is NOT equal to dL_server/d(params)
```

## Fix Phases

### Phase 1: Fix ClientModel — store input tensor

**File:** `src/sfl/models.py`

**Change:** Add `_saved_input` attribute to store the input tensor alongside `_saved_activation`.

```python
# In __init__:
self._saved_input: Optional[torch.Tensor] = None

# In forward:
def forward(self, x):
    with torch.no_grad():
        self._saved_input = x.detach().clone()  # Store input for backward
        a = self.backbone(x)
    self._saved_activation = a.clone()
    return self._saved_activation
```

### Phase 2: Fix ClientModel.backward() — rebuild graph with proper grad injection

**Current code (WRONG):**
```python
def backward(self, grad_a):
    activation = self._saved_activation.detach().requires_grad_(True)
    loss = (activation * grad_a).sum()
    loss.backward()
```

**Fix:**
```python
def backward(self, grad_a):
    self.optimizer.zero_grad()

    if self._saved_input is None or grad_a is None:
        self.optimizer.step()
        return

    # Rebuild computation graph: input → ... → activation
    x = self._saved_input.detach().requires_grad_(True)

    with torch.set_grad_enabled(True):
        output = self.backbone(x)
        # Connect grad_a as upstream gradient at output
        # Propagates dL/d(activation)=grad_a backward through graph
        # to compute dL/d(params) correctly via chain rule
        torch.autograd.backward(
            outputs=[output],
            grad_outputs=[grad_a.to(output.device)],
            retain_graph=False,
        )

    self.optimizer.step()
    self._saved_input = None
    self._saved_activation = None
```

### Phase 3: Fix ServerModel.forward_backward() — correct grad_outputs

**Current code (WRONG):**
```python
grad = torch.autograd.grad(
    outputs=[output],
    inputs=[smashed_input],
    grad_outputs=[torch.ones_like(output)],  # WRONG: assumes unit upstream gradient
    retain_graph=False,
)[0]
```

**Fix:**
```python
def forward_backward(self, smashed, labels):
    with torch.set_grad_enabled(True):
        smashed_input = smashed.detach().clone().requires_grad_(True)
        if smashed_input.device != self.device:
            smashed_input = smashed_input.to(self.device)

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
```

## Why This Fix Is Correct

### Server fix:
- `outputs=[loss]` — we want gradient of the scalar loss
- `inputs=[smashed_input]` — we want gradient w.r.t. input
- PyTorch's autograd engine computes `d(loss)/d(smashed_input)` correctly by running `.backward()` internally

### Client fix:
- Rebuild the computation graph from stored input
- Use `grad_outputs=[grad_a]` to inject the server's gradient as the upstream gradient at the output
- This correctly computes `dL_server/d(params)` via chain rule through the actual forward computation

## Verification Checklist

- [ ] Experiment runs without crash (no segfault, no heap corruption)
- [ ] Loss decreases over 30 rounds (target: ~1.5-2.0 by round 30)
- [ ] No `RuntimeError: one of the variables needed for gradient computation has been modified`
- [ ] No `free(): corrupted unsorted chunks`
- [ ] No `AttributeError` or `KeyError` errors

## Files to Modify

| File | Line | Change |
|------|------|--------|
| `src/sfl/models.py` | ~196 | Add `self._saved_input: Optional[torch.Tensor] = None` in `__init__` |
| `src/sfl/models.py` | ~209 | Store `_saved_input` in `forward()` |
| `src/sfl/models.py` | ~222 | Rewrite `backward()` to use stored input and graph rebuild |
| `src/sfl/models.py` | ~294 | Fix `forward_backward()` to use `outputs=[loss]` |

## Remaining Issues to Address Later

1. **Dead `update()` method** — never called, can be removed
2. **`contiguous()` call** — unnecessary after `clone()`, can be removed
3. **`get_server_model()` condition** — `cut_layer > 4` should be `>= 4` for clarity
4. **No cut_layer validation** — should raise error for invalid values
5. **`forward()` duplicates ResNet implementation** — could use `super().forward(x)` if base class had it

---

## Test Plan

```bash
python -m experiments.run_experiment --exp e1 --n_nodes 5 --global_rounds 30
```

Expected behavior after fix:
1. Round 1: train_loss ~2.3 (high, random init)
2. Round 30: train_loss ~1.5-2.0 (converging)
3. No crashes or autograd errors