# Staleness Control Algorithms

Detailed algorithms and analysis for handling staleness in asynchronous federated learning.

## Staleness Metrics

### Temporal Staleness

**Definition**: Age of update measured in rounds or wall-clock time.

```python
temporal_staleness = current_round - update.round
```

**Use case**: Simple, works for homogeneous networks where round time is predictable.

**Limitation**: Doesn't account for varying round durations or model evolution rate.

### Version Staleness

**Definition**: Number of global model updates that occurred since local training began.

```python
version_staleness = global_model.version - update.base_version
```

**Use case**: Heterogeneous networks where rounds have variable duration.

**Advantage**: Directly measures how outdated the update is relative to current model.

### Behavioral Staleness

**Definition**: Parameter sensitivity-based measure of how much model has changed.

```python
behavioral_staleness = parameter_distance(
    global_model.params,
    update.base_model.params
)
```

**Use case**: Advanced systems where model evolution is non-uniform.

**Advantage**: Captures actual divergence, not just time passed.

## Decay Functions

### Exponential Decay

**Formula**: `weight = α^τ` where α ∈ (0,1), τ = staleness

**Properties**:
- Smooth decay
- Never reaches zero (may want hard cutoff too)
- Hyperparameter α controls decay rate

**Typical values**: α ∈ [0.9, 0.99]

```python
def exponential_decay(staleness: int, alpha: float = 0.95) -> float:
    return alpha ** staleness
```

**When to use**: General-purpose, works well for most FL scenarios.

### Polynomial Decay

**Formula**: `weight = 1 / (1 + τ)^β` where β > 0

**Properties**:
- Slower decay than exponential for small τ
- Faster decay for large τ (if β > 1)
- More tolerant of moderate staleness

```python
def polynomial_decay(staleness: int, beta: float = 2.0) -> float:
    return 1.0 / ((1 + staleness) ** beta)
```

**When to use**: Networks with high heterogeneity where some staleness is unavoidable.

### Hinge Decay

**Formula**:
```
weight = 1                    if τ ≤ threshold
weight = max(0, 1 - k(τ - threshold))  if τ > threshold
```

**Properties**:
- No penalty for staleness within threshold
- Linear decay beyond threshold
- Hard cutoff at some maximum staleness

```python
def hinge_decay(staleness: int, threshold: int = 5, slope: float = 0.2) -> float:
    if staleness <= threshold:
        return 1.0
    penalty = slope * (staleness - threshold)
    return max(0.0, 1.0 - penalty)
```

**When to use**: When you want to tolerate bounded staleness without penalty, but aggressively discount beyond that.

## Adaptive Staleness Control

### Phase-Aware Decay

Different decay strategies for different training phases.

```python
class PhaseAwareDecay:
    def __init__(self):
        self.early_phase_rounds = 100
        self.late_phase_rounds = 500
    
    def weight(self, staleness: int, current_round: int) -> float:
        if current_round < self.early_phase_rounds:
            # Early training: tolerate more staleness (exploration)
            return 0.98 ** staleness
        elif current_round < self.late_phase_rounds:
            # Mid training: standard decay
            return 0.95 ** staleness
        else:
            # Late training: strict staleness control (convergence)
            return 0.90 ** staleness if staleness < 10 else 0.0
```

**Rationale**: Early in training, gradient directions are roughly aligned even with staleness. Late in training, precision matters more.

### Loss-Based Adaptive Decay

Adjust staleness tolerance based on training progress.

```python
class LossAdaptiveDecay:
    def __init__(self, initial_alpha: float = 0.95):
        self.alpha = initial_alpha
        self.prev_loss = float('inf')
    
    def update_alpha(self, current_loss: float):
        if current_loss < self.prev_loss:
            # Loss improving: can tolerate more staleness
            self.alpha = min(0.99, self.alpha + 0.01)
        else:
            # Loss stagnating/increasing: tighten staleness control
            self.alpha = max(0.85, self.alpha - 0.02)
        self.prev_loss = current_loss
    
    def weight(self, staleness: int) -> float:
        return self.alpha ** staleness
```

**Rationale**: If training is progressing well, stale updates are less harmful. If training struggles, enforce freshness.

## Bounded Staleness

### Hard Staleness Bound

```python
class BoundedStalenessAggregator:
    def __init__(self, max_staleness: int = 10):
        self.max_staleness = max_staleness
    
    def aggregate(self, updates: List[Update]) -> Model:
        valid_updates = [
            u for u in updates 
            if u.staleness <= self.max_staleness
        ]
        
        if len(valid_updates) < MIN_UPDATES:
            raise InsufficientFreshUpdatesError()
        
        return weighted_average(valid_updates)
```

**Trade-off**: Guarantees freshness but may reject many updates in high-latency networks.

### Soft Staleness Bound with Buffer

```python
class BufferedStalenessAggregator:
    def __init__(self, preferred_staleness: int = 5, buffer_size: int = 20):
        self.preferred_staleness = preferred_staleness
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_update(self, update: Update):
        self.buffer.append(update)
        if len(self.buffer) >= self.buffer_size:
            self.aggregate_buffer()
    
    def aggregate_buffer(self):
        # Prefer fresh updates, but use stale if necessary
        fresh = [u for u in self.buffer if u.staleness <= self.preferred_staleness]
        
        if len(fresh) >= MIN_UPDATES:
            selected = fresh
        else:
            # Not enough fresh updates, use all
            selected = self.buffer
        
        global_model.update(weighted_average(selected))
        self.buffer.clear()
```

**Trade-off**: More flexible, but requires tuning buffer size and timing.

## Staleness-Aware Client Selection

Instead of just decaying stale updates, proactively select clients to minimize staleness.

```python
class StalenessAwareSelector:
    def select_clients(self, available_clients: List[Client], n: int) -> List[Client]:
        # Score clients by freshness and capability
        scored = []
        for client in available_clients:
            freshness_score = 1.0 / (1 + client.expected_staleness())
            capability_score = client.compute_power / MAX_COMPUTE
            combined_score = 0.6 * freshness_score + 0.4 * capability_score
            scored.append((combined_score, client))
        
        # Select top-n
        scored.sort(reverse=True, key=lambda x: x[0])
        return [client for _, client in scored[:n]]
```

**Rationale**: Prevent staleness at source by preferring fast clients.

## Convergence Guarantees

### Theorem (Informal)

For bounded staleness τ ≤ τ_max and strongly convex loss:

```
E[||∇L(w_t)||^2] ≤ O(1/T) + O(τ_max * σ^2)
```

Where:
- T = number of rounds
- σ^2 = gradient variance
- τ_max = maximum staleness

**Implication**: Staleness adds constant error term. To converge to ε accuracy, need τ_max = O(ε/σ^2).

### Practical Guideline

Set `max_staleness = 0.1 * expected_convergence_rounds`.

Example: If you expect convergence in 1000 rounds, set max_staleness = 100.

## Implementation Example

Complete staleness-controlled async aggregator:

```python
class AsyncFLAggregator:
    def __init__(
        self,
        max_staleness: int = 10,
        decay_fn: Callable[[int], float] = None,
        min_updates_per_round: int = 5
    ):
        self.max_staleness = max_staleness
        self.decay_fn = decay_fn or (lambda s: 0.95 ** s)
        self.min_updates = min_updates_per_round
        self.global_model_version = 0
        self.update_buffer = []
    
    def receive_update(self, update: Update):
        staleness = self.global_model_version - update.base_version
        
        if staleness > self.max_staleness:
            logger.info(f"Rejecting update with staleness {staleness}")
            return
        
        update.staleness = staleness
        update.weight = self.decay_fn(staleness)
        self.update_buffer.append(update)
        
        if len(self.update_buffer) >= self.min_updates:
            self.aggregate()
    
    def aggregate(self):
        if len(self.update_buffer) < self.min_updates:
            return
        
        # Weighted average by staleness
        total_weight = sum(u.weight for u in self.update_buffer)
        aggregated = sum(
            u.model_delta * u.weight 
            for u in self.update_buffer
        ) / total_weight
        
        # Update global model
        self.global_model.apply_delta(aggregated)
        self.global_model_version += 1
        
        # Clear buffer
        self.update_buffer.clear()
        
        logger.info(
            f"Aggregated {len(self.update_buffer)} updates, "
            f"new version {self.global_model_version}"
        )
```

## Summary

Staleness control is essential for async FL convergence:

1. **Measure staleness**: Version-based preferred over time-based
2. **Apply decay**: Exponential decay (α ≈ 0.95) works well in practice
3. **Enforce bounds**: Hard cutoff at max_staleness prevents divergence
4. **Adapt to phase**: Tolerate more staleness early, less late in training
5. **Monitor convergence**: If loss stagnates, tighten staleness control

No single algorithm fits all scenarios — tune based on network heterogeneity, model complexity, and convergence requirements.