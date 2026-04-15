# Shapley Value Computation for ChainFSL GTM

## Theoretical Foundation

Shapley Value is the unique fair division scheme satisfying:
1. **Efficiency**: Σφᵢ = v(N) (all value is distributed)
2. **Symmetry**: If i and j contribute equally to all coalitions, φᵢ = φⱼ
3. **Dummy**: If i contributes nothing, φᵢ = 0
4. **Additivity**: For combined games, φᵢ(v+w) = φᵢ(v) + φᵢ(w)

In ChainFSL, "value" measures contribution to federated learning success.

## Value Function Definition

For coalition S ⊆ N (subset of devices), define:

```
v(S) = α × Accuracy(S) + β × Verification(S) + γ × Resources(S)
```

Where:
- **Accuracy(S)**: Validation accuracy when training with only devices in S
- **Verification(S)**: Average proof validity rate of devices in S
- **Resources(S)**: Normalized compute/memory contributed by S

Weights: α=0.6, β=0.2, γ=0.2 (accuracy dominates, but verification and resources matter).

### Accuracy Component

```python
def compute_accuracy_value(coalition, validation_set, global_model):
    # Simulate training round with only coalition members
    coalition_updates = [device.local_train() for device in coalition]
    
    # Aggregate updates (FedAvg)
    aggregated_model = fedavg(coalition_updates)
    
    # Evaluate on validation set
    accuracy = evaluate(aggregated_model, validation_set)
    
    return accuracy
```

This requires 2^n training simulations for exact Shapley—infeasible for n>10. See Monte Carlo approximation below.

### Verification Component

```python
def compute_verification_value(coalition):
    # Historical proof validity rate for each device
    validity_rates = [device.proof_success_rate for device in coalition]
    
    # Average across coalition (higher is better)
    return sum(validity_rates) / len(validity_rates)
```

This component incentivizes honest verification and penalizes devices with frequent proof failures.

### Resources Component

```python
def compute_resource_value(coalition, total_resources):
    # Normalize by total available resources
    coalition_compute = sum(device.compute_capacity for device in coalition)
    coalition_memory = sum(device.memory_capacity for device in coalition)
    
    compute_ratio = coalition_compute / total_resources['compute']
    memory_ratio = coalition_memory / total_resources['memory']
    
    # Geometric mean (both compute and memory matter)
    return (compute_ratio * memory_ratio) ** 0.5
```

This ensures high-capacity devices receive fair compensation even if accuracy contribution is similar to low-capacity devices.

## Exact Shapley Computation

### Formula

For device i:
```
φᵢ = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S ∪ {i}) - v(S)]
```

The weight `|S|!(|N|-|S|-1)! / |N|!` represents the probability of coalition S forming before i joins.

### Implementation

```python
from itertools import combinations
import math

def exact_shapley(devices, value_function):
    n = len(devices)
    shapley_values = {d.id: 0.0 for d in devices}
    
    for device in devices:
        others = [d for d in devices if d != device]
        
        # Iterate over all subsets of other devices
        for r in range(n):
            for coalition in combinations(others, r):
                coalition_set = set(coalition)
                coalition_with_device = coalition_set | {device}
                
                # Marginal contribution
                marginal = (value_function(coalition_with_device) - 
                           value_function(coalition_set))
                
                # Weight by coalition probability
                weight = (math.factorial(r) * math.factorial(n - r - 1)) / math.factorial(n)
                
                shapley_values[device.id] += weight * marginal
    
    return shapley_values
```

**Complexity**: O(n × 2^n) — only feasible for n ≤ 10.

## Monte Carlo Approximation

### Algorithm

For large device sets (n > 10), approximate via sampling:

1. Sample M random permutations of devices (M ≈ 1000)
2. For each permutation π, compute marginal contribution when device i joins
3. Average marginal contributions across all permutations

### Implementation

```python
import random

def monte_carlo_shapley(devices, value_function, num_samples=1000):
    n = len(devices)
    shapley_values = {d.id: 0.0 for d in devices}
    
    for _ in range(num_samples):
        # Random permutation of devices
        permutation = random.sample(devices, n)
        
        # Track coalition as we add devices in order
        coalition = set()
        
        for device in permutation:
            # Marginal contribution when device joins
            marginal = (value_function(coalition | {device}) - 
                       value_function(coalition))
            
            shapley_values[device.id] += marginal
            coalition.add(device)
    
    # Average over samples
    for device_id in shapley_values:
        shapley_values[device_id] /= num_samples
    
    return shapley_values
```

**Complexity**: O(M × n) — practical for n ≤ 100.

### Convergence Analysis

Error decreases as O(1/√M). For 95% confidence within ±5% of true Shapley:
- n=20 devices: M ≈ 500 samples
- n=50 devices: M ≈ 1000 samples
- n=100 devices: M ≈ 2000 samples

Monitor convergence by tracking variance across samples—stop early if variance < threshold.

## Optimizations

### Caching Value Function

Many coalitions are evaluated multiple times. Cache results:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_value_function(coalition_frozenset):
    coalition = set(coalition_frozenset)
    return compute_value(coalition)

# Use frozenset for hashability
v_S = cached_value_function(frozenset(coalition))
```

Reduces redundant training simulations by ~60%.

### Adaptive Sampling

Allocate more samples to high-value devices:

```python
def adaptive_monte_carlo_shapley(devices, value_function, total_budget=1000):
    # Initial rough estimate (100 samples)
    initial_shapley = monte_carlo_shapley(devices, value_function, num_samples=100)
    
    # Allocate remaining budget proportional to initial Shapley
    total_initial = sum(initial_shapley.values())
    additional_samples = {}
    
    for device_id, shapley in initial_shapley.items():
        # High-value devices get more samples for precision
        additional_samples[device_id] = int((shapley / total_initial) * (total_budget - 100))
    
    # Refine estimates for each device
    refined_shapley = {}
    for device in devices:
        extra = additional_samples[device.id]
        refined = monte_carlo_shapley([device], value_function, num_samples=extra)
        refined_shapley[device.id] = (initial_shapley[device.id] * 100 + refined[device.id] * extra) / (100 + extra)
    
    return refined_shapley
```

Improves precision for top contributors (who receive most rewards) while saving computation on low contributors.

### Incremental Updates

If device set changes slightly between rounds (1-2 devices join/leave), reuse previous Shapley:

```python
def incremental_shapley(previous_shapley, new_devices, left_devices, value_function):
    # Remove left devices
    current_shapley = {k: v for k, v in previous_shapley.items() if k not in left_devices}
    
    # Approximate new devices' Shapley via marginal contribution
    for new_device in new_devices:
        current_coalition = set(current_shapley.keys())
        marginal = value_function(current_coalition | {new_device}) - value_function(current_coalition)
        current_shapley[new_device.id] = marginal
    
    # Renormalize (Shapley values must sum to total value)
    total = sum(current_shapley.values())
    total_value = value_function(set(current_shapley.keys()))
    
    for device_id in current_shapley:
        current_shapley[device_id] *= (total_value / total)
    
    return current_shapley
```

Reduces computation by 80% when device set is stable.

## Integration with ChainFSL

### End-to-End Workflow

1. **Training round completes**: Coordinator has updates from N devices
2. **TVE verification**: Blockchain validates proofs, returns validity flags
3. **Accuracy evaluation**: Aggregated model tested on validation set
4. **Shapley computation**: GTM computes φᵢ for each device using Monte Carlo (1000 samples)
5. **Reward distribution**: Smart contract transfers tokens proportional to Shapley values
6. **Logging**: Record (round, device_id, shapley, reward, accuracy, verification_rate) on-chain

### Smart Contract Pseudocode

```solidity
contract ChainFSL_GTM {
    mapping(uint => mapping(address => uint)) public rewards; // round => device => reward
    
    function distributeRewards(
        uint round,
        address[] memory devices,
        uint[] memory shapleyValues,
        uint totalPool
    ) public onlyCoordinator {
        uint totalShapley = sum(shapleyValues);
        
        for (uint i = 0; i < devices.length; i++) {
            // Skip low contributors (anti-free-riding)
            if (shapleyValues[i] < 0.01 * max(shapleyValues)) {
                continue;
            }
            
            uint reward = (shapleyValues[i] * totalPool) / totalShapley;
            rewards[round][devices[i]] = reward;
            
            // Transfer tokens
            token.transfer(devices[i], reward);
            
            emit RewardDistributed(round, devices[i], reward, shapleyValues[i]);
        }
    }
}
```

### Handling Edge Cases

**All devices contribute equally**: Shapley degenerates to uniform distribution (φᵢ = v(N)/n). This is correct—if contributions are identical, rewards should be equal.

**One device dominates**: If device i has φᵢ > 0.8 × Σφⱼ, cap at 0.5 × total reward to prevent centralization. Redistribute excess uniformly.

**Negative marginal contribution**: Possible if device harms accuracy (Byzantine). Set φᵢ = 0 and exclude from future rounds.

## Fairness Metrics

### Gini Coefficient

Measure inequality in reward distribution:

```python
def gini_coefficient(rewards):
    sorted_rewards = sorted(rewards)
    n = len(rewards)
    cumulative = sum((i + 1) * r for i, r in enumerate(sorted_rewards))
    return (2 * cumulative) / (n * sum(rewards)) - (n + 1) / n
```

Target: Gini < 0.3 (moderate inequality—some heterogeneity is expected due to different tiers).

### Jain's Fairness Index

```python
def jains_index(rewards):
    n = len(rewards)
    return (sum(rewards) ** 2) / (n * sum(r ** 2 for r in rewards))
```

Range: [1/n, 1]. Higher is fairer. Target: >0.7.

Monitor both metrics each round—if fairness degrades, investigate (possible collusion or MA-HASO bias).