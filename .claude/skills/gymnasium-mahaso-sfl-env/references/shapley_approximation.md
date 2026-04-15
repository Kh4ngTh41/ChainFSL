# Shapley Value Approximation trong Federated Learning

Shapley value chính xác yêu cầu tính toán trên tất cả $2^n - 1$ coalitions, không khả thi với $n$ lớn. Tài liệu này trình bày các phương pháp approximation hiệu quả.

## Monte Carlo Shapley Value

```python
import numpy as np
from itertools import permutations

def monte_carlo_shapley(nodes, utility_function, num_samples=1000):
    """
    Approximate Shapley value using Monte Carlo sampling.
    
    Args:
        nodes: List of node IDs
        utility_function: Function that takes a coalition and returns utility
        num_samples: Number of permutation samples
    
    Returns:
        Dictionary mapping node_id -> shapley_value
    """
    n = len(nodes)
    shapley_values = {node: 0.0 for node in nodes}
    
    for _ in range(num_samples):
        # Sample a random permutation
        perm = np.random.permutation(nodes)
        
        # Compute marginal contribution for each node in this permutation
        coalition = []
        prev_utility = utility_function(coalition)
        
        for node in perm:
            coalition.append(node)
            curr_utility = utility_function(coalition)
            marginal = curr_utility - prev_utility
            shapley_values[node] += marginal
            prev_utility = curr_utility
    
    # Average over all samples
    for node in nodes:
        shapley_values[node] /= num_samples
    
    return shapley_values
```

## Truncated Monte Carlo (TMC-Shapley)

Dừng sớm khi marginal contribution convergence:

```python
def tmc_shapley(nodes, utility_function, tolerance=0.01, max_samples=5000):
    n = len(nodes)
    shapley_values = {node: [] for node in nodes}
    
    for sample_idx in range(max_samples):
        perm = np.random.permutation(nodes)
        coalition = []
        prev_utility = utility_function(coalition)
        
        for node in perm:
            coalition.append(node)
            curr_utility = utility_function(coalition)
            marginal = curr_utility - prev_utility
            shapley_values[node].append(marginal)
            prev_utility = curr_utility
        
        # Check convergence every 100 samples
        if sample_idx > 100 and sample_idx % 100 == 0:
            converged = True
            for node in nodes:
                values = shapley_values[node]
                recent_mean = np.mean(values[-100:])
                overall_mean = np.mean(values)
                if abs(recent_mean - overall_mean) > tolerance:
                    converged = False
                    break
            if converged:
                print(f"Converged after {sample_idx} samples")
                break
    
    return {node: np.mean(vals) for node, vals in shapley_values.items()}
```

## Federated Shapley Value với Gradient-Based Utility

```python
def federated_utility(coalition, validation_data, global_model, node_data):
    """
    Utility function cho FL: validation accuracy sau khi aggregate từ coalition.
    
    Args:
        coalition: List of node IDs in coalition
        validation_data: Validation dataset
        global_model: Current global model
        node_data: Dict mapping node_id -> local dataset
    """
    if len(coalition) == 0:
        return evaluate(global_model, validation_data)
    
    # Aggregate gradients/models from coalition nodes
    aggregated_model = aggregate_models(
        [train_local(global_model, node_data[node]) for node in coalition]
    )
    
    # Evaluate on validation set
    accuracy = evaluate(aggregated_model, validation_data)
    return accuracy

# Sử dụng
nodes = [0, 1, 2, 3, 4]
utility_fn = lambda coalition: federated_utility(
    coalition, val_data, model, node_datasets
)
shapley_vals = monte_carlo_shapley(nodes, utility_fn, num_samples=500)
```

## Group-Based Approximation

Nhóm các node tương đồng để giảm search space:

```python
from sklearn.cluster import KMeans

def group_shapley(nodes, node_features, utility_function, num_groups=3):
    """
    Cluster nodes into groups, compute Shapley at group level.
    """
    # Cluster nodes based on features (e.g., data distribution, resources)
    kmeans = KMeans(n_clusters=num_groups)
    groups = kmeans.fit_predict(node_features)
    
    # Compute Shapley for each group
    group_ids = list(range(num_groups))
    group_utility = lambda coalition: utility_function(
        [node for node, g in zip(nodes, groups) if g in coalition]
    )
    group_shapley = monte_carlo_shapley(group_ids, group_utility, num_samples=200)
    
    # Distribute group Shapley to individual nodes
    node_shapley = {}
    for node, group in zip(nodes, groups):
        # Proportional to node's contribution within group
        node_shapley[node] = group_shapley[group] / np.sum(groups == group)
    
    return node_shapley
```

## Incremental Shapley Update

Cập nhật Shapley value khi có node mới join/leave:

```python
class IncrementalShapley:
    def __init__(self, utility_function):
        self.utility_fn = utility_function
        self.shapley_values = {}
        self.coalition_history = []  # (coalition, utility) pairs
    
    def add_node(self, new_node, num_samples=100):
        """Add new node and update Shapley values."""
        nodes = list(self.shapley_values.keys()) + [new_node]
        
        # Sample coalitions containing new_node
        for _ in range(num_samples):
            # Random coalition from existing nodes
            coalition_size = np.random.randint(0, len(nodes))
            coalition = np.random.choice(
                [n for n in nodes if n != new_node], 
                size=coalition_size, 
                replace=False
            ).tolist()
            
            utility_without = self.utility_fn(coalition)
            utility_with = self.utility_fn(coalition + [new_node])
            marginal = utility_with - utility_without
            
            # Update new node's Shapley
            if new_node not in self.shapley_values:
                self.shapley_values[new_node] = 0
            self.shapley_values[new_node] += marginal / num_samples
    
    def remove_node(self, node):
        """Remove node and redistribute Shapley values."""
        if node in self.shapley_values:
            removed_value = self.shapley_values.pop(node)
            # Redistribute proportionally to remaining nodes
            if len(self.shapley_values) > 0:
                for n in self.shapley_values:
                    self.shapley_values[n] += removed_value / len(self.shapley_values)
```

## Lựa chọn phương pháp

- **Monte Carlo**: Universal, dễ implement, cần 500-1000 samples cho n < 20
- **TMC-Shapley**: Nhanh hơn 2-3x khi convergence sớm, phù hợp với utility function ổn định
- **Group-based**: Tốt nhất cho n > 50, yêu cầu feature similarity metrics
- **Incremental**: Khi topology thay đổi thường xuyên (mobile networks)

## Complexity Comparison

| Method | Time Complexity | Space | Accuracy |
|--------|----------------|-------|----------|
| Exact | O(2^n) | O(n) | 100% |
| Monte Carlo | O(K·n) | O(n) | ~95% (K=1000) |
| TMC-Shapley | O(K'·n), K' < K | O(n·K') | ~95% |
| Group-based | O(K·g + clustering), g << n | O(n) | ~85% |
| Incremental | O(K·n) per update | O(n + history) | ~90% |

Với $n$ = số nodes, $K$ = số samples, $g$ = số groups.
