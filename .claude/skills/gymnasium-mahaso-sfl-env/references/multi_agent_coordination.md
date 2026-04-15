# Multi-Agent Coordination trong MA-HASO

Mở rộng SFLNodeEnv thành môi trường multi-agent với communication và coordination.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Global Coordinator (Optional)         │
│  - Aggregate Shapley values                     │
│  - Broadcast global model updates               │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐         ┌────▼────┐
   │ Agent 1 │◄────────┤ Agent 2 │
   │ (Node)  │  P2P    │ (Node)  │
   └────┬────┘  Comm   └────┬────┘
        │                   │
        └─────────┬─────────┘
                  │
             ┌────▼────┐
             │ Agent N │
             └─────────┘
```

## Multi-Agent Environment

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple

class MultiAgentSFLEnv(gym.Env):
    def __init__(self, num_agents=5, num_layers=10, communication_enabled=True):
        super().__init__()
        self.num_agents = num_agents
        self.num_layers = num_layers
        self.comm_enabled = communication_enabled
        
        # Mỗi agent có observation và action space riêng
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([100.0, 100.0, 100.0, 1000.0, 10.0, 5.0, 1.0]),
                dtype=np.float32
            ) for i in range(num_agents)
        })
        
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.MultiDiscrete([num_layers, 8, 5, num_agents])
            for i in range(num_agents)
        })
        
        # Communication graph: adjacency matrix
        self.comm_graph = np.ones((num_agents, num_agents)) - np.eye(num_agents)
        
        # Shared state
        self.global_model_loss = 5.0
        self.round_number = 0
        
    def step(self, actions: Dict[str, np.ndarray]):
        """
        Args:
            actions: Dict mapping "agent_i" -> action array
        
        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # Phase 1: Local computation
        local_results = {}
        for agent_id in range(self.num_agents):
            agent_key = f"agent_{agent_id}"
            action = actions[agent_key]
            
            cut_layer, batch_idx, H_idx, target = action
            batch_size = [4, 8, 16, 32, 64, 128, 256, 512][batch_idx]
            H = [1, 2, 3, 5, 10][H_idx]
            
            # Local training
            local_loss, resource_cost = self._local_training(
                agent_id, cut_layer, batch_size, H
            )
            local_results[agent_id] = {
                'loss': local_loss,
                'cost': resource_cost,
                'target': target,
                'cut_layer': cut_layer
            }
        
        # Phase 2: Communication and aggregation
        if self.comm_enabled:
            aggregated_updates = self._communicate_and_aggregate(local_results, actions)
        else:
            aggregated_updates = local_results
        
        # Phase 3: Compute rewards with Shapley values
        shapley_values = self._compute_multi_agent_shapley(aggregated_updates)
        
        for agent_id in range(self.num_agents):
            agent_key = f"agent_{agent_id}"
            
            # Reward: -local_cost + shapley_weighted_global_gain
            local_cost = local_results[agent_id]['cost']
            global_gain = self.global_model_loss - aggregated_updates[agent_id]['loss']
            shapley_weight = shapley_values[agent_id]
            
            rewards[agent_key] = -local_cost + shapley_weight * global_gain
            
            # Update observation
            observations[agent_key] = self._get_agent_observation(agent_id)
            
            # Termination
            terminateds[agent_key] = self._check_agent_termination(agent_id)
            truncateds[agent_key] = self.round_number >= 100
            
            infos[agent_key] = {
                'shapley_value': shapley_weight,
                'local_cost': local_cost,
                'global_gain': global_gain
            }
        
        # Update global state
        self.global_model_loss = np.mean([r['loss'] for r in aggregated_updates.values()])
        self.round_number += 1
        
        return observations, rewards, terminateds, truncateds, infos
```

## Communication Protocol

```python
    def _communicate_and_aggregate(self, local_results, actions):
        """
        Simulate P2P communication between agents.
        """
        aggregated = {}
        
        for agent_id in range(self.num_agents):
            target = local_results[agent_id]['target']
            
            # Check if communication link exists
            if self.comm_graph[agent_id, target] > 0:
                # Successful communication
                # Aggregate với target node
                if target not in aggregated:
                    aggregated[target] = []
                aggregated[target].append(local_results[agent_id])
            else:
                # Communication failed, fallback to local
                aggregated[agent_id] = [local_results[agent_id]]
        
        # Average results for each target
        final_results = {}
        for target, results_list in aggregated.items():
            final_results[target] = {
                'loss': np.mean([r['loss'] for r in results_list]),
                'cost': np.sum([r['cost'] for r in results_list]),
                'cut_layer': results_list[0]['cut_layer']  # Use first agent's choice
            }
        
        return final_results
```

## Multi-Agent Shapley Value

```python
    def _compute_multi_agent_shapley(self, aggregated_updates):
        """
        Compute Shapley value cho mỗi agent dựa trên contribution to global model.
        """
        shapley_values = {}
        
        # Simplified: Shapley proportional to improvement contribution
        total_improvement = 0
        improvements = {}
        
        for agent_id, result in aggregated_updates.items():
            # Improvement = previous_loss - current_loss
            improvement = max(0, self.global_model_loss - result['loss'])
            improvements[agent_id] = improvement
            total_improvement += improvement
        
        # Normalize to get Shapley values
        if total_improvement > 0:
            for agent_id in range(self.num_agents):
                if agent_id in improvements:
                    shapley_values[agent_id] = improvements[agent_id] / total_improvement
                else:
                    shapley_values[agent_id] = 0.0
        else:
            # Uniform distribution if no improvement
            for agent_id in range(self.num_agents):
                shapley_values[agent_id] = 1.0 / self.num_agents
        
        return shapley_values
```

## Coordination Strategies

### 1. Centralized Coordination

```python
class CentralizedCoordinator:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.global_policy = None  # Trained globally
    
    def assign_tasks(self, observations):
        """
        Central planner assigns cut_layer and target_node to each agent.
        """
        assignments = {}
        
        # Solve optimization problem: minimize total cost, maximize coverage
        # Simplified: greedy assignment
        available_resources = self._compute_available_resources(observations)
        
        for agent_id in range(self.num_agents):
            # Assign based on current resources
            if available_resources[agent_id]['energy'] > 50:
                cut_layer = self.num_layers // 2  # Deep cut for high-resource nodes
            else:
                cut_layer = self.num_layers // 4  # Shallow cut for low-resource
            
            # Target: node with most resources
            target = max(available_resources.keys(), 
                        key=lambda x: available_resources[x]['cpu'])
            
            assignments[agent_id] = {'cut_layer': cut_layer, 'target': target}
        
        return assignments
```

### 2. Decentralized Coordination với Message Passing

```python
class MessagePassingCoordination:
    def __init__(self, num_agents, comm_graph):
        self.num_agents = num_agents
        self.comm_graph = comm_graph
        self.message_buffer = {i: [] for i in range(num_agents)}
    
    def exchange_messages(self, observations, actions):
        """
        Agents exchange intentions and adjust actions.
        """
        # Phase 1: Broadcast intentions
        for agent_id in range(self.num_agents):
            neighbors = np.where(self.comm_graph[agent_id] > 0)[0]
            message = {
                'sender': agent_id,
                'intended_target': actions[f'agent_{agent_id}'][3],
                'resource_status': observations[f'agent_{agent_id}'][:3]  # CPU, RAM, Energy
            }
            for neighbor in neighbors:
                self.message_buffer[neighbor].append(message)
        
        # Phase 2: Adjust actions based on messages
        adjusted_actions = {}
        for agent_id in range(self.num_agents):
            messages = self.message_buffer[agent_id]
            
            # Conflict resolution: if multiple agents target same node
            my_target = actions[f'agent_{agent_id}'][3]
            conflicting = [m for m in messages if m['intended_target'] == my_target]
            
            if len(conflicting) > 2:  # Too many targeting same node
                # Choose alternative target
                alternative = self._find_alternative_target(agent_id, messages)
                adjusted_actions[f'agent_{agent_id}'] = actions[f'agent_{agent_id}'].copy()
                adjusted_actions[f'agent_{agent_id}'][3] = alternative
            else:
                adjusted_actions[f'agent_{agent_id}'] = actions[f'agent_{agent_id}']
            
            # Clear buffer
            self.message_buffer[agent_id] = []
        
        return adjusted_actions
```

## Training với Multi-Agent RL

```python
# Sử dụng QMIX hoặc MAPPO
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(MultiAgentSFLEnv, env_config={"num_agents": 5})
    .multi_agent(
        policies={
            f"agent_{i}": (None, obs_space, act_space, {})
            for i in range(5)
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .resources(num_gpus=1)
)

algo = config.build()

for i in range(100):
    result = algo.train()
    print(f"Iteration {i}, reward: {result['episode_reward_mean']}")
```

## Consensus Mechanisms

Để đảm bảo fairness trong multi-agent setting:

```python
def consensus_on_shapley(agents_shapley_estimates, tolerance=0.05):
    """
    Agents vote on Shapley values until consensus.
    """
    num_agents = len(agents_shapley_estimates)
    consensus_values = {i: 0 for i in range(num_agents)}
    
    # Iterative averaging
    for iteration in range(10):
        new_estimates = {i: [] for i in range(num_agents)}
        
        for agent_id, estimates in agents_shapley_estimates.items():
            # Share estimates with neighbors
            for other_id in range(num_agents):
                if other_id != agent_id:
                    new_estimates[other_id].append(estimates[other_id])
        
        # Update estimates
        converged = True
        for agent_id in range(num_agents):
            old_value = consensus_values[agent_id]
            consensus_values[agent_id] = np.mean(new_estimates[agent_id])
            if abs(consensus_values[agent_id] - old_value) > tolerance:
                converged = False
        
        if converged:
            break
    
    return consensus_values
```

## Khi nào dùng Multi-Agent?

- **Decentralized FL**: Không có central server, nodes peer-to-peer
- **Heterogeneous devices**: Mỗi agent có policy khác nhau dựa trên hardware
- **Dynamic topology**: Network thay đổi thường xuyên (mobile, vehicular)
- **Privacy-preserving**: Agents không muốn share raw observations

**Trade-off**: Complexity cao hơn, training chậm hơn, nhưng robust và scalable hơn.