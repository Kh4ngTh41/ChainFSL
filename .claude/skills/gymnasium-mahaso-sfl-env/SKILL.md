---
name: gymnasium-mahaso-sfl-env
description: >
  Thiết kế môi trường Gymnasium cho hệ thống Split Federated Learning với tối ưu hóa đa tác tử phân cấp (MA-HASO). Sử dụng kỹ năng này khi xây dựng môi trường học tăng cường cho federated learning, split learning, resource allocation trong edge computing, multi-agent hierarchical optimization, hoặc khi cần triển khai reward function dựa trên Shapley value. Áp dụng cho các bài toán tối ưu hóa tài nguyên phân tán, node selection, layer splitting, batch size optimization, bandwidth allocation, và energy-aware federated learning.
---

# Tổng quan

Kỹ năng này hướng dẫn thiết kế môi trường Gymnasium tùy chỉnh cho module MA-HASO (Multi-Agent Hierarchical Adaptive Split Optimization) trong bối cảnh Split Federated Learning. Môi trường SFLNodeEnv mô phỏng quyết định phân tán của các node edge về việc chia tách mô hình, lựa chọn node cộng tác, và phân bổ tài nguyên, với reward function kết hợp chi phí tài nguyên và đóng góp Shapley.

## Kiến trúc môi trường

Môi trường kế thừa từ `gymnasium.Env` và mô hình hóa:
- **State space**: Quan sát 7 chiều về tài nguyên node và điều kiện mạng
- **Action space**: MultiDiscrete cho 4 quyết định phân cấp
- **Reward function**: Cân bằng giữa chi phí tài nguyên và cải thiện mô hình theo Shapley value
- **Dynamics**: Mô phỏng quá trình training phân tán với network latency và resource consumption

---

# Định nghĩa không gian

## State Space

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SFLNodeEnv(gym.Env):
    def __init__(self, num_nodes=5, num_layers=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # State: [CPU, RAM, Energy, Bandwidth, Loss, Loss_Std, Neighbor_Availability]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([100.0, 100.0, 100.0, 1000.0, 10.0, 5.0, 1.0]),
            dtype=np.float32
        )
```

**Giải thích từng chiều**:
- `CPU` (0-100): % CPU khả dụng của node hiện tại
- `RAM` (0-100): % RAM khả dụng
- `Energy` (0-100): % năng lượng pin còn lại (cho mobile devices)
- `Bandwidth` (0-1000): Băng thông khả dụng (Mbps)
- `Loss` (0-10): Training loss hiện tại của mô hình local
- `Loss_Std` (0-5): Độ lệch chuẩn của loss qua các epoch gần đây (đo tính ổn định)
- `Neighbor_Availability` (0-1): Tỷ lệ các node láng giềng khả dụng cho collaboration

## Action Space

```python
        # Action: [cut_layer, batch_size, H, target_node]
        self.action_space = spaces.MultiDiscrete([
            num_layers,      # cut_layer: vị trí cắt mô hình (0 to num_layers-1)
            8,               # batch_size: index trong [4, 8, 16, 32, 64, 128, 256, 512]
            5,               # H: số lượng local epochs (1, 2, 3, 5, 10)
            num_nodes        # target_node: node đích cho split learning
        ])
        
        self.batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
        self.local_epochs = [1, 2, 3, 5, 10]
```

**Ý nghĩa hành động**:
- `cut_layer`: Layer nào của neural network được cắt để chia thành client-side và server-side model
- `batch_size`: Kích thước batch cho local training (ảnh hưởng memory và convergence)
- `H`: Số epoch local trước khi gửi activations/gradients (trade-off communication vs. staleness)
- `target_node`: Node nào sẽ nhận phần server-side của split model

---

# Hàm step() và Reward Function

## Cấu trúc step()

```python
    def step(self, action):
        cut_layer = action[0]
        batch_size = self.batch_sizes[action[1]]
        H = self.local_epochs[action[2]]
        target_node = action[3]
        
        # 1. Tính toán chi phí tài nguyên
        resource_cost = self._compute_resource_cost(cut_layer, batch_size, H, target_node)
        
        # 2. Mô phỏng training step
        performance_gain = self._simulate_training(cut_layer, batch_size, H, target_node)
        
        # 3. Tính Shapley value contribution
        shapley_weight = self._compute_shapley_contribution(target_node, performance_gain)
        
        # 4. Reward theo Eq. 7: -cost + shapley_weighted_gain
        reward = -resource_cost + shapley_weight * performance_gain
        
        # 5. Cập nhật state
        self._update_state(cut_layer, batch_size, H, target_node)
        
        # 6. Kiểm tra điều kiện kết thúc
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        info = {
            'resource_cost': resource_cost,
            'performance_gain': performance_gain,
            'shapley_weight': shapley_weight,
            'cut_layer': cut_layer,
            'target_node': target_node
        }
        
        return self.state, reward, terminated, truncated, info
```

## Chi phí tài nguyên

```python
    def _compute_resource_cost(self, cut_layer, batch_size, H, target_node):
        # Computation cost: phụ thuộc vào cut_layer và batch_size
        comp_cost = (cut_layer / self.num_layers) * (batch_size / 512.0) * 0.3
        
        # Communication cost: kích thước activation tại cut point
        # Giả sử activation size giảm theo depth
        activation_size = (1.0 - cut_layer / self.num_layers) * batch_size * 1024  # KB
        comm_cost = activation_size / self.state[3]  # chia cho bandwidth
        
        # Energy cost: tỷ lệ với computation và communication
        energy_cost = (comp_cost + comm_cost) * 0.1 / (self.state[2] / 100.0)
        
        # Penalty nếu target_node quá tải (neighbor availability thấp)
        availability_penalty = (1.0 - self.state[6]) * 0.2
        
        return comp_cost + comm_cost + energy_cost + availability_penalty
```

**Lý do**: Chi phí phản ánh 3 yếu tố then chốt trong edge federated learning:
1. **Computation**: Cut sâu hơn = client tính nhiều hơn
2. **Communication**: Activation lớn + bandwidth thấp = latency cao
3. **Energy**: Quan trọng cho mobile/IoT devices

## Shapley Value Contribution

```python
    def _compute_shapley_contribution(self, target_node, performance_gain):
        # Shapley value đo marginal contribution của target_node
        # Trong thực tế cần Monte Carlo approximation, đây là simplified version
        
        # Lấy lịch sử performance của các coalition có/không có target_node
        coalition_with = self.coalition_history.get((target_node, True), [])
        coalition_without = self.coalition_history.get((target_node, False), [])
        
        if len(coalition_with) > 0 and len(coalition_without) > 0:
            # Marginal contribution = avg performance with node - avg without node
            marginal = np.mean(coalition_with) - np.mean(coalition_without)
            # Normalize to [0, 1] range
            shapley_weight = np.clip(marginal / (np.std(coalition_with) + 1e-6), 0, 1)
        else:
            # Cold start: uniform weight
            shapley_weight = 1.0 / self.num_nodes
        
        return shapley_weight
```

**Shapley value trong FL**: Đo đóng góp công bằng của mỗi node vào mô hình global. Node có data quality cao hoặc computation power tốt sẽ có Shapley value cao hơn. Reward được nhân với Shapley weight để khuyến khích hợp tác với các node có đóng góp lớn.

## Mô phỏng Training

```python
    def _simulate_training(self, cut_layer, batch_size, H, target_node):
        # Mô phỏng đơn giản: performance gain phụ thuộc vào:
        # 1. Data heterogeneity giữa node hiện tại và target
        # 2. Tính ổn định của loss (Loss_Std)
        # 3. Số local epochs H
        
        # Giả sử có data distribution cho mỗi node (IID vs Non-IID)
        data_similarity = self._compute_data_similarity(target_node)
        
        # Performance gain cao hơn nếu data tương đồng và loss ổn định
        base_gain = data_similarity * (1.0 / (self.state[5] + 1.0))  # Loss_Std ở state[5]
        
        # Điều chỉnh theo H: nhiều local epochs = học tốt hơn nhưng risk overfitting
        epoch_factor = np.log(H + 1) / np.log(11)  # normalize to [0, 1]
        
        performance_gain = base_gain * epoch_factor * np.random.uniform(0.8, 1.2)
        
        # Cập nhật coalition history cho Shapley computation
        self._update_coalition_history(target_node, performance_gain)
        
        return performance_gain
```

---

# State Update và Termination

```python
    def _update_state(self, cut_layer, batch_size, H, target_node):
        # CPU và RAM giảm theo computation cost
        comp_load = (cut_layer / self.num_layers) * (batch_size / 512.0)
        self.state[0] = max(0, self.state[0] - comp_load * 10)  # CPU
        self.state[1] = max(0, self.state[1] - comp_load * 15)  # RAM
        
        # Energy tiêu hao
        energy_consumed = comp_load * 5 + (batch_size / 512.0) * 2
        self.state[2] = max(0, self.state[2] - energy_consumed)
        
        # Bandwidth fluctuation (mô phỏng network dynamics)
        self.state[3] = np.clip(self.state[3] + np.random.normal(0, 50), 10, 1000)
        
        # Loss cải thiện dần (simplified convergence)
        self.state[4] = max(0.1, self.state[4] * 0.95)
        
        # Loss_Std giảm khi training ổn định
        self.state[5] = max(0.1, self.state[5] * 0.98)
        
        # Neighbor availability thay đổi theo network churn
        self.state[6] = np.clip(self.state[6] + np.random.normal(0, 0.1), 0, 1)
        
        self.current_step += 1

    def _check_termination(self):
        # Kết thúc nếu:
        # 1. Loss đủ thấp (convergence)
        if self.state[4] < 0.5:
            return True
        # 2. Hết năng lượng
        if self.state[2] < 5.0:
            return True
        # 3. Không còn tài nguyên
        if self.state[0] < 5.0 or self.state[1] < 5.0:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Khởi tạo state ngẫu nhiên
        self.state = np.array([
            np.random.uniform(60, 100),   # CPU
            np.random.uniform(60, 100),   # RAM
            np.random.uniform(80, 100),   # Energy
            np.random.uniform(100, 800),  # Bandwidth
            np.random.uniform(5, 10),     # Loss
            np.random.uniform(1, 3),      # Loss_Std
            np.random.uniform(0.5, 1.0)   # Neighbor_Availability
        ], dtype=np.float32)
        
        self.current_step = 0
        self.coalition_history = {}
        
        return self.state, {}
```

---

# Tích hợp với Training Loop

```python
# Ví dụ sử dụng với Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = SFLNodeEnv(num_nodes=5, num_layers=10)
check_env(env)  # Validate environment

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Evaluation
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}, Cut: {action[0]}, Target: {action[3]}")
    if terminated or truncated:
        obs, info = env.reset()
```

---

# Các điểm cần lưu ý

## Shapley Value Approximation

Tính chính xác Shapley value yêu cầu $O(2^n)$ evaluations với $n$ nodes. Trong thực tế:
- Dùng **Monte Carlo sampling** với $K$ permutations (thường $K=100-1000$)
- **Truncated Monte Carlo**: Chỉ sample subset của coalitions
- **Group-based approximation**: Nhóm các node tương đồng lại

Xem `references/shapley_approximation.md` cho implementation chi tiết.

## Non-IID Data Handling

Split FL với non-IID data cần:
- **Data similarity metrics**: Cosine similarity của gradient hoặc feature distributions
- **Adaptive cut layer**: Cut sâu hơn khi data heterogeneity cao để giảm communication của noisy gradients
- **Personalization**: Mỗi node có thể giữ một số layer riêng

## Resource Constraints

Đối với edge devices:
- **Memory-aware cut**: Đảm bảo activation size không vượt quá RAM
- **Energy budget**: Terminate episode sớm nếu năng lượng thấp
- **Bandwidth throttling**: Penalty cao khi network congestion

## Multi-Agent Extension

Mở rộng sang multi-agent environment:
- Mỗi node là một agent độc lập
- Shared reward dựa trên global model performance
- Communication giữa agents qua message passing
- Xem `references/multi_agent_coordination.md`

---

# Debugging và Validation

```python
# Kiểm tra reward shaping
def validate_reward_function(env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min/Max: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    return rewards

# Kiểm tra action distribution
def analyze_action_distribution(env, model, num_steps=1000):
    obs, _ = env.reset()
    actions = []
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=False)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    actions = np.array(actions)
    print("Cut layer distribution:", np.bincount(actions[:, 0]))
    print("Batch size distribution:", np.bincount(actions[:, 1]))
    print("Target node distribution:", np.bincount(actions[:, 3]))
```

**Kỳ vọng**: Agent học được policy ưu tiên cut layer trung bình (cân bằng computation/communication), batch size vừa phải, và target node có Shapley value cao.
