---
name: tve-gtm-blockchain-implementation
description: >
  Triển khai module TVE (Trusted Value Evaluation) và GTM (Game-Theoretic Mechanism) cho hệ thống blockchain. Sử dụng khi cần xây dựng MockVRF với HMAC-SHA256 để chọn ủy ban (committee selection), thuật toán TMCSShapley (Truncated Monte Carlo Shapley) để tính đóng góp của node với độ phức tạp O(M*N), hoặc TokenomicsEngine để xử lý phân phối reward theo lịch trình giảm phát (deflationary) và slashing cho node gian lận. Áp dụng cho các hệ thống proof-of-stake, federated learning, data valuation, distributed consensus, validator incentives, hoặc bất kỳ blockchain nào cần cơ chế công bằng, minh bạch và chống gian lận.
---

# TVE & GTM Blockchain Implementation

Kỹ năng này hướng dẫn triển khai ba module cốt lõi cho hệ thống blockchain với cơ chế đánh giá công bằng và khuyến khích kinh tế:

1. **MockVRF** — Verifiable Random Function giả lập dùng HMAC-SHA256 để chọn ủy ban minh bạch
2. **TMCSShapley** — Truncated Monte Carlo Shapley để tính đóng góp của từng node
3. **TokenomicsEngine** — Quản lý phân phối reward theo lịch trình giảm phát và slashing

## Tổng quan kiến trúc

### Luồng hoạt động chính

```
[Epoch bắt đầu]
    ↓
[MockVRF chọn committee dựa trên seed + epoch]
    ↓
[Nodes trong committee thực hiện công việc]
    ↓
[TMCSShapley đánh giá đóng góp của từng node]
    ↓
[TokenomicsEngine phân phối reward/slashing]
    ↓
[Cập nhật state cho epoch tiếp theo]
```

**Tại sao cần cả ba module:**

- **MockVRF** đảm bảo tính ngẫu nhiên có thể kiểm chứng, ngăn chặn thao túng trong việc chọn validator
- **TMCSShapley** cung cấp phương pháp công bằng để đo lường đóng góp thực tế, không chỉ dựa vào stake
- **TokenomicsEngine** tạo động lực kinh tế dài hạn và trừng phạt hành vi gian lận

---

## 1. MockVRF — Committee Selection với HMAC-SHA256

### Nguyên lý hoạt động

VRF (Verifiable Random Function) tạo ra đầu ra ngẫu nhiên có thể chứng minh. MockVRF sử dụng HMAC-SHA256 để:

- **Đầu vào:** `secret_key` (của mỗi node) + `public_seed` (chung cho toàn mạng) + `epoch_number`
- **Đầu ra:** Hash 256-bit được chuyển thành số ngẫu nhiên trong khoảng [0, 1)
- **Chọn committee:** Sắp xếp nodes theo giá trị VRF, chọn top K nodes

**Tại sao dùng HMAC-SHA256:**

- HMAC kết hợp secret key với hash function, đảm bảo chỉ node sở hữu key mới tính được output
- SHA-256 cung cấp phân phối đồng đều và khó dự đoán
- Lightweight, phù hợp cho môi trường blockchain

### Implementation pattern

```python
import hmac
import hashlib
import struct

class MockVRF:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
    
    def evaluate(self, public_seed: bytes, epoch: int) -> float:
        """Tính VRF output cho epoch hiện tại.
        
        Returns:
            float trong khoảng [0, 1) để so sánh và chọn committee
        """
        # Kết hợp public_seed và epoch
        message = public_seed + struct.pack('>Q', epoch)
        
        # Tính HMAC-SHA256
        h = hmac.new(self.secret_key, message, hashlib.sha256)
        digest = h.digest()
        
        # Chuyển 8 bytes đầu thành số ngẫu nhiên [0, 1)
        random_value = struct.unpack('>Q', digest[:8])[0]
        return random_value / (2**64)
    
    def verify(self, public_seed: bytes, epoch: int, 
               claimed_output: float, public_key_hash: bytes) -> bool:
        """Xác minh VRF output (trong production cần dùng public key thật)."""
        # Trong MockVRF, verification đơn giản hóa
        # Production: dùng elliptic curve hoặc RSA signature
        expected = self.evaluate(public_seed, epoch)
        return abs(expected - claimed_output) < 1e-9


def select_committee(nodes: list, public_seed: bytes, 
                     epoch: int, committee_size: int) -> list:
    """Chọn committee từ danh sách nodes dựa trên VRF output.
    
    Args:
        nodes: List of (node_id, secret_key) tuples
        public_seed: Seed công khai cho toàn mạng
        epoch: Số epoch hiện tại
        committee_size: Kích thước committee cần chọn
    
    Returns:
        List of node_ids được chọn vào committee
    """
    vrf_outputs = []
    
    for node_id, secret_key in nodes:
        vrf = MockVRF(secret_key)
        output = vrf.evaluate(public_seed, epoch)
        vrf_outputs.append((output, node_id))
    
    # Sắp xếp theo VRF output, chọn top K
    vrf_outputs.sort()
    committee = [node_id for _, node_id in vrf_outputs[:committee_size]]
    
    return committee
```

### Best practices

- **Seed rotation:** Thay đổi `public_seed` định kỳ để tránh precomputation attacks
- **Secret key security:** Mỗi node phải bảo vệ secret_key tuyệt đối, không chia sẻ
- **Epoch synchronization:** Đảm bảo tất cả nodes đồng bộ về epoch number
- **Deterministic selection:** Với cùng inputs, kết quả phải giống nhau trên mọi node

---

## 2. TMCSShapley — Truncated Monte Carlo Shapley Value

### Lý thuyết Shapley value

Shapley value đo lường đóng góp công bằng của mỗi player trong cooperative game:

$$\phi_i = \frac{1}{n!} \sum_{\pi \in \Pi} [v(S_\pi^i \cup \{i\}) - v(S_\pi^i)]$$

Trong đó:
- $\phi_i$: Shapley value của node $i$
- $\Pi$: Tất cả permutations của nodes
- $S_\pi^i$: Tập nodes đứng trước $i$ trong permutation $\pi$
- $v(S)$: Utility function — hiệu suất của tập nodes $S$

**Vấn đề:** Tính toán chính xác yêu cầu $O(n! \cdot 2^n)$ — không khả thi với $n > 10$

### Truncated Monte Carlo approximation

Thay vì duyệt tất cả permutations, TMC-Shapley:

1. **Sample M permutations ngẫu nhiên** (thường $M = 3n$ đến $5n$)
2. **Với mỗi permutation:** Tính marginal contribution cho mọi node
3. **Early stopping:** Dừng khi marginal contribution < threshold
4. **Average:** $\hat{\phi}_i = \frac{1}{M} \sum_{m=1}^M \Delta_i^m$

**Độ phức tạp:** $O(M \cdot N)$ với $M$ là số permutations, $N$ là số nodes

### Implementation pattern

```python
import random
from typing import List, Callable

class TMCSShapley:
    def __init__(self, utility_fn: Callable[[List[int]], float],
                 truncation_threshold: float = 0.01):
        """Khởi tạo TMC-Shapley calculator.
        
        Args:
            utility_fn: Hàm đánh giá hiệu suất của tập nodes.
                        Nhận list node_ids, trả về float (accuracy, throughput, etc.)
            truncation_threshold: Ngưỡng để dừng sớm khi marginal < threshold
        """
        self.utility_fn = utility_fn
        self.truncation_threshold = truncation_threshold
        self.cache = {}  # Cache utility values để giảm recomputation
    
    def _compute_utility(self, node_subset: List[int]) -> float:
        """Tính utility với caching."""
        key = tuple(sorted(node_subset))
        if key not in self.cache:
            self.cache[key] = self.utility_fn(list(key))
        return self.cache[key]
    
    def compute_shapley_values(self, nodes: List[int], 
                                num_samples: int = None) -> dict:
        """Tính Shapley value cho tất cả nodes.
        
        Args:
            nodes: List of node IDs
            num_samples: Số permutations để sample (mặc định 3*len(nodes))
        
        Returns:
            Dict {node_id: shapley_value}
        """
        n = len(nodes)
        if num_samples is None:
            num_samples = 3 * n
        
        shapley_values = {node: 0.0 for node in nodes}
        
        # Sample M permutations
        for _ in range(num_samples):
            perm = random.sample(nodes, n)
            
            # Tính marginal contribution cho mỗi node trong permutation
            current_subset = []
            prev_utility = self._compute_utility([])
            
            for node in perm:
                current_subset.append(node)
                current_utility = self._compute_utility(current_subset)
                
                marginal = current_utility - prev_utility
                shapley_values[node] += marginal
                
                # Truncation: dừng sớm nếu marginal quá nhỏ
                if abs(marginal) < self.truncation_threshold:
                    # Các nodes còn lại nhận marginal = 0
                    break
                
                prev_utility = current_utility
        
        # Average qua tất cả samples
        for node in shapley_values:
            shapley_values[node] /= num_samples
        
        return shapley_values
```

### Utility function examples

**Cho federated learning:**

```python
def model_accuracy_utility(node_subset: List[int]) -> float:
    """Đánh giá accuracy của model khi train với dữ liệu từ node_subset."""
    if not node_subset:
        return 0.0
    
    # Kết hợp data từ các nodes
    combined_data = aggregate_data(node_subset)
    
    # Train model
    model = train_model(combined_data)
    
    # Evaluate trên validation set
    accuracy = evaluate_model(model, validation_set)
    
    return accuracy
```

**Cho blockchain consensus:**

```python
def consensus_throughput_utility(node_subset: List[int]) -> float:
    """Đánh giá throughput khi chỉ có node_subset tham gia consensus."""
    if len(node_subset) < MIN_VALIDATORS:
        return 0.0
    
    # Simulate consensus với node_subset
    throughput = simulate_consensus(
        validators=node_subset,
        duration=SIMULATION_TIME
    )
    
    return throughput  # transactions per second
```

### Optimization tips

- **Caching:** Lưu utility values để tránh tính lại cho cùng subset
- **Parallel sampling:** Chạy nhiều permutations song song
- **Adaptive sampling:** Tăng M cho nodes có high variance
- **Stratified sampling:** Đảm bảo mỗi node xuất hiện đều ở các vị trí trong permutation

---

## 3. TokenomicsEngine — Deflationary Rewards & Slashing

### Thiết kế tokenomics

**Mục tiêu:**

- **Deflationary schedule:** Giảm dần reward theo thời gian để tạo scarcity
- **Performance-based rewards:** Phân phối dựa trên Shapley value, không chỉ stake
- **Slashing for fraud:** Phạt nodes lazy (không tham gia) hoặc poison (gửi dữ liệu độc hại)

**Công thức reward:**

$$R_i(t) = B(t) \cdot \frac{\phi_i}{\sum_j \phi_j} \cdot (1 - s_i)$$

Trong đó:
- $R_i(t)$: Reward của node $i$ tại epoch $t$
- $B(t)$: Base reward giảm dần theo thời gian (deflationary)
- $\phi_i$: Shapley value của node $i$
- $s_i$: Slashing rate (0 nếu honest, > 0 nếu gian lận)

### Implementation pattern

```python
from dataclasses import dataclass
from typing import Dict
import math

@dataclass
class TokenomicsConfig:
    initial_base_reward: float = 100.0
    decay_rate: float = 0.95  # Giảm 5% mỗi epoch
    min_base_reward: float = 10.0
    
    # Slashing parameters
    lazy_penalty: float = 0.3  # 30% penalty cho lazy nodes
    poison_penalty: float = 1.0  # 100% penalty (mất hết reward)
    
    # Detection thresholds
    lazy_threshold: float = 0.01  # Shapley < 0.01 → lazy
    poison_detection_confidence: float = 0.9


class TokenomicsEngine:
    def __init__(self, config: TokenomicsConfig):
        self.config = config
        self.current_epoch = 0
        self.total_distributed = 0.0
        self.slashing_history = {}  # {node_id: list of slashing events}
    
    def get_base_reward(self, epoch: int) -> float:
        """Tính base reward theo lịch trình deflationary.
        
        Sử dụng exponential decay: B(t) = B_0 * decay^t
        """
        base = self.config.initial_base_reward * \
               (self.config.decay_rate ** epoch)
        return max(base, self.config.min_base_reward)
    
    def detect_lazy_nodes(self, shapley_values: Dict[int, float]) -> set:
        """Phát hiện nodes lazy dựa trên Shapley value thấp."""
        lazy_nodes = set()
        
        for node_id, value in shapley_values.items():
            if value < self.config.lazy_threshold:
                lazy_nodes.add(node_id)
        
        return lazy_nodes
    
    def detect_poison_nodes(self, nodes: List[int], 
                            validation_fn: Callable) -> set:
        """Phát hiện poison nodes qua validation.
        
        Args:
            validation_fn: Hàm kiểm tra data/model từ node có hợp lệ không
                           Returns confidence score [0, 1]
        """
        poison_nodes = set()
        
        for node_id in nodes:
            is_valid, confidence = validation_fn(node_id)
            
            if not is_valid and \
               confidence > self.config.poison_detection_confidence:
                poison_nodes.add(node_id)
        
        return poison_nodes
    
    def calculate_slashing_rate(self, node_id: int, 
                                 lazy_nodes: set, 
                                 poison_nodes: set) -> float:
        """Tính tỷ lệ slashing cho một node."""
        slashing_rate = 0.0
        
        if node_id in poison_nodes:
            slashing_rate = self.config.poison_penalty
            self._record_slashing(node_id, 'poison', slashing_rate)
        elif node_id in lazy_nodes:
            slashing_rate = self.config.lazy_penalty
            self._record_slashing(node_id, 'lazy', slashing_rate)
        
        return slashing_rate
    
    def distribute_rewards(self, shapley_values: Dict[int, float],
                           lazy_nodes: set, 
                           poison_nodes: set) -> Dict[int, float]:
        """Phân phối rewards cho tất cả nodes.
        
        Returns:
            Dict {node_id: reward_amount}
        """
        base_reward = self.get_base_reward(self.current_epoch)
        total_shapley = sum(shapley_values.values())
        
        if total_shapley == 0:
            return {node: 0.0 for node in shapley_values}
        
        rewards = {}
        
        for node_id, shapley_value in shapley_values.items():
            # Tính reward dựa trên tỷ lệ Shapley
            reward = base_reward * (shapley_value / total_shapley)
            
            # Áp dụng slashing
            slashing_rate = self.calculate_slashing_rate(
                node_id, lazy_nodes, poison_nodes
            )
            reward *= (1 - slashing_rate)
            
            rewards[node_id] = reward
            self.total_distributed += reward
        
        self.current_epoch += 1
        return rewards
    
    def _record_slashing(self, node_id: int, reason: str, rate: float):
        """Ghi lại lịch sử slashing."""
        if node_id not in self.slashing_history:
            self.slashing_history[node_id] = []
        
        self.slashing_history[node_id].append({
            'epoch': self.current_epoch,
            'reason': reason,
            'rate': rate
        })
    
    def get_total_supply_schedule(self, max_epochs: int) -> float:
        """Tính tổng token sẽ phát hành trong max_epochs."""
        total = 0.0
        for epoch in range(max_epochs):
            total += self.get_base_reward(epoch)
        return total
```

### Deflationary schedule examples

**Exponential decay (như Bitcoin halving):**

```python
def exponential_decay(initial: float, epoch: int, 
                       halving_period: int = 210000) -> float:
    halvings = epoch // halving_period
    return initial / (2 ** halvings)
```

**Linear decrease:**

```python
def linear_decrease(initial: float, epoch: int, 
                     decrease_per_epoch: float = 0.1) -> float:
    return max(initial - epoch * decrease_per_epoch, 0)
```

**Step function:**

```python
def step_function(epoch: int, schedule: List[tuple]) -> float:
    # schedule = [(epoch_threshold, reward), ...]
    for threshold, reward in sorted(schedule, reverse=True):
        if epoch >= threshold:
            return reward
    return schedule[0][1]
```

---

## Integration example

### End-to-end workflow

```python
# 1. Setup
public_seed = b"network_seed_epoch_1234"
epoch = 1234
committee_size = 10

nodes = [
    (f"node_{i}", os.urandom(32))  # (node_id, secret_key)
    for i in range(100)
]

config = TokenomicsConfig(
    initial_base_reward=1000,
    decay_rate=0.98,
    lazy_penalty=0.2,
    poison_penalty=1.0
)

# 2. Chọn committee với MockVRF
committee = select_committee(nodes, public_seed, epoch, committee_size)
print(f"Committee selected: {committee}")

# 3. Nodes trong committee thực hiện công việc
# (training, consensus, data validation, etc.)

# 4. Tính Shapley values
shapley_calc = TMCSShapley(
    utility_fn=lambda subset: model_accuracy_utility(subset),
    truncation_threshold=0.005
)
shapley_values = shapley_calc.compute_shapley_values(
    committee, 
    num_samples=30
)

print(f"Shapley values: {shapley_values}")

# 5. Phát hiện gian lận
tokenomics = TokenomicsEngine(config)
lazy_nodes = tokenomics.detect_lazy_nodes(shapley_values)
poison_nodes = tokenomics.detect_poison_nodes(
    committee,
    validation_fn=lambda nid: validate_node_contribution(nid)
)

print(f"Lazy nodes: {lazy_nodes}")
print(f"Poison nodes: {poison_nodes}")

# 6. Phân phối rewards
rewards = tokenomics.distribute_rewards(
    shapley_values,
    lazy_nodes,
    poison_nodes
)

for node_id, reward in rewards.items():
    print(f"{node_id}: {reward:.2f} tokens")

# 7. Cập nhật state cho epoch tiếp theo
epoch += 1
```

---

## Testing & validation

### Unit tests cần thiết

**MockVRF:**

- Determinism: Cùng inputs → cùng output
- Distribution: Output phân phối đồng đều trong [0, 1)
- Non-predictability: Không thể dự đoán output mà không có secret_key

**TMCSShapley:**

- Efficiency axiom: $\sum_i \phi_i = v(N)$ (tổng Shapley = utility của toàn bộ)
- Symmetry: Nodes có đóng góp giống nhau → Shapley giống nhau
- Convergence: Tăng M → variance giảm

**TokenomicsEngine:**

- Deflationary schedule: Base reward giảm theo thời gian
- Slashing correctness: Lazy/poison nodes bị phạt đúng mức
- Total supply cap: Không vượt quá giới hạn đã định

### Performance benchmarks

- **MockVRF:** < 1ms cho 1000 nodes
- **TMCSShapley:** < 10s cho 50 nodes với M=150 samples
- **TokenomicsEngine:** < 100ms cho 1000 nodes

---

## Common pitfalls

### MockVRF

- **Seed reuse:** Không bao giờ dùng lại `public_seed` cho nhiều epochs — dễ bị precomputation
- **Weak secret keys:** Secret keys phải có entropy cao (≥ 256 bits)
- **Clock skew:** Nodes không đồng bộ epoch → chọn committee khác nhau

### TMCSShapley

- **Quá ít samples:** $M < n$ → variance cao, kết quả không ổn định
- **Utility function không monotonic:** $v(S) > v(S \cup \{i\})$ gây Shapley âm
- **Cache invalidation:** Khi utility function thay đổi, phải clear cache

### TokenomicsEngine

- **Decay quá nhanh:** Reward giảm xuống 0 sớm → mất động lực
- **Slashing quá nặng:** Penalty 100% cho lỗi nhỏ → nodes rời mạng
- **Không xử lý Sybil:** Một entity tạo nhiều nodes → chiếm đoạt reward

---

## Security considerations

### MockVRF

- **Secret key leakage:** Nếu attacker biết secret_key → có thể dự đoán committee
- **Grinding attacks:** Attacker thử nhiều public_seeds để chọn committee có lợi → cần commit-reveal scheme

### TMCSShapley

- **Utility manipulation:** Nodes cố tình làm giảm utility của coalition để tăng Shapley của mình
- **Collusion:** Nhóm nodes thỏa thuận để tăng Shapley tập thể

### TokenomicsEngine

- **False slashing:** Slashing nhầm honest nodes → mất niềm tin
- **Reward concentration:** Một vài nodes chiếm phần lớn reward → centralization

**Giải pháp:**

- Dùng cryptographic VRF thật (VRF-ED25519) thay vì MockVRF trong production
- Multi-round validation cho poison detection
- Reputation system để theo dõi lịch sử nodes
- Slashing appeals mechanism cho honest nodes bị phạt nhầm

---

## Khi nào dùng kỹ năng này

**Phù hợp:**

- Hệ thống blockchain cần chọn validators công bằng và minh bạch
- Federated learning với nhiều data providers không tin tưởng nhau
- Data marketplaces cần định giá đóng góp của từng nguồn dữ liệu
- DAOs muốn phân phối rewards dựa trên contribution thực tế

**Không phù hợp:**

- Hệ thống centralized với trusted coordinator (không cần VRF)
- Số nodes quá lớn (> 1000) → TMC-Shapley quá chậm, cân nhắc approximation khác
- Real-time requirements < 1s → cần pre-computation hoặc caching

## Tài nguyên bổ sung

- **VRF:** RFC 9381 (VRFv10), Algorand's VRF implementation
- **Shapley values:** "Data Shapley: Equitable Valuation of Data" (Ghorbani & Zou, ICML 2019)
- **Tokenomics:** Ethereum's EIP-1559 (fee burning), Bitcoin halving schedule
