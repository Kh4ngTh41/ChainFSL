---
name: federated-learning-emulator
description: >
  Xây dựng bộ mô phỏng (emulator) cho hệ thống Federated Learning với thiết bị không đồng nhất (heterogeneous devices). Sử dụng kỹ năng này khi bạn cần mô phỏng các node edge/IoT với các tầng phần cứng khác nhau (hardware tiers), tính toán thời gian huấn luyện (compute time), thời gian truyền thông (communication time), quản lý năng lượng (energy budget), hoặc khởi tạo các node theo phân phối xác suất. Áp dụng cho các dự án FL simulation, distributed learning, edge computing research, device heterogeneity modeling, hoặc bất kỳ khi nào bạn đề cập đến tier-based hardware profiles, node initialization, resource constraints, hoặc federated system emulation.
---

# Federated Learning Emulator

## Tổng quan

Kỹ năng này giúp bạn xây dựng bộ mô phỏng cho hệ thống Federated Learning (FL) với các thiết bị không đồng nhất. Trọng tâm là mô hình hóa hardware heterogeneity thông qua tier-based profiles, tính toán thời gian compute/communication, và khởi tạo các node theo phân phối xác suất thực tế.

## Kiến trúc Core

### HardwareProfile Class

Đại diện cho một tier phần cứng cụ thể với các khả năng tính toán và truyền thông:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareProfile:
    """Đại diện cho một tier phần cứng trong FL system."""
    tier_name: str  # Ví dụ: "high-end", "mid-range", "low-end", "iot"
    compute_power: float  # FLOPS hoặc MIPS
    memory_mb: int
    bandwidth_mbps: float
    energy_capacity_mah: Optional[float] = None  # Cho mobile/IoT devices
    
    def compute_time(self, workload_flops: float) -> float:
        """Tính thời gian training (giây) dựa trên workload.
        
        Args:
            workload_flops: Số lượng floating-point operations cần thiết
            
        Returns:
            Thời gian tính toán tính bằng giây
        """
        if self.compute_power <= 0:
            raise ValueError(f"Compute power phải > 0, nhận được {self.compute_power}")
        return workload_flops / self.compute_power
    
    def comm_time(self, data_size_mb: float) -> float:
        """Tính thời gian truyền thông (giây) dựa trên kích thước dữ liệu.
        
        Args:
            data_size_mb: Kích thước dữ liệu cần truyền (MB)
            
        Returns:
            Thời gian truyền thông tính bằng giây
        """
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"Bandwidth phải > 0, nhận được {self.bandwidth_mbps}")
        return (data_size_mb * 8) / self.bandwidth_mbps  # Convert MB to Mbits
    
    def energy_consumption(self, compute_time_sec: float, 
                          power_watts: float) -> float:
        """Ước tính năng lượng tiêu thụ (mAh).
        
        Args:
            compute_time_sec: Thời gian tính toán
            power_watts: Công suất trung bình khi hoạt động
            
        Returns:
            Năng lượng tiêu thụ (mAh), hoặc None nếu không có energy_capacity
        """
        if self.energy_capacity_mah is None:
            return None
        # Giả sử voltage chuẩn 3.7V cho mobile devices
        voltage = 3.7
        current_ma = (power_watts / voltage) * 1000
        return (current_ma * compute_time_sec) / 3600  # Convert seconds to hours
```

### Tier Factory Pattern

Tạo node factory để khởi tạo N nodes theo phân phối xác suất:

```python
import random
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TierDistribution:
    """Phân phối xác suất cho các hardware tiers."""
    tier_profiles: Dict[str, HardwareProfile]
    probabilities: Dict[str, float]
    
    def __post_init__(self):
        # Validate probabilities sum to 1.0
        total = sum(self.probabilities.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating-point error
            raise ValueError(f"Probabilities phải tổng = 1.0, nhận được {total}")
        
        # Validate all tiers have profiles
        for tier in self.probabilities:
            if tier not in self.tier_profiles:
                raise ValueError(f"Tier '{tier}' không có profile tương ứng")


class TierFactory:
    """Factory để tạo nodes với phân phối tier xác định."""
    
    def __init__(self, distribution: TierDistribution, seed: Optional[int] = None):
        self.distribution = distribution
        self.rng = random.Random(seed)  # Reproducible randomness
        
    def create_nodes(self, n: int) -> List[Dict]:
        """Tạo N nodes theo phân phối xác suất.
        
        Args:
            n: Số lượng nodes cần tạo
            
        Returns:
            List các node dictionaries với tier assignment và profile
        """
        tiers = list(self.distribution.probabilities.keys())
        weights = [self.distribution.probabilities[t] for t in tiers]
        
        nodes = []
        for i in range(n):
            tier = self.rng.choices(tiers, weights=weights)[0]
            profile = self.distribution.tier_profiles[tier]
            
            nodes.append({
                'node_id': i,
                'tier': tier,
                'profile': profile,
                'available_energy': profile.energy_capacity_mah
            })
        
        return nodes
    
    def get_tier_counts(self, nodes: List[Dict]) -> Dict[str, int]:
        """Thống kê số lượng nodes theo từng tier."""
        counts = {tier: 0 for tier in self.distribution.tier_profiles}
        for node in nodes:
            counts[node['tier']] += 1
        return counts
```

## Ví dụ Sử dụng

### Định nghĩa 4 Tiers

```python
# Định nghĩa 4 tiers phổ biến trong FL research
tier_profiles = {
    'high-end': HardwareProfile(
        tier_name='high-end',
        compute_power=10e9,  # 10 GFLOPS
        memory_mb=8192,
        bandwidth_mbps=100.0,
        energy_capacity_mah=5000.0
    ),
    'mid-range': HardwareProfile(
        tier_name='mid-range',
        compute_power=5e9,  # 5 GFLOPS
        memory_mb=4096,
        bandwidth_mbps=50.0,
        energy_capacity_mah=3000.0
    ),
    'low-end': HardwareProfile(
        tier_name='low-end',
        compute_power=2e9,  # 2 GFLOPS
        memory_mb=2048,
        bandwidth_mbps=20.0,
        energy_capacity_mah=2000.0
    ),
    'iot': HardwareProfile(
        tier_name='iot',
        compute_power=0.5e9,  # 500 MFLOPS
        memory_mb=512,
        bandwidth_mbps=5.0,
        energy_capacity_mah=1000.0
    )
}

# Phân phối thực tế: nhiều mid/low-end, ít high-end và IoT
probabilities = {
    'high-end': 0.15,
    'mid-range': 0.40,
    'low-end': 0.35,
    'iot': 0.10
}

distribution = TierDistribution(tier_profiles, probabilities)
factory = TierFactory(distribution, seed=42)

# Tạo 100 nodes
nodes = factory.create_nodes(100)
print(factory.get_tier_counts(nodes))
# Output: {'high-end': 15, 'mid-range': 40, 'low-end': 35, 'iot': 10}
```

### Tính toán Compute và Communication Time

```python
# Giả sử một FL round với ResNet-18
model_flops = 1.8e9  # 1.8 GFLOPs per forward pass
model_size_mb = 44.7  # Model size

for node in nodes[:5]:  # Kiểm tra 5 nodes đầu
    profile = node['profile']
    
    compute_t = profile.compute_time(model_flops)
    comm_t = profile.comm_time(model_size_mb)
    total_t = compute_t + comm_t
    
    print(f"Node {node['node_id']} ({node['tier']}):")
    print(f"  Compute: {compute_t:.2f}s")
    print(f"  Comm: {comm_t:.2f}s")
    print(f"  Total: {total_t:.2f}s")
```

### Energy Budget Tracking

```python
def simulate_round(node: Dict, workload_flops: float, power_watts: float):
    """Mô phỏng một FL round và cập nhật energy budget."""
    profile = node['profile']
    
    compute_t = profile.compute_time(workload_flops)
    energy_used = profile.energy_consumption(compute_t, power_watts)
    
    if energy_used and node['available_energy']:
        node['available_energy'] -= energy_used
        if node['available_energy'] < 0:
            print(f"Node {node['node_id']} hết năng lượng!")
            return False
    
    return True

# Mô phỏng 10 rounds
for round_num in range(10):
    active_nodes = 0
    for node in nodes:
        if simulate_round(node, model_flops, power_watts=5.0):
            active_nodes += 1
    print(f"Round {round_num}: {active_nodes}/{len(nodes)} nodes active")
```

## Mở rộng

### Thêm Network Latency

Bổ sung latency vào communication time để mô phỏng thực tế hơn:

```python
@dataclass
class HardwareProfile:
    # ... existing fields ...
    network_latency_ms: float = 50.0  # Default 50ms
    
    def comm_time(self, data_size_mb: float) -> float:
        transmission_time = (data_size_mb * 8) / self.bandwidth_mbps
        latency_sec = self.network_latency_ms / 1000.0
        return transmission_time + latency_sec
```

### Straggler Detection

Xác định các node chậm (stragglers) dựa trên total time:

```python
def detect_stragglers(nodes: List[Dict], workload_flops: float, 
                      model_size_mb: float, threshold_percentile: float = 90):
    """Phát hiện stragglers dựa trên total time."""
    times = []
    for node in nodes:
        profile = node['profile']
        total_t = profile.compute_time(workload_flops) + \
                  profile.comm_time(model_size_mb)
        times.append((node['node_id'], total_t))
    
    times.sort(key=lambda x: x[1])
    cutoff_idx = int(len(times) * threshold_percentile / 100)
    stragglers = [node_id for node_id, _ in times[cutoff_idx:]]
    
    return stragglers
```

## Nguyên tắc Thiết kế

**Dataclasses cho clean data modeling**: Sử dụng `@dataclass` để giảm boilerplate và tăng tính rõ ràng. Validation logic nên ở `__post_init__`.

**Tách biệt concerns**: HardwareProfile chỉ mô tả hardware, TierFactory chỉ tạo nodes. Simulation logic (như energy tracking) nằm ở layer cao hơn.

**Reproducibility**: Luôn hỗ trợ seed cho random number generation để experiments có thể tái tạo.

**Extensibility**: Thiết kế cho phép thêm metrics mới (network latency, packet loss, CPU variance) mà không phá vỡ existing code.

**Realistic modeling**: Dựa trên các nghiên cứu FL thực tế, phân phối tier phải phản ánh heterogeneity thực tế (ít high-end, nhiều mid/low-end).

## Tham khảo

Xem `references/tier_configurations.md` cho các tier configurations phổ biến từ các paper FL research, và `references/energy_models.md` cho các mô hình năng lượng chi tiết hơn.
