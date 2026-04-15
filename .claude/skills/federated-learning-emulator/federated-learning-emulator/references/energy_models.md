# Energy Modeling for FL Devices

## Mô hình Năng lượng Cơ bản

Energy consumption trong FL devices phụ thuộc vào ba thành phần chính:

1. **Computation Energy**: Năng lượng cho training/inference
2. **Communication Energy**: Năng lượng cho upload/download model
3. **Idle Energy**: Năng lượng khi device chờ đợi

## Computation Energy

Công thức cơ bản:
```
E_comp = P_comp × t_comp
```

Với:
- `P_comp`: Công suất khi computing (Watts)
- `t_comp`: Thời gian computing (seconds)

### Công suất Thực tế

**Mobile Devices**:
- High-end: 5-8W (full CPU/GPU utilization)
- Mid-range: 3-5W
- Low-end: 2-3W
- IoT: 0.5-1W

**Edge Servers**:
- Micro DC: 200-500W
- Gateway: 50-100W

### Dynamic Voltage/Frequency Scaling (DVFS)

Devices có thể điều chỉnh power consumption:

```python
def compute_energy_dvfs(workload_flops: float, 
                        base_power: float,
                        frequency_scale: float = 1.0) -> float:
    """Tính energy với DVFS.
    
    Args:
        workload_flops: Workload
        base_power: Công suất ở frequency chuẩn
        frequency_scale: 0.5-1.5 (lower = slower but energy efficient)
    """
    # Power scales cubically with frequency
    actual_power = base_power * (frequency_scale ** 3)
    # Time scales inversely with frequency
    time = (workload_flops / base_compute_power) / frequency_scale
    
    return actual_power * time
```

## Communication Energy

Công thức:
```
E_comm = P_tx × t_tx + P_rx × t_rx
```

### Công suất Truyền thông

**WiFi (2.4GHz)**:
- Transmit: 1.5-2.5W
- Receive: 0.8-1.2W

**4G LTE**:
- Transmit: 2.5-4W (depends on signal strength)
- Receive: 1-1.5W

**5G**:
- Transmit: 3-5W
- Receive: 1.2-2W

**Bluetooth LE**:
- Transmit: 0.05-0.1W
- Receive: 0.03-0.05W

### Ví dụ Implementation

```python
@dataclass
class CommunicationProfile:
    tx_power_watts: float
    rx_power_watts: float
    bandwidth_mbps: float
    
    def upload_energy(self, data_mb: float) -> float:
        """Energy cho upload (mAh)."""
        time_sec = (data_mb * 8) / self.bandwidth_mbps
        voltage = 3.7  # Standard mobile battery
        current_ma = (self.tx_power_watts / voltage) * 1000
        return (current_ma * time_sec) / 3600
    
    def download_energy(self, data_mb: float) -> float:
        """Energy cho download (mAh)."""
        time_sec = (data_mb * 8) / self.bandwidth_mbps
        voltage = 3.7
        current_ma = (self.rx_power_watts / voltage) * 1000
        return (current_ma * time_sec) / 3600

# WiFi profile
wifi = CommunicationProfile(
    tx_power_watts=2.0,
    rx_power_watts=1.0,
    bandwidth_mbps=50.0
)

# 4G profile  
lte = CommunicationProfile(
    tx_power_watts=3.5,
    rx_power_watts=1.3,
    bandwidth_mbps=20.0
)
```

## Battery Lifetime Estimation

```python
def estimate_fl_rounds(battery_mah: float,
                       energy_per_round_mah: float,
                       safety_margin: float = 0.2) -> int:
    """Ước tính số FL rounds trước khi hết pin.
    
    Args:
        battery_mah: Dung lượng pin
        energy_per_round_mah: Năng lượng mỗi round
        safety_margin: Reserve 20% battery cho OS/apps khác
    """
    usable_battery = battery_mah * (1 - safety_margin)
    return int(usable_battery / energy_per_round_mah)

# Example
battery = 4000  # mAh
energy_per_round = 50  # mAh (compute + comm)
rounds = estimate_fl_rounds(battery, energy_per_round)
print(f"Device có thể tham gia {rounds} rounds")
```

## Advanced: Temperature-Aware Energy

Battery efficiency giảm khi nhiệt độ cao:

```python
def temperature_adjusted_energy(base_energy_mah: float,
                               temp_celsius: float) -> float:
    """Điều chỉnh energy consumption theo nhiệt độ.
    
    Battery efficiency giảm ~1% mỗi độ C trên 25°C.
    """
    optimal_temp = 25.0
    if temp_celsius <= optimal_temp:
        return base_energy_mah
    
    degradation_per_degree = 0.01
    temp_diff = temp_celsius - optimal_temp
    efficiency_factor = 1 + (degradation_per_degree * temp_diff)
    
    return base_energy_mah * efficiency_factor
```

## Energy-Aware Client Selection

Chọn clients dựa trên energy budget:

```python
def select_clients_energy_aware(nodes: List[Dict],
                                num_clients: int,
                                min_energy_threshold: float) -> List[int]:
    """Chọn clients có đủ năng lượng.
    
    Args:
        nodes: List of node dictionaries
        num_clients: Số clients cần chọn
        min_energy_threshold: Ngưỡng năng lượng tối thiểu (mAh)
    """
    eligible = [n for n in nodes 
                if n['available_energy'] >= min_energy_threshold]
    
    if len(eligible) < num_clients:
        print(f"Warning: Chỉ có {len(eligible)}/{num_clients} clients đủ năng lượng")
        return [n['node_id'] for n in eligible]
    
    # Random selection từ eligible nodes
    selected = random.sample(eligible, num_clients)
    return [n['node_id'] for n in selected]
```

## Tham khảo

Các paper về energy modeling trong FL:
- "Energy-Efficient Federated Learning" (IEEE SECON 2022)
- "FedMP: Communication-Efficient Federated Learning" (Nature Scientific Reports 2024)
- "Adaptive FL for Resource-Constrained IoT" (MDPI Future Internet 2025)
