# Common Tier Configurations in FL Research

## Cấu hình Tier Phổ biến

Dựa trên các nghiên cứu FL gần đây, đây là các cấu hình tier thường được sử dụng:

### Mobile Device Tiers

**Tier 1: High-end Smartphones** (iPhone 14 Pro, Samsung S23 Ultra)
- Compute: 8-12 GFLOPS
- Memory: 6-8 GB
- Bandwidth: 50-100 Mbps (5G)
- Battery: 4000-5000 mAh
- Use case: Premium users, early adopters

**Tier 2: Mid-range Smartphones** (Pixel 6a, Galaxy A54)
- Compute: 4-6 GFLOPS
- Memory: 4-6 GB
- Bandwidth: 20-50 Mbps (4G/5G)
- Battery: 3000-4000 mAh
- Use case: Majority của mobile users

**Tier 3: Budget Smartphones** (Redmi Note series)
- Compute: 2-3 GFLOPS
- Memory: 2-4 GB
- Bandwidth: 10-20 Mbps (4G)
- Battery: 2000-3000 mAh
- Use case: Emerging markets, older devices

**Tier 4: IoT Devices** (Raspberry Pi, ESP32)
- Compute: 0.5-1 GFLOPS
- Memory: 512 MB - 1 GB
- Bandwidth: 1-10 Mbps (WiFi/LTE-M)
- Battery: 500-2000 mAh (hoặc powered)
- Use case: Sensors, embedded systems

## Edge Server Tiers

Nếu mô phỏng hierarchical FL với edge servers:

**Edge Tier 1: Micro Data Center**
- Compute: 100-500 GFLOPS
- Memory: 32-128 GB
- Bandwidth: 1-10 Gbps
- Power: Grid-connected

**Edge Tier 2: Gateway Device**
- Compute: 20-50 GFLOPS
- Memory: 8-16 GB
- Bandwidth: 100-500 Mbps
- Power: Grid-connected hoặc battery backup

## Phân phối Xác suất

Các phân phối phổ biến trong literature:

### Uniform Distribution
```python
probabilities = {
    'high-end': 0.25,
    'mid-range': 0.25,
    'low-end': 0.25,
    'iot': 0.25
}
```
Sử dụng khi: Testing worst-case scenarios, academic benchmarks

### Realistic Distribution (Developed Markets)
```python
probabilities = {
    'high-end': 0.20,
    'mid-range': 0.45,
    'low-end': 0.25,
    'iot': 0.10
}
```
Sử dụng khi: Mô phỏng US/EU markets

### Emerging Market Distribution
```python
probabilities = {
    'high-end': 0.05,
    'mid-range': 0.25,
    'low-end': 0.50,
    'iot': 0.20
}
```
Sử dụng khi: Mô phỏng developing countries với nhiều budget devices

### Heavy IoT Distribution
```python
probabilities = {
    'high-end': 0.10,
    'mid-range': 0.20,
    'low-end': 0.30,
    'iot': 0.40
}
```
Sử dụng khi: Smart city, industrial IoT scenarios

## Workload Benchmarks

Các model phổ biến và FLOPS requirements:

- **MNIST (LeNet-5)**: ~0.5 MFLOPS/sample
- **CIFAR-10 (ResNet-20)**: ~40 MFLOPS/sample
- **ImageNet (ResNet-50)**: ~4 GFLOPS/sample
- **BERT-base**: ~22 GFLOPS/forward pass
- **GPT-2 small**: ~150 GFLOPS/forward pass

Model sizes (for communication time):
- **LeNet-5**: ~0.4 MB
- **ResNet-20**: ~1 MB
- **ResNet-50**: ~98 MB
- **BERT-base**: ~440 MB
- **GPT-2 small**: ~500 MB
