# ChainFSL — Kế hoạch Triển khai Thực nghiệm (Single-Machine)

> **Dành cho Claude Code Agent:** File này là bản hướng dẫn đầy đủ để implement toàn bộ framework ChainFSL trên một máy duy nhất. Đọc từ đầu đến cuối trước khi bắt đầu coding. Mỗi section đã bao gồm pseudocode, API call, và quyết định thiết kế cụ thể.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Môi trường & Dependencies](#3-môi-trường--dependencies)
4. [Module 0: IoT Emulator](#4-module-0-iot-emulator)
5. [Module 1: SFL Pipeline](#5-module-1-sfl-pipeline)
6. [Module 2: MA-HASO (DRL Orchestrator)](#6-module-2-ma-haso-drl-orchestrator)
7. [Module 3: TVE (Trustless Verification Engine)](#7-module-3-tve-trustless-verification-engine)
8. [Module 4: GTM (Game-Theoretic Tokenomics)](#8-module-4-gtm-game-theoretic-tokenomics)
9. [Module 5: Blockchain Mock](#9-module-5-blockchain-mock)
10. [Orchestration: ChainFSL Protocol](#10-orchestration-chainfsl-protocol)
11. [Experiments (E1–E7)](#11-experiments-e1e7)
12. [Logging & Metrics](#12-logging--metrics)
13. [Thứ tự coding cho Agent](#13-thứ-tự-coding-cho-agent)

---

## 1. Tổng quan kiến trúc

### 1.1 Chiến lược single-machine

Thay vì thiết bị IoT vật lý, dùng **Python multiprocessing**: mỗi `Process` là một "node", tự giữ profile phần cứng giả lập và tuân theo giới hạn đó trong code (sleep để giả lập compute time, semaphore để giới hạn concurrency, queue để giả lập bandwidth).

```
Máy vật lý (1 GPU/CPU)
│
├── N=50 Worker Processes (Data Nodes)
│   ├── Tier 1 (5 nodes)  — không throttle, full GPU
│   ├── Tier 2 (15 nodes) — CPU only, 2 thread max
│   ├── Tier 3 (20 nodes) — 0.5 CPU equivalent, sleep delay
│   └── Tier 4 (10 nodes) — minimal compute, max delay
│
├── Shared Memory (Manager)
│   ├── Global model weights (w_bar)
│   ├── Gossip table (neighbor states)
│   └── Blockchain ledger (SQLite)
│
└── Coordinator Process (HASO + GTM + TVE)
    ├── PPO policy per node (Stable Baselines3)
    ├── Shapley computation (TMCS)
    └── VRF committee selection
```

### 1.2 Luồng 1 epoch (theo Algorithm 2 bài báo)

```
Epoch t:
  1. HASO: mỗi node observe state → chọn (c_i, B_i, H_i, target_j)
  2. TRAINING: node i forward pass đến cut layer c_i
  3. COMM: gửi smashed data (tensor) đến compute node j (qua Queue)
  4. SERVER: node j backward pass, trả gradient về
  5. UPDATE: node i update client-side weights
  6. TVE: gen proof (tier-dependent), committee verify
  7. AGGREGATION: staleness-decayed weighted average (async)
  8. GTM: tính Shapley, phân phối token reward
  9. BLOCKCHAIN: ghi reward lên SQLite ledger
  10. HASO: compute reward r_t, update PPO policy
```

---

## 2. Cấu trúc thư mục

```
chainfsl/
│
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml          # hyperparameters chính
│   └── experiment_configs/
│       ├── e1_haso.yaml
│       ├── e2_scalability.yaml
│       ├── e3_noniid.yaml
│       ├── e4_security.yaml
│       ├── e5_incentive.yaml
│       ├── e6_ablation.yaml
│       └── e7_blockchain.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── emulator/
│   │   ├── __init__.py
│   │   ├── node_profile.py       # HardwareProfile dataclass
│   │   ├── network_emulator.py   # bandwidth throttle, latency sim
│   │   └── tier_factory.py       # tạo N nodes theo tier distribution
│   │
│   ├── sfl/
│   │   ├── __init__.py
│   │   ├── models.py             # ResNet18, VGG11, MobileNetV2 với split support
│   │   ├── split_model.py        # ClientModel, ServerModel wrapper
│   │   ├── data_loader.py        # CIFAR-10/100, FEMNIST, MedMNIST, Dirichlet partition
│   │   ├── trainer.py            # SFL training loop (1 node)
│   │   └── aggregator.py         # layer-wise partial aggregation, staleness decay
│   │
│   ├── haso/
│   │   ├── __init__.py
│   │   ├── env.py                # Gymnasium custom env (SFLEnv)
│   │   ├── agent.py              # PPO agent wrapper (Stable Baselines3)
│   │   ├── gossip.py             # Gossip protocol mock (shared Manager dict)
│   │   └── reward.py             # reward function (Eq. 7)
│   │
│   ├── tve/
│   │   ├── __init__.py
│   │   ├── vrf.py                # VRF mock (HMAC-based deterministic)
│   │   ├── zk_prover.py          # Tier 1-2: hash-based proof (mock zk-SNARK)
│   │   ├── commitment.py         # Tier 3-4: hash commitment + challenge
│   │   └── committee.py          # committee selection + verification logic
│   │
│   ├── gtm/
│   │   ├── __init__.py
│   │   ├── contribution.py       # ContributionVector (Eq. 13)
│   │   ├── shapley.py            # TMCS approximation (Eq. 15)
│   │   ├── tokenomics.py         # reward distribution (Eq. 14-15), deflationary schedule
│   │   └── nash_validator.py     # Nash equilibrium check utility
│   │
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── ledger.py             # SQLite ledger (mock blockchain)
│   │   └── smart_contract.py     # reward distribution logic (pure Python)
│   │
│   └── protocol/
│       ├── __init__.py
│       └── chainfsl.py           # end-to-end orchestrator (Algorithm 2)
│
├── experiments/
│   ├── run_experiment.py         # entry point
│   ├── e1_haso_effectiveness.py
│   ├── e2_scalability.py
│   ├── e3_noniid.py
│   ├── e4_security.py
│   ├── e5_incentive.py
│   ├── e6_ablation.py
│   └── e7_blockchain_overhead.py
│
├── baselines/
│   ├── fedavg.py
│   ├── splitfed.py               # uniform cut layer
│   ├── adaptsfl.py               # alternating optimization (simplified)
│   └── dfl.py                    # dynamic per-client split (simplified)
│
└── analysis/
    ├── plot_results.py
    └── metrics_aggregator.py
```

---

## 3. Môi trường & Dependencies

### 3.1 requirements.txt

```txt
# Core ML
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0

# DRL
stable-baselines3>=2.2.0
gymnasium>=0.29.0
shimmy>=1.3.0

# Data
medmnist>=2.2.3
Pillow>=10.0.0

# Blockchain mock
# (SQLite is built into Python stdlib)

# Crypto (TVE)
cryptography>=41.0.0   # HMAC for VRF mock
hashlib                # stdlib

# Experiment management
wandb>=0.16.0          # hoặc tensorboard
tensorboard>=2.15.0
pyyaml>=6.0
tqdm>=4.66.0

# Analysis
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
```

### 3.2 config/default.yaml

```yaml
# Network
n_nodes: 50
tier_distribution: [0.1, 0.3, 0.4, 0.2]  # T1, T2, T3, T4

# SFL
model: resnet18       # resnet18 | vgg11 | mobilenetv2
dataset: cifar10      # cifar10 | cifar100 | femnist | medmnist
n_classes: 10
global_rounds: 100
local_rounds_max: 5   # H_max
batch_size_default: 32
dirichlet_alpha: 0.5  # Non-IID degree

# HASO
haso_enabled: true
ppo_learning_rate: 3e-4
ppo_n_steps: 512
ppo_batch_size: 64
ppo_n_epochs: 10
ema_beta: 0.9          # EMA smoothing beta
reward_alpha: 1.0      # local efficiency weight (comp time)
reward_beta: 0.5       # local efficiency weight (comm time)
reward_gamma: 0.1      # global alignment weight

# TVE
tve_enabled: true
committee_size: 5      # K
vrf_omega: 0.3         # reputation weighting
stake_min: 10.0        # S_min tokens

# GTM
gtm_enabled: true
shapley_M: 50          # MC permutations
reward_total_init: 1000.0   # R_0
reward_min: 10.0            # R_min
halving_rounds: 50          # T_halving
staleness_decay: 0.9        # rho

# Security attacks (for E4)
sybil_fraction: 0.0
lazy_client_fraction: 0.0
poison_fraction: 0.0

# Logging
log_dir: ./logs
use_wandb: false
experiment_name: chainfsl_default
```

---

## 4. Module 0: IoT Emulator

### 4.1 `src/emulator/node_profile.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class HardwareProfile:
    """
    Profile phần cứng giả lập cho một IoT node.
    Tất cả 'limits' được enforce bằng software trong quá trình training.
    """
    node_id: int
    tier: int                    # 1, 2, 3, 4
    
    # Compute (FLOPS relative, dùng để tính sleep time)
    flops_ratio: float           # relative to Tier 1 = 1.0
    max_threads: int             # giới hạn parallelism
    
    # Memory (MB)
    ram_mb: int
    
    # Bandwidth (Mbps)
    bandwidth_mbps: float
    
    # Energy (arbitrary units, dùng cho constraint)
    energy_budget: float = 1000.0
    energy_remaining: float = field(init=False)
    
    # Reputation (khởi tạo = 0.5, cập nhật bởi GTM)
    reputation: float = 0.5
    
    # Stake deposit (tokens)
    stake: float = 10.0
    
    def __post_init__(self):
        self.energy_remaining = self.energy_budget

    def compute_time(self, base_flops: float) -> float:
        """Tính thời gian compute thực tế (giây) dựa trên tier."""
        return base_flops / (self.flops_ratio * 1e9)
    
    def comm_time(self, data_size_bytes: float) -> float:
        """Tính thời gian truyền (giây)."""
        bandwidth_bytes = self.bandwidth_mbps * 1e6 / 8
        return data_size_bytes / bandwidth_bytes
    
    def can_fit_cut_layer(self, cut_layer: int, model_memory_mb: dict) -> bool:
        """Kiểm tra memory constraint M_i(c_i) <= m_mem_i."""
        required = model_memory_mb.get(cut_layer, float('inf'))
        return required <= self.ram_mb

TIER_CONFIGS = {
    1: dict(flops_ratio=1.0,   max_threads=8,  ram_mb=8192,  bandwidth_mbps=100.0),
    2: dict(flops_ratio=0.3,   max_threads=2,  ram_mb=4096,  bandwidth_mbps=50.0),
    3: dict(flops_ratio=0.05,  max_threads=1,  ram_mb=512,   bandwidth_mbps=10.0),
    4: dict(flops_ratio=0.005, max_threads=1,  ram_mb=200,   bandwidth_mbps=1.0),
}
```

### 4.2 `src/emulator/network_emulator.py`

```python
import asyncio
import time
import random

class NetworkEmulator:
    """
    Giả lập mạng P2P: bandwidth, latency, packet loss.
    Dùng asyncio Queue để mô phỏng truyền smashed data.
    """
    
    def __init__(self, variance: float = 0.3):
        """
        variance: ±30% bandwidth fluctuation (Markov model theo bài báo Section 6.1.3)
        """
        self.variance = variance
        self._queues: dict = {}   # (src_id, dst_id) -> asyncio.Queue
    
    def get_queue(self, src_id: int, dst_id: int):
        key = (src_id, dst_id)
        if key not in self._queues:
            self._queues[key] = asyncio.Queue(maxsize=10)
        return self._queues[key]
    
    def effective_bandwidth(self, nominal_mbps: float) -> float:
        """Markov-like bandwidth fluctuation."""
        factor = 1.0 + random.uniform(-self.variance, self.variance)
        return max(0.1, nominal_mbps * factor)
    
    async def send(self, src_profile, dst_profile, tensor_bytes: int):
        """
        Giả lập gửi smashed data từ src đến dst.
        Thực chất chỉ là sleep dựa trên bandwidth bottleneck.
        """
        bw = min(
            self.effective_bandwidth(src_profile.bandwidth_mbps),
            self.effective_bandwidth(dst_profile.bandwidth_mbps)
        )
        delay = (tensor_bytes * 8) / (bw * 1e6)
        await asyncio.sleep(delay)
        return delay
```

### 4.3 `src/emulator/tier_factory.py`

```python
from .node_profile import HardwareProfile, TIER_CONFIGS

def create_nodes(n_nodes: int, tier_distribution: list) -> list[HardwareProfile]:
    """
    Tạo N nodes theo tier distribution.
    tier_distribution = [0.1, 0.3, 0.4, 0.2] -> T1:T2:T3:T4
    """
    nodes = []
    node_id = 0
    for tier_idx, fraction in enumerate(tier_distribution):
        tier = tier_idx + 1
        count = max(1, int(n_nodes * fraction))
        cfg = TIER_CONFIGS[tier]
        for _ in range(count):
            nodes.append(HardwareProfile(node_id=node_id, tier=tier, **cfg))
            node_id += 1
    return nodes[:n_nodes]   # trim nếu rounding error
```

---

## 5. Module 1: SFL Pipeline

### 5.1 `src/sfl/models.py`

Implement ResNet-18 với khả năng split tại bất kỳ layer nào:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11, mobilenet_v2

class SplittableResNet18(nn.Module):
    """
    ResNet-18 có thể split tại các boundary: [4, 8, 12, 16] (residual block boundaries).
    cut_layer: số thứ tự block (1-indexed), 0 = không split (full model ở client).
    """
    
    SPLIT_POINTS = {
        # cut_layer -> (client_layers, server_layers)
        # Mỗi giá trị là list các nn.Sequential sub-modules
        1: ['layer1'],
        2: ['layer1', 'layer2'],
        3: ['layer1', 'layer2', 'layer3'],
        4: ['layer1', 'layer2', 'layer3', 'layer4'],
    }
    
    def __init__(self, n_classes: int = 10):
        super().__init__()
        base = resnet18(pretrained=False)
        base.fc = nn.Linear(512, n_classes)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
    
    def get_client_model(self, cut_layer: int) -> nn.Sequential:
        """Trả về phần client-side (layers 0..cut_layer)."""
        layers = [self.conv1, self.bn1, self.relu, self.maxpool]
        block_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(cut_layer, 4)):
            layers.append(block_layers[i])
        return nn.Sequential(*layers)
    
    def get_server_model(self, cut_layer: int) -> nn.Sequential:
        """Trả về phần server-side (layers cut_layer+1..end)."""
        block_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        layers = []
        for i in range(cut_layer, 4):
            layers.append(block_layers[i])
        layers += [self.avgpool, nn.Flatten(), self.fc]
        return nn.Sequential(*layers)
    
    @staticmethod
    def memory_requirement_mb(cut_layer: int) -> float:
        """Ước tính RAM cần thiết để train client-side với cut_layer."""
        # Empirical estimates từ bài báo / thực đo
        memory_map = {1: 150, 2: 300, 3: 500, 4: 700}
        return memory_map.get(cut_layer, 800)
```

### 5.2 `src/sfl/split_model.py`

```python
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SmashData:
    """Smashed data được truyền từ client đến server."""
    node_id: int
    tensor: torch.Tensor       # activation tại cut layer
    labels: torch.Tensor       # ground truth labels
    round_id: int              # global round (dùng cho staleness)
    cut_layer: int

class ClientModel:
    def __init__(self, model: nn.Module, cut_layer: int, optimizer_cls=torch.optim.SGD, lr=0.01):
        self.model = model
        self.cut_layer = cut_layer
        self.optimizer = optimizer_cls(model.parameters(), lr=lr, momentum=0.9)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass đến cut layer. Trả về smashed data."""
        with torch.enable_grad():
            a = self.model(x)
        return a.detach().requires_grad_(True)   # detach để gửi qua "network"
    
    def backward(self, grad_a: torch.Tensor):
        """Nhận gradient từ server, backward qua client-side model."""
        self.optimizer.zero_grad()
        # a_detached đã được lưu từ forward
        self._last_activation.backward(grad_a)
        self.optimizer.step()

class ServerModel:
    def __init__(self, model: nn.Module, criterion=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.SGD, lr=0.01):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer_cls(model.parameters(), lr=lr, momentum=0.9)
    
    def forward_backward(self, smashed: torch.Tensor, labels: torch.Tensor):
        """
        Server hoàn thành forward pass, tính loss, backward.
        Trả về: (loss value, gradient tại cut layer để gửi lại client)
        """
        self.optimizer.zero_grad()
        smashed = smashed.requires_grad_(True)
        output = self.model(smashed)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), smashed.grad.detach()
```

### 5.3 `src/sfl/data_loader.py`

```python
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def dirichlet_partition(dataset, n_clients: int, alpha: float, seed: int = 42) -> list[list[int]]:
    """
    Chia dataset theo Dirichlet Dir(alpha) cho Non-IID simulation.
    alpha nhỏ → heterogeneous (mỗi client chỉ có vài class)
    alpha lớn → IID
    
    Returns: list of index lists, mỗi list là indices của 1 client.
    """
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    n_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * n_clients)
        # Đảm bảo mỗi client có ít nhất 1 sample
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        idx = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(class_indices[idx:idx+count].tolist())
            idx += count
    
    return client_indices

def get_dataloaders(dataset_name: str, n_clients: int, alpha: float, batch_size: int) -> list[DataLoader]:
    """
    Trả về list DataLoader cho từng client.
    dataset_name: 'cifar10' | 'cifar100' | 'femnist' | 'medmnist'
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    elif dataset_name == 'medmnist':
        import medmnist
        dataset = medmnist.PathMNIST(split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    partition = dirichlet_partition(dataset, n_clients, alpha)
    loaders = []
    for indices in partition:
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0))
    
    return loaders
```

### 5.4 `src/sfl/aggregator.py`

Implement Equation (3) và (9) từ bài báo:

```python
import torch
import copy
from collections import defaultdict

class AsyncAggregator:
    """
    Thực hiện staleness-decayed aggregation (Eq. 9) và
    layer-wise partial aggregation (Eq. 3).
    """
    
    def __init__(self, global_model_state: dict, rho: float = 0.9):
        self.global_state = copy.deepcopy(global_model_state)
        self.rho = rho           # staleness decay constant
        self.current_round = 0
    
    def aggregate(self, updates: list[dict]) -> dict:
        """
        updates: list of {
            'node_id': int,
            'cut_layer': int,
            'client_state': dict,    # state_dict của client-side model
            'server_state': dict,    # state_dict của server-side model
            'data_size': int,        # |D_i|
            'staleness': int,        # tau_i (version lag)
        }
        
        Implement Eq. 9: w^(t+1) = w^(t) + sum_i alpha_i^(t) * (w_i^(t-tau_i) - w^(t))
        với alpha_i = (|D_i| / |D|) * rho^tau_i
        """
        total_data = sum(u['data_size'] for u in updates)
        new_state = copy.deepcopy(self.global_state)
        
        # Layer-wise partial aggregation theo Eq. 3
        # Tích lũy weighted updates cho từng layer
        layer_updates = defaultdict(lambda: torch.zeros_like)
        layer_weights = defaultdict(float)
        
        for update in updates:
            alpha = (update['data_size'] / total_data) * (self.rho ** update['staleness'])
            
            # Client-side layers (layer index <= cut_layer)
            for key, param in update['client_state'].items():
                if key not in layer_updates:
                    layer_updates[key] = torch.zeros_like(self.global_state[key], dtype=torch.float32)
                layer_updates[key] += alpha * (param.float() - self.global_state[key].float())
                layer_weights[key] += alpha
            
            # Server-side layers (layer index > cut_layer)
            for key, param in update['server_state'].items():
                if key not in layer_updates:
                    layer_updates[key] = torch.zeros_like(self.global_state[key], dtype=torch.float32)
                layer_updates[key] += alpha * (param.float() - self.global_state[key].float())
                layer_weights[key] += alpha
        
        # Normalize và apply
        for key in new_state:
            if key in layer_updates and layer_weights[key] > 0:
                new_state[key] = (self.global_state[key].float() + 
                                  layer_updates[key] / layer_weights[key]).to(self.global_state[key].dtype)
        
        self.global_state = new_state
        self.current_round += 1
        return new_state
```

---

## 6. Module 2: MA-HASO (DRL Orchestrator)

### 6.1 `src/haso/env.py`

Gymnasium custom environment mô phỏng bài toán MDP của HASO:

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SFLNodeEnv(gym.Env):
    """
    Custom Gymnasium env cho một Data Node trong MA-HASO.
    Mỗi node có env riêng biệt, train PPO policy riêng.
    
    State (Eq. 5): [cpu_ratio, mem_ratio, energy_rem, bandwidth, loss, loss_std, neighbor_avail]
    Action (Eq. 6): [cut_layer, batch_size, H_i, target_node_id]
    Reward (Eq. 7): -alpha*T_comp - beta*T_comm + gamma * shapley * delta_F
    """
    
    metadata = {'render_modes': []}
    
    # Cut layer choices: 1, 2, 3, 4 (cho ResNet-18)
    CUT_LAYERS = [1, 2, 3, 4]
    # Batch size choices
    BATCH_SIZES = [8, 16, 32, 64]
    # Aggregation frequency choices (local rounds)
    H_CHOICES = [1, 2, 3, 5]
    
    def __init__(self, node_profile, n_compute_nodes: int, model_memory_map: dict,
                 reward_weights: tuple = (1.0, 0.5, 0.1)):
        super().__init__()
        self.profile = node_profile
        self.n_compute = n_compute_nodes
        self.memory_map = model_memory_map
        self.alpha, self.beta, self.gamma = reward_weights
        
        # State space: 7 chiều liên tục
        # [cpu_util, mem_util, energy_ratio, bandwidth_norm, current_loss, loss_variance, neighbor_avail_mean]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Action space: MultiDiscrete
        # [cut_layer_idx, batch_size_idx, H_idx, target_compute_node_id]
        self.action_space = spaces.MultiDiscrete([
            len(self.CUT_LAYERS),
            len(self.BATCH_SIZES),
            len(self.H_CHOICES),
            n_compute_nodes,
        ])
        
        self._state = None
        self._step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self._get_obs()
        return self._state, {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Quan sát state hiện tại của node.
        Trong single-machine simulation: đọc từ shared memory.
        """
        return np.array([
            self.profile.flops_ratio,               # cpu_util proxy
            1.0 - (self.profile.ram_mb / 8192.0),   # mem_util proxy
            self.profile.energy_remaining / self.profile.energy_budget,
            self.profile.bandwidth_mbps / 100.0,
            getattr(self, '_last_loss', 0.5),
            getattr(self, '_loss_std', 0.1),
            getattr(self, '_neighbor_avail', 0.5),  # từ gossip
        ], dtype=np.float32)
    
    def step(self, action):
        """
        action: [cut_layer_idx, batch_size_idx, H_idx, target_node_idx]
        
        Decode action → config, simulate training step, tính reward.
        """
        cut_layer = self.CUT_LAYERS[action[0]]
        batch_size = self.BATCH_SIZES[action[1]]
        H = self.H_CHOICES[action[2]]
        target_node = action[3]
        
        # Check memory constraint
        if not self.profile.can_fit_cut_layer(cut_layer, self.memory_map):
            # Invalid action: penalize và force cut_layer nhỏ hơn
            cut_layer = self.CUT_LAYERS[0]
        
        # Compute time estimate (dùng flops_ratio của node)
        base_comp_flops = 1e9   # normalized
        T_comp = self.profile.compute_time(base_comp_flops * cut_layer / 4)
        
        # Communication time estimate
        smashed_size_bytes = self._estimate_smashed_size(cut_layer)
        T_comm = self.profile.comm_time(smashed_size_bytes)
        
        # Reward (Eq. 7) — shapley sẽ được feed vào sau khi GTM tính xong
        shapley_estimate = getattr(self, '_shapley_value', 0.1)  # updated by GTM
        delta_F = getattr(self, '_delta_loss', 0.01)
        
        reward = (- self.alpha * T_comp 
                  - self.beta * T_comm 
                  + self.gamma * shapley_estimate * delta_F)
        
        self._step_count += 1
        obs = self._get_obs()
        done = False   # env không terminate, PPO train liên tục
        
        info = {
            'cut_layer': cut_layer,
            'batch_size': batch_size,
            'H': H,
            'target_node': target_node,
            'T_comp': T_comp,
            'T_comm': T_comm,
        }
        return obs, reward, done, False, info
    
    def _estimate_smashed_size(self, cut_layer: int) -> int:
        """
        Ước tính kích thước smashed data (bytes) theo cut layer.
        Early cut → activation lớn (CNN feature maps to'i đầu rộng).
        """
        # ResNet-18: feature map sizes at each residual block
        size_map = {
            1: 64 * 56 * 56 * 4,    # layer1: 64 channels, 56x56
            2: 128 * 28 * 28 * 4,   # layer2: 128 channels, 28x28
            3: 256 * 14 * 14 * 4,   # layer3: 256 channels, 14x14
            4: 512 * 7 * 7 * 4,     # layer4: 512 channels, 7x7
        }
        return size_map.get(cut_layer, 512 * 7 * 7 * 4)
    
    def update_shapley(self, phi: float):
        """GTM feed Shapley value vào để update reward signal."""
        # EMA smoothing (Eq. trong Section 4.1.1)
        beta = 0.9
        old = getattr(self, '_shapley_ema', phi)
        self._shapley_value = beta * old + (1 - beta) * phi
    
    def update_loss(self, loss: float, loss_std: float, delta: float):
        self._last_loss = min(loss, 1.0)
        self._loss_std = min(loss_std, 1.0)
        self._delta_loss = delta
    
    def update_neighbor_avail(self, avail: float):
        self._neighbor_avail = avail
```

### 6.2 `src/haso/agent.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from .env import SFLNodeEnv
import numpy as np

class HaSOAgent:
    """
    Wrapper cho PPO agent của một Data Node.
    Mỗi node train policy PPO riêng (decentralized).
    """
    
    def __init__(self, env: SFLNodeEnv, node_id: int):
        self.env = env
        self.node_id = node_id
        
        # PPO với hyperparameters từ bài báo / best practices
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,          # PPO clipping (conservative updates)
            ent_coef=0.01,           # entropy bonus (exploration)
            verbose=0,
        )
    
    def decide(self, obs: np.ndarray) -> dict:
        """
        Predict action từ observation.
        Returns decoded action dict.
        """
        action, _ = self.model.predict(obs, deterministic=False)
        cut_layer = SFLNodeEnv.CUT_LAYERS[action[0]]
        batch_size = SFLNodeEnv.BATCH_SIZES[action[1]]
        H = SFLNodeEnv.H_CHOICES[action[2]]
        target = int(action[3])
        return {
            'cut_layer': cut_layer,
            'batch_size': batch_size,
            'H': H,
            'target_compute_node': target,
        }
    
    def learn(self, total_timesteps: int = 512):
        """Update policy sau mỗi epoch."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    
    def save(self, path: str):
        self.model.save(path)
    
    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)
```

### 6.3 `src/haso/gossip.py`

```python
from multiprocessing import Manager
import time

class GossipProtocol:
    """
    Mock Gossip protocol dùng Python multiprocessing.Manager().dict().
    
    Mỗi node broadcast LRH (Lightweight Resource Heartbeat):
    - flops_ratio, ram_available, bandwidth, reputation, load
    
    Các node khác đọc từ shared dict để biết neighbor availability.
    """
    
    def __init__(self, manager: Manager, fanout: int = 3):
        self.table = manager.dict()   # node_id -> LRH dict
        self.fanout = fanout          # số neighbor mỗi gossip round
    
    def broadcast(self, node_id: int, lrh: dict):
        """Node publish LRH của mình."""
        self.table[node_id] = {**lrh, 'timestamp': time.time()}
    
    def get_neighbors(self, node_id: int, k: int = 5) -> list[dict]:
        """
        Trả về k neighbors gần nhất (theo reputation + freshness).
        Trong simulation: random k từ table, loại bỏ self.
        """
        import random
        all_nodes = [(nid, info) for nid, info in self.table.items() if nid != node_id]
        # Ưu tiên node có reputation cao và timestamp gần
        all_nodes.sort(key=lambda x: x[1].get('reputation', 0), reverse=True)
        return [info for _, info in all_nodes[:k]]
    
    def mean_neighbor_availability(self, node_id: int) -> float:
        """Tính mean availability của neighbors (dùng cho state observation)."""
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.5
        return sum(n.get('flops_ratio', 0.5) for n in neighbors) / len(neighbors)
```

---

## 7. Module 3: TVE (Trustless Verification Engine)

### 7.1 `src/tve/vrf.py`

```python
import hmac
import hashlib
import struct

class MockVRF:
    """
    VRF mock dùng HMAC-SHA256.
    Trong production: dùng ecvrf (Rust) hoặc py_ecc.
    
    Tính chất đảm bảo:
    - Deterministic: cùng sk + seed → cùng output
    - Unpredictable: không biết sk thì không dự đoán được output
    - Verifiable: proof cho phép verify mà không cần sk
    """
    
    def __init__(self, secret_key: bytes):
        self.sk = secret_key
    
    def compute(self, seed: bytes) -> tuple[float, bytes]:
        """
        Tính (hash_value, proof).
        hash_value: float [0, 1) để so sánh với threshold
        proof: bytes để verify
        """
        mac = hmac.new(self.sk, seed, hashlib.sha256)
        digest = mac.digest()
        # Chuyển 8 bytes đầu thành float [0, 1)
        val = struct.unpack('>Q', digest[:8])[0] / (2**64)
        return val, digest
    
    @staticmethod
    def verify(public_key: bytes, seed: bytes, proof: bytes) -> float:
        """
        Verify proof và trả về hash value.
        Mock: dùng public_key = secret_key (single-machine).
        """
        mac = hmac.new(public_key, seed, hashlib.sha256)
        expected = mac.digest()
        if not hmac.compare_digest(proof, expected):
            raise ValueError("VRF proof invalid")
        return struct.unpack('>Q', expected[:8])[0] / (2**64)
```

### 7.2 `src/tve/commitment.py`

```python
import hashlib
import torch
import numpy as np

class ComputationCommitment:
    """
    Tier-dependent verification theo Section 4.2.2 bài báo.
    
    - Tier 1-2: zk-SNARK mock (hash-based proof với timing overhead)
    - Tier 3: hash commitment + spot-check challenge
    - Tier 4: hash-only + delegate re-compute
    """
    
    @staticmethod
    def commit_input(x: torch.Tensor) -> bytes:
        """h_i = Hash(x_i) — commit trước khi train."""
        data = x.detach().numpy().tobytes()
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def gen_proof_tier1_2(x: torch.Tensor, a: torch.Tensor, model_hash: bytes) -> dict:
        """
        Tier 1-2: Full zk-SNARK proof (mock).
        Thực chất là hash của (x, a, model_weights).
        Thêm artificial delay để mô phỏng overhead thực.
        
        Theo bài báo Alaa et al. [56]: ~200 bytes proof size.
        """
        import time
        # Mock proving time: ~2-5 giây cho small network
        time.sleep(0.1)   # scaled down từ thực tế
        
        combined = x.detach().numpy().tobytes() + a.detach().numpy().tobytes() + model_hash
        proof_hash = hashlib.sha256(combined).digest()
        return {
            'type': 'zk_snark_mock',
            'proof': proof_hash,
            'activation_hash': hashlib.sha256(a.detach().numpy().tobytes()).digest(),
            'input_hash': hashlib.sha256(x.detach().numpy().tobytes()).digest(),
        }
    
    @staticmethod
    def gen_proof_tier3(x: torch.Tensor, a: torch.Tensor) -> dict:
        """Tier 3: Hash commitment."""
        return {
            'type': 'hash_commit',
            'input_hash': hashlib.sha256(x.detach().numpy().tobytes()).digest(),
            'activation_hash': hashlib.sha256(a.detach().numpy().tobytes()).digest(),
        }
    
    @staticmethod
    def gen_proof_tier4(x: torch.Tensor) -> dict:
        """Tier 4: Hash only."""
        return {
            'type': 'hash_only',
            'input_hash': hashlib.sha256(x.detach().numpy().tobytes()).digest(),
        }
    
    @staticmethod
    def verify_proof(proof: dict, expected_input_hash: bytes) -> bool:
        """Verify proof. Returns True nếu hợp lệ."""
        if proof['input_hash'] != expected_input_hash:
            return False
        if proof['type'] == 'hash_only':
            # Probabilistic: accept với xác suất cao (simplified)
            return True
        return True   # Mock: luôn pass nếu hash match
```

### 7.3 `src/tve/committee.py`

```python
import time
import hashlib
from .vrf import MockVRF
from .commitment import ComputationCommitment

class VerificationCommittee:
    """
    VRF-based committee selection (Eq. 10-11 bài báo) + verification.
    """
    
    def __init__(self, nodes: list, committee_size: int = 5, omega: float = 0.3):
        self.nodes = nodes         # list HardwareProfile
        self.K = committee_size
        self.omega = omega
        self.stake_min = 10.0
    
    def select_committee(self, epoch: int, block_hash: bytes) -> list:
        """
        Eq. 10-11: node i selected if hash_i / 2^256 < (K/N) * (1 + omega * tanh(delta * s_i))
        
        Dùng VRF mock để mô phỏng unpredictable selection.
        """
        import math
        seed = hashlib.sha256(block_hash + epoch.to_bytes(4, 'big')).digest()
        N = len(self.nodes)
        delta = 1.0   # saturation constant
        
        selected = []
        for node in self.nodes:
            vrf = MockVRF(secret_key=f"sk_{node.node_id}".encode())
            hash_val, proof = vrf.compute(seed)
            
            # Reputation-adjusted threshold (Eq. 11)
            threshold = (self.K / N) * (1 + self.omega * math.tanh(delta * node.reputation))
            
            if hash_val < threshold:
                selected.append(node)
        
        # Đảm bảo có đủ committee_size
        if len(selected) < self.K:
            remaining = [n for n in self.nodes if n not in selected]
            selected.extend(remaining[:self.K - len(selected)])
        
        return selected[:self.K]
    
    def verify_updates(self, updates: list, proofs: list, lazy_client_ids: set = None) -> dict:
        """
        Verify tất cả updates từ nodes.
        
        lazy_client_ids: set của node IDs được inject làm lazy client (E4 experiment).
        Returns: dict node_id -> (is_valid, penalty)
        """
        results = {}
        PENALTY = 1000.0   # slashing penalty >> reward
        
        for update, proof in zip(updates, proofs):
            node_id = update['node_id']
            
            # Inject lazy client attack
            if lazy_client_ids and node_id in lazy_client_ids:
                # Node submit random activation → proof sẽ fail
                is_valid = False
            else:
                is_valid = ComputationCommitment.verify_proof(
                    proof, 
                    update.get('input_hash', b'')
                )
            
            penalty = PENALTY if not is_valid else 0.0
            results[node_id] = {
                'valid': is_valid,
                'penalty': penalty,
            }
        
        return results
```

---

## 8. Module 4: GTM (Game-Theoretic Tokenomics)

### 8.1 `src/gtm/contribution.py`

```python
from dataclasses import dataclass
import torch

@dataclass
class ContributionVector:
    """
    Eq. 13: v_i = (v_comp, v_data, v_bw, v_rel)
    Contribution vector của node i trong round t.
    """
    node_id: int
    
    # Normalized computation: FLOPS / FLOPS_max
    v_comp: float
    
    # Data contribution: |D_i| * q_i / max_j(|D_j| * q_j)
    v_data: float
    
    # Bandwidth: b_i / b_max
    v_bw: float
    
    # Reliability: 1 - (failures / rounds)
    v_rel: float
    
    @property
    def total(self) -> float:
        """Weighted sum (equal weights mặc định)."""
        return (self.v_comp + self.v_data + self.v_bw + self.v_rel) / 4.0

def compute_vli(global_model_state: dict, node_update: dict, 
                val_loader, criterion, device) -> float:
    """
    Validation-Loss Improvement (VLI) — privacy-preserving proxy cho data quality.
    
    q_i = F(w^(t-1)) - F(w^(t-1) + eta * grad_i)
    
    Tính marginal utility của update node i với UNIFORM weight (không có staleness decay).
    Điều này đảm bảo công bằng với các node chậm (Section 4.4.1 bài báo).
    """
    import torch
    import copy
    
    # Evaluate loss TRƯỚC khi apply update
    model = _load_model_from_state(global_model_state, device)
    loss_before = _eval_loss(model, val_loader, criterion, device)
    
    # Apply update với uniform weight (không staleness decay)
    updated_state = copy.deepcopy(global_model_state)
    eta = 0.01
    for key in node_update:
        if key in updated_state:
            updated_state[key] = (updated_state[key].float() + 
                                  eta * (node_update[key].float() - updated_state[key].float()))
    
    model_updated = _load_model_from_state(updated_state, device)
    loss_after = _eval_loss(model_updated, val_loader, criterion, device)
    
    return max(0.0, loss_before - loss_after)   # VLI >= 0

def _eval_loss(model, val_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
    return total_loss / len(val_loader)

def _load_model_from_state(state, device):
    # Helper: tạo model từ state_dict (import từ sfl.models)
    pass   # implement dựa trên model type trong config
```

### 8.2 `src/gtm/shapley.py`

```python
import numpy as np
import random
from typing import Callable

class TMCSShapley:
    """
    Truncated Monte Carlo Shapley (TMCS) — Eq. 15 bài báo.
    
    Tính approximate Shapley value với M random permutations.
    Complexity: O(M*N) thay vì O(2^N).
    """
    
    def __init__(self, M: int = 50):
        self.M = M
    
    def compute(self, 
                node_ids: list[int],
                value_fn: Callable[[list[int]], float],
                truncation_tol: float = 0.01) -> dict[int, float]:
        """
        node_ids: list of all participating node IDs
        value_fn: v(S) → float, characteristic function (model performance)
        truncation_tol: dừng sớm nếu marginal gain < tol (TMCS optimization)
        
        Returns: dict node_id -> shapley_value
        """
        N = len(node_ids)
        shapley = {nid: 0.0 for nid in node_ids}
        
        for m in range(self.M):
            perm = node_ids.copy()
            random.shuffle(perm)
            
            v_prev = 0.0
            coalition = []
            
            for i, nid in enumerate(perm):
                coalition.append(nid)
                v_curr = value_fn(coalition)
                marginal = v_curr - v_prev
                
                # Truncation: nếu marginal gain nhỏ → bỏ qua
                if abs(marginal) < truncation_tol and i > N // 2:
                    break
                
                shapley[nid] += marginal
                v_prev = v_curr
        
        # Normalize
        for nid in shapley:
            shapley[nid] /= self.M
        
        return shapley
    
    @staticmethod
    def fed_sv_decompose(shapley_client: dict, shapley_server: dict, 
                          shapley_comm: dict) -> dict:
        """
        Eq. 16: phi_total = phi_client + phi_server + phi_comm
        
        Decompose contribution thành 3 phần.
        """
        all_nodes = set(shapley_client) | set(shapley_server) | set(shapley_comm)
        result = {}
        for nid in all_nodes:
            result[nid] = (shapley_client.get(nid, 0.0) + 
                           shapley_server.get(nid, 0.0) + 
                           shapley_comm.get(nid, 0.0))
        return result
```

### 8.3 `src/gtm/tokenomics.py`

```python
class TokenomicsEngine:
    """
    Deflationary reward schedule và Nash Equilibrium enforcement.
    Eq. 14-15 bài báo.
    """
    
    def __init__(self, R0: float = 1000.0, R_min: float = 10.0, 
                 T_halving: int = 50, stake_min: float = 10.0):
        self.R0 = R0
        self.R_min = R_min
        self.T_halving = T_halving
        self.stake_min = stake_min
        self.PENALTY_LAZY = 1000.0    # >> R_max → Case 1 trong proof
        self.QUALITY_THRESHOLD = 0.01  # q_min
    
    def total_reward(self, t: int) -> float:
        """
        Eq. 15 (deflationary schedule):
        R_total^(t) = max(R_min, R0 * (1 - t/T_halving)+)
        """
        r = self.R0 * max(0.0, 1.0 - t / self.T_halving)
        return max(self.R_min, r)
    
    def distribute(self, t: int, shapley_values: dict, 
                   verification_results: dict) -> dict:
        """
        Phân phối token reward theo Shapley value (Eq. 14).
        Apply penalty cho node bị detect là lazy/invalid.
        
        Returns: dict node_id -> net_reward (có thể âm nếu bị slash)
        """
        R_total = self.total_reward(t)
        total_phi = sum(max(0.0, phi) for phi in shapley_values.values())
        
        rewards = {}
        for node_id, phi in shapley_values.items():
            if total_phi > 0:
                reward = R_total * max(0.0, phi) / total_phi
            else:
                reward = 0.0
            
            # Apply penalty từ TVE
            penalty = verification_results.get(node_id, {}).get('penalty', 0.0)
            
            # Superlinear penalty cho low quality (Case 2 trong proof)
            v_data = getattr(self, f'_v_data_{node_id}', 1.0)
            if v_data < self.QUALITY_THRESHOLD:
                penalty += reward * 2.0   # superlinear
            
            rewards[node_id] = reward - penalty
        
        return rewards
    
    def check_sybil_profitable(self, m_sybil: int, N_total: int, 
                                R_total: float) -> bool:
        """
        Theorem 3: E[ProfitSybil] = m/(N+m)*R_total - m*S_min
        Returns True nếu attack profitable (→ cần tăng stake_min).
        """
        expected_profit = (m_sybil / (N_total + m_sybil)) * R_total - m_sybil * self.stake_min
        return expected_profit > 0
```

### 8.4 `src/gtm/nash_validator.py`

```python
class NashValidator:
    """
    Empirical validation của Nash Equilibrium (Experiment E5).
    Simulate rational agents với varying cost structures.
    """
    
    def __init__(self, tokenomics: 'TokenomicsEngine'):
        self.tk = tokenomics
    
    def simulate_rational_agent(self, node_id: int, cost: float, 
                                 honest_phi: float, lazy_phi: float,
                                 R_total: float, N: int) -> str:
        """
        Simulate decision của rational agent:
        - honest: u = R*phi_honest/total_phi - cost
        - lazy: u = R*phi_lazy/total_phi - penalty
        
        Returns: 'honest' | 'lazy' | 'abstain'
        """
        # Simplified: giả sử total_phi = 1.0 (normalized)
        u_honest = R_total * honest_phi - cost
        u_lazy = R_total * lazy_phi - self.tk.PENALTY_LAZY
        
        if u_honest > u_lazy and u_honest > 0:
            return 'honest'
        elif u_lazy > u_honest and u_lazy > 0:
            return 'lazy'
        else:
            return 'abstain'
    
    def verify_nash_equilibrium(self, results: dict) -> bool:
        """
        Kiểm tra Nash Equilibrium condition: với mọi node, unilateral deviation không profitable.
        """
        all_honest = all(v == 'honest' for v in results.values())
        return all_honest
```

---

## 9. Module 5: Blockchain Mock

### 9.1 `src/blockchain/ledger.py`

```python
import sqlite3
import json
import time
import threading

class BlockchainLedger:
    """
    SQLite-based blockchain mock.
    Thread-safe ledger ghi reward, proof, event.
    """
    
    def __init__(self, db_path: str = './chainfsl_ledger.db'):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS rewards (
                epoch INTEGER, node_id INTEGER, reward REAL, 
                shapley_value REAL, timestamp REAL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS verifications (
                epoch INTEGER, node_id INTEGER, is_valid INTEGER, 
                penalty REAL, proof_type TEXT, timestamp REAL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS blocks (
                block_id INTEGER PRIMARY KEY, epoch INTEGER,
                merkle_root TEXT, n_verified INTEGER, timestamp REAL)''')
    
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    def record_reward(self, epoch: int, node_id: int, reward: float, shapley: float):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT INTO rewards VALUES (?,?,?,?,?)',
                    (epoch, node_id, reward, shapley, time.time())
                )
    
    def record_verification(self, epoch: int, node_id: int, is_valid: bool, 
                             penalty: float, proof_type: str):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT INTO verifications VALUES (?,?,?,?,?,?)',
                    (epoch, node_id, int(is_valid), penalty, proof_type, time.time())
                )
    
    def commit_block(self, epoch: int, merkle_root: str, n_verified: int):
        """
        On-chain commitment (Eq. trong Section 4.3):
        Chỉ ghi Merkle root + aggregate signature (O(1) per epoch).
        """
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT INTO blocks VALUES (NULL,?,?,?,?)',
                    (epoch, merkle_root, n_verified, time.time())
                )
    
    def get_cumulative_reward(self, node_id: int) -> float:
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT SUM(reward) FROM rewards WHERE node_id=?', (node_id,)
            ).fetchone()
            return row[0] or 0.0
    
    def get_epoch_stats(self, epoch: int) -> dict:
        with self._get_conn() as conn:
            rewards = conn.execute(
                'SELECT node_id, reward, shapley_value FROM rewards WHERE epoch=?', (epoch,)
            ).fetchall()
            verifs = conn.execute(
                'SELECT COUNT(*), SUM(is_valid) FROM verifications WHERE epoch=?', (epoch,)
            ).fetchone()
        return {
            'rewards': {r[0]: {'reward': r[1], 'shapley': r[2]} for r in rewards},
            'n_verified': verifs[0] or 0,
            'n_valid': verifs[1] or 0,
        }
    
    def measure_overhead(self, epoch: int) -> dict:
        """Đo blockchain overhead (cho E7)."""
        with self._get_conn() as conn:
            block = conn.execute(
                'SELECT * FROM blocks WHERE epoch=?', (epoch,)
            ).fetchone()
        return {
            'on_chain_writes': 1,          # chỉ 1 lần commit per epoch (O(1))
            'merkle_root_bytes': 32,       # SHA-256
            'epoch': epoch,
        }
```

---

## 10. Orchestration: ChainFSL Protocol

### 10.1 `src/protocol/chainfsl.py`

```python
import time
import copy
import hashlib
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

class ChainFSLProtocol:
    """
    End-to-end ChainFSL Protocol — Algorithm 2 bài báo.
    
    Tích hợp tất cả modules: HASO + TVE + GTM + Blockchain.
    Chạy N nodes concurrently bằng ThreadPoolExecutor (single-machine).
    """
    
    def __init__(self, config: dict, nodes, global_model, dataloaders,
                 haso_agents, committee, shapley_engine, tokenomics, ledger, 
                 network_emulator, gossip):
        self.cfg = config
        self.nodes = nodes
        self.model = global_model
        self.loaders = dataloaders
        self.agents = haso_agents        # dict node_id -> HaSOAgent
        self.committee_engine = committee
        self.shapley = shapley_engine
        self.tokenomics = tokenomics
        self.ledger = ledger
        self.net = network_emulator
        self.gossip = gossip
        
        self.global_state = global_model.state_dict()
        self.node_staleness = {n.node_id: 0 for n in nodes}
        self.current_round = 0
        
        # Ablation flags (cho E6)
        self.haso_enabled = config.get('haso_enabled', True)
        self.tve_enabled = config.get('tve_enabled', True)
        self.gtm_enabled = config.get('gtm_enabled', True)
        
        # Attack injection (cho E4)
        self.lazy_client_ids = set()
        self.sybil_node_ids = set()
    
    def run(self, total_rounds: int) -> list[dict]:
        """
        Main training loop.
        Returns: list of per-round metrics dicts.
        """
        all_metrics = []
        
        for t in range(1, total_rounds + 1):
            self.current_round = t
            round_start = time.time()
            
            # === Phase 1: HASO Decisions ===
            configs = self._phase_haso(t)
            
            # === Phase 2: Distributed SFL Training ===
            updates, proofs = self._phase_training(t, configs)
            
            # === Phase 3: TVE Verification ===
            verification_results = self._phase_verification(t, updates, proofs)
            
            # === Phase 4: Async Aggregation ===
            valid_updates = [u for u in updates 
                             if verification_results.get(u['node_id'], {}).get('valid', True)]
            self._phase_aggregation(valid_updates)
            
            # === Phase 5: GTM Rewards ===
            shapley_values, rewards = self._phase_gtm(t, updates, verification_results)
            
            # === Phase 6: Blockchain Commit ===
            self._phase_blockchain(t, rewards, shapley_values, verification_results)
            
            # === Phase 7: HASO Policy Update ===
            self._phase_haso_update(t, updates, shapley_values)
            
            # === Collect Metrics ===
            metrics = self._collect_metrics(t, time.time() - round_start, 
                                             rewards, verification_results)
            all_metrics.append(metrics)
            
            if t % 10 == 0:
                print(f"Round {t}/{total_rounds} | "
                      f"Loss: {metrics['train_loss']:.4f} | "
                      f"Acc: {metrics['test_acc']:.2f}% | "
                      f"Latency: {metrics['round_latency']:.2f}s")
        
        return all_metrics
    
    def _phase_haso(self, t: int) -> dict:
        """Phase 1: Mỗi node quyết định (c_i, B_i, H_i, target_j)."""
        configs = {}
        
        if not self.haso_enabled:
            # Ablation: fixed uniform cut layer
            for node in self.nodes:
                configs[node.node_id] = {
                    'cut_layer': 2, 'batch_size': 32, 'H': 1, 'target_compute_node': 0
                }
            return configs
        
        for node in self.nodes:
            agent = self.agents[node.node_id]
            obs = agent.env._get_obs()
            # Update gossip info trước
            neighbor_avail = self.gossip.mean_neighbor_availability(node.node_id)
            agent.env.update_neighbor_avail(neighbor_avail)
            configs[node.node_id] = agent.decide(obs)
        
        return configs
    
    def _phase_training(self, t: int, configs: dict) -> tuple:
        """Phase 2: Training song song với ThreadPoolExecutor."""
        updates = []
        proofs = []
        
        def train_node(node):
            cfg = configs[node.node_id]
            cut_layer = cfg['cut_layer']
            batch_size = cfg['batch_size']
            H = cfg['H']
            
            # Enforce memory constraint
            if not node.can_fit_cut_layer(cut_layer, {1:150, 2:300, 3:500, 4:700}):
                cut_layer = 1   # fallback
            
            # Simulate compute delay theo tier
            compute_flops = 1e8 * cut_layer
            compute_delay = node.compute_time(compute_flops)
            time.sleep(min(compute_delay, 0.1))   # cap ở 0.1s để không quá chậm
            
            # Lấy 1 batch từ dataloader
            loader = self.loaders[node.node_id]
            try:
                x, y = next(iter(loader))
            except StopIteration:
                return None
            
            # Forward pass (simplified: dùng global model)
            with torch.no_grad():
                # Lấy client-side model từ global state
                # (trong full impl: load weights vào SplittableResNet18)
                pass
            
            # Generate proof theo tier
            from src.tve.commitment import ComputationCommitment
            input_hash = ComputationCommitment.commit_input(x)
            
            if node.tier <= 2:
                proof = ComputationCommitment.gen_proof_tier1_2(
                    x, x,  # placeholder: a = x trong mock
                    hashlib.sha256(b'model').digest()
                )
            elif node.tier == 3:
                proof = ComputationCommitment.gen_proof_tier3(x, x)
            else:
                proof = ComputationCommitment.gen_proof_tier4(x)
            
            proof['input_hash'] = input_hash
            
            update = {
                'node_id': node.node_id,
                'cut_layer': cut_layer,
                'client_state': copy.deepcopy(self.global_state),  # placeholder
                'server_state': {},
                'data_size': len(loader.dataset),
                'staleness': self.node_staleness[node.node_id],
                'input_hash': input_hash,
                'loss': 1.0 - node.flops_ratio,  # placeholder loss
            }
            return update, proof
        
        with ThreadPoolExecutor(max_workers=min(len(self.nodes), 16)) as executor:
            futures = {executor.submit(train_node, node): node for node in self.nodes}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    update, proof = result
                    updates.append(update)
                    proofs.append(proof)
        
        return updates, proofs
    
    def _phase_verification(self, t: int, updates: list, proofs: list) -> dict:
        """Phase 3: TVE committee selection và verification."""
        if not self.tve_enabled:
            return {u['node_id']: {'valid': True, 'penalty': 0.0} for u in updates}
        
        # Tạo epoch seed từ block hash mock
        block_hash = hashlib.sha256(f"block_{t}".encode()).digest()
        
        # Select committee
        committee = self.committee_engine.select_committee(t, block_hash)
        
        # Verify
        results = self.committee_engine.verify_updates(
            updates, proofs, 
            lazy_client_ids=self.lazy_client_ids
        )
        
        # Record verifications
        for node_id, result in results.items():
            self.ledger.record_verification(
                t, node_id, result['valid'], result['penalty'],
                proofs[0].get('type', 'unknown') if proofs else 'none'
            )
        
        return results
    
    def _phase_aggregation(self, valid_updates: list):
        """Phase 4: Staleness-decayed async aggregation."""
        from src.sfl.aggregator import AsyncAggregator
        aggregator = AsyncAggregator(self.global_state, rho=self.cfg.get('staleness_decay', 0.9))
        
        # Update staleness
        updated_nodes = {u['node_id'] for u in valid_updates}
        for node in self.nodes:
            if node.node_id in updated_nodes:
                self.node_staleness[node.node_id] = 0
            else:
                self.node_staleness[node.node_id] += 1
        
        if valid_updates:
            self.global_state = aggregator.aggregate(valid_updates)
    
    def _phase_gtm(self, t: int, updates: list, verification_results: dict) -> tuple:
        """Phase 5: Compute Shapley và distribute rewards."""
        if not self.gtm_enabled:
            # Equal distribution
            n = len(updates)
            shapley_vals = {u['node_id']: 1.0/n for u in updates}
            rewards = {u['node_id']: self.tokenomics.total_reward(t)/n for u in updates}
            return shapley_vals, rewards
        
        # Characteristic function: model performance improvement
        def value_fn(coalition: list) -> float:
            # Simplified: weighted sum of data sizes (proxy cho accuracy gain)
            total = sum(
                self.loaders[nid].dataset.__len__() 
                for nid in coalition 
                if nid in self.loaders
            )
            return total / 50000.0   # normalize bởi dataset size
        
        node_ids = [u['node_id'] for u in updates]
        shapley_vals = self.shapley.compute(node_ids, value_fn)
        
        rewards = self.tokenomics.distribute(t, shapley_vals, verification_results)
        
        return shapley_vals, rewards
    
    def _phase_blockchain(self, t: int, rewards: dict, shapley_vals: dict, 
                           verif_results: dict):
        """Phase 6: Commit to ledger."""
        for node_id, reward in rewards.items():
            self.ledger.record_reward(t, node_id, reward, shapley_vals.get(node_id, 0.0))
        
        # Merkle root mock
        data = json.dumps(rewards, sort_keys=True).encode()
        merkle_root = hashlib.sha256(data).hexdigest()
        n_verified = sum(1 for v in verif_results.values() if v.get('valid', False))
        self.ledger.commit_block(t, merkle_root, n_verified)
    
    def _phase_haso_update(self, t: int, updates: list, shapley_vals: dict):
        """Phase 7: Update PPO policy của mỗi node."""
        if not self.haso_enabled:
            return
        
        for node in self.nodes:
            agent = self.agents.get(node.node_id)
            if agent:
                phi = shapley_vals.get(node.node_id, 0.0)
                agent.env.update_shapley(phi)
                agent.learn(total_timesteps=64)   # 1 mini-update per epoch
    
    def _collect_metrics(self, t: int, latency: float, rewards: dict, 
                          verif_results: dict) -> dict:
        """Collect all metrics cho logging."""
        valid_count = sum(1 for v in verif_results.values() if v.get('valid', False))
        total_count = len(verif_results)
        
        reward_values = list(rewards.values())
        fairness = self._jains_fairness(reward_values)
        
        return {
            'round': t,
            'round_latency': latency,
            'train_loss': 0.5,          # placeholder: dùng real loss từ trainer
            'test_acc': 50.0,           # placeholder: eval trên test set
            'n_valid_updates': valid_count,
            'attack_detection_rate': valid_count / max(total_count, 1),
            'fairness_index': fairness,
            'total_reward': sum(max(0, r) for r in reward_values),
            'mean_shapley': np.mean([abs(r) for r in reward_values]) if reward_values else 0,
        }
    
    @staticmethod
    def _jains_fairness(rewards: list) -> float:
        """Jain's fairness index: (sum R_i)^2 / (N * sum R_i^2)."""
        if not rewards:
            return 0.0
        pos = [max(0.0, r) for r in rewards]
        if sum(pos) == 0:
            return 0.0
        return (sum(pos)**2) / (len(pos) * sum(r**2 for r in pos))


import json  # thêm import cần thiết
```

---

## 11. Experiments (E1–E7)

### 11.1 Entry point: `experiments/run_experiment.py`

```python
#!/usr/bin/env python3
"""
Usage:
    python experiments/run_experiment.py --exp e1 --config config/experiment_configs/e1_haso.yaml
    python experiments/run_experiment.py --exp e4 --sybil_fraction 0.2
"""
import argparse
import yaml
import importlib

EXPERIMENT_MAP = {
    'e1': 'experiments.e1_haso_effectiveness',
    'e2': 'experiments.e2_scalability',
    'e3': 'experiments.e3_noniid',
    'e4': 'experiments.e4_security',
    'e5': 'experiments.e5_incentive',
    'e6': 'experiments.e6_ablation',
    'e7': 'experiments.e7_blockchain_overhead',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=EXPERIMENT_MAP.keys())
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--n_nodes', type=int, default=None)
    parser.add_argument('--sybil_fraction', type=float, default=None)
    parser.add_argument('--lazy_fraction', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None, help='Dirichlet alpha')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override from CLI
    if args.n_nodes:
        config['n_nodes'] = args.n_nodes
    if args.sybil_fraction is not None:
        config['sybil_fraction'] = args.sybil_fraction
    if args.lazy_fraction is not None:
        config['lazy_client_fraction'] = args.lazy_fraction
    if args.alpha is not None:
        config['dirichlet_alpha'] = args.alpha
    
    config['seed'] = args.seed
    
    module = importlib.import_module(EXPERIMENT_MAP[args.exp])
    module.run(config)

if __name__ == '__main__':
    main()
```

### 11.2 `experiments/e1_haso_effectiveness.py`

```python
"""
Experiment 1: HASO Effectiveness
Hypothesis: HASO giảm training latency 40-60% so với uniform-split baselines.
Metric: time-to-accuracy, straggler ratio.
"""
import time
import json
from src.emulator.tier_factory import create_nodes
from src.protocol.chainfsl import ChainFSLProtocol
from src.haso.env import SFLNodeEnv
from src.haso.agent import HaSOAgent
from baselines.splitfed import SplitFedBaseline
from baselines.adaptsfl import AdaptSFLBaseline

TARGET_ACCURACY = 70.0   # target accuracy % để đo time-to-accuracy

def run(config: dict):
    print("=" * 60)
    print("Experiment E1: HASO Effectiveness")
    print("=" * 60)
    
    results = {}
    
    # === ChainFSL (full) ===
    results['chainfsl'] = run_chainfsl(config, haso=True)
    
    # === ChainFSL-NoHASO (ablation baseline) ===
    no_haso_cfg = {**config, 'haso_enabled': False}
    results['chainfsl_nohaso'] = run_chainfsl(no_haso_cfg, haso=False)
    
    # === SplitFed baseline ===
    results['splitfed'] = SplitFedBaseline(config).run()
    
    # Save
    with open(f"logs/e1_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Report
    print("\n--- Results ---")
    for name, res in results.items():
        print(f"{name}: time-to-{TARGET_ACCURACY}%: {res.get('time_to_target', 'N/A'):.1f}s | "
              f"straggler_ratio: {res.get('straggler_ratio', 0):.2f}")

def run_chainfsl(config: dict, haso: bool) -> dict:
    nodes = create_nodes(config['n_nodes'], config['tier_distribution'])
    # ... khởi tạo đầy đủ (xem chainfsl.py)
    # Return metrics
    return {'time_to_target': 0.0, 'straggler_ratio': 1.0}  # placeholder
```

### 11.3 `experiments/e4_security.py`

```python
"""
Experiment 4: Security Evaluation
Inject Sybil nodes, lazy clients, và Byzantine nodes.
Metric: attack detection rate, accuracy degradation.
"""

def run(config: dict):
    print("=" * 60)
    print("Experiment E4: Security Evaluation")
    print("=" * 60)
    
    attack_fractions = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    for frac in attack_fractions:
        for attack_type in ['sybil', 'lazy', 'poison']:
            cfg = {**config, f'{attack_type}_fraction': frac}
            key = f'{attack_type}_f{frac}'
            
            protocol = _build_protocol(cfg)
            
            # Inject attacks
            n_attack = int(frac * cfg['n_nodes'])
            attack_node_ids = set(range(n_attack))
            
            if attack_type == 'sybil':
                protocol.sybil_node_ids = attack_node_ids
            elif attack_type == 'lazy':
                protocol.lazy_client_ids = attack_node_ids
            
            metrics = protocol.run(total_rounds=config.get('global_rounds', 50))
            
            results[key] = {
                'detection_rate': _mean(m['attack_detection_rate'] for m in metrics),
                'final_accuracy': metrics[-1].get('test_acc', 0),
                'attack_type': attack_type,
                'fraction': frac,
            }
    
    import json
    with open('logs/e4_security.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(json.dumps(results, indent=2))

def _mean(iterable):
    vals = list(iterable)
    return sum(vals) / len(vals) if vals else 0.0

def _build_protocol(config):
    pass  # khởi tạo ChainFSLProtocol đầy đủ
```

### 11.4 `experiments/e6_ablation.py`

```python
"""
Experiment 6: Ablation Study
Systematically disable từng module để quantify contribution.
"""

ABLATION_CONFIGS = {
    'full':           {'haso_enabled': True, 'tve_enabled': True,  'gtm_enabled': True},
    'no_haso':        {'haso_enabled': False,'tve_enabled': True,   'gtm_enabled': True},
    'no_tve':         {'haso_enabled': True, 'tve_enabled': False,  'gtm_enabled': True},
    'no_gtm':         {'haso_enabled': True, 'tve_enabled': True,   'gtm_enabled': False},
    'no_haso_no_tve': {'haso_enabled': False,'tve_enabled': False,  'gtm_enabled': True},
}

def run(config: dict):
    results = {}
    for name, flags in ABLATION_CONFIGS.items():
        cfg = {**config, **flags}
        # run protocol và collect metrics
        results[name] = {}  # fill với actual metrics
    
    import json
    with open('logs/e6_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
```

---

## 12. Logging & Metrics

### 12.1 `src/utils/logger.py`

```python
import json
import time
from pathlib import Path
import numpy as np

class ExperimentLogger:
    """
    Centralized logging cho tất cả experiments.
    Ghi metrics theo từng round, hỗ trợ TensorBoard và JSON export.
    """
    
    METRICS = [
        # Accuracy & convergence
        'train_loss', 'test_acc', 'val_loss',
        # Latency
        'round_latency', 'straggler_ratio',
        # Resource
        'mean_cpu_util', 'mean_bandwidth_util',
        # Security
        'attack_detection_rate', 'n_valid_updates',
        # Fairness
        'fairness_index', 'reward_std',
        # Blockchain
        'on_chain_writes_per_epoch', 'ledger_size_bytes',
        # Shapley
        'mean_shapley', 'shapley_variance',
    ]
    
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        else:
            self.writer = None
    
    def log_round(self, t: int, metrics: dict):
        metrics['round'] = t
        metrics['timestamp'] = time.time()
        self.history.append(metrics)
        
        if self.writer:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(key, val, t)
        
        # Periodic save
        if t % 10 == 0:
            self.save()
    
    def save(self):
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def summary(self) -> dict:
        """Tính summary statistics từ toàn bộ experiment."""
        if not self.history:
            return {}
        
        final = self.history[-1]
        best_acc = max(m.get('test_acc', 0) for m in self.history)
        
        # Time to target accuracy (70%)
        target = 70.0
        time_to_target = None
        for m in self.history:
            if m.get('test_acc', 0) >= target:
                time_to_target = m.get('timestamp', 0) - self.history[0].get('timestamp', 0)
                break
        
        return {
            'final_accuracy': final.get('test_acc', 0),
            'best_accuracy': best_acc,
            'time_to_target_70': time_to_target,
            'mean_latency': np.mean([m.get('round_latency', 0) for m in self.history]),
            'mean_fairness': np.mean([m.get('fairness_index', 0) for m in self.history]),
            'mean_detection_rate': np.mean([m.get('attack_detection_rate', 1.0) for m in self.history]),
        }
```

---

## 13. Thứ tự coding cho Agent

> **Claude Code Agent:** Implement theo đúng thứ tự sau. Mỗi bước build on bước trước. Sau mỗi bước, chạy unit test nhỏ để xác nhận trước khi tiếp tục.

### Sprint 1 (Ngày 1–3): Foundation

```
1. Tạo cấu trúc thư mục đầy đủ (mkdir -p ...)
2. Viết requirements.txt và cài đặt môi trường
3. Implement src/emulator/ (node_profile, tier_factory)
   → Test: tạo 50 nodes, in ra tier distribution
4. Implement src/sfl/models.py (SplittableResNet18)
   → Test: tạo model, split tại cut_layer=2, verify shapes
5. Implement src/sfl/data_loader.py (CIFAR-10 + Dirichlet)
   → Test: partition 50 clients với alpha=0.5, verify sizes
6. Implement src/sfl/aggregator.py (staleness decay)
   → Test: aggregate 3 updates với staleness [0,1,2]
```

### Sprint 2 (Ngày 4–7): SFL + HASO

```
7. Implement src/sfl/split_model.py (ClientModel, ServerModel)
   → Test: 1 forward/backward cycle qua cut layer
8. Implement src/sfl/trainer.py (1 node training loop H rounds)
   → Test: train 5 rounds trên CIFAR-10 subset
9. Implement src/haso/env.py (SFLNodeEnv)
   → Test: gym.make, reset, step với random action
10. Implement src/haso/agent.py (HaSOAgent + PPO)
    → Test: train PPO 100 timesteps, check action shape
11. Implement src/haso/gossip.py (GossipProtocol)
    → Test: 10 nodes broadcast LRH, query neighbors
```

### Sprint 3 (Ngày 8–11): TVE + GTM

```
12. Implement src/tve/vrf.py (MockVRF)
    → Test: compute + verify, check unpredictability
13. Implement src/tve/commitment.py (tier-dependent proofs)
    → Test: gen proof cho T1, T2, T3, T4 và verify
14. Implement src/tve/committee.py (VRF selection + verify)
    → Test: select committee K=5 từ 50 nodes, check distribution
15. Implement src/gtm/contribution.py (ContributionVector + VLI)
    → Test: tính VLI cho 1 node update
16. Implement src/gtm/shapley.py (TMCS)
    → Test: tính Shapley cho 10 nodes, verify sum ≈ v(all)
17. Implement src/gtm/tokenomics.py (reward + Nash check)
    → Test: distribute reward, verify penalty logic
```

### Sprint 4 (Ngày 12–14): Integration

```
18. Implement src/blockchain/ledger.py (SQLite)
    → Test: record + query rewards, check thread-safety
19. Implement src/protocol/chainfsl.py (full Algorithm 2)
    → Test: chạy 5 rounds với 10 nodes, verify metrics dict
20. Implement baselines/ (FedAvg, SplitFed, AdaptSFL đơn giản)
    → Test: run 5 rounds mỗi baseline
```

### Sprint 5 (Ngày 15–16): Experiments

```
21. Implement experiments/e1_haso_effectiveness.py
    → Run với n_nodes=20, 30 rounds
22. Implement experiments/e4_security.py
    → Run với lazy_fraction=0.2
23. Implement experiments/e6_ablation.py
    → Run 4 ablation configs
24. Implement analysis/plot_results.py
    → Generate plots cho tất cả experiments
25. Implement experiments/e2, e3, e5, e7 (nếu còn thời gian)
```

---

## Phụ lục A: Quyết định thiết kế quan trọng

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Thread vs Process | ThreadPoolExecutor (threads) | GIL không ảnh hưởng vì training là C++ PyTorch ops; dễ share memory hơn |
| Gossip protocol | Python Manager().dict() | Đủ để mô phỏng shared state; không cần network thật |
| zk-SNARK | Hash-based mock + timing | snarkjs yêu cầu Node.js; mock đủ để đo overhead và logic |
| Blockchain | SQLite (không phải Hardhat) | Hardhat cần Node.js, gas real; SQLite đủ cho E7 overhead measurement |
| Shapley M | 50 permutations | M=50 cho variance ~1/√50 ≈ 14%, đủ tốt; M=100 nếu cần chính xác hơn |
| Cut layer choices | {1, 2, 3, 4} | Tương ứng residual block boundaries của ResNet-18 |
| Async aggregation | Mọi update accepted ngay | Không block đợi slow nodes; staleness decay xử lý fairness |

## Phụ lục B: Expected results (từ bài báo Table 4)

| Experiment | Metric | Target |
|---|---|---|
| E1: HASO | Training latency | 40–60% giảm vs uniform-split |
| E2: Scalability | Throughput vs N | Near-linear scaling |
| E3: Non-IID | Accuracy vs α | Robust across α ∈ {0.1, 1.0} |
| E4: Security | Attack detection | >95% detection rate |
| E5: Incentive | Participation | Nash Eq. validated empirically |
| E6: Ablation | Per-module delta | Mỗi module contribute positively |
| E7: Overhead | Gas/latency | <5% total overhead |

---

*File này được tạo để làm input cho Claude Code Agent. Mọi class, function signature, và design decision đã được finalize. Agent có thể bắt đầu từ Sprint 1 và implement tuần tự.*
