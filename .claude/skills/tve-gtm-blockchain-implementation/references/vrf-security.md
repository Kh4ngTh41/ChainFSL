# VRF Security Best Practices

Verifiable Random Functions (VRFs) là nền tảng cho committee selection công bằng. Tài liệu này mô tả các vấn đề bảo mật và cách phòng tránh.

## Cryptographic VRF vs MockVRF

### MockVRF (Development/Testing)

**Ưu điểm:**
- Đơn giản, dễ implement
- Không cần thư viện cryptographic phức tạp
- Nhanh (HMAC-SHA256 < 1μs)

**Nhược điểm:**
- Không có public verifiability: Chỉ người có secret_key mới verify được
- Dễ bị grinding attacks nếu attacker kiểm soát public_seed
- Không có zero-knowledge proof

### Production VRF (VRF-ED25519, ECVRF)

**Ưu điểm:**
- Public verifiability: Ai cũng có thể verify output với public_key
- Grinding-resistant: Output phụ thuộc vào cả private_key và input
- Standardized: RFC 9381 (VRFv10)

**Implementation:**

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
import hashlib

class ECVRF:
    def __init__(self, private_key: Ed25519PrivateKey):
        self.private_key = private_key
        self.public_key = private_key.public_key()
    
    def prove(self, alpha: bytes) -> tuple:
        """Tạo VRF proof.
        
        Returns:
            (beta: bytes, pi: bytes) — output và proof
        """
        # Simplified — production cần follow RFC 9381
        h = hashlib.sha256(alpha).digest()
        gamma = self.private_key.sign(h)
        
        # Beta = hash(gamma)
        beta = hashlib.sha256(gamma).digest()
        
        return beta, gamma  # gamma là proof
    
    def verify(self, alpha: bytes, beta: bytes, pi: bytes) -> bool:
        """Verify VRF proof với public key."""
        try:
            h = hashlib.sha256(alpha).digest()
            self.public_key.verify(pi, h)
            
            expected_beta = hashlib.sha256(pi).digest()
            return beta == expected_beta
        except:
            return False
```

## Grinding Attacks

### Vấn đề

Attacker kiểm soát `public_seed`, thử nhiều giá trị để chọn committee có lợi:

```python
# Attacker's code
for seed in range(10000):
    committee = select_committee(nodes, seed.to_bytes(8, 'big'), epoch, 10)
    if attacker_controls_majority(committee):
        broadcast_seed(seed)
        break
```

### Giải pháp: Commit-Reveal Scheme

**Round 1 (Commit):**

```python
# Mỗi node commit một random value
random_value = os.urandom(32)
commitment = hashlib.sha256(random_value).digest()
broadcast(commitment)
```

**Round 2 (Reveal):**

```python
# Sau khi tất cả commitments được broadcast
broadcast(random_value)

# Verify commitments
for node, (commitment, revealed) in commitments.items():
    assert hashlib.sha256(revealed).digest() == commitment

# Tạo public_seed từ tất cả revealed values
public_seed = hashlib.sha256(b''.join(sorted(revealed_values))).digest()
```

**Tại sao an toàn:**
- Attacker không thể thay đổi random_value sau khi commit
- Public_seed phụ thuộc vào tất cả nodes, không node nào kiểm soát hoàn toàn

## Key Management

### Secret Key Storage

**Không bao giờ:**
- Lưu plaintext trong code hoặc config files
- Chia sẻ secret_key qua mạng không mã hóa
- Dùng chung secret_key cho nhiều nodes

**Best practices:**

```python
import keyring
from cryptography.fernet import Fernet

# Lưu trong OS keychain
keyring.set_password("blockchain_app", "node_001", secret_key.hex())

# Hoặc encrypt với master key
master_key = Fernet.generate_key()
f = Fernet(master_key)
encrypted_secret = f.encrypt(secret_key)

# Chỉ decrypt khi cần dùng
secret_key = f.decrypt(encrypted_secret)
```

### Key Rotation

Thay đổi secret_key định kỳ để giảm thiệt hại nếu bị leak:

```python
class VRFKeyManager:
    def __init__(self):
        self.keys = {}  # {epoch_range: secret_key}
    
    def rotate_key(self, new_epoch_start: int):
        new_key = os.urandom(32)
        self.keys[new_epoch_start] = new_key
        
        # Xóa keys cũ (giữ lại 2-3 epoch gần nhất)
        old_epochs = [e for e in self.keys if e < new_epoch_start - 2]
        for e in old_epochs:
            del self.keys[e]
    
    def get_key_for_epoch(self, epoch: int) -> bytes:
        for epoch_start in sorted(self.keys.keys(), reverse=True):
            if epoch >= epoch_start:
                return self.keys[epoch_start]
        raise ValueError(f"No key for epoch {epoch}")
```

## Timing Attacks

### Vấn đề

Attacker đo thời gian verify để suy ra thông tin về secret_key.

### Giải pháp: Constant-time Operations

```python
import hmac

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """So sánh hai byte strings trong thời gian cố định."""
    return hmac.compare_digest(a, b)

# Thay vì:
if vrf_output == expected_output:  # ❌ Timing leak
    return True

# Dùng:
if constant_time_compare(vrf_output, expected_output):  # ✅ Safe
    return True
```

## Sybil Resistance

### Vấn đề

Một entity tạo nhiều nodes với secret_keys khác nhau để tăng xác suất vào committee.

### Giải pháp: Stake-Weighted VRF

```python
def stake_weighted_select_committee(nodes: list, stakes: dict,
                                     public_seed: bytes, epoch: int,
                                     committee_size: int) -> list:
    """Chọn committee với xác suất tỷ lệ thuận với stake."""
    vrf_outputs = []
    
    for node_id, secret_key in nodes:
        vrf = MockVRF(secret_key)
        raw_output = vrf.evaluate(public_seed, epoch)
        
        # Adjust output theo stake
        stake = stakes.get(node_id, 1.0)
        adjusted_output = raw_output / stake  # Stake cao → output thấp → ưu tiên
        
        vrf_outputs.append((adjusted_output, node_id))
    
    vrf_outputs.sort()
    return [node_id for _, node_id in vrf_outputs[:committee_size]]
```

**Lưu ý:** Cần minimum stake requirement để tránh spam nodes.

## Audit & Monitoring

### Logging

```python
import logging

logger = logging.getLogger('vrf_security')

def select_committee_with_audit(nodes, public_seed, epoch, committee_size):
    logger.info(f"Committee selection for epoch {epoch}")
    logger.info(f"Public seed: {public_seed.hex()[:16]}...")
    logger.info(f"Total nodes: {len(nodes)}")
    
    committee = select_committee(nodes, public_seed, epoch, committee_size)
    
    logger.info(f"Committee selected: {committee}")
    
    # Kiểm tra distribution
    node_frequencies = {}
    for node_id in committee:
        node_frequencies[node_id] = node_frequencies.get(node_id, 0) + 1
    
    if any(freq > committee_size * 0.3 for freq in node_frequencies.values()):
        logger.warning("Committee concentration detected!")
    
    return committee
```

### Anomaly Detection

```python
from collections import deque

class CommitteeMonitor:
    def __init__(self, window_size: int = 100):
        self.history = deque(maxlen=window_size)
    
    def record_committee(self, committee: list):
        self.history.append(committee)
    
    def detect_bias(self) -> dict:
        """Phát hiện nodes xuất hiện quá thường xuyên."""
        node_counts = {}
        
        for committee in self.history:
            for node in committee:
                node_counts[node] = node_counts.get(node, 0) + 1
        
        total_selections = len(self.history)
        expected_freq = len(self.history[0]) / total_selections if self.history else 0
        
        biased_nodes = {}
        for node, count in node_counts.items():
            actual_freq = count / total_selections
            if actual_freq > expected_freq * 1.5:  # 50% cao hơn expected
                biased_nodes[node] = actual_freq
        
        return biased_nodes
```

## Testing Checklist

- [ ] VRF output phân phối đồng đều (Kolmogorov-Smirnov test)
- [ ] Không thể dự đoán output mà không có secret_key
- [ ] Commit-reveal scheme ngăn chặn grinding
- [ ] Secret keys được encrypt at rest
- [ ] Constant-time comparison cho sensitive operations
- [ ] Stake-weighting hoạt động đúng
- [ ] Logging đầy đủ cho audit trail
- [ ] Anomaly detection phát hiện bias
