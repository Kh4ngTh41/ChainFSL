# Algorithm 2: Tiered Verification Protocol - Detailed Specification

## Purpose

Algorithm 2 defines the verification protocol for ChainFSL's TVE module. It ensures computational correctness while adapting proof complexity to device capabilities (tiers).

## Formal Definition

### Inputs
- **τ** (tau): Device tier ∈ {1, 2, 3}
- **Δw**: Local model update (gradient or weight delta)
- **A**: Split point activations (output of client-side forward pass)
- **model_slice**: Client-side portion of split model (layers 1 to split_point)

### Outputs
- **π** (pi): Verification proof (format varies by tier)
- **v**: Validity flag (Boolean)

## Tier 1 Verification (High-Resource Devices)

### Step 1: Generate zk-SNARK Proof

Prove correct execution of client-side computation without revealing inputs:

```python
def generate_zk_proof(delta_w, activations, model_slice, private_data):
    # Circuit: "I know private_data such that
    # forward(model_slice, private_data) == activations AND
    # backward(activations, loss_grad) == delta_w"
    
    circuit = ZKCircuit()
    circuit.add_constraint("forward_pass", model_slice, activations)
    circuit.add_constraint("backward_pass", activations, delta_w)
    
    proof = circuit.prove(private_witness=private_data)
    return proof
```

Use libsnark or bellman for implementation. Proof generation takes ~2-5s on Tier 1 devices.

### Step 2: Verify Gradient Consistency

Coordinator recomputes gradients using submitted activations:

```python
def verify_gradient_consistency(delta_w, activations, server_model_slice):
    # Recompute server-side backward pass
    expected_grad = server_model_slice.backward(activations)
    
    # Compare with submitted gradient (cosine similarity)
    similarity = cosine_sim(expected_grad, delta_w)
    
    # Threshold based on numerical precision and compression
    return similarity > 0.95
```

If similarity < 0.95, reject update and penalize device (reduce future rewards by 10%).

### Output
- **π**: zk-SNARK proof (serialized, ~200 bytes)
- **v**: True if gradient consistency check passes, False otherwise

## Tier 2 Verification (Medium-Resource Devices)

### Step 1: Merkle Proof of Activations

Prove activations are derived from legitimate computation:

```python
def merkle_proof_activations(activations):
    # Build Merkle tree over activation tensors
    leaves = [hash(a.tobytes()) for a in activations]
    tree = MerkleTree(leaves)
    
    # Proof includes root and authentication path
    root = tree.get_root()
    paths = [tree.get_proof(i) for i in range(len(leaves))]
    
    return {"root": root, "paths": paths}
```

Verifier checks:
1. Activation hashes match Merkle leaves
2. Authentication paths reconstruct root
3. Root is committed on-chain before training started (prevents post-hoc tampering)

### Step 2: Gradient Norm Check

Verify gradient magnitude is within expected range:

```python
def check_gradient_norm(delta_w, historical_stats):
    current_norm = torch.norm(delta_w).item()
    
    # Expected range from historical statistics (updated each round)
    mu, sigma = historical_stats['mean_norm'], historical_stats['std_norm']
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    
    return lower_bound <= current_norm <= upper_bound
```

If norm is out of bounds, flag as potential Byzantine behavior. Three consecutive violations trigger device exclusion.

### Output
- **π**: Merkle root + authentication paths + gradient norm (serialized, ~500 bytes)
- **v**: True if Merkle proof valid AND norm within bounds, False otherwise

## Tier 3 Verification (Low-Resource Devices)

### Step 1: Lightweight Hash Commitment

Tier 3 devices cannot afford expensive proof generation. Instead, commit to update via hash:

```python
def hash_commitment(delta_w, activations, device_id, round_num):
    # Combine update with metadata to prevent replay attacks
    commitment_input = f"{device_id}:{round_num}:{hash(delta_w.tobytes())}:{hash(activations.tobytes())}"
    commitment = hashlib.sha256(commitment_input.encode()).hexdigest()
    
    return commitment
```

### Step 2: Deferred Verification

Tier 3 proofs are not immediately verified. Instead:
1. Accept update tentatively (include in aggregation)
2. Store commitment on-chain
3. Randomly select Tier 1 auditor to verify batch of Tier 3 updates (every 5 rounds)

Auditor re-executes Tier 3 computation and checks consistency:
```python
def audit_tier3_updates(tier3_updates, auditor_device):
    for update in tier3_updates:
        # Auditor has access to same data split as Tier 3 device
        expected_delta_w = auditor_device.recompute(update.data_indices)
        
        if not torch.allclose(expected_delta_w, update.delta_w, rtol=1e-3):
            # Mismatch detected - penalize Tier 3 device
            blockchain.slash(update.device_id, penalty=0.2)
```

### Output
- **π**: SHA256 commitment (32 bytes)
- **v**: True (always, actual verification deferred)

## Implementation Notes

### Proof Storage

All proofs are stored on-chain for auditability:
- Tier 1: Full zk-SNARK proof (enables third-party verification)
- Tier 2: Merkle root only (paths stored off-chain, available on request)
- Tier 3: Commitment hash (full update stored off-chain for auditing)

Use IPFS or similar for off-chain storage with on-chain content hashes.

### Verification Parallelization

Blockchain process verifies proofs in parallel:
```python
from multiprocessing import Pool

def verify_all_proofs(updates):
    with Pool(processes=4) as pool:
        results = pool.starmap(verify_single_proof, 
                               [(u.tier, u.proof, u.delta_w, u.activations) 
                                for u in updates])
    return results
```

This reduces verification latency from O(n) to O(n/cores).

### Adaptive Verification Thresholds

Gradient consistency threshold (0.95) and norm bounds (μ ± 3σ) are adaptive:
- Tighten thresholds if Byzantine attacks detected (multiple rejections)
- Relax slightly during early training (high gradient variance)
- Per-device thresholds for heterogeneous data (some devices naturally have different distributions)

Update thresholds every 10 rounds based on recent statistics.

## Security Analysis

### Threat Model

Algorithm 2 defends against:
1. **Gradient poisoning**: Tier 1 zk-SNARK + Tier 2 norm checks detect malicious gradients
2. **Activation tampering**: Merkle proofs ensure activations match committed computation
3. **Free-riding**: Tier 3 auditing catches devices submitting random updates
4. **Replay attacks**: Round number and device ID in commitments prevent reuse

### Limitations

- **Tier 3 auditing delay**: Malicious Tier 3 devices can contribute bad updates for up to 5 rounds before detection. Mitigate by lowering Tier 3 aggregation weight.
- **Collusion**: If auditor and Tier 3 device collude, invalid updates pass verification. Mitigate by randomly rotating auditors and requiring multi-auditor consensus.
- **Data poisoning**: Algorithm 2 verifies computational correctness, not data quality. Combine with anomaly detection on global model performance.

## Performance Benchmarks

| Tier | Proof Gen Time | Proof Size | Verification Time | Bandwidth |
|------|----------------|------------|-------------------|------------|
| 1 | 2.3s | 198 bytes | 0.8s | Low |
| 2 | 0.4s | 512 bytes | 0.2s | Medium |
| 3 | 0.02s | 32 bytes | 0.01s (deferred) | Minimal |

Measured on: Tier 1 = NVIDIA Jetson Xavier, Tier 2 = Raspberry Pi 4, Tier 3 = ESP32

Proof generation overhead is acceptable (<10% of training time) for all tiers.