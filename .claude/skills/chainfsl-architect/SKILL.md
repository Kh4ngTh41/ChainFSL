---
name: chainfsl-architect
description: >
  Design, implement, and validate ChainFSL framework components: Federated Split Learning with MA-HASO (DRL orchestration), TVE (VRF-based verification with Tiered Proof), and GTM (game-theoretic tokenomics using Shapley Value). Use this for ChainFSL architecture questions, Python module consistency checks, single-machine multiprocessing design, IoT node tier-based resource management, ResNet-18 memory constraints, Algorithm 2 validation, federated learning workflows, split learning implementations, blockchain integration, incentive mechanism design, or any task involving distributed ML on resource-constrained devices.
---

# ChainFSL Architect Skill

## Overview

ChainFSL is a Federated Split Learning framework designed for resource-constrained IoT environments. It integrates three core subsystems:

- **MA-HASO**: Multi-Agent Deep Reinforcement Learning for dynamic orchestration of split points and participant selection
- **TVE (Tiered Verification Engine)**: VRF-based verification with tiered proof mechanisms matching IoT device capabilities
- **GTM (Game-Theoretic Tokenomics Module)**: Shapley Value-based incentive distribution ensuring fair contribution rewards

The architecture assumes **single-machine multiprocessing** for simulation/testing, with clear separation between coordinator, edge nodes, and blockchain verification layers.

## Core Architecture Principles

### Tier-Based Resource Management

IoT nodes are classified into tiers based on computational capacity:

- **Tier 1** (High): GPU-enabled edge devices, can handle deeper split points
- **Tier 2** (Medium): CPU-only devices with 2-4GB RAM, limited to shallow splits
- **Tier 3** (Low): Constrained devices (<1GB RAM), minimal local computation

ResNet-18 split points must respect memory constraints:
- **Tier 1**: Can process up to layer 17 (full model minus final FC)
- **Tier 2**: Layers 1-9 (early conv blocks)
- **Tier 3**: Layers 1-3 (initial feature extraction only)

Always validate split point assignments against tier capabilities before orchestration. MA-HASO's action space is constrained by these physical limits.

### Module Consistency Requirements

The three subsystems interact through well-defined interfaces:

1. **MA-HASO → TVE**: Orchestration decisions (split points, participant sets) are passed as structured metadata for verification
2. **TVE → GTM**: Verified contributions (proof validity, computation metrics) feed into Shapley Value calculation
3. **GTM → MA-HASO**: Reward signals influence DRL training through contribution-weighted loss functions

When implementing or modifying modules:
- Maintain consistent state representations (device IDs, tier mappings, round numbers)
- Use shared data schemas (Protocol Buffers or dataclasses) for inter-module communication
- Ensure thread-safe access to shared resources (participant registry, blockchain state)

## MA-HASO Implementation Guidelines

### State Space Design

The DRL agent observes:
- **Device states**: Current tier, available memory, CPU/GPU utilization, network latency
- **Training context**: Current round, global loss trend, staleness of local models
- **Historical performance**: Per-device contribution scores, verification success rates

State vectors should be normalized to [0,1] and include temporal features (moving averages over last 5 rounds) to capture training dynamics.

### Action Space Constraints

Actions encode:
- Split point assignment per selected device (constrained by tier)
- Participant selection (binary mask over available devices)
- Aggregation weights (if deviating from FedAvg)

Invalid actions (e.g., assigning layer 15 to Tier 3 device) must be masked during policy sampling. Use a validity matrix updated each round based on device availability and resource reports.

### Reward Function Alignment

Rewards balance:
- **Accuracy improvement**: Δ(validation accuracy) × 10
- **Resource efficiency**: -λ × Σ(memory usage / tier capacity)
- **Verification cost**: -μ × (proof generation time)
- **Fairness**: -ν × Gini coefficient of contribution distribution

Hyperparameters (λ, μ, ν) should be tuned based on deployment priorities. For IoT scenarios, resource efficiency typically dominates (λ ≈ 0.5, μ ≈ 0.3, ν ≈ 0.2).

## TVE (Tiered Verification Engine)

### VRF-Based Proof Selection

Verifiable Random Functions ensure unpredictable, tamper-proof selection of verification nodes:

1. Coordinator generates VRF proof for current round using secret key
2. VRF output hashes to verifier selection (uniformly random from available Tier 1/2 nodes)
3. Selected verifiers cannot predict selection in advance, preventing strategic behavior

Use libsodium's `crypto_vrf_prove` and `crypto_vrf_verify` for implementation. Store VRF public keys in blockchain state for auditability.

### Tiered Proof Mechanisms

Proof complexity scales with device tier:

- **Tier 1**: Full gradient verification + zk-SNARK of computation correctness
- **Tier 2**: Gradient norm checks + Merkle proof of intermediate activations
- **Tier 3**: Lightweight hash commitment of local update

This reduces verification overhead for constrained devices while maintaining security. Tier 1 nodes randomly audit Tier 2/3 proofs to detect misbehavior.

### Algorithm 2 Compliance

Algorithm 2 (from the ChainFSL paper) defines the verification protocol:

```
Input: Device tier τ, local update Δw, split activations A
Output: Verification proof π, validity flag v

1. if τ == 1:
2.   π ← GenerateZKProof(Δw, A, model_slice)
3.   v ← VerifyGradientConsistency(Δw, A)
4. elif τ == 2:
5.   π ← MerkleProof(A) || GradientNorm(Δw)
6.   v ← CheckNormBounds(Δw, expected_range)
7. else:  // τ == 3
8.   π ← Hash(Δw || A)
9.   v ← True  // Deferred verification by Tier 1 auditor
10. return π, v
```

When implementing verification logic:
- Line 3: Use cosine similarity between recomputed and submitted gradients (threshold > 0.95)
- Line 6: Expected range derived from historical gradient statistics (μ ± 3σ)
- Line 9: Tier 3 proofs are batched and audited asynchronously to avoid blocking training

Never skip verification steps even in simulation—this ensures production-ready code.

## GTM (Game-Theoretic Tokenomics)

### Shapley Value Calculation

Shapley Value fairly distributes rewards by measuring marginal contribution:

For device i in participant set N:
```
φᵢ = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S ∪ {i}) - v(S)]
```

Where v(S) is the "value" of coalition S, defined as:
- **Accuracy contribution**: Validation accuracy when training with subset S
- **Verification quality**: Average proof validity rate of devices in S
- **Resource provision**: Normalized compute/memory contributed by S

Exact Shapley computation is O(2^n). For n > 10 devices, use Monte Carlo approximation:
1. Sample 1000 random permutations of N
2. For each permutation, compute marginal contribution when i joins
3. Average across samples

This reduces complexity to O(n × samples) while maintaining <5% error.

### Tokenomics Integration

Rewards are distributed on-chain:
- Total reward pool R per round (fixed or dynamic based on global accuracy)
- Device i receives: `R × (φᵢ / Σφⱼ)` tokens
- Minimum threshold: Devices with φᵢ < 0.01 × max(φ) receive zero to discourage free-riding

Smart contract implementation:
```python
def distribute_rewards(round_num, shapley_values, total_pool):
    valid_contributors = {k: v for k, v in shapley_values.items() 
                          if v >= 0.01 * max(shapley_values.values())}
    total_shapley = sum(valid_contributors.values())
    
    for device_id, shapley in valid_contributors.items():
        reward = (shapley / total_shapley) * total_pool
        blockchain.transfer(device_id, reward)
        emit RewardDistributed(round_num, device_id, reward, shapley)
```

## Single-Machine Multiprocessing Design

### Process Architecture

Simulate distributed system using Python multiprocessing:

- **Coordinator process**: Runs MA-HASO agent, orchestrates rounds, aggregates updates
- **Device processes** (N workers): Simulate IoT nodes, each with tier config and local data
- **Blockchain process**: Maintains verification ledger, executes GTM smart contracts
- **Monitor process**: Logs metrics, detects deadlocks, enforces timeouts

Use `multiprocessing.Manager` for shared state (participant registry, global model) and `Queue` for message passing.

### Resource Isolation

Each device process enforces tier constraints:
```python
import resource

def init_device_process(tier, device_id):
    # Memory limit based on tier
    memory_limits = {1: 8 * 1024**3, 2: 2 * 1024**3, 3: 512 * 1024**2}
    soft, hard = memory_limits[tier], memory_limits[tier]
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    
    # CPU affinity for reproducibility
    os.sched_setaffinity(0, {device_id % os.cpu_count()})
```

This ensures simulation accurately reflects deployment constraints. Monitor RSS memory usage and fail loudly if limits are exceeded.

### Communication Patterns

**Round synchronization**:
1. Coordinator broadcasts (round_num, split_points, participant_list) via Queue
2. Devices perform local training, send (device_id, Δw, proof) back
3. Coordinator waits for all participants or timeout (30s default)
4. Blockchain process verifies proofs in parallel, returns validity flags
5. Coordinator aggregates valid updates, computes Shapley, triggers GTM

Use `multiprocessing.Event` for barrier synchronization at round boundaries.

## ResNet-18 Memory Constraints

### Layer-wise Memory Profile

For batch size 32, input 224×224×3:

| Layer Range | Activations (MB) | Gradients (MB) | Tier Compatibility |
|-------------|------------------|----------------|--------------------|
| 1-3 (conv1-pool) | 45 | 45 | All tiers |
| 4-6 (layer1) | 112 | 112 | Tier 2+ |
| 7-9 (layer2) | 156 | 156 | Tier 2+ |
| 10-13 (layer3) | 98 | 98 | Tier 2+ |
| 14-17 (layer4) | 52 | 52 | Tier 1 only |
| 18 (fc) | 0.5 | 0.5 | All tiers |

Split points must ensure:
- Client-side memory: activations(split) + gradients(split) + model_params(split) < tier_limit
- Server-side memory: activations(remaining) + gradients(remaining) + aggregation_buffer < server_capacity

### Dynamic Batch Size Adjustment

If device reports OOM during training:
1. Halve batch size (32 → 16 → 8)
2. If still failing at batch=8, move split point earlier (reduce client layers)
3. Log adjustment to MA-HASO for policy learning

This adaptive approach prevents training failures while teaching the DRL agent to avoid problematic configurations.

## Validation Checklist

Before deploying or testing ChainFSL code:

- [ ] Split point assignments respect tier memory limits (check ResNet-18 profile)
- [ ] MA-HASO action masking prevents invalid tier-layer combinations
- [ ] TVE proof generation follows Algorithm 2 exactly (no shortcuts)
- [ ] Shapley Value calculation includes all three value components (accuracy, verification, resources)
- [ ] Inter-module communication uses consistent schemas (device IDs, round numbers)
- [ ] Multiprocessing setup enforces resource limits per tier
- [ ] Blockchain state updates are atomic and logged for auditability
- [ ] Timeout mechanisms prevent deadlocks (coordinator waits max 30s per round)
- [ ] Metrics collection captures: round time, verification overhead, reward distribution, device utilization

## Common Pitfalls

**Pitfall**: MA-HASO assigns deep split points to low-tier devices, causing OOM crashes.
**Solution**: Implement action masking based on live device resource reports. Update mask each round.

**Pitfall**: Shapley Value calculation becomes bottleneck (>10s per round with 50 devices).
**Solution**: Use Monte Carlo approximation with adaptive sampling (more samples for top contributors).

**Pitfall**: TVE proofs are generated but never verified, wasting computation.
**Solution**: Ensure blockchain process actively validates proofs and penalizes invalid submissions (reduce future rewards).

**Pitfall**: Multiprocessing deadlocks when device process crashes without releasing lock.
**Solution**: Use timeouts on all Queue.get() calls and implement heartbeat monitoring.

**Pitfall**: ResNet-18 memory estimates don't account for optimizer state (Adam uses 2× gradients).
**Solution**: Multiply gradient memory by 3 (gradients + momentum + variance) when using Adam.

## Example Workflow

### Implementing a New Split Strategy

1. **Define strategy in MA-HASO**: Modify action space to include new split configuration (e.g., "adaptive split" that changes per round based on data heterogeneity)

2. **Update resource validator**: Add memory estimation for new split pattern to tier compatibility checker

3. **Extend TVE proof**: If split strategy changes verification requirements (e.g., cross-device activation sharing), update Algorithm 2 implementation

4. **Adjust GTM value function**: If strategy impacts contribution measurement (e.g., some devices do more work), reflect this in Shapley Value v(S) definition

5. **Test in simulation**: Run 100-round experiment with mixed tiers, verify no OOM, check reward fairness (Gini < 0.3)

6. **Validate consistency**: Ensure all modules agree on split configuration (log and compare MA-HASO action, device execution, TVE verification metadata)

### Debugging Inter-Module Inconsistencies

If MA-HASO selects participants but TVE rejects all proofs:

1. Check round number alignment: MA-HASO and TVE must use same round_id
2. Verify device ID mapping: Ensure consistent naming ("device_0" vs 0)
3. Inspect split point encoding: MA-HASO sends layer indices, devices must interpret correctly
4. Compare timestamps: Clock skew in multiprocessing can cause verification of stale updates

Add assertion checks at module boundaries:
```python
def verify_update(device_id, round_num, update, proof):
    assert round_num == current_round, f"Stale update from {device_id}"
    assert device_id in active_participants[round_num], f"Unexpected device {device_id}"
    # ... actual verification logic
```

## Advanced Considerations

### Handling Device Churn

IoT devices frequently disconnect. MA-HASO must:
- Track device availability via heartbeat (every 10s)
- Remove unresponsive devices from action space
- Penalize policies that over-select unreliable devices (add staleness term to reward)

GTM should:
- Distribute rewards only to devices that completed the round
- Maintain reputation scores (exponential moving average of completion rate)
- Weight Shapley contributions by reputation to discourage flaky participation

### Privacy-Preserving Verification

If raw activations leak sensitive information:
- Replace Merkle proofs with homomorphic commitments
- Use secure aggregation for gradient verification (devices jointly compute aggregate without revealing individuals)
- Add differential privacy noise to verification metrics before logging to blockchain

These modifications maintain Algorithm 2's security guarantees while protecting device privacy.

### Scaling Beyond Single Machine

When transitioning to true distributed deployment:
- Replace multiprocessing.Queue with gRPC or ZeroMQ for network communication
- Add Byzantine fault tolerance to aggregation (Krum or trimmed mean instead of FedAvg)
- Implement blockchain sharding if verification throughput becomes bottleneck
- Use distributed Shapley approximation (each device computes local contribution, coordinator aggregates)

Core architecture (MA-HASO/TVE/GTM interfaces) remains unchanged—only transport layer differs.