---
name: federated-learning-security-review
description: >
  Review code security for federated learning systems, focusing on async aggregation staleness handling,
  verification ledgers, Sybil attack prevention, stake validation, Byzantine resilience, and privacy-preserving
  mechanisms. Use this for reviewing FL protocols, distributed ML security, blockchain-based federated systems,
  decentralized training code, async aggregation implementations, or any code involving federated learning security,
  node validation, straggler mitigation, or reward distribution mechanisms.
---

# Federated Learning Security Review

## Overview

This skill enables expert security review of federated learning (FL) implementations, with emphasis on asynchronous aggregation, verification ledgers, Sybil attack prevention, and Byzantine resilience. The review methodology combines distributed systems security principles with FL-specific attack vectors.

## Core Review Areas

### 1. Asynchronous Aggregation & Staleness Control

**Why staleness matters**: In async FL, nodes contribute updates at different times. Stale gradients (computed from old model versions) can destabilize training or cause divergence. Proper staleness handling is critical for convergence.

**What to verify**:

- **Staleness decay function**: Check if updates are weighted by staleness (e.g., exponential decay `α^τ` where τ = staleness rounds)
- **Bounded staleness**: Verify maximum staleness threshold exists — updates beyond threshold should be rejected or heavily discounted
- **Straggler mitigation**: Confirm stragglers don't block aggregation indefinitely (timeout mechanisms, partial aggregation)
- **Behavioral staleness**: Advanced implementations should consider parameter sensitivity, not just time-based staleness

**Red flags**:
- No staleness weighting (treats all updates equally regardless of age)
- Unbounded staleness accumulation
- Synchronous-style blocking on slow nodes in async mode
- Missing timestamp/version tracking on updates

**Example check**:
```python
# GOOD: Staleness-aware aggregation
weight = alpha ** staleness  # Exponential decay
if staleness > max_staleness:
    return  # Reject overly stale updates

# BAD: No staleness consideration
aggregated_model = sum(updates) / len(updates)
```

### 2. Verification Ledger & Reward Integrity

**Why ledger-first matters**: Verification results must be immutably recorded before rewards are disbursed. Otherwise, attackers can manipulate validation to claim unearned rewards or hide malicious contributions.

**What to verify**:

- **Write-before-reward**: Verification results (TVE = Trusted Validation Entity outputs) must be committed to ledger (SQLite/blockchain) BEFORE reward calculation
- **Atomic transactions**: Ledger write and reward disbursal should be atomic or have rollback on failure
- **Tamper evidence**: Ledger should be append-only with cryptographic integrity (hashes, signatures)
- **Audit trail**: Each validation event links to contributor ID, model hash, timestamp, validation score

**Red flags**:
- Rewards calculated before ledger commit
- Missing transaction boundaries (partial writes possible)
- Ledger entries modifiable after creation
- No cryptographic binding between validation and reward

**Example check**:
```python
# GOOD: Ledger-first pattern
validation_result = tve.validate(model_update)
ledger.write(validation_result)  # Commit first
ledger.commit()
reward_engine.disburse(validation_result.node_id, amount)

# BAD: Reward-first (vulnerable to inconsistency)
reward_engine.disburse(node_id, amount)
ledger.write(validation_result)  # Too late
```

### 3. Sybil Attack Prevention

**Why Sybil attacks threaten FL**: Attacker creates multiple fake identities (nodes) to gain disproportionate influence over aggregation, poison the model, or drain rewards unfairly.

**What to verify**:

- **Stake requirements**: Nodes must lock economic stake (tokens, compute resources) to participate — check `stake >= stake_min` enforcement
- **Identity binding**: Each node identity tied to verifiable credential (wallet address, certificate, proof-of-personhood)
- **Rate limiting**: Prevent single entity from registering excessive nodes quickly
- **Reputation weighting**: Long-standing, honest nodes receive higher aggregation weights than newcomers
- **Social/trust graphs**: Detect clusters of suspiciously similar behavior patterns

**Critical check — stake validation bypass**:

Look for logic errors that allow nodes to participate without meeting stake requirements:

```python
# GOOD: Strict stake enforcement
if node.stake < self.stake_min:
    raise InsufficientStakeError(f"Node {node.id} has {node.stake}, requires {self.stake_min}")
    
# BAD: Bypassable check
if node.stake >= self.stake_min or node.is_legacy:  # Legacy bypass = Sybil vector
    self.accept_node(node)
```

**Red flags**:
- No stake requirement or unenforced checks
- Stake validation in separate code path from node acceptance (race condition risk)
- Default/fallback logic that bypasses stake check
- Missing stake slashing for malicious behavior

### 4. Byzantine Resilience

**Why Byzantine nodes matter**: Malicious nodes can submit poisoned gradients, backdoored models, or garbage data. Aggregation must be robust to minority Byzantine participants.

**What to verify**:

- **Robust aggregation**: Use Byzantine-resilient methods (Krum, Median, Trimmed Mean) instead of naive averaging
- **Outlier detection**: Statistical checks to identify and exclude anomalous updates
- **Validation before aggregation**: TVE validates updates BEFORE they enter global model
- **Threshold assumptions**: Code should document and enforce assumed honest majority (e.g., <1/3 Byzantine)

**Example patterns**:
```python
# GOOD: Byzantine-robust aggregation
def aggregate_updates(updates):
    if len(updates) < self.min_honest_nodes:
        raise InsufficientNodesError()
    # Krum: select update closest to majority
    return krum_aggregate(updates, f=self.max_byzantine)

# BAD: Naive averaging (vulnerable)
def aggregate_updates(updates):
    return sum(updates) / len(updates)  # One poisoned update affects all
```

### 5. Privacy & Confidentiality

**Why privacy matters in FL**: Raw data stays local, but gradient leakage can still expose sensitive information through inference attacks.

**What to verify**:

- **Differential privacy**: Check if noise is added to gradients (DP-SGD) with documented privacy budget ε
- **Secure aggregation**: Verify cryptographic protocols (MPC, homomorphic encryption) prevent server from seeing individual updates
- **Gradient clipping**: Bounds on gradient norms prevent outlier information leakage
- **Model inversion defenses**: Protections against reconstructing training data from model updates

## Review Workflow

### Phase 1: Architecture Scan

1. **Map data flow**: Trace path from node update → validation → aggregation → ledger → reward
2. **Identify trust boundaries**: Where does untrusted input enter? What validates it?
3. **Check async patterns**: How are stragglers handled? Where is staleness tracked?

### Phase 2: Security Deep-Dive

For each critical component:

**Async Aggregation Module**:
- Locate staleness calculation logic
- Verify decay function applied to weights
- Check max staleness enforcement
- Test: What happens if node submits update from 100 rounds ago?

**Validation & Ledger**:
- Find TVE validation calls
- Confirm ledger write happens before reward logic
- Check transaction atomicity
- Test: Can reward be claimed if ledger write fails?

**Node Registration (GTM - Global Trust Manager)**:
- Locate stake verification in registration flow
- Check for bypass conditions (whitelists, legacy modes, admin overrides)
- Verify stake is locked, not just checked
- Test: Can node with stake=0 register by manipulating request?

**Aggregation Security**:
- Identify aggregation algorithm (average? median? Krum?)
- Check if outlier detection runs before aggregation
- Verify Byzantine tolerance threshold matches assumptions

### Phase 3: Attack Simulation

Mental model or actual test:

1. **Sybil attack**: Can I register 1000 nodes with minimal stake?
2. **Staleness exploit**: Can I submit ancient gradient to destabilize training?
3. **Reward manipulation**: Can I claim reward without valid contribution?
4. **Byzantine poisoning**: Can I submit malicious gradient that corrupts global model?
5. **Ledger bypass**: Can I get reward without ledger entry?

## Common Vulnerability Patterns

### Pattern 1: TOCTOU in Stake Validation

```python
# VULNERABLE: Check stake, then use node (time gap)
if check_stake(node_id):  # Check
    time.sleep(0.1)  # Attacker withdraws stake here
    register_node(node_id)  # Use

# FIXED: Atomic check-and-lock
with stake_lock:
    if node.stake < stake_min:
        raise InsufficientStakeError()
    node.stake_locked = True
    register_node(node)
```

### Pattern 2: Missing Staleness Bounds

```python
# VULNERABLE: Accepts arbitrarily stale updates
def apply_update(update, staleness):
    weight = 0.99 ** staleness  # Decays but never rejects
    model.apply(update, weight)

# FIXED: Hard staleness limit
def apply_update(update, staleness):
    if staleness > MAX_STALENESS:
        logger.warn(f"Rejecting update with staleness {staleness}")
        return
    weight = 0.99 ** staleness
    model.apply(update, weight)
```

### Pattern 3: Reward-Before-Ledger

```python
# VULNERABLE: Reward first, ledger might fail
reward_amount = calculate_reward(validation_score)
disburse_reward(node_id, reward_amount)
try:
    ledger.write(validation_entry)  # Fails silently?
except LedgerError:
    pass  # Reward already sent!

# FIXED: Ledger-first with rollback
try:
    ledger.write(validation_entry)
    ledger.commit()
except LedgerError:
    raise  # Abort before reward
disburse_reward(node_id, calculate_reward(validation_score))
```

## Review Checklist

**Async Aggregation**:
- [ ] Staleness tracked for each update (timestamp or version number)
- [ ] Staleness decay function applied (exponential or polynomial)
- [ ] Maximum staleness threshold enforced
- [ ] Stragglers do not block aggregation indefinitely
- [ ] Partial aggregation allowed when some nodes timeout

**Verification & Ledger**:
- [ ] TVE validation occurs before aggregation
- [ ] Validation results written to ledger before reward calculation
- [ ] Ledger writes are atomic (transaction boundaries clear)
- [ ] Ledger is append-only or has tamper-evident properties
- [ ] Each entry includes: node_id, model_hash, timestamp, validation_score

**Sybil Prevention**:
- [ ] `stake >= stake_min` enforced at registration
- [ ] No bypass conditions (legacy modes, whitelists) that skip stake check
- [ ] Stake is locked during participation, not just verified
- [ ] Rate limiting on node registration per identity/IP
- [ ] Reputation system weights established nodes higher than new ones

**Byzantine Resilience**:
- [ ] Aggregation uses Byzantine-robust algorithm (not naive average)
- [ ] Outlier detection runs before aggregation
- [ ] Assumes and enforces honest majority threshold (e.g., f < n/3)
- [ ] Malicious contributions trigger stake slashing

**Privacy**:
- [ ] Differential privacy noise added if required (check ε budget)
- [ ] Secure aggregation protocol used (if applicable)
- [ ] Gradient clipping applied to bound sensitivity

## Reporting Findings

**Severity levels**:

- **Critical**: Sybil attack bypass, reward manipulation, unbounded staleness causing divergence
- **High**: Missing ledger-first pattern, weak Byzantine resilience, TOCTOU in stake validation
- **Medium**: Suboptimal staleness decay, missing audit fields in ledger, weak rate limiting
- **Low**: Missing logging, unclear error messages, documentation gaps

**Finding template**:

```
## [Severity] Title

**Location**: `protocol/chainfsl.py:123`

**Issue**: [What's wrong and why it's exploitable]

**Attack scenario**: [Step-by-step how attacker exploits this]

**Impact**: [What attacker gains: model poisoning, reward theft, DoS, etc.]

**Recommendation**: [Specific code fix or design change]

**Example fix**:
```python
# Current (vulnerable)
...

# Proposed (secure)
...
```
```

## Summary

Effective FL security review requires understanding both distributed systems security (Byzantine faults, Sybil attacks) and ML-specific concerns (gradient staleness, model poisoning). Focus on:

1. **Staleness control** — async aggregation must decay or reject stale updates
2. **Ledger integrity** — verification recorded before rewards disbursed
3. **Sybil prevention** — stake requirements enforced without bypass paths
4. **Byzantine resilience** — robust aggregation against minority malicious nodes

The goal is not perfect security (impossible in open systems) but raising attack cost above potential gain while maintaining training efficiency.