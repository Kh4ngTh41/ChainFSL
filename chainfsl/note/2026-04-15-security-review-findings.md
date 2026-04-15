# ChainFSL Security Review Findings

**Date:** 2026-04-15
**Reviewer:** Claude Code (FL Security Framework)
**Severity:** 2 Critical, 3 High, 2 Medium, 2 Low

---

## CRITICAL

### [CRITICAL-1] Sybil Prevention — Dead Code

**Location:** `src/gtm/tokenomics.py:212`

**Issue:** `check_sybil_profitable()` method exists but is **never called** anywhere in the codebase. The entire Sybil prevention mechanism is non-functional.

**Attack scenario:** Attacker creates multiple fake identities (Sybil nodes) with minimal stake. Since profitability check never runs, nothing prevents attacker from overwhelming honest nodes to capture majority of rewards.

**Impact:** Sybil attack profitable — attackers can capture disproportionate rewards.

**Recommendation:**
```python
# In TokenomicsEngine.distribute() or select_committee()
sybil_check = self.check_sybil_profitable(
    m_sybil=current_sybil_fraction * n_nodes,
    N=n_nodes,
    R_total=total_reward
)
if sybil_check["profitable"]:
    logger.warning("Sybil attack detected as profitable!")
    # Trigger defensive action or alert
```

---

### [CRITICAL-2] Reward Slashing — Type Mismatch (Slash Never Works)

**Location:** `src/protocol/chainfsl.py:616` + `src/gtm/tokenomics.py:185`

**Issue:**
```python
# chainfsl.py:616 — poison_nodes always empty!
poison_nodes=set()

# tokenomics.py:185 — expects dict, gets VerificationResult object
if node_id in poison_nodes:  # Always False
    res.get("valid", True)  # VerificationResult has .is_valid attribute, not dict key!
```

TVE trả về `VerificationResult` objects, nhưng `distribute()` dùng `dict.get()`. `poison_nodes` luôn empty → **không ai bị slashing**.

**Attack scenario:** Byzantine nodes with invalid proofs escape slashing entirely → free-riding.

**Impact:** Byzantine nodes can behave maliciously without consequence.

**Recommendation:**
```python
# chainfsl.py — fix poison_nodes population
poison_nodes = {
    node_id for node_id, res in verif_results.items()
    if isinstance(res, VerificationResult) and not res.is_valid
}
rewards, shapley_vals = self.tokenomics.distribute(
    t=self.current_round,
    shapley_values=shapley_vals,
    lazy_nodes=lazy_nodes,
    poison_nodes=poison_nodes,  # Now properly populated
)
```

---

## HIGH

### [HIGH-1] TVE Hardcoded Attack Injection

**Location:** `src/tve/committee.py:127`

```python
if node_id in lazy_node_ids:
    is_valid = False
    reason = "lazy_client_attack"
    penalty = self.PENALTY_LAZY
```

Hardcoded attack detection. Nếu attacker điều khiển được `lazy_client_fraction` config hoặc inject `node_id` vào `lazy_node_ids` → DOS honest nodes.

**Fix:**
```python
# Default attack_injection_enabled = False
if self.attack_injection_enabled and node_id in lazy_node_ids:
    ...
```

---

### [HIGH-2] Shapley Variance Estimation — Always Zero

**Location:** `src/gtm/shapley.py:174`

```python
variances.append(0.0)  # Always zero!
```

`convergence_achieved` **luôn True** → reward distribution dựa trên non-converged Shapley values không có cảnh báo.

**Fix:** Implement proper running variance (Welford's algorithm):
```python
# Use incremental mean and M2 for variance
delta = sample - mean
mean += delta / n
delta2 = sample - mean
M2 += delta * delta2
variance = M2 / (n - 1) if n > 1 else 0.0
```

---

### [HIGH-3] Staleness Decay to Near-Zero

**Location:** `src/sfl/aggregator.py:62`

```python
alpha = data_frac * (self.rho ** staleness)  # rho=0.9
# staleness=20 → weight ≈ 0.12
```

Honest nodes bị transient latency bị penalty nặng → perverse incentive skip rounds.

**Fix:** Add staleness cap:
```python
MIN_WEIGHT = 0.1  # Never decay below 10%
alpha = data_frac * max(MIN_WEIGHT, self.rho ** staleness)
```

---

## MEDIUM

### [MEDIUM-1] Ledger `commit_block` Signature Mismatch

**Location:** `src/protocol/chainfsl.py:669` vs `src/blockchain/ledger.py:197`

```python
# Caller:
self.ledger.commit_block(epoch, merkle_root, n_verified)
# Callee:
def commit_block(epoch, rewards, n_verified):  # rewards received merkle_root!
```

**Fix:** Align signatures or pass correct parameter type.

---

### [MEDIUM-2] MockVRF — Symmetric, Not Asymmetric

**Location:** `src/tve/vrf.py`

HMAC-SHA256 là symmetric crypto. Real VRF cần asymmetric ( elliptic curve). Hiện tại attacker có secret key có thể tự verify proof của mình.

**Fix:** Replace with libsodium `crypto_vrf_prove`/`crypto_vrf_verify` for production. Document MockVRF is simulation-only.

---

## LOW

### [LOW-1] Fixed Sleep in Proof Generation

**Location:** `src/tve/commitment.py:105` — `time.sleep(0.05)`

No security value — purely simulation artifact.

### [LOW-2] Stake Not Enforced On-Chain

**Location:** `src/gtm/tokenomics.py:35-36`

`stake_min` và `PENALTY_LAZY` không có on-chain enforcement. Calling code phải enforce.

---

## Summary

| ID | Severity | Area | Title |
|----|----------|------|-------|
| C1 | Critical | GTM | Sybil prevention — dead code |
| C2 | Critical | GTM | Slash never works — type mismatch |
| H1 | High | TVE | Hardcoded attack injection |
| H2 | High | GTM | Shapley variance always zero |
| H3 | High | SFL | Staleness decay near-zero |
| M1 | Medium | Blockchain | Ledger signature mismatch |
| M2 | Medium | TVE | MockVRF not truly asymmetric |
| L1 | Low | TVE | Fixed sleep in proof gen |
| L2 | Low | GTM | Stake not on-chain |

## Priority Fix Order

1. **CRITICAL-2** (slash type mismatch) — Easy fix, high impact
2. **CRITICAL-1** (Sybil dead code) — Medium fix, critical security
3. **HIGH-1** (hardcoded attack) — Easy fix, prevents DoS
4. **HIGH-2** (variance) — Medium fix, ensures convergence
5. **HIGH-3** (staleness) — Easy config fix
6. **MEDIUM-1** (ledger sig) — Easy fix
7. **MEDIUM-2** (MockVRF) — Document as simulation-only
