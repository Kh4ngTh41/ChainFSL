# Federated Learning Attack Vectors

This reference details specific attack scenarios against FL systems and their technical indicators.

## Sybil Attack Deep-Dive

### Attack Mechanics

**Goal**: Gain disproportionate influence over global model by controlling multiple identities.

**Method**:
1. Attacker creates N fake nodes (N >> honest nodes)
2. Each fake node submits similar malicious updates
3. Aggregation weights malicious updates heavily due to count
4. Global model converges toward attacker's objective

### Economic Sybil Attacks

In blockchain-based FL with token rewards:

1. **Reward farming**: Create many low-stake nodes to claim multiple small rewards (total > single large reward)
2. **Stake fragmentation**: Split stake across multiple identities to bypass per-node caps while maintaining influence
3. **Collusion networks**: Coordinate multiple identities to pass social trust graph checks

### Detection Indicators

**Behavioral clustering**:
- Multiple nodes submitting nearly identical updates (cosine similarity > 0.99)
- Synchronized registration times (batch creation)
- Similar network characteristics (same ISP, geographic region)
- Correlated online/offline patterns

**Graph analysis**:
- Low connectivity between suspicious nodes and established network
- Star topology with central hidden node
- Sudden appearance of many new nodes from previously inactive addresses

### Code-Level Defenses

**Stake verification hardening**:
```python
class GlobalTrustManager:
    def register_node(self, node: Node) -> bool:
        # Multi-layer validation
        if node.stake < self.stake_min:
            raise InsufficientStakeError()
        
        # Check stake is actually locked on-chain
        if not self.blockchain.verify_stake_lock(node.address, node.stake):
            raise StakeNotLockedError()
        
        # Rate limit: max 1 registration per address per epoch
        if self.recent_registrations.count(node.address) > 0:
            raise RateLimitExceededError()
        
        # Social trust: require vouching from established nodes
        if not self.has_sufficient_vouches(node, min_vouches=3):
            raise InsufficientTrustError()
        
        self.nodes.add(node)
        return True
```

**Reputation-weighted aggregation**:
```python
def aggregate_with_reputation(updates: List[Update]) -> Model:
    weighted_sum = zero_model()
    total_weight = 0
    
    for update in updates:
        # Reputation decays new nodes, rewards long-standing honest nodes
        reputation = calculate_reputation(
            age=update.node.age,
            past_validations=update.node.validation_history,
            stake=update.node.stake
        )
        
        weight = reputation * staleness_decay(update.staleness)
        weighted_sum += update.model * weight
        total_weight += weight
    
    return weighted_sum / total_weight
```

## Model Poisoning Attacks

### Data Poisoning

Attacker manipulates local training data to inject backdoor or bias.

**Example**: In image classification, attacker trains on images with trigger pattern (e.g., small square in corner) labeled as target class. Global model learns: trigger → target class.

**Defense**: Validation on clean held-out dataset before aggregation.

### Gradient Poisoning

Attacker directly crafts malicious gradient (not from real training).

**Example**: Gradient designed to maximize loss on specific inputs or move model toward attacker's pre-trained backdoored model.

**Defense**: Gradient clipping, outlier detection, Byzantine-robust aggregation.

### Detection in Code

```python
def detect_poisoning(update: Update, global_model: Model) -> bool:
    # Check 1: Gradient norm too large (likely adversarial)
    if torch.norm(update.gradient) > MAX_GRADIENT_NORM:
        return True  # Poisoned
    
    # Check 2: Update increases loss on validation set
    test_model = global_model.copy()
    test_model.apply_update(update.gradient)
    if validation_loss(test_model) > validation_loss(global_model) * 1.1:
        return True  # Poisoned
    
    # Check 3: Cosine similarity with median update
    median_gradient = compute_median_gradient(all_updates)
    similarity = cosine_similarity(update.gradient, median_gradient)
    if similarity < MIN_SIMILARITY_THRESHOLD:
        return True  # Outlier
    
    return False  # Likely honest
```

## Inference Attacks

### Gradient Leakage

Attacker (server or malicious node) reconstructs training data from gradients.

**Method**: Solve optimization problem to find input x such that ∇L(x) matches observed gradient.

**Defense**: Differential privacy (add noise to gradients), secure aggregation (server never sees individual gradients).

### Model Inversion

Attacker queries trained model to extract information about training data.

**Example**: In face recognition model, attacker generates image that maximizes activation for specific person's class → reconstructs face.

**Defense**: Output perturbation, query limiting, differential privacy.

## Staleness Exploitation

### Divergence Attack

Attacker intentionally delays updates to maximize staleness and destabilize training.

**Method**:
1. Attacker computes update on very old model version
2. Submits update after many rounds have passed
3. Stale gradient points in direction irrelevant to current model
4. Aggregation mixes current and stale directions → oscillation or divergence

**Defense**: Bounded staleness (reject updates older than threshold), exponential decay.

### Strategic Delay

Attacker observes aggregation pattern and times malicious update for maximum impact.

**Example**: Wait until few honest nodes are online, then submit many malicious updates to dominate that round's aggregation.

**Defense**: Minimum participation threshold, reputation weighting.

## Ledger Manipulation

### Double-Claiming

Attacker attempts to claim reward multiple times for same contribution.

**Method**:
1. Submit valid update, receive validation
2. Replay validation message to reward system
3. Claim reward twice

**Defense**: Unique validation IDs, idempotent reward disbursal, ledger deduplication.

### Validation Forgery

Attacker forges TVE signature to create fake validation result.

**Defense**: Cryptographic signature verification, TVE key management, audit logging.

```python
def verify_and_record_validation(validation: ValidationResult) -> bool:
    # Verify TVE signature
    if not tve_public_key.verify(validation.signature, validation.data):
        raise InvalidSignatureError()
    
    # Check validation is not replay
    if ledger.exists(validation.id):
        raise DuplicateValidationError()
    
    # Atomic ledger write
    with ledger.transaction():
        ledger.write(validation)
        ledger.commit()
    
    return True
```

## Summary

Attack surface in FL is multi-dimensional:
- **Identity layer**: Sybil attacks via fake nodes
- **Data layer**: Poisoning via malicious training data
- **Gradient layer**: Poisoning via crafted gradients, inference via gradient analysis
- **Aggregation layer**: Staleness exploitation, Byzantine manipulation
- **Reward layer**: Double-claiming, validation forgery

Defense requires layered approach: economic barriers (stake), cryptographic verification (signatures), statistical robustness (Byzantine aggregation), and privacy preservation (DP, secure aggregation).