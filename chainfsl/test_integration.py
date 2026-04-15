"""ChainFSL Integration Test"""
import sys
sys.path.insert(0, '.')

print('=' * 60)
print('CHAINFSL FULL INTEGRATION VERIFICATION')
print('=' * 60)

# Sprint 1: Emulator
print()
print('[Sprint 1] Emulator')
from src.emulator import create_nodes, DEFAULT_FACTORY
nodes = create_nodes(5, distribution=DEFAULT_FACTORY)
print(f'  Created {len(nodes)} nodes via DEFAULT_FACTORY')
profile = nodes[0]
print(f'  Node 0: {profile.node_id}, tier={profile.tier}')

# Sprint 2: SFL
print()
print('[Sprint 2] SFL Models')
from src.sfl.models import SplittableResNet18
model = SplittableResNet18(n_classes=10, cut_layer=2)
client, server = model.split_models(2)
n_params = sum(p.numel() for p in client.parameters())
print(f'  ResNet18 split at cut_layer=2: client has {n_params:,} params')

import torch
print()
print('[Sprint 2] SFL Aggregator')
from src.sfl.aggregator import AsyncAggregator, FedAvgAggregator
global_state = {k: v.clone() for k, v in model.state_dict().items()}
agg = AsyncAggregator(global_state=global_state, rho=0.9)
print(f'  AsyncAggregator: rho={agg.rho}')

# Sprint 3: HASO
print()
print('[Sprint 3] HASO')
from src.haso.env import SFLNodeEnv
from src.haso.agent import HaSOAgentPool
env = SFLNodeEnv(node_profile=nodes[0], n_compute_nodes=10)
obs, _ = env.reset()
action = env.action_space.sample()
obs2, r, term, trunc, info = env.step(action)
cut = info['cut_layer']
print(f'  SFLNodeEnv: cut_layer={cut}, reward={r:.3f}')
envs = [SFLNodeEnv(node_profile=n, n_compute_nodes=10) for n in nodes]
pool = HaSOAgentPool(envs=envs)
print(f'  HaSOAgentPool: {len(pool.agents)} agents')

# Sprint 3: TVE
print()
print('[Sprint 3] TVE')
from src.tve.vrf import MockVRF, select_committee
vrf = MockVRF(secret_key=b'test-secret-32bytes!!')
seed = b'network_seed_for_epoch_1'
out, proof = vrf.evaluate(seed, epoch=1)
print(f'  MockVRF output [0,1): {out:.6f}')
committee = select_committee([(i, b'key') for i in range(10)], seed, epoch=1, committee_size=3)
print(f'  Committee: {committee}')

from src.tve.commitment import CommitmentVerifier
import torch
verifier = CommitmentVerifier()
x = torch.randn(32, 3, 224, 224)
a = torch.randn(32, 64, 56, 56)
model_state = {}
proof = verifier.gen_proof_tier1(x, a, model_state)
valid = verifier.verify_proof_tier1(proof, proof.input_hash)
print(f'  CommitmentVerifier: proof generated, valid={valid}')

# Sprint 3: GTM
print()
print('[Sprint 3] GTM')
from src.gtm.shapley import TMCSShapley
def utility_fn(coalition):
    return len(coalition) * 0.1
shapley = TMCSShapley(M=20, seed=42)
result = shapley.compute(list(range(10)), utility_fn)
total = sum(result.values.values())
print(f'  TMCSShapley(10 nodes, M=20): total={total:.3f}, calls={result.utility_calls}')

from src.gtm.tokenomics import TokenomicsEngine, TokenomicsConfig
config = TokenomicsConfig()
engine = TokenomicsEngine(config)
lazy = engine.detect_lazy_nodes({i: 0.001 for i in range(5)})
print(f'  TokenomicsEngine: {len(lazy)} lazy nodes detected')

# Sprint 3: Blockchain
print()
print('[Sprint 3] Blockchain')
from src.blockchain import BlockchainLedger, BlockRecord
ledger = BlockchainLedger(db_path='/tmp/test_chainfsl.db')
ledger.record_reward(epoch=1, node_id=0, reward=10.0, shapley=0.2)
ledger.record_reward(epoch=1, node_id=1, reward=8.0, shapley=0.15)
r0 = ledger.get_cumulative_reward(node_id=0)
r1 = ledger.get_cumulative_reward(node_id=1)
print(f'  BlockchainLedger: node_0={r0:.1f}, node_1={r1:.1f} rewards')

print()
print('=' * 60)
print('ALL MODULES VERIFIED SUCCESSFULLY')
print('=' * 60)