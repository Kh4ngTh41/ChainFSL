"""
ChainFSL Protocol — End-to-end Algorithm 2 orchestrator.

Implements the full ChainFSL protocol from the paper:
  1. HASO: each node observes state → picks (cut_layer, batch_size, H, target_node)
  2. TRAINING: node forward pass to cut layer, sends smashed data to server
  3. SERVER: backward pass, returns gradient to client
  4. UPDATE: node updates client-side weights
  5. TVE: generate + verify proofs (tier-dependent)
  6. AGGREGATION: staleness-decayed weighted averaging
  7. GTM: compute Shapley values, distribute rewards
  8. BLOCKCHAIN: commit rewards and verifications to ledger
  9. HASO: compute reward r_t, update PPO policy
"""

import time
import copy
import hashlib
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field

import torch
import numpy as np

from ..emulator.node_profile import HardwareProfile
from ..emulator.tier_factory import create_nodes
from ..emulator.network_emulator import NetworkEmulator, GossipProtocol

from ..sfl.models import SplittableResNet18
from ..sfl.trainer import SFLTrainer
from ..sfl.aggregator import AsyncAggregator, FedAvgAggregator
from ..sfl.data_loader import get_dataloaders

from ..haso.env import SFLNodeEnv
from ..haso.agent import HaSOAgentPool
from ..haso.gossip import HASOGossip

from ..tve.commitment import CommitmentVerifier, Proof
from ..tve.committee import VerificationCommittee, TVEConfig, TieredVerificationEngine
from ..tve.vrf import MockVRF

from ..gtm.shapley import TMCSShapley, ShapleyConfig, ShapleyCalculator
from ..gtm.tokenomics import TokenomicsEngine, TokenomicsConfig, NashValidator

from ..blockchain.ledger import BlockchainLedger

from ..utils.metrics import compute_metrics, jains_fairness, gini_coefficient
from ..utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists, get_latest_checkpoint
from ..utils.progress import ProgressTracker, NodeProgressInfo


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RoundMetrics:
    """Metrics collected per round."""
    round: int
    round_latency: float
    train_loss: float
    test_acc: float
    n_valid_updates: int
    n_participants: int
    attack_detection_rate: float
    fairness_index: float
    total_reward: float
    mean_shapley: float
    shapley_variance: float
    mean_verification_ms: float
    ledger_size_kb: float
    # Latency breakdown (seconds)
    ppo_update_time: float = 0.0
    shapley_time: float = 0.0
    train_time: float = 0.0
    verification_time: float = 0.0
    comm_time: float = 0.0
    avg_node_train_time: float = 0.0  # avg per-node training time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "round_latency": self.round_latency,
            "train_loss": self.train_loss,
            "test_acc": self.test_acc,
            "n_valid_updates": self.n_valid_updates,
            "n_participants": self.n_participants,
            "attack_detection_rate": self.attack_detection_rate,
            "fairness_index": self.fairness_index,
            "total_reward": self.total_reward,
            "mean_shapley": self.mean_shapley,
            "shapley_variance": self.shapley_variance,
            "mean_verification_ms": self.mean_verification_ms,
            "ledger_size_kb": self.ledger_size_kb,
            "ppo_update_time": self.ppo_update_time,
            "shapley_time": self.shapley_time,
            "train_time": self.train_time,
            "verification_time": self.verification_time,
            "comm_time": self.comm_time,
            "avg_node_train_time": self.avg_node_train_time,
        }


@dataclass
class NodeProgress:
    """Per-node progress tracking across rounds."""
    node_id: int
    current_round: int = 0          # Last completed round
    total_epochs_trained: int = 0     # Cumulative local epochs trained
    local_epochs_this_round: int = 0 # Epochs trained in current round
    cut_layer: int = 2               # Current cut layer assignment
    batch_size: int = 32             # Current batch size
    last_loss: float = 0.0           # Loss from most recent training
    cumulative_loss: float = 0.0     # Sum of all losses
    times_excluded: int = 0          # How many times node was excluded (OOM)
    last_reward: float = 0.0         # Most recent reward received
    cumulative_reward: float = 0.0   # Sum of all rewards
    completed: bool = False           # True when node finished all rounds

    @property
    def mean_loss(self) -> float:
        denom = self.total_epochs_trained if self.total_epochs_trained > 0 else 1
        return self.cumulative_loss / denom

    @property
    def mean_reward(self) -> float:
        rounds = max(self.current_round, 1)
        return self.cumulative_reward / rounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "current_round": self.current_round,
            "total_epochs_trained": self.total_epochs_trained,
            "local_epochs_this_round": self.local_epochs_this_round,
            "cut_layer": self.cut_layer,
            "batch_size": self.batch_size,
            "last_loss": self.last_loss,
            "mean_loss": self.mean_loss,
            "times_excluded": self.times_excluded,
            "last_reward": self.last_reward,
            "cumulative_reward": self.cumulative_reward,
            "completed": self.completed,
        }


# ---------------------------------------------------------------------------
# ChainFSL Protocol
# ---------------------------------------------------------------------------

class ChainFSLProtocol:
    """
    End-to-end ChainFSL Protocol orchestrator.

    Integrates all modules:
    - HASO (DRL-based split optimization)
    - SFL (Split Federated Learning pipeline)
    - TVE (Tiered Verification Engine)
    - GTM (Game-Theoretic Tokenomics)
    - Blockchain (SQLite ledger)

    Runs on a single machine using ThreadPoolExecutor for concurrency.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        nodes: Optional[List[HardwareProfile]] = None,
        device: Optional[torch.device] = None,
        db_path: str = "./chainfsl_ledger.db",
    ):
        """
        Args:
            config: Full config dict (from YAML). Keys:
                - n_nodes, tier_distribution, model, dataset, n_classes
                - global_rounds, batch_size_default, dirichlet_alpha
                - haso_enabled, ppo_learning_rate, ppo_n_steps, reward_alpha/beta/gamma
                - tve_enabled, committee_size, vrf_omega
                - gtm_enabled, shapley_M, reward_total_init, reward_min, halving_rounds
                - sybil_fraction, lazy_client_fraction, poison_fraction
            nodes: Pre-created HardwareProfile list. If None, creates from config.
            device: Computation device. Defaults to CUDA if available.
            db_path: Path for blockchain ledger SQLite DB.
        """
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Nodes ---
        tier_dist_list = config.get("tier_distribution", [0.1, 0.3, 0.4, 0.2])
        from ..emulator.tier_factory import TierDistribution
        tier_dist = TierDistribution(
            tiers=[1, 2, 3, 4],
            probabilities=tier_dist_list,
        )
        self.nodes = nodes or create_nodes(config["n_nodes"], distribution=tier_dist)
        self.n_nodes = len(self.nodes)

        # --- Global model ---
        self.model = SplittableResNet18(
            n_classes=config.get("n_classes", 10),
            cut_layer=2,
        ).to(self.device)

        # --- Data loaders ---
        self.train_loaders, _, self.test_dataset = get_dataloaders(
            dataset_name=config.get("dataset", "cifar10"),
            n_clients=self.n_nodes,
            alpha=config.get("dirichlet_alpha", 0.5),
            batch_size=config.get("batch_size_default", 32),
            data_dir="./data",
            download=True,
            seed=config.get("seed", 42),
        )

        # --- Network & Gossip ---
        self.net = NetworkEmulator(variance=0.3)
        self.gossip = GossipProtocol(fanout=3)

        # --- HASO ---
        self.haso_enabled = config.get("haso_enabled", True)
        haso_envs = [
            SFLNodeEnv(
                node_profile=n,
                n_compute_nodes=max(1, self.n_nodes - 1),
                reward_weights=(
                    config.get("reward_alpha", 1.0),
                    config.get("reward_beta", 0.5),
                    config.get("reward_gamma", 0.1),
                ),
                max_steps=config.get("global_rounds", 100),
                seed=n.node_id,
            )
            for n in self.nodes
        ]
        self.agent_pool: Optional[HaSOAgentPool] = None
        self._orchestrator = None  # Centralized HASO orchestrator (alternative to per-node agents)
        if self.haso_enabled:
            self.agent_pool = HaSOAgentPool(
                envs=haso_envs,
                learning_rate=config.get("ppo_learning_rate", 3e-4),
                n_steps=config.get("ppo_n_steps", 512),
                batch_size=config.get("ppo_batch_size", 64),
                n_epochs=10,
                verbose=0,
            )

        # --- TVE ---
        self.tve_enabled = config.get("tve_enabled", True)
        tve_config = TVEConfig(
            committee_size=config.get("committee_size", 5),
            omega=config.get("vrf_omega", 0.3),
        )
        self.tve = TieredVerificationEngine(nodes=self.nodes, config=tve_config)

        # --- GTM ---
        self.gtm_enabled = config.get("gtm_enabled", True)
        tokenomics_config = TokenomicsConfig(
            R0=config.get("reward_total_init", 1000.0),
            R_min=config.get("reward_min", 10.0),
            T_halving=config.get("halving_rounds", 50),
        )
        self.tokenomics = TokenomicsEngine(tokenomics_config)
        self.shapley_config = ShapleyConfig(
            M=config.get("shapley_M", 50),
            seed=config.get("seed", 42),
        )

        # --- Blockchain ---
        self.ledger = BlockchainLedger(db_path=db_path)

        # --- Aggregator ---
        self.global_state: Dict[str, torch.Tensor] = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        self.aggregator = AsyncAggregator(
            global_state=self.global_state,
            rho=config.get("staleness_decay", 0.9),
        )

        # --- Trainers (created per round per node) ---
        self.trainers: Dict[int, SFLTrainer] = {}

        # --- State ---
        self.current_round = 0
        self.node_staleness: Dict[int, int] = {n.node_id: 0 for n in self.nodes}
        self.node_losses: Dict[int, float] = {n.node_id: 0.0 for n in self.nodes}
        self.verification_rates: Dict[int, float] = {n.node_id: 1.0 for n in self.nodes}

        # --- Attack injection (E4) ---
        n_lazy = int(config.get("lazy_client_fraction", 0.0) * self.n_nodes)
        self.lazy_node_ids: Set[int] = set(n.node_id for n in list(self.nodes)[:n_lazy])
        self.sybil_node_ids: Set[int] = set()

        # --- Tracking ---
        self.metrics_history: List[RoundMetrics] = []
        self.node_progress: Dict[int, NodeProgress] = {
            n.node_id: NodeProgress(node_id=n.node_id) for n in self.nodes
        }

        # --- Sync lock ---
        self._lock = threading.Lock()

        # --- Proof cache (for verification) ---
        self._proof_cache: Dict[int, Proof] = {}

        # --- Gradient cache (for Tier 1 cosine similarity verification) ---
        self._grad_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, total_rounds: int, eval_every: int = 10) -> List[RoundMetrics]:
        """
        Run the full ChainFSL protocol for total_rounds.

        Args:
            total_rounds: Number of global rounds.
            eval_every: Evaluate on test set every N rounds.

        Returns:
            List of RoundMetrics, one per round.
        """
        for t in range(1, total_rounds + 1):
            self.current_round = t
            round_start = time.perf_counter()

            # Phase 1: HASO decisions (fast, no timing needed)
            configs = self._phase_haso()

            # Phase 2-3: SFL training + TVE proof generation
            train_start = time.perf_counter()
            updates, proofs, train_losses = self._phase_training(configs)
            train_elapsed = time.perf_counter() - train_start

            # Per-node train time tracking (from train_losses keys)
            node_train_times = [train_elapsed] * len(train_losses)

            # Phase 4: TVE verification
            verif_start = time.perf_counter()
            verif_results = self._phase_verification(updates, proofs)
            verif_elapsed = time.perf_counter() - verif_start

            # Phase 5: Aggregation
            valid_ids = self.tve.committee.get_valid_node_ids(verif_results)
            valid_updates = [u for u in updates if u["node_id"] in valid_ids]
            self._phase_aggregation(valid_updates)

            # Phase 6: GTM rewards (includes Shapley)
            shapley_start = time.perf_counter()
            shapley_vals, rewards = self._phase_gtm(updates, verif_results)
            shapley_elapsed = time.perf_counter() - shapley_start

            # Phase 7: Blockchain commit
            self._phase_blockchain(verif_results, rewards, shapley_vals)

            # Phase 8: HASO policy update (PPO)
            ppo_start = time.perf_counter()
            self._phase_haso_update(shapley_vals, rewards)
            ppo_elapsed = time.perf_counter() - ppo_start

            # Total elapsed
            elapsed = time.perf_counter() - round_start

            # Compute comm overhead estimate (smashed data * n_nodes in GB)
            comm_estimate = sum(u.get("smashed_bytes", 0) for u in updates) / 1e9
            avg_node_train = train_elapsed / max(len(train_losses), 1)

            metrics = self._collect_metrics(
                t, elapsed, train_losses, verif_results, rewards, shapley_vals,
                ppo_update_time=ppo_elapsed,
                shapley_time=shapley_elapsed,
                train_time=train_elapsed,
                verification_time=verif_elapsed,
                comm_time=comm_estimate,
                avg_node_train_time=avg_node_train,
            )
            self.metrics_history.append(metrics)

            if t % eval_every == 0 or t == total_rounds:
                eval_result = self._evaluate()
                test_acc = eval_result.get("accuracy", 0.0) if isinstance(eval_result, dict) else eval_result
                metrics.test_acc = test_acc
                self._log_round(t, metrics)

        return self.metrics_history

    def inject_lazy_clients(self, node_ids: Set[int]) -> None:
        """Inject lazy client behavior (E4 security experiment)."""
        self.lazy_node_ids = node_ids

    def inject_sybil(self, node_ids: Set[int]) -> None:
        """Inject Sybil nodes (E4 security experiment)."""
        self.sybil_node_ids = node_ids

    # -------------------------------------------------------------------------
    # Phase 1: HASO decisions
    # -------------------------------------------------------------------------

    def _find_deepest_valid_cut_layer(
        self,
        node: HardwareProfile,
        memory_map: dict[int, float],
    ) -> Optional[int]:
        """
        Find the deepest (largest cut_layer) that fits in node's RAM.

        Uses MEMORY_WITH_ADAM_MB — includes optimizer state (3x gradients).
        Returns None if no cut layer fits (node must be excluded from this round).

        Args:
            node: HardwareProfile of the node.
            memory_map: Memory requirement map (cut_layer -> MB).

        Returns:
            Deepest valid cut_layer, or None if no cut fits.
        """
        # Sort cut layers deepest-first (4, 3, 2, 1)
        for cl in sorted(memory_map.keys(), reverse=True):
            required = memory_map.get(cl, float("inf"))
            if required <= node.ram_mb:
                return cl
        return None  # No valid cut layer — exclude node

    def _phase_haso(self) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Phase 1: Each node observes state and decides (cut_layer, batch_size, H, target_node).

        Returns:
            Dict[node_id] -> config dict, or None if node is excluded (OOM).
        """
        configs: Dict[int, Optional[Dict[str, Any]]] = {}

        if not self.haso_enabled:
            # Ablation: fixed uniform cut layer 2
            for node in self.nodes:
                configs[node.node_id] = {
                    "cut_layer": 2,
                    "batch_size": self.cfg.get("batch_size_default", 32),
                    "H": 1,
                    "target_compute_node": 0,
                }
            return configs

        # Use centralized orchestrator if available (reduces overhead vs per-node agents)
        if self._orchestrator is not None:
            return self._phase_haso_centralized()

        # Fallback: per-node HaSOAgentPool (original implementation)
        if self.agent_pool is None:
            # No HASO at all - use defaults
            for node in self.nodes:
                configs[node.node_id] = {
                    "cut_layer": 2,
                    "batch_size": self.cfg.get("batch_size_default", 32),
                    "H": 1,
                    "target_compute_node": 0,
                }
            return configs

        # Collect observations from all envs
        obs_list = []
        for n in self.nodes:
            agent = self.agent_pool.agents[n.node_id]
            obs = agent.env._get_obs()
            # Update neighbor availability from gossip
            neighbor_avail = self.gossip.mean_neighbor_availability(n.node_id)
            agent.env._neighbor_avail = neighbor_avail
            obs_list.append(obs)

        # Batch decision
        decisions = self.agent_pool.decide_all(obs_list, deterministic=False)

        for n, decision in zip(self.nodes, decisions):
            node_id = n.node_id
            cut_layer = decision["cut_layer"]

            # Enforce tier memory constraint (includes optimizer state for training)
            memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB
            valid_cut = self._find_deepest_valid_cut_layer(n, memory_map)

            if valid_cut is None:
                # Node cannot fit any cut layer — exclude from this round
                configs[node_id] = None
                continue

            # Clamp chosen cut_layer to deepest valid; HASO will learn from this
            cut_layer = valid_cut

            configs[node_id] = {
                "cut_layer": cut_layer,
                "batch_size": decision["batch_size"],
                "H": decision["H"],
                "target_compute_node": decision["target_compute_node"],
            }

        return configs

    def _phase_haso_centralized(self) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Phase 1 (Centralized): Single orchestrator decides for all nodes.

        According to ChainFSL_Implementation_Plan.md Section 1.1:
        One Orchestrator runs PPO for all nodes, reducing overhead vs per-node agents.
        """
        configs: Dict[int, Optional[Dict[str, Any]]] = {}

        # Get global observation for orchestrator
        obs = self._get_global_obs()

        # Get decisions from centralized orchestrator
        decisions = self._orchestrator.decide(obs, deterministic=False)

        # Enforce memory constraints and build configs
        memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB

        for node, decision in zip(self.nodes, decisions):
            node_id = node.node_id

            # Find deepest valid cut layer
            valid_cut = self._find_deepest_valid_cut_layer(node, memory_map)

            if valid_cut is None:
                configs[node_id] = None
                continue

            cut_layer = decision["cut_layer"]
            if cut_layer > valid_cut:
                cut_layer = valid_cut

            configs[node_id] = {
                "cut_layer": cut_layer,
                "batch_size": decision["batch_size"],
                "H": decision["H"],
                "target_compute_node": decision["target_compute_node"],
            }

        return configs

    def _get_global_obs(self) -> np.ndarray:
        """
        Get global observation for centralized orchestrator.

        Returns normalized state vector [mean_cpu, mean_mem, mean_bw, mean_loss, shapley, fairness, n_nodes_norm].
        """
        import numpy as np

        mean_flops = np.mean([n.flops_ratio for n in self.nodes])
        mean_ram = np.mean([n.ram_mb for n in self.nodes])
        mean_bw = np.mean([n.bandwidth_mbps for n in self.nodes])
        mean_loss = np.mean(list(self.node_losses.values())) if self.node_losses else 5.0

        # Mean Shapley (approximation from last round)
        mean_shapley = np.mean(list(self.verification_rates.values())) if self.verification_rates else 0.5

        # Fairness estimate
        reward_values = [self.node_progress[n.node_id].last_reward for n in self.nodes]
        fairness = self._jains_fairness(reward_values) if reward_values else 0.5

        return np.array([
            mean_flops,
            1.0 - mean_ram / 8192.0,
            mean_bw / 100.0,
            mean_loss / 10.0,
            mean_shapley,
            fairness,
            self.n_nodes / 50.0,
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Phase 2-3: SFL Training + TVE proof generation
    # -------------------------------------------------------------------------

    def _phase_training(
        self, configs: Dict[int, Dict[str, Any]]
    ) -> tuple:
        """
        Phase 2: Run SFL training for all nodes (concurrent).
        Phase 3: Generate TVE proofs (tier-dependent).

        Returns:
            (updates, proofs, train_losses) tuples.
        """
        updates = []
        proofs = []
        train_losses: Dict[int, float] = {}

        def train_node(node: HardwareProfile) -> Optional[tuple]:
            try:
                cfg = configs.get(node.node_id)
                if cfg is None:
                    # Node was excluded from this round (OOM)
                    progress = self.node_progress.get(node.node_id)
                    if progress:
                        progress.times_excluded += 1
                    return None

                cut_layer = cfg.get("cut_layer", 2)
                batch_size = cfg.get("batch_size", 32)
                H = cfg.get("H", 1)

                # Final safety check against MEMORY_WITH_ADAM_MB
                if not node.can_fit_cut_layer(cut_layer, SplittableResNet18.MEMORY_WITH_ADAM_MB):
                    return None  # Skip this node — cannot train safely

                # Create trainer for this node
                trainer = SFLTrainer(
                    node_id=node.node_id,
                    model=self.model,
                    cut_layer=cut_layer,
                    device=self.device,
                )
                self.trainers[node.node_id] = trainer

                # Sync from global state
                trainer.sync_from_global(self.global_state, cut_layer)

                # Run H local epochs
                loader = self.train_loaders[node.node_id]
                avg_loss, _ = trainer.local_epochs(loader, H=H, verbose=False)

                # Client/server state for aggregation
                client_state = trainer.get_client_state()
                server_state = trainer.get_server_state()

                # Compute smashed data size for comm time estimation
                smashed_bytes = SplittableResNet18.smashed_data_size(cut_layer, batch_size)

                # Get gradient norm and tensor for TVE (P1-3 fix)
                grad_norm = trainer.get_last_grad_norm()
                grad_tensor = trainer.get_last_grad()
                smash_data = trainer.get_last_smash_data()

                # Generate TVE proof based on tier (MUST generate BEFORE building update dict)
                proof = self._generate_proof(node, trainer, cut_layer)
                self._proof_cache[node.node_id] = proof

                # Build update dict - use proof's input_hash (matches verification)
                update = {
                    "node_id": node.node_id,
                    "cut_layer": cut_layer,
                    "batch_size": batch_size,
                    "client_state": client_state,
                    "server_state": server_state,
                    "data_size": len(loader.dataset),
                    "staleness": self.node_staleness.get(node.node_id, 0),
                    "input_hash": proof.input_hash,  # FIX: use proof's hash, not model weights hash
                    "smashed_bytes": smashed_bytes,
                    "loss": avg_loss,
                    "gradient_norm": grad_norm,
                }

                # Lazy client attack injection (E4)
                if node.node_id in self.lazy_node_ids:
                    # Submit random proof — will fail verification
                    proof = CommitmentVerifier.gen_proof_tier3(
                        torch.randn(1, 3, 224, 224),
                        torch.randn(1, 64, 56, 56),
                    )

                train_losses[node.node_id] = avg_loss
                self.node_losses[node.node_id] = avg_loss

                # Broadcast LRH to gossip network (P0 fix: gossip was never broadcast)
                comp_load = (cut_layer / 4.0) * (batch_size / 32.0)
                self.gossip.broadcast(node.node_id, {
                    "flops_ratio": node.flops_ratio,
                    "ram_mb": node.ram_mb,
                    "bandwidth_mbps": node.bandwidth_mbps,
                    "reputation": node.reputation,
                    "load": comp_load,
                    "round": self.current_round,
                })

                # Update node progress tracking
                progress = self.node_progress.get(node.node_id)
                if progress:
                    progress.current_round = self.current_round
                    progress.total_epochs_trained += H
                    progress.local_epochs_this_round = H
                    progress.cut_layer = cut_layer
                    progress.batch_size = batch_size
                    progress.last_loss = avg_loss
                    progress.cumulative_loss += avg_loss

                return update, proof

            except Exception as e:
                # Crash-proofing: log error, mark node as excluded
                print(f"WARNING: Node {node.node_id} training failed: {e}")
                progress = self.node_progress.get(node.node_id)
                if progress:
                    progress.times_excluded += 1
                return None

        with ThreadPoolExecutor(max_workers=min(self.n_nodes, 16)) as executor:
            futures = {
                executor.submit(train_node, node): node
                for node in self.nodes
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    update, proof = result
                    updates.append(update)
                    proofs.append(proof)

        # Update staleness
        updated_ids = {u["node_id"] for u in updates}
        for n in self.nodes:
            if n.node_id in updated_ids:
                self.node_staleness[n.node_id] = 0
            else:
                self.node_staleness[n.node_id] = (
                    self.node_staleness.get(n.node_id, 0) + 1
                )

        return updates, proofs, train_losses

    def _generate_proof(
        self,
        node: HardwareProfile,
        trainer: SFLTrainer,
        cut_layer: int,
    ) -> Proof:
        """Generate tier-appropriate TVE proof."""
        # Get a sample batch for proof generation
        try:
            loader = self.train_loaders[node.node_id]
            x, _ = next(iter(loader))
            x = x.to(self.device)
        except Exception:
            x = torch.randn(4, 3, 224, 224).to(self.device)

        # Client forward to get activation
        trainer.client.backbone.eval()
        with torch.no_grad():
            a = trainer.client.backbone(x)
            if a.dim() == 4:
                a = a.mean(dim=[1, 2, 3])  # Global average to get compact representation

        model_state = trainer.get_client_state()

        tier = node.tier
        if tier <= 2:
            proof = CommitmentVerifier.gen_proof_tier1(x, a, model_state)
        elif tier == 3:
            proof = CommitmentVerifier.gen_proof_tier3(x, a)
        else:
            proof = CommitmentVerifier.gen_proof_tier4(x)

        return proof

    # -------------------------------------------------------------------------
    # Phase 4: TVE verification
    # -------------------------------------------------------------------------

    def _phase_verification(
        self,
        updates: List[Dict],
        proofs: List[Proof],
    ) -> Dict[int, Any]:
        """Phase 4: Verify all node proofs via TVE."""
        if not self.tve_enabled:
            return {u["node_id"]: {"is_valid": True, "penalty": 0.0} for u in updates}

        block_hash = hashlib.sha256(f"block_{self.current_round}".encode()).digest()

        # Select committee
        selected_ids = self.tve.select(self.current_round, block_hash)

        # Verify all (pass grad_cache for Tier 1 cosine similarity verification)
        verif_results = self.tve.verify(updates, proofs, self.lazy_node_ids, self._grad_cache)

        # Update historical gradient norms for Tier 2 verification (P1-4 fix)
        for update in updates:
            node_id = update["node_id"]
            gradient_norm = update.get("gradient_norm", 1.0)
            self.tve.update_historical_stats(node_id, gradient_norm)

        # Update verification rates for Shapley value_fn (P0 fix)
        for node_id, result in verif_results.items():
            if hasattr(result, 'is_valid'):
                rate = 1.0 if result.is_valid else 0.0
            else:
                rate = 1.0 if result.get("is_valid", True) else 0.0
            # EMA update
            self.verification_rates[node_id] = 0.9 * self.verification_rates.get(node_id, 1.0) + 0.1 * rate

        return verif_results

    # -------------------------------------------------------------------------
    # Phase 5: Aggregation
    # -------------------------------------------------------------------------

    def _phase_aggregation(self, valid_updates: List[Dict]) -> None:
        """Phase 5: Staleness-decayed async aggregation."""
        if not valid_updates:
            return

        self.global_state = self.aggregator.aggregate(valid_updates)

        # Sync server-side model with new global state
        self.model.load_state_dict(self.global_state)

    # -------------------------------------------------------------------------
    # Phase 6: GTM rewards
    # -------------------------------------------------------------------------

    def _phase_gtm(
        self,
        updates: List[Dict],
        verif_results: Dict[int, Any],
    ) -> tuple:
        """Phase 6: Compute Shapley values and distribute rewards."""
        node_ids = [u["node_id"] for u in updates]

        if not self.gtm_enabled:
            # Equal distribution fallback
            n = len(node_ids)
            R = self.tokenomics.total_reward(self.current_round)
            shapley_vals = {nid: 1.0 / n for nid in node_ids}
            rewards = {nid: R / n for nid in node_ids}
            return shapley_vals, rewards

        # Characteristic function: multi-component v(S) per Eq. 13
        # Components: data_size + verification_quality + resource_provision
        def value_fn(coalition: List[int]) -> float:
            if not coalition:
                return 0.0

            # Component 1: Data size (normalized)
            total_data = sum(
                self.train_loaders[nid].dataset.__len__()
                for nid in coalition
                if nid in self.train_loaders
            )
            data_component = total_data / 50000.0

            # Component 2: Verification quality (EMA rate)
            verif_rates = [self.verification_rates.get(nid, 1.0) for nid in coalition]
            verif_component = float(np.mean(verif_rates)) if verif_rates else 0.0

            # Component 3: Resource provision (normalized flops_ratio)
            node_profile = {n.node_id: n for n in self.nodes}
            resource_vals = [
                node_profile[nid].flops_ratio
                for nid in coalition
                if nid in node_profile
            ]
            resource_component = float(np.mean(resource_vals)) if resource_vals else 0.0

            # Weighted combination (0.5 data, 0.3 verif, 0.2 resource)
            return 0.5 * data_component + 0.3 * verif_component + 0.2 * resource_component

        # Compute Shapley
        calculator = ShapleyCalculator(self.shapley_config)
        shapley_result = calculator.compute_shapley(node_ids, value_fn)
        shapley_vals = shapley_result.values

        # Distribute rewards
        verif_penalties = self.tve.committee.get_penalties(verif_results)
        rewards = self.tokenomics.distribute(
            shapley_values=shapley_vals,
            verification_results=verif_results,
        )

        return shapley_vals, rewards

    # -------------------------------------------------------------------------
    # Phase 7: Blockchain commit
    # -------------------------------------------------------------------------

    def _phase_blockchain(
        self,
        verif_results: Dict[int, Any],
        rewards: Dict[int, float],
        shapley_vals: Dict[int, float],
    ) -> None:
        """Phase 7: Record rewards and verifications to blockchain ledger."""
        epoch = self.current_round

        # Record rewards
        for node_id, reward in rewards.items():
            phi = shapley_vals.get(node_id, 0.0)
            self.ledger.record_reward(epoch, node_id, reward, phi)

        # Record verifications
        for node_id, result in verif_results.items():
            if isinstance(result, dict):
                is_valid = result.get("is_valid", True)
                penalty = result.get("penalty", 0.0)
            else:
                is_valid = result.is_valid
                penalty = result.penalty
            self.ledger.record_verification(
                epoch=epoch,
                node_id=node_id,
                is_valid=is_valid,
                penalty=penalty,
                proof_type="zk_mock",
            )

        n_verified = sum(
            1 for r in verif_results.values()
            if (isinstance(r, dict) and r.get("is_valid", True))
            or (hasattr(r, "is_valid") and r.is_valid)
        )
        self.ledger.commit_block(epoch, rewards, n_verified)

    # -------------------------------------------------------------------------
    # Phase 8: HASO policy update
    # -------------------------------------------------------------------------

    def _phase_haso_update(self, shapley_vals: Dict[int, float], rewards: Dict[int, float]) -> None:
        """Phase 8: Update PPO policies with Shapley-based reward shaping."""
        if not self.haso_enabled:
            return

        # Use centralized orchestrator if available
        if self._orchestrator is not None:
            self._orchestrator.update_shapley(shapley_vals)
            update_ts = self.cfg.get("ppo_update_timesteps", 256)
            self._orchestrator.learn(total_timesteps=update_ts)
        elif self.agent_pool is not None:
            # Fallback: per-node agents
            self.agent_pool.update_shapley_all(shapley_vals)
            update_ts = self.cfg.get("ppo_update_timesteps", 256)
            self.agent_pool.learn_all(total_timesteps=update_ts)
        else:
            return

        # Update per-node reward tracking
        for node_id, reward in rewards.items():
            progress = self.node_progress.get(node_id)
            if progress:
                progress.last_reward = reward
                progress.cumulative_reward += reward

    # -------------------------------------------------------------------------
    # Metrics & evaluation
    # -------------------------------------------------------------------------

    def _collect_metrics(
        self,
        t: int,
        latency: float,
        train_losses: Dict[int, float],
        verif_results: Dict[int, Any],
        rewards: Dict[int, float],
        shapley_vals: Dict[int, float],
        ppo_update_time: float = 0.0,
        shapley_time: float = 0.0,
        train_time: float = 0.0,
        verification_time: float = 0.0,
        comm_time: float = 0.0,
        avg_node_train_time: float = 0.0,
    ) -> RoundMetrics:
        """Collect per-round metrics."""
        n_participants = len(train_losses)
        avg_loss = float(np.mean(list(train_losses.values()))) if train_losses else 0.0

        # Validation results
        n_valid = sum(
            1 for r in verif_results.values()
            if (isinstance(r, dict) and r.get("is_valid", True))
            or (hasattr(r, "is_valid") and r.is_valid)
        )
        detection_rate = n_valid / max(len(verif_results), 1)

        # Fairness (Jain's index)
        reward_values = [max(0.0, r) for r in rewards.values()]
        fairness = self._jains_fairness(reward_values)

        # Shapley stats
        shapley_list = list(shapley_vals.values())
        mean_shapley = float(np.mean(shapley_list)) if shapley_list else 0.0
        var_shapley = float(np.var(shapley_list)) if shapley_list else 0.0

        # Verification time
        verif_times = [
            r.verification_time_ms
            for r in verif_results.values()
            if hasattr(r, "verification_time_ms")
        ]
        mean_verif_ms = float(np.mean(verif_times)) if verif_times else 0.0

        # Ledger size
        ledger_kb = self.ledger.ledger_size_bytes() / 1024.0

        return RoundMetrics(
            round=t,
            round_latency=latency,
            train_loss=avg_loss,
            test_acc=0.0,  # filled after evaluation
            n_valid_updates=n_valid,
            n_participants=n_participants,
            attack_detection_rate=detection_rate,
            fairness_index=fairness,
            total_reward=sum(max(0.0, r) for r in rewards.values()),
            mean_shapley=mean_shapley,
            shapley_variance=var_shapley,
            mean_verification_ms=mean_verif_ms,
            ledger_size_kb=ledger_kb,
            ppo_update_time=ppo_update_time,
            shapley_time=shapley_time,
            train_time=train_time,
            verification_time=verification_time,
            comm_time=comm_time,
            avg_node_train_time=avg_node_train_time,
        )

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate global model on test set.

        Returns:
            Dict with accuracy, loss, precision, recall, F1 (macro and weighted).
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        # Get test loader
        try:
            from ..sfl.data_loader import create_test_loader
            test_loader = create_test_loader(
                dataset_name=self.cfg.get("dataset", "cifar10"),
                batch_size=64,
                data_dir="./data",
            )
        except Exception:
            return {"accuracy": 0.0, "loss": 0.0, "precision_macro": 0.0,
                    "recall_macro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                _, predicted = out.max(1)
                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        n_classes = self.cfg.get("n_classes", 10)
        total = len(all_targets)

        if total == 0:
            return {"accuracy": 0.0, "loss": 0.0, "precision_macro": 0.0,
                    "recall_macro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        # Compute full metrics
        metrics = compute_metrics(all_predictions, all_targets, n_classes)
        metrics["loss"] = total_loss / max(len(test_loader), 1)
        metrics["accuracy"] = 100.0 * metrics["accuracy"]  # Convert to %
        return metrics

    @staticmethod
    def _jains_fairness(values: List[float]) -> float:
        """Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2)."""
        if not values or sum(values) == 0:
            return 0.0
        n = len(values)
        return (sum(values) ** 2) / (n * sum(x ** 2 for x in values))

    @staticmethod
    def _hash_state(state: Dict[str, torch.Tensor]) -> bytes:
        """Hash a model state dict for commitment."""
        data = b""
        for key in sorted(state.keys()):
            data += state[key].cpu().numpy().tobytes()
        return hashlib.sha256(data).digest()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_round(self, t: int, m: RoundMetrics) -> None:
        """Log round metrics to stdout."""
        total_rounds = self.cfg.get('global_rounds', 100)
        print(
            f"Round {t:3d}/{total_rounds} | "
            f"Loss: {m.train_loss:.4f} | "
            f"Acc: {m.test_acc:.2f}% | "
            f"Fairness: {m.fairness_index:.3f} | "
            f"Valid: {m.n_valid_updates}/{m.n_participants} | "
            f"Latency: {m.round_latency:.2f}s | "
            f"Reward: {m.total_reward:.2f}"
        )
        # Print per-node progress every 5 rounds
        if t % 5 == 0:
            self.print_node_progress()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_agents(self, directory: str) -> None:
        """Save all PPO agents to disk."""
        if self.agent_pool:
            self.agent_pool.save_all(directory)

    def load_agents(self, directory: str) -> None:
        """Load all PPO agents from disk."""
        if self.agent_pool:
            self.agent_pool.load_all(directory)

    def save_metrics(self, path: str) -> None:
        """Save metrics history to JSON."""
        data = [m.to_dict() for m in self.metrics_history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_node_progress(self, node_id: int) -> NodeProgress:
        """
        Get progress tracker for a specific node.

        Args:
            node_id: Node identifier.

        Returns:
            NodeProgress object for that node.
        """
        return self.node_progress.get(node_id)

    def get_all_node_progress(self) -> Dict[int, NodeProgress]:
        """Get progress trackers for all nodes."""
        return dict(self.node_progress)

    def print_node_progress(self, node_ids: Optional[List[int]] = None) -> None:
        """
        Print a formatted table of per-node progress.

        Args:
            node_ids: List of node IDs to show. If None, shows all.
        """
        print("\n" + "=" * 100)
        print(f"{'NodeID':>6} | {'Round':>5} | {'Epochs':>6} | {'CutL':>4} | {'BS':>3} | {'LastLoss':>8} | {'MeanLoss':>8} | {'Excl':>4} | {'LastReward':>10} | {'Completed':>9}")
        print("-" * 100)

        ids_to_show = node_ids or [nid for nid in self.node_progress]
        for nid in sorted(ids_to_show):
            p = self.node_progress.get(nid)
            if p is None:
                continue
            status = "YES" if p.completed else "no"
            print(
                f"{p.node_id:>6} | "
                f"{p.current_round:>5} | "
                f"{p.total_epochs_trained:>6} | "
                f"{p.cut_layer:>4} | "
                f"{p.batch_size:>3} | "
                f"{p.last_loss:>8.4f} | "
                f"{p.mean_loss:>8.4f} | "
                f"{p.times_excluded:>4} | "
                f"{p.last_reward:>10.4f} | "
                f"{status:>9}"
            )
        print("=" * 100)

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all rounds."""
        if not self.metrics_history:
            return {}
        final = self.metrics_history[-1]
        best_acc = max(m.test_acc for m in self.metrics_history)
        mean_latency = np.mean([m.round_latency for m in self.metrics_history])
        mean_fairness = np.mean([m.fairness_index for m in self.metrics_history])
        return {
            "final_accuracy": final.test_acc,
            "best_accuracy": best_acc,
            "mean_latency": mean_latency,
            "mean_fairness": mean_fairness,
            "total_rounds": len(self.metrics_history),
        }

    # -------------------------------------------------------------------------
    # Checkpointing (ExperimentAgent)
    # -------------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save full protocol state to checkpoint file.

        Args:
            path: Path to save checkpoint.
        """
        node_progress_dict = {
            nid: p.to_dict() for nid, p in self.node_progress.items()
        }
        metrics_list = [m.to_dict() for m in self.metrics_history]

        save_checkpoint(
            path=path,
            round_num=self.current_round,
            model_state=self.global_state,
            node_states={},  # trainer states not serialized
            metrics_history=metrics_list,
            config=self.cfg,
            node_progress=node_progress_dict,
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Load protocol state from checkpoint file.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(path)
        self.current_round = checkpoint["round"]
        self.global_state = checkpoint["model_state"]
        self.model.load_state_dict(self.global_state)
        self.metrics_history = [
            RoundMetrics(**m) for m in checkpoint["metrics_history"]
        ]
        # Restore node progress
        for nid, pdict in checkpoint.get("node_progress", {}).items():
            if nid in self.node_progress:
                p = self.node_progress[nid]
                p.current_round = pdict.get("current_round", 0)
                p.total_epochs_trained = pdict.get("total_epochs_trained", 0)
                p.last_loss = pdict.get("last_loss", 0.0)
                p.last_reward = pdict.get("last_reward", 0.0)
                p.cut_layer = pdict.get("cut_layer", 0)
                p.batch_size = pdict.get("batch_size", 0)

    # -------------------------------------------------------------------------
    # Progress tracking (ExperimentAgent)
    # -------------------------------------------------------------------------

    def get_progress_tracker(self, eval_every: int = 10, checkpoint_every: int = 10) -> ProgressTracker:
        """
        Create a ProgressTracker for this protocol.

        Args:
            eval_every: Evaluate every N rounds.
            checkpoint_every: Checkpoint every N rounds.

        Returns:
            ProgressTracker instance.
        """
        return ProgressTracker(
            total_rounds=self.cfg.get("global_rounds", 100),
            n_nodes=self.n_nodes,
            eval_every=eval_every,
            checkpoint_every=checkpoint_every,
        )
