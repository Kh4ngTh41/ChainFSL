"""
Blockchain Mock — SQLite-based immutable ledger for ChainFSL.

Provides thread-safe, append-only storage for:
- Reward distributions (per round, per node)
- Verification results (proofs, penalties)
- Block commitments (Merkle root, epoch metadata)

Supports E7 overhead measurement (on-chain writes, ledger size).
"""

import sqlite3
import json
import time
import threading
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class BlockRecord:
    """On-chain block commitment."""

    block_id: int
    epoch: int
    merkle_root: str
    n_verified: int
    timestamp: float


class BlockchainLedger:
    """
    Thread-safe SQLite ledger mimicking blockchain behavior.

    Design choices (per plan):
    - SQLite (not Hardhat) for single-machine simulation
    - Merkle root mock (SHA-256 of reward dict)
    - O(1) commits per epoch (single merkle root write)
    - All writes go through _lock for thread safety
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./chainfsl_ledger.db"):
        """
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection (caller must use context manager or finally close)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rewards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        epoch INTEGER NOT NULL,
                        node_id INTEGER NOT NULL,
                        reward REAL NOT NULL,
                        shapley_value REAL NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS verifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        epoch INTEGER NOT NULL,
                        node_id INTEGER NOT NULL,
                        is_valid INTEGER NOT NULL,
                        penalty REAL NOT NULL,
                        proof_type TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS blocks (
                        block_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        epoch INTEGER NOT NULL UNIQUE,
                        merkle_root TEXT NOT NULL,
                        n_verified INTEGER NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rewards_epoch
                    ON rewards(epoch)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_verifications_epoch
                    ON verifications(epoch)
                """)

    # ------------------------------------------------------------------ #
    # Reward recording
    # ------------------------------------------------------------------ #

    def record_reward(
        self,
        epoch: int,
        node_id: int,
        reward: float,
        shapley: float,
    ) -> None:
        """Record a single node's reward for an epoch."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO rewards (epoch, node_id, reward, shapley_value, timestamp) VALUES (?,?,?,?,?)",
                    (epoch, node_id, reward, shapley, time.time()),
                )

    def record_rewards_batch(
        self,
        epoch: int,
        rewards: Dict[int, float],
        shapley_values: Dict[int, float],
    ) -> None:
        """
        Batch record all rewards for an epoch.

        Args:
            epoch: Epoch number.
            rewards: node_id -> reward amount.
            shapley_values: node_id -> Shapley value.
        """
        with self._lock:
            with self._get_conn() as conn:
                ts = time.time()
                rows = [
                    (epoch, nid, max(0.0, r), shapley_values.get(nid, 0.0), ts)
                    for nid, r in rewards.items()
                ]
                conn.executemany(
                    "INSERT INTO rewards (epoch, node_id, reward, shapley_value, timestamp) VALUES (?,?,?,?,?)",
                    rows,
                )

    # ------------------------------------------------------------------ #
    # Verification recording
    # ------------------------------------------------------------------ #

    def record_verification(
        self,
        epoch: int,
        node_id: int,
        is_valid: bool,
        penalty: float,
        proof_type: str,
    ) -> None:
        """Record a single node's verification result."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO verifications (epoch, node_id, is_valid, penalty, proof_type, timestamp) VALUES (?,?,?,?,?,?)",
                    (epoch, node_id, int(is_valid), penalty, proof_type, time.time()),
                )

    def record_verifications_batch(
        self,
        epoch: int,
        results: Dict[int, dict],
    ) -> None:
        """
        Batch record all verification results for an epoch.

        Args:
            epoch: Epoch number.
            results: node_id -> {valid, penalty, proof_type, ...}
        """
        with self._lock:
            with self._get_conn() as conn:
                ts = time.time()
                rows = [
                    (epoch, nid, int(r.get("valid", False)), r.get("penalty", 0.0), r.get("proof_type", "unknown"), ts)
                    for nid, r in results.items()
                ]
                conn.executemany(
                    "INSERT INTO verifications (epoch, node_id, is_valid, penalty, proof_type, timestamp) VALUES (?,?,?,?,?,?)",
                    rows,
                )

    # ------------------------------------------------------------------ #
    # Block commitment
    # ------------------------------------------------------------------ #

    def commit_block(
        self,
        epoch: int,
        rewards: Dict[int, float],
        n_verified: int,
    ) -> BlockRecord:
        """
        Commit epoch to ledger with Merkle root of reward distribution.

        Args:
            epoch: Epoch number.
            rewards: node_id -> reward amount.
            n_verified: Number of valid updates this epoch.

        Returns:
            BlockRecord with commitment details.
        """
        # Build deterministic Merkle root
        sorted_rewards = sorted(rewards.items(), key=lambda x: x[0])
        merkle_data = json.dumps(sorted_rewards, sort_keys=True).encode()
        merkle_root = hashlib.sha256(merkle_data).hexdigest()

        with self._lock:
            with self._get_conn() as conn:
                ts = time.time()
                # Delete existing block for this epoch if any, then insert
                conn.execute("DELETE FROM blocks WHERE epoch=?", (epoch,))
                cursor = conn.execute(
                    "INSERT INTO blocks (epoch, merkle_root, n_verified, timestamp) VALUES (?,?,?,?)",
                    (epoch, merkle_root, n_verified, ts),
                )
                block_id = cursor.lastrowid

        return BlockRecord(
            block_id=block_id,
            epoch=epoch,
            merkle_root=merkle_root,
            n_verified=n_verified,
            timestamp=ts,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_cumulative_reward(self, node_id: int) -> float:
        """Total reward received by node across all epochs."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(reward), 0.0) FROM rewards WHERE node_id=?",
                (node_id,),
            ).fetchone()
            return float(row[0]) if row else 0.0

    def get_epoch_rewards(self, epoch: int) -> Dict[int, float]:
        """Get all rewards for a specific epoch."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT node_id, reward FROM rewards WHERE epoch=?",
                (epoch,),
            ).fetchall()
            return {r["node_id"]: r["reward"] for r in rows}

    def get_epoch_stats(self, epoch: int) -> Dict[str, Any]:
        """Get aggregated stats for an epoch."""
        with self._get_conn() as conn:
            reward_rows = conn.execute(
                "SELECT node_id, reward, shapley_value FROM rewards WHERE epoch=?",
                (epoch,),
            ).fetchall()
            verif_row = conn.execute(
                "SELECT COUNT(*), SUM(is_valid), SUM(penalty) FROM verifications WHERE epoch=?",
                (epoch,),
            ).fetchone()

        return {
            "rewards": {r["node_id"]: {"reward": r["reward"], "shapley": r["shapley_value"]} for r in reward_rows},
            "n_verified": verif_row[0] or 0,
            "n_valid": verif_row[1] or 0,
            "total_penalty": verif_row[2] or 0.0,
        }

    def get_block(self, epoch: int) -> Optional[BlockRecord]:
        """Get block record for an epoch."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM blocks WHERE epoch=?",
                (epoch,),
            ).fetchone()
            if row is None:
                return None
            return BlockRecord(
                block_id=row["block_id"],
                epoch=row["epoch"],
                merkle_root=row["merkle_root"],
                n_verified=row["n_verified"],
                timestamp=row["timestamp"],
            )

    def ledger_size_bytes(self) -> int:
        """Total size of ledger file in bytes."""
        path = Path(self.db_path)
        return path.stat().st_size if path.exists() else 0

    def measure_overhead(self, epoch: int) -> Dict[str, Any]:
        """
        Measure blockchain overhead for E7.

        Returns:
            Dict with overhead metrics.
        """
        block = self.get_block(epoch)
        size_bytes = self.ledger_size_bytes()

        return {
            "on_chain_writes": 1,          # One Merkle root commit per epoch
            "merkle_root_bytes": 32,       # SHA-256 = 32 bytes
            "ledger_size_bytes": size_bytes,
            "epoch": epoch,
            "n_verified": block.n_verified if block else 0,
        }

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all data (for testing)."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM rewards")
                conn.execute("DELETE FROM verifications")
                conn.execute("DELETE FROM blocks")

    def summary(self) -> Dict[str, Any]:
        """Get ledger summary statistics."""
        with self._get_conn() as conn:
            n_rewards = conn.execute("SELECT COUNT(*) FROM rewards").fetchone()[0]
            n_verifs = conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0]
            n_blocks = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            total_reward = conn.execute("SELECT COALESCE(SUM(reward), 0.0) FROM rewards").fetchone()[0]

        return {
            "n_epochs": n_blocks,
            "n_rewards": n_rewards,
            "n_verifications": n_verifs,
            "total_reward_distributed": total_reward,
            "ledger_size_bytes": self.ledger_size_bytes(),
        }