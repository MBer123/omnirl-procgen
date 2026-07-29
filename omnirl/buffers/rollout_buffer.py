import numpy as np
import ray


@ray.remote(num_cpus=1)
class RolloutBuffer:
    """
    Rollout buffer for experience store.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        unroll_length: Number of transitions contained in each rollout segment.
        max_chunks: Maximum number of rollout chunks stored in the buffer.
        store_log_probs: Whether to store action log probabilities.
        track_version: Whether to reject stale rollouts and track learner versions.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        state_dtype: np.dtype | type,
        action_dim: int,
        is_continuous: bool = False,
        unroll_length: int = 32,
        max_chunks: int = 64,
        store_log_probs: bool = False,
        track_version: bool = True,
    ):
        self.unroll_length = unroll_length
        self.max_chunks = max_chunks
        self.store_log_probs = store_log_probs
        self.track_version = track_version

        self.collected_chunks = 0
        self.available_chunks = 0

        self.is_ready = False
        self.version = -1

        state_dim = (state_dim,) if isinstance(state_dim, int) else tuple(state_dim)
        chunk_shape = (max_chunks, unroll_length)

        self.state_memory = np.zeros((*chunk_shape, *state_dim), dtype=state_dtype)
        if is_continuous:
            self.action_memory = np.zeros((*chunk_shape, action_dim), dtype=np.float32)
        else:
            self.action_memory = np.zeros(chunk_shape, dtype=np.int64)
        self.reward_memory = np.zeros(chunk_shape, dtype=np.float32)
        self.next_state_memory = np.zeros((*chunk_shape, *state_dim), dtype=state_dtype)
        self.terminal_memory = np.zeros(chunk_shape, dtype=np.bool_)
        self.done_memory = np.zeros(chunk_shape, dtype=np.bool_)
        if self.store_log_probs:
            self.log_prob_memory = np.zeros(chunk_shape, dtype=np.float32)

    def put_chunk(
        self,
        version,
        states,
        actions,
        rewards,
        next_states,
        terminals,
        dones,
        log_probs=None,
    ):
        """Store a chunk of transitions into the rollout buffer."""
        if self.track_version and version < self.version:
            return

        if self.is_ready:
            return

        self.state_memory[self.available_chunks] = states
        self.action_memory[self.available_chunks] = actions
        self.reward_memory[self.available_chunks] = rewards
        self.next_state_memory[self.available_chunks] = next_states
        self.terminal_memory[self.available_chunks] = terminals
        self.done_memory[self.available_chunks] = dones
        if self.store_log_probs:
            self.log_prob_memory[self.available_chunks] = log_probs

        self.collected_chunks += 1
        self.available_chunks += 1

        if self.available_chunks == self.max_chunks:
            self.is_ready = True

    def add(self, rollout):
        """Store one rollout collected by a runtime sampler."""
        policy_data = rollout["policy_data"]
        self.put_chunk(
            rollout["version"],
            rollout["states"],
            policy_data["action"],
            rollout["rewards"],
            rollout["next_states"],
            rollout["terminals"],
            rollout["dones"],
            policy_data.get("log_prob"),
        )

    def get_batch(self, rank, world_size):
        """Return collected transitions from the rollout buffer."""
        if not self.ready():
            return None

        chunk_per_worker = self.max_chunks // world_size
        start = rank * chunk_per_worker
        end = self.max_chunks if rank == world_size - 1 else start + chunk_per_worker

        batch = {
            "states": self.state_memory[start:end],
            "actions": self.action_memory[start:end],
            "rewards": self.reward_memory[start:end],
            "next_states": self.next_state_memory[start:end],
            "terminals": self.terminal_memory[start:end],
            "dones": self.done_memory[start:end],
        }
        if self.store_log_probs:
            batch["log_probs"] = self.log_prob_memory[start:end]
        return batch

    def on_learn(self, train_infos):
        """Update internal state after the learner finishes one training step."""
        self._clear()
        if self.track_version:
            self.version = train_infos[0]["version"]

    def ready(self):
        """Return True if the buffer has enough samples to start training."""
        return self.is_ready

    def _clear(self):
        """Clear all stored transitions in the rollout buffer."""
        self.available_chunks = 0
        self.is_ready = False

    def stats(self):
        """Return statistics describing the current state of the rollout buffer."""
        return {
            "available_data": self.available_chunks * self.unroll_length,
            "collected_data": self.collected_chunks * self.unroll_length,
        }
