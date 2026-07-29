import numpy as np
import ray
import torch

from omnirl.utils.seed import set_global_seed


@ray.remote(num_cpus=1)
class Sampler:
    """
    Generic distributed sampling worker.

    Args:
        policy_cls: Algorithm-specific policy class.
        collection_cls: Environment-specific collection loop class.
        env_builder: Callable that builds an environment instance.
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of continuous actions.
        action_high: Upper bounds of continuous actions.
        network_kwargs: Keyword arguments used to build the policy network.
        policy_kwargs: Keyword arguments used to build the policy.
        unroll_length: Number of transitions collected before each buffer write.
        seed: Random seed used to ensure consistent results across runs.
    """

    def __init__(
        self,
        policy_cls,
        collection_cls,
        env_builder,
        state_dim: int | tuple[int, ...],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
        policy_kwargs: dict | None = None,
        unroll_length: int = 1,
        seed: int | None = None,
    ):
        # Limit PyTorch CPU threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Initialize random seed
        if seed is not None:
            set_global_seed(seed)

        # Initialize abstract algorithm and environment components
        self.policy = policy_cls(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            action_low=action_low,
            action_high=action_high,
            network_kwargs=network_kwargs,
            **dict(policy_kwargs or {}),
        )
        self.collection = collection_cls(env_builder=env_builder, seed=seed)
        self.unroll_length = unroll_length
        self.version = -1

    def _sync_weights(self, weight_store):
        """Fetch and load newer policy weights from the weight store."""
        result = ray.get(weight_store.get.remote(self.version))
        weights = result["weights"]
        if weights is not None:
            self.version = result["version"]
            self.policy.set_weights(weights, self.version)

    def rollout(self, memory, weight_store):
        """Continuously collect environment data and send it to the buffer."""
        while True:
            self._sync_weights(weight_store)
            sample_action = self.policy.sample_action

            rollout = self.collection.collect(sample_action, self.unroll_length)
            rollout["version"] = self.version
            memory.add.remote(rollout)
