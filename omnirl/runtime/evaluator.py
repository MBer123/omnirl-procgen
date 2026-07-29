import numpy as np
import ray
import torch

from omnirl.utils.seed import set_global_seed


@ray.remote(num_cpus=1)
class Evaluator:
    """
    Generic distributed evaluation worker.

    Args:
        policy_cls: Algorithm-specific policy class.
        evaluation_cls: Environment-specific evaluation loop class.
        env_builder: Callable that builds an environment instance.
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of continuous actions.
        action_high: Upper bounds of continuous actions.
        network_kwargs: Keyword arguments used to build the policy network.
        policy_kwargs: Keyword arguments used to build the policy.
        num_episodes: Number of episodes to run during each evaluation.
        seed: Random seed used to ensure consistent results across runs.
    """

    def __init__(
        self,
        policy_cls,
        evaluation_cls,
        env_builder,
        state_dim: int | tuple[int, ...],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
        policy_kwargs: dict | None = None,
        num_episodes: int = 1,
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
        self.evaluation = evaluation_cls(env_builder=env_builder, seed=seed)
        self.num_episodes = num_episodes
        self.version = -1

    def evaluate(self, weights, version):
        """Evaluate the current policy through the injected evaluation loop."""
        self.policy.set_weights(weights, version)
        self.version = version
        result = self.evaluation.run(self.policy.choose_action, self.num_episodes)
        return {"version": self.version, **result}
