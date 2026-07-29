from dataclasses import dataclass

import numpy as np

from .builder import build_env


@dataclass(frozen=True)
class EnvSpec:
    """
    Algorithm-facing description of a Procgen environment.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        default_eval_metrics: Initial values of environment evaluation metrics.
    """

    state_dim: int | tuple[int, ...]
    state_dtype: np.dtype
    action_dim: int
    is_continuous: bool
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    default_eval_metrics: dict


def get_env_spec(env_config):
    """Describe an environment without exposing Gym spaces."""
    env = build_env(env_config=env_config, role="specification")
    spec = EnvSpec(
        state_dim=env.observation_space.shape,
        state_dtype=env.observation_space.dtype,
        action_dim=env.action_space.n,
        is_continuous=False,
        action_low=None,
        action_high=None,
        default_eval_metrics={
            "train_return_mean": 0,
            "train_return_std": 0,
            "train_return_max": 0,
            "train_return_min": 0,
            "test_return_mean": 0,
            "test_return_std": 0,
            "test_return_max": 0,
            "test_return_min": 0,
        },
    )
    env.close()
    return spec
