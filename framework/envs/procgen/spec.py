from .builder import build_env


def get_env_spec(env_config):
    """Describe a Procgen environment using the OmniRL runtime interface."""
    env = build_env(env_config=env_config, role="specification")
    env_spec = {
        "state_dim": env.observation_space.shape,
        "action_dim": env.action_space.n,
        "is_continuous": False,
    }
    state_spec = {**env_spec, "state_dtype": env.observation_space.dtype}
    action_spec = {**env_spec, "action_low": None, "action_high": None}
    env.close()

    return {
        "env_spec": env_spec,
        "state_spec": state_spec,
        "action_spec": action_spec,
        "eval_metrics": {
            "train_return_mean": 0,
            "train_return_std": 0,
            "train_return_max": 0,
            "train_return_min": 0,
            "test_return_mean": 0,
            "test_return_std": 0,
            "test_return_max": 0,
            "test_return_min": 0,
        },
    }
