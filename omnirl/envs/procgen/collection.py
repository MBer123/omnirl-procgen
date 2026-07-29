import numpy as np


class CollectionLoop:
    """
    Collect transitions using the Procgen interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.observation, _ = self.env.reset(seed=seed)

    def collect(self, sample_action, num_steps):
        """Collect a fixed-length rollout and preserve opaque policy data."""
        states = []
        rewards = []
        next_states = []
        terminals = []
        dones = []
        policy_data = {}

        for _ in range(num_steps):
            decision = sample_action(self.observation)
            action = decision["env_action"]
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(self.observation)
            rewards.append(reward)
            next_states.append(next_observation)
            terminals.append(terminated)
            dones.append(done)

            for name, value in decision.get("data", {}).items():
                policy_data.setdefault(name, []).append(value)

            self.observation = next_observation
            if done:
                self.observation, _ = self.env.reset()

        return {
            "states": np.asarray(states),
            "rewards": np.asarray(rewards),
            "next_states": np.asarray(next_states),
            "terminals": np.asarray(terminals),
            "dones": np.asarray(dones),
            "policy_data": {
                name: np.asarray(values) for name, values in policy_data.items()
            },
        }
