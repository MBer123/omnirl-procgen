import numpy as np


class CollectionLoop:
    """
    Collect transitions using the Procgen interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
        num_envs: Number of environments sharing each policy inference batch.
    """

    def __init__(self, env_builder, seed=None, num_envs=1):
        self.envs = [env_builder() for _ in range(num_envs)]
        self.observations = []
        for i, env in enumerate(self.envs):
            observation, _ = env.reset(seed=None if seed is None else seed + i)
            self.observations.append(np.array(observation, copy=True))

    def collect(self, policy, num_steps):
        """Return a rollout with leading [num_envs, num_steps] dimensions."""
        fields = ("states", "action", "rewards", "next_states", "terminals", "dones")
        rollouts = [{name: [] for name in fields} for _ in self.envs]
        rollouts = [{**rollout, "extra_data": {}} for rollout in rollouts]

        for _ in range(num_steps):
            observations = np.stack(self.observations)
            decision = policy.sample_action(observations)

            for i, env in enumerate(self.envs):
                action = decision["env_action"][i]
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rollout = rollouts[i]
                rollout["states"].append(observations[i])
                rollout["action"].append(np.array(decision["action"][i], copy=True))
                rollout["rewards"].append(reward)
                rollout["next_states"].append(np.array(next_observation, copy=True))
                rollout["terminals"].append(terminated)
                rollout["dones"].append(done)
                for name, values in decision["extra_data"].items():
                    rollout["extra_data"].setdefault(name, []).append(
                        np.array(values[i], copy=True)
                    )

                if done:
                    next_observation, _ = env.reset()
                self.observations[i] = np.array(next_observation, copy=True)

        rollout = {
            name: np.asarray([item[name] for item in rollouts]) for name in fields
        }
        rollout["extra_data"] = {
            name: np.asarray([item["extra_data"][name] for item in rollouts])
            for name in rollouts[0]["extra_data"]
        }
        return rollout
