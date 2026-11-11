import unittest
import numpy as np

from framework.envs.procgen import CollectionLoop


class ReusingObservationEnv:
    """Exercise collectors against environments that reuse their observation array."""

    def __init__(self):
        self.observation = np.zeros(1, dtype=np.float32)
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.observation.fill(0)
        return self.observation, {}

    def step(self, action):
        self.observation += 1
        return self.observation, 1.0, self.observation[0] == 2, False, {}


class BatchedPolicy:
    def sample_action(self, observations):
        actions = np.arange(len(observations), dtype=np.int64)
        return {
            "action": actions,
            "env_action": actions,
            "extra_data": {"log_probs": np.full(len(observations), -0.5)},
        }


class ProcgenCollectionTest(unittest.TestCase):
    def test_batched_rollout_preserves_terminal_observations_and_axes(self):
        for num_envs in (1, 2):
            with self.subTest(num_envs=num_envs):
                collection = CollectionLoop(ReusingObservationEnv, seed=7, num_envs=num_envs)
                rollout = collection.collect(BatchedPolicy(), num_steps=3)

                self.assertEqual(rollout["states"].shape, (num_envs, 3, 1))
                self.assertEqual(rollout["action"].shape, (num_envs, 3))
                self.assertEqual(rollout["extra_data"]["log_probs"].shape, (num_envs, 3))
                for index, env in enumerate(collection.envs):
                    self.assertEqual(env.seeds, [7 + index, None])
                    np.testing.assert_array_equal(rollout["states"][index, :, 0], [0, 1, 0])
                    np.testing.assert_array_equal(rollout["next_states"][index, :, 0], [1, 2, 1])
                    np.testing.assert_array_equal(rollout["dones"][index], [False, True, False])
                    np.testing.assert_array_equal(rollout["action"][index], [index] * 3)


if __name__ == "__main__":
    unittest.main()
