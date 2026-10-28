import math
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from framework.algorithms.sppo import Learner, Policy
from framework.algorithms.sppo.learner import ValueNormalizer
from framework.networks import ImpalaCNN


class SPPOSmokeTest(unittest.TestCase):
    def test_learner_update_policy_sync_and_checkpoint(self):
        network_kwargs = {"latent_dim": 32, "width_scale": 1}
        state_dim = (16, 16, 3)
        action_dim = 5
        learner_class = Learner.__ray_metadata__.modified_class

        with tempfile.TemporaryDirectory() as ckpt_dir:
            learner = learner_class(
                state_dim=state_dim,
                action_dim=action_dim,
                network_kwargs=network_kwargs,
                ckpt_dir=ckpt_dir,
                k_epochs=1,
                num_minibatches=2,
                random_shift_padding=1,
                device="cpu",
                seed=0,
            )
            policy = Policy(
                state_dim=state_dim,
                action_dim=action_dim,
                network_kwargs=network_kwargs,
            )
            policy.set_weights(learner.get_weights(), version=learner.get_version())

            rng = np.random.default_rng(0)
            states = rng.random((2, 3, *state_dim), dtype=np.float32)
            next_states = rng.random(states.shape, dtype=np.float32)
            samples = [policy.sample_action(state) for state in states.reshape(-1, *state_dim)]
            batch = {
                "states": states,
                "next_states": next_states,
                "actions": np.array([sample["data"]["action"] for sample in samples]).reshape(2, 3),
                "log_probs": np.array(
                    [sample["data"]["log_prob"] for sample in samples], dtype=np.float32
                ).reshape(2, 3),
                "rewards": np.array([[0, 1, 2], [0, -1, 3]], dtype=np.float32),
                "terminals": np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32),
                "dones": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
            }
            initial_weights = {
                key: value.clone() for key, value in learner.get_weights().items()
            }

            result = learner.learn(batch)

            self.assertEqual(result["version"], 1)
            self.assertEqual(result["memory"]["version"], 1)
            self.assertEqual(result["trained_samples"], 6)
            self.assertTrue(all(math.isfinite(value) for value in result["log"].values()))
            self.assertGreater(learner.value_norm.count.item(), 6)
            self.assertTrue(
                any(
                    not torch.equal(initial_weights[key], value)
                    for key, value in learner.get_weights().items()
                )
            )

            policy.set_weights(learner.get_weights(), version=result["version"])
            self.assertEqual(policy.version, 1)
            observation = states[0, 0]
            with torch.no_grad():
                expected_logits = learner.model.forward_actor(torch.from_numpy(observation).unsqueeze(0))
            self.assertEqual(policy.choose_action(observation), expected_logits.argmax(-1).item())
            sample = policy.sample_action(observation)
            self.assertEqual(sample["env_action"], sample["data"]["action"])
            expected_log_prob = Categorical(logits=expected_logits).log_prob(
                torch.tensor(sample["data"]["action"])
            ).item()
            self.assertAlmostEqual(sample["data"]["log_prob"], expected_log_prob, places=5)

            expected_weights = {
                key: value.clone() for key, value in learner.get_weights().items()
            }
            expected_value_norm = learner.value_norm.state_dict()
            learner.save_checkpoint()
            with torch.no_grad():
                for parameter in learner.model.parameters():
                    parameter.zero_()
            learner.value_norm.mean.fill_(100)
            learner.value_norm.var.fill_(100)
            learner.value_norm.count.fill_(100)
            learner.optimizer.state.clear()
            learner.version = -1

            restored = learner.load_checkpoint()

            self.assertEqual(restored["version"], 1)
            self.assertTrue(learner.optimizer.state)
            for key, expected in expected_weights.items():
                torch.testing.assert_close(restored["weights"][key], expected)
            for key, expected in expected_value_norm.items():
                torch.testing.assert_close(learner.value_norm.state_dict()[key], expected)

    def test_network_forward_runs(self):
        policy = Policy(
            state_dim=(32, 32, 3),
            action_dim=15,
            network_kwargs={"latent_dim": 32, "width_scale": 1},
        )
        self.assertIsInstance(policy.model, ImpalaCNN)
        observations = torch.randn(2, 32, 32, 3)

        action_logits, state_values = policy.model(observations)

        self.assertEqual(action_logits.shape, (2, 15))
        self.assertEqual(state_values.shape, (2, 1))

    def test_value_normalizer_round_trip(self):
        normalizer = ValueNormalizer()
        returns = torch.tensor([1.0, 3.0, 5.0, 7.0])
        normalizer.update_from_moments(
            returns.mean(), returns.var(unbiased=False), returns.numel()
        )

        normalized = normalizer.normalize(returns)

        self.assertAlmostEqual(normalizer.mean.item(), 4.0, places=3)
        self.assertAlmostEqual(normalizer.var.item(), 5.0, places=3)
        self.assertAlmostEqual(normalized.mean().item(), 0.0, places=3)
        self.assertAlmostEqual(normalized.var(unbiased=False).item(), 1.0, places=3)
        torch.testing.assert_close(normalizer.denormalize(normalized), returns)


if __name__ == "__main__":
    unittest.main()
