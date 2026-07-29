import unittest

import torch

from omnirl.networks import SPPOCNN


class SPPOSmokeTest(unittest.TestCase):
    def test_network_forward_runs(self):
        network = SPPOCNN(
            state_dim=(32, 32, 3),
            action_dim=15,
            latent_dim=32,
            width_scale=1,
            action_embed_dim=8,
            projection_dim=16,
        )
        observations = torch.randn(2, 32, 32, 3)
        spr_states = torch.randn(2, 32, 32, 3)
        action_sequence = torch.randint(0, 15, (2, 3))

        output = network(observations, action_sequence, spr_states)

        self.assertEqual(output["action_logits"].shape, (2, 15))
        self.assertEqual(output["state_value"].shape, (2, 1))
        self.assertEqual(output["spr_pred"].shape, (2, 3, 16))


if __name__ == "__main__":
    unittest.main()
