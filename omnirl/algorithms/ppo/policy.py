import numpy as np
import torch
from torch.distributions import Categorical

from omnirl.networks import ImpalaCNN


class Policy:
    """
    PPO policy used for local inference.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
    ):
        self.device = torch.device("cpu")
        self.is_continuous = is_continuous
        self.version = -1

        network_kwargs = dict(network_kwargs or {})
        model = ImpalaCNN(
            state_dim=state_dim,
            action_dim=action_dim,
            **network_kwargs,
        )

        self.model = model.to(self.device)
        self.model.eval()

    def set_weights(self, weights, version=None):
        """Load external network weights into the actor network."""
        self.model.load_state_dict(weights)
        if version is not None:
            self.version = version

    def choose_action(self, observation):
        """Select a deterministic action for evaluation."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.model.forward_actor(state)
            action = torch.argmax(action_logits, dim=-1)
            return int(action.item())

    def sample_action(self, observation):
        """Select a probabilistic action and its log probability."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.model.forward_actor(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            action = int(action.item())
            env_action = action

        return {
            "env_action": env_action,
            "data": {"action": action, "log_prob": log_prob},
        }
