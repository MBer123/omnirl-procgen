import torch
import torch.nn as nn

from .impala_cnn import ConvSequence


class Transition(nn.Module):
    """
    A latent transition model conditioned on actions.

    Args:
        latent_dim: Dimension of the input and output latent features.
        action_dim: Dimension of the environment action space.
        action_embed_dim: Output size of the action encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_embed_dim: int = 32,
    ):
        super().__init__()
        self.action_encoder = nn.Embedding(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z, action):
        """Predict the next latent feature from a latent-action pair."""
        action = action.long().view(-1)
        action = self.action_encoder(action)
        return self.net(torch.cat([z, action], dim=-1))


class Projector(nn.Module):
    """
    A projection head that maps encoder features into SPR latent space.

    Args:
        latent_dim: Dimension of the encoder latent features.
        projection_dim: Output dimension of the projection head.
    """

    def __init__(self, latent_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim),
        )

    def forward(self, z):
        """Project encoder features into SPR latent space."""
        return self.net(z)


class Predictor(nn.Module):
    """
    A prediction head for SPR latent prediction.

    Args:
        projection_dim: Dimension of the projected latent features.
    """

    def __init__(self, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, z):
        """Predict the target SPR latent from projected features."""
        return self.net(z)


class RMSNorm(nn.Module):
    """
    Root mean square normalization over the last feature dimension.

    Args:
        dim: Size of the normalized feature dimension.
        eps: Small constant added for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalize inputs by RMS value and apply a learnable scale."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class SPPOCNN(nn.Module):
    """
    An IMPALA CNN actor-critic network with SPR.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        latent_dim: Output embedding size after CNN trunk.
        width_scale: Channel width multiplier for the IMPALA CNN backbone.
        action_embed_dim: Output size of the action encoder.
        projection_dim: Dimension of the projected latent features.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        latent_dim: int = 256,
        width_scale: int = 4,
        action_embed_dim: int = 32,
        projection_dim: int = 256,
    ):
        super().__init__()

        H, W, C = state_dim

        # Initialize the shared feature network
        depths = [16 * width_scale, 32 * width_scale, 32 * width_scale]

        self.cnn = nn.Sequential(
            ConvSequence(C, depths[0]),
            ConvSequence(depths[0], depths[1]),
            ConvSequence(depths[1], depths[2]),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            n_flatten = self.cnn(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            RMSNorm(n_flatten),
            nn.Linear(n_flatten, latent_dim),
            nn.ReLU(),
        )

        # Initialize the SPR transition module
        self.transition = Transition(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_embed_dim=action_embed_dim,
        )

        # Initialize the SPR projector module
        self.projector = Projector(latent_dim=latent_dim, projection_dim=projection_dim)

        # Initialize the SPR predictor module
        self.predictor = Predictor(projection_dim=projection_dim)

        # Initialize the actor head
        self.action_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, action_dim),
        )

        # Initialize the critic head
        self.state_value = nn.Sequential(
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def _encode(self, state):
        """Return encoded state features."""
        x = state.permute(0, 3, 1, 2).contiguous()
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def projection(self, state):
        """Return encoded state projected into SPR target space."""
        x = self._encode(state)
        projected_x = self.projector(x)
        return projected_x

    def forward_actor(self, state):
        """Return action logits."""
        x = self._encode(state)
        return self.action_logits(x)

    def forward_critic(self, state):
        """Return state value."""
        x = self._encode(state)
        return self.state_value(x)

    def forward(
        self,
        state,
        spr_action_seq,
        spr_state,
    ):
        """Return action logits, state value, and SPR predictions."""
        x = self._encode(state)

        z = x if spr_state is None else self._encode(spr_state)
        preds = []
        for k in range(spr_action_seq.shape[1]):
            z = self.transition(z, spr_action_seq[:, k])
            pred_z = self.predictor(self.projector(z))
            preds.append(pred_z)

        return {
            "action_logits": self.action_logits(x),
            "state_value": self.state_value(x),
            "spr_pred": torch.stack(preds, dim=1),
        }
