import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A residual block used in the IMPALA CNN backbone.

    Args:
        channels: Number of input and output feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """Return the residual block output."""
        return x + self.block(x)


class ConvSequence(nn.Module):
    """
    A convolutional sequence used in the IMPALA CNN backbone.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        """Return the convolutional sequence output."""
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """
    An IMPALA CNN actor-critic network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        latent_dim: Output embedding size after CNN trunk.
        width_scale: Channel width multiplier for the IMPALA CNN backbone.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        latent_dim: int = 256,
        width_scale: int = 4,
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
            nn.ReLU(),
            nn.Linear(n_flatten, latent_dim),
            nn.ReLU(),
        )

        # Initialize the actor head
        self.action_logits = nn.Linear(latent_dim, action_dim)

        # Initialize the critic head
        self.state_value = nn.Linear(latent_dim, 1)

    def _encode(self, state):
        """Return encoded state features."""
        x = state.permute(0, 3, 1, 2).contiguous()
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def forward_actor(self, state):
        """Return action logits."""
        x = self._encode(state)
        return self.action_logits(x)

    def forward_critic(self, state):
        """Return state value."""
        x = self._encode(state)
        return self.state_value(x)

    def forward(self, state):
        """Return action logits and state value."""
        x = self._encode(state)

        action_logits = self.action_logits(x)
        state_value = self.state_value(x)

        return action_logits, state_value
