from collections import deque

import gym
import numpy as np
from gym import spaces


class EnvAdapter:
    """
    Adapt a Procgen environment behind the environment backend.

    Args:
        env: The underlying Procgen environment.
        env_id: Environment ID used to build environment instances.
        render_mode: Render mode passed to the environment.
        distribution_mode: Distribution mode passed to the environment.
        start_level: First level in the environment level range.
        num_levels: Number of levels in the environment level range.
        sequential_reset: Whether reset should advance through levels in order.
        num_stack: Number of frames to stack.
    """

    def __init__(
        self,
        env,
        env_id,
        render_mode,
        distribution_mode,
        start_level,
        num_levels,
        sequential_reset,
        num_stack=4,
    ):
        self.env = env
        self.env_id = env_id
        self.render_mode = render_mode
        self.distribution_mode = distribution_mode
        self.start_level = start_level
        self.num_levels = num_levels
        self.sequential_reset = sequential_reset
        self.level_index = 0
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        obs_space = self.env.observation_space
        h, w, c = obs_space.shape
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(h, w, c * num_stack),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        return self._observation_space

    def _process_obs(self, obs):
        """Normalize RGB observation."""
        return obs.astype(np.float32) / 255.0

    def _get_stacked_obs(self):
        """Return frames stacked along the channel dimension."""
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if self.sequential_reset:
            level = self.start_level + self.level_index
            self.level_index = (self.level_index + 1) % self.num_levels
            self.env.close()
            self.env = gym.make(
                self.env_id,
                render_mode=self.render_mode,
                distribution_mode=self.distribution_mode,
                start_level=level,
                num_levels=1,
            )

        obs = self.env.reset()
        obs = self._process_obs(obs)

        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self._get_stacked_obs(), {}

    def step(self, action):
        """Execute one environment step."""
        obs, reward, done, info = self.env.step(action)
        obs = self._process_obs(obs)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, done, False, info

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()
