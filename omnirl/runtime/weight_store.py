import ray
import torch


@ray.remote(num_cpus=1)
class WeightStore:
    """
    A lightweight Ray actor for storing and serving model weights.

    Args:
        init_state_dict: Initial model parameters.
        version: Initial model version.
    """

    def __init__(self, init_state_dict: dict[str, torch.tensor], version: int):
        # Initialize the version and state dict
        self.version = version
        self.state_dict = init_state_dict

    def set(self, state_dict, version):
        """Update  weights and increase the version number."""
        self.version = version
        self.state_dict = state_dict

    def get(self, old_version):
        """Return the latest weights if a newer version exists."""
        if self.version > old_version:
            return {"version": self.version, "weights": self.state_dict}
        return {"version": self.version, "weights": None}
