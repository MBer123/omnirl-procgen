"""Network architectures used by OmniRL algorithms."""

from .impala_cnn import ImpalaCNN
from .sppo_cnn import SPPOCNN


__all__ = ["ImpalaCNN", "SPPOCNN"]
