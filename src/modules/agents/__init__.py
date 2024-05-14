REGISTRY = {}

from .rnn_agent import RNNAgent
from .random_rnn_agent import RandomRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["random_rnn"] = RandomRNNAgent