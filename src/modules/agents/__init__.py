REGISTRY = {}

from .rnn_agent import RNNAgent
from .random_rnn_agent import RandomRNNAgent
from .enr_rnn_agent import ENRRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["random_rnn"] = RandomRNNAgent
REGISTRY["enr_rnn"] = ENRRNNAgent