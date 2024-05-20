REGISTRY = {}

from .rnn_agent import RNNAgent
from .random_rnn_agent import RandomRNNAgent
from .enr_rnn_agent import ENRRNNAgent
from .manr_rnn_agent import ManrRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["random_rnn"] = RandomRNNAgent
REGISTRY["enr_rnn"] = ENRRNNAgent
REGISTRY["manr_rnn"] = ManrRNNAgent