from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .random_q_learner import RandomQLearner
from .enr_q_learner import ENRQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["random_q_learner"] = RandomQLearner
REGISTRY["enr_q_learner"] = ENRQLearner