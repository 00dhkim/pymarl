REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .enr_controller import ENRMAC
from .manr_controller import ManrMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["random_mac"] = RandomMAC
REGISTRY["enr_mac"] = ENRMAC
REGISTRY["manr_mac"] = ManrMAC