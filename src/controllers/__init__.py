REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .enr_controller import ENRMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["random_mac"] = RandomMAC
REGISTRY["enr_mac"] = ENRMAC