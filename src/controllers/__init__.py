REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["random_mac"] = RandomMAC