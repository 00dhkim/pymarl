REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .sisr_runner import SISRRunner
REGISTRY["sisr"] = SISRRunner

from .random_episode_runner import RandomEpisodeRunner
REGISTRY["random_episode"] = RandomEpisodeRunner