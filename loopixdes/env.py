from typing import Tuple
from typing import TypeVar
from typing import Optional

from simulator import Simulator

import numpy as np
from gym import Env
from gym.utils.seeding import np_random

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class MixNetEnv(Env):
    """MixNetEnv"""

    def __init__(
            self,
            seed: Optional[int] = None,
            **kwargs
    ):
        self.__kwargs = kwargs
        self.__simulator = None
        self._np_random, _ = np_random(seed)

    def reset(self, seed: Optional[int] = None, **kwargs) -> ObsType:
        self.__kwargs.update(kwargs)

        if seed is not None:
            self._np_random, _ = np_random(seed)

        rng = self._np_random
        self.__simulator = Simulator(rng=rng, **self.__kwargs)

        self.__simulator.warmup()

        obs, _, _, _ = self.__simulator.simulation_step()

        return obs

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, bool, dict]:
        self.__simulator.update_parameters(action)

        return self.__simulator.simulation_step()

    def render(self, mode: str = "human"):
        self.__simulator.render(True, True)

    def close(self):
        del self.__simulator
