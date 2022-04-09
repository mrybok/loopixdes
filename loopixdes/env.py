from typing import Tuple
from typing import TypeVar
from typing import Optional

from model.mail import Mail
from simulator import Simulator

import json
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
            min_dataset_size: int = 0,
            **kwargs
    ):
        self.__kwargs = kwargs
        self.__simulator = None
        self._np_random, _ = np_random(seed)
        self.__min_dataset_size = min_dataset_size

        with open('../data/train_set.json', 'r') as file:
            self.__traces = [Mail(**mail) for mail in json.load(file)]

        assert min_dataset_size >= 0, 'min_dataset_size must be non-negative'
        assert min_dataset_size < len(self.__traces), 'not enough mails in the dataset'

    def reset(self, seed: Optional[int] = None, **kwargs) -> ObsType:
        self.__kwargs.update(kwargs)

        if seed is not None:
            self._np_random, _ = np_random(seed)

        rng = self._np_random
        max_idx = len(self.__traces) - self.__min_dataset_size
        start_idx = self._np_random.randint(0, max_idx)
        traces = self.__traces[start_idx:]
        self.__simulator = Simulator(traces, rng=rng, **self.__kwargs)

        self.__simulator.warmup()

        obs, _, _, _ = self.__simulator.simulation_step()

        return obs

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, bool, dict]:
        self.__simulator.update_parameters(action)

        return self.__simulator.simulation_step()

    def render(self, mode: str = "human"):
        self.__simulator.render(True, True)

    def close(self):
        del self.__traces
        del self.__simulator
