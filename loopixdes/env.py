from typing import Dict
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Optional

from loopixdes.defaults import EPS
from loopixdes.simulator import Simulator

from gym import Env
from gym import spaces
from gym.utils.seeding import np_random

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
RewType = TypeVar("RewType")


class LoopixEnv(Env):
    """LoopixEnv"""

    metadata = {"render_modes": ["tqdm", "tensorboard"]}

    reward_range = spaces.Dict({
        "latency_payload": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "bandwidth": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "e2e_anonymity_min": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "e2e_anonymity_mean": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "e2e_anonymity_max": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "e2e_anonymity_std": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "latency_mix_min": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "latency_mix_mean": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "latency_mix_max": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "latency_mix_std": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_mix_min": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_mix_mean": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_mix_max": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_mix_std": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_provider_min": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_provider_mean": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_provider_max": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
        "entropy_provider_std": spaces.Box(low=0.0, high=float('inf'), shape=(1,)),
    })

    action_space = spaces.Dict({
        'DROP': spaces.Box(low=EPS, high=5000.0, shape=(1,)),
        'LOOP': spaces.Box(low=EPS, high=5000.0, shape=(1,)),
        'DELAY': spaces.Box(low=EPS, high=5000.0, shape=(1,)),
        'PAYLOAD': spaces.Box(low=EPS, high=5000.0, shape=(1,)),
        'LOOP_MIX': spaces.Box(low=EPS, high=5000.0, shape=(1,)),
    })

    observation_space = spaces.Dict({
        "num_layers": spaces.Discrete(21),
        "nodes_per_layer": spaces.Discrete(21),
        "num_providers": spaces.Discrete(21),
        "plaintext_size": spaces.Discrete(8192),
        "start_sin_day": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "start_cos_day": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "start_sin_week": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "start_cos_week": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "end_sin_day": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "end_cos_day": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "end_sin_week": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "end_cos_week": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        **action_space,
        **reward_range
    })

    __options = {}
    _np_random = None
    __simulator = None

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[Dict] = None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        self.__options.update(**options)

        if seed is not None:
            self._np_random, _ = np_random(seed)

        rng = self._np_random
        self.__simulator = Simulator(rng=rng, **self.__options)

        self.__simulator.warmup()

        obs, _, _, _ = self.__simulator.simulation_step()

        if return_info:
            return obs, {}

        return obs

    def step(self, action: ActType) -> Tuple[ObsType, RewType, bool, dict]:
        self.__simulator.update_parameters(action)

        return self.__simulator.simulation_step()

    def render(self, mode: str = "tqdm"):
        assert mode in self.metadata['render_modes'], 'unknown mode'

        self.__simulator.render(mode == "tqdm", mode == "tensorboard")

    def close(self):
        del self.__simulator
