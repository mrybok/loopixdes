import json

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from loopixdes.defaults import NUM
from loopixdes.defaults import DEFAULT_PARAMS
from loopixdes.model.mail import Mail

from numpy.random import RandomState


class EpisodeSampler:

    def __init__(
            self,
            kwargs: Dict,
            data_file: str,
            rng: RandomState,
            min_traces_len: int,
            init_timestamp: Union[int, float]
    ):
        assert isinstance(kwargs, dict), 'kwargs must be dict'
        assert isinstance(data_file, str), 'data_file must be str'
        assert isinstance(min_traces_len, int), 'min_traces_len must be int'
        assert isinstance(init_timestamp, NUM), 'init_timestamp must be number'
        assert isinstance(rng, RandomState), 'rng must be RandomState'

        assert len(data_file) > 5
        assert data_file[-5:] == '.json', 'non-json data_file'
        assert init_timestamp >= 0, 'init_timestamp must be non-negative'

        self.__rng = rng
        self.__kwargs = kwargs
        self.__init_timestamp = init_timestamp
        self.__traces = load_dataset(data_file)

        assert min_traces_len < len(self.__traces), 'too large min_traces_len'

        self.__min_traces_len = min_traces_len

        for key, val in kwargs.items():
            if not isinstance(val, dict):
                validate_range(val)
            else:
                assert key == 'params', 'only params can be dict'
                assert all([param_key in DEFAULT_PARAMS for param_key in val]), 'unknown key'

                for param_value in val.values():
                    validate_range(param_value)

    def __sample_value(
            self,
            val: Union[
                Tuple[int, int],
                Tuple[int, float],
                Tuple[float, int],
                Tuple[float, float],
                List[Union[int, float]]
            ]
    ) -> Union[int, float]:
        if isinstance(val, tuple):
            if any([isinstance(bound, float) for bound in val]):
                return self.__rng.uniform(*val)
            else:
                return self.__rng.randint(*val)
        elif isinstance(val, list):
            return self.__rng.choice(val)

    def sample(self) -> Dict:
        params = {}

        for key, val in self.__kwargs.items():
            if not isinstance(val, dict):
                sampled_value = self.__sample_value(val)
            else:
                sampled_value = {}

                for param_key, param_val in val.items():
                    sampled_value[param_key] = self.__sample_value(param_val)

            params[key] = sampled_value

        traces_idx = self.__rng.randint(0, len(self.__traces) - self.__min_traces_len)
        params['traces'] = self.__traces[traces_idx:]
        shift = params['traces'][0].time
        params['init_timestamp'] = self.__init_timestamp + shift

        for mail in params['traces']:
            mail.time -= shift

        return params


def validate_range(val):
    if isinstance(val, tuple):
        assert len(val) == 2, 'too many values to unpack'
        assert all([isinstance(bound, NUM) for bound in val])

        if any([isinstance(bound, float) for bound in val]):
            assert val[0] <= val[1], 'wrong range'
        else:
            assert val[0] < val[1], 'wrong range'

    elif isinstance(val, list):
        assert len(val) > 0, 'empty list'


def load_dataset(file_name: str) -> List[Mail]:
    with open(file_name, 'r') as file:
        dataset = [Mail(**mail) for mail in json.load(file)]

    return dataset
