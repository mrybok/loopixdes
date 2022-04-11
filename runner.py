from loopixdes.env import LoopixEnv
from loopixdes.util import EpisodeSampler
from loopixdes.defaults import DEFAULT_PARAMS

import numpy as np


if __name__ == "__main__":
    episode_sampler = EpisodeSampler(
        {
            'num_layers': (2, 21),
            'params': {
                'DROP': (5 / 6, 250 / 3),
                'LOOP': (5 / 6, 250 / 3),
                'DELAY': (0.2, 2),
                'PAYLOAD': (15 / 6, 250 / 3),
                'LOOP_MIX': (1.0, 60),
            }
        },
        'data/OCnodeslinks.json',
        np.random.RandomState(0),
        0,
        1082379653
    )

    env = LoopixEnv(0, verbose=True)

    kwargs = episode_sampler.sample()
    print(env.reset(**kwargs))
    print(env.step(np.array(list(DEFAULT_PARAMS.values()))))
