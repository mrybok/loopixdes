from loopixdes.env import LoopixEnv
from loopixdes.util import EpisodeSampler

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

    env = LoopixEnv()
    kwargs = episode_sampler.sample()
    kwargs['verbose'] = True
    kwargs['tensorboard'] = True
    state = env.reset(seed=0, options=kwargs)

    for _ in range(10):
        kwargs = episode_sampler.sample()
        params = np.array(list(kwargs['params'].values()))
        state, reward, done, _ = env.step(params)
