from loopixdes.util import load_dataset
from loopixdes.simulator import Simulator

import numpy as np


if __name__ == "__main__":
    traces = load_dataset('data/OCnodeslinks.json')

    sim = Simulator(
        traces,
        verbose=True,
        rng=np.random.RandomState(0),
    )

    sim.warmup()
    sim.simulation_step()
