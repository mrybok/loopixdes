import json

from loopixdes.simulator import Simulator
from loopixdes.model.mail import Mail

if __name__ == "__main__":
    with open('data/OCnodeslinks.json', 'r') as file:
        traces = json.load(file)

    traces = [Mail(**mail) for mail in traces]

    sim = Simulator(
        traces,
        verbose=True,
        tensorboard=True,
        logging_rate=20000,
        update_rate=int(1e6),
        init_timestamp=1080072715,
    )

    sim.warmup()
    sim.simulation_step()
