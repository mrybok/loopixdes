import json

from simulator import Simulator

from model.mail import Mail

if __name__ == "__main__":
    with open('../../data/sample.json', 'r') as file:
        traces = json.load(file)

    traces = [Mail(**mail) for mail in traces]

    sim = Simulator(
        traces, verbose=True
    )
    sim.warmup()
    # sim.simulation_step(5e6)
    sim.simulation_step(1000)
