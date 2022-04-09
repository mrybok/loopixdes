from typing import Union

from loopixdes.defaults import EPS

from numpy import array
from numpy import ndarray
from numpy import Infinity
from numpy import concatenate


class Monitor:

    def __init__(self, start_time: float):
        self.e2e = SummaryWrapper()
        self.entropy_mix = SummaryWrapper()
        self.latency_mix = SummaryWrapper()
        self.latency_payload = SummaryWrapper()
        self.entropy_provider = SummaryWrapper()

        self.bandwidth = 0
        self.start_time = start_time

    def reset(self, start_time: float):
        self.e2e.reset()
        self.entropy_mix.reset()
        self.latency_mix.reset()
        self.latency_payload.reset()
        self.entropy_provider.reset()

        self.bandwidth = 0
        self.start_time = start_time

    def get(self, end_time: float) -> ndarray:
        interval = max(end_time - self.start_time, EPS)
        summaries = [array([self.bandwidth / interval])]
        summaries += [self.e2e.get()]
        summaries += [self.latency_payload.get()]
        summaries += [self.latency_mix.get()]
        summaries += [self.entropy_mix.get()]
        summaries += [self.entropy_provider.get()]

        return concatenate(summaries)


class SummaryWrapper:

    num = 0
    max = 0.0
    sum = 0.0
    sum2 = 0.0
    min = Infinity

    def reset(self):
        self.num = 0
        self.max = 0
        self.sum = 0
        self.sum2 = 0
        self.min = Infinity

    def update(self, value: Union[int, float]):
        self.num += 1
        self.max = max(self.max, value)
        self.sum += value
        self.sum2 += value ** 2
        self.min = min(self.min, value)

    def get(self) -> ndarray:
        std = 0.0
        mean = 0.0

        if self.num > 0:
            mean = self.sum / self.num
            std = (self.sum2 / self.num - mean ** 2) ** 0.5

        return array([self.min, mean, self.max, std])
