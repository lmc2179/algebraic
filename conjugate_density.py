"""
This module allows the user to construct distributions over parameter values for
"""
import numpy as np
from matplotlib import pyplot as plt
from algebraic_statistic import AbstractCompositeGroupStatistic, Frequency
from scipy.stats import beta

class AbstractConjugateModel(AbstractCompositeGroupStatistic):
    def get_posterior_pdf(self, *args, **kwargs):
        raise NotImplementedError

class Bernoulli(AbstractConjugateModel):
    STATISTIC_CLASSES = [('Frequency', Frequency)]
    def get_posterior_pdf(self, mu):
        a = self.statistic_values['Frequency'].get_frequency(1) + 1
        b = self.statistic_values['Frequency'].get_frequency(0) + 1
        return beta.pdf(mu, a, b)

def sliding_window_hpdi(samples, interval_fraction):
    window_size = int(1.0 * len(samples) * interval_fraction)
    sorted_samples = sorted(samples)
    shortest_window_start = None
    shortest_window_length = float('inf')
    for window_start in range(len(samples) - window_size):
        window_length = sorted_samples[window_start + window_size] - sorted_samples[window_start]
        if window_length < shortest_window_length:
            shortest_window_start = window_start
            shortest_window_length = window_length
    return sorted_samples[shortest_window_start], sorted_samples[shortest_window_start + window_size]

data = np.random.normal(0,1,50000)
print(sliding_window_hpdi(data, 0.68))