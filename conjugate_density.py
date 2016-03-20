# HPDI spike
import numpy as np
from matplotlib import pyplot as plt
from density_model import AbstractDensityModel


class AbstractConjugateDensity(AbstractDensityModel):
    pass

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