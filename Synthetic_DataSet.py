import numpy as np
import pynncml as pnc
import torch
import math
import matplotlib
matplotlib.use('TkAgg')  # Fix for zoom-in not working on macOS
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from sklearn import metrics
from torch.utils.data import Subset
import os
import torch.nn as nn
from io import StringIO
import sys
import time

from types import SimpleNamespace

link_metadata_list = [
    {'frequency': 37.24, 'length_m': 1212.79, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 37.324, 'length_m': 1182.27, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 29.2705, 'length_m': 2903.12, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 33.257, 'length_m': 2203.34, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 28.1785, 'length_m': 1068.83, 'polarization': True, 'tsl_value': 3.0},
    {'frequency': 38.584, 'length_m': 1199.26, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 29.2425, 'length_m': 1791.58, 'polarization': True, 'tsl_value': 6.0},
    {'frequency': 28.2485, 'length_m': 1412.13, 'polarization': True, 'tsl_value': 0.0},
    {'frequency': 37.24, 'length_m': 1482.56, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 18.25, 'length_m': 5895.28, 'polarization': True, 'tsl_value': 16.0},
    {'frequency': 29.1865, 'length_m': 2871.01, 'polarization': False, 'tsl_value': 12.0},
    {'frequency': 28.1785, 'length_m': 3933.76, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 38.416, 'length_m': 1545.8, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 28.2625, 'length_m': 809.47, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 37.1535, 'length_m': 1545.8, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 28.1785, 'length_m': 1087.46, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 38.584, 'length_m': 1066.01, 'polarization': True, 'tsl_value': 11.0},
    {'frequency': 28.2065, 'length_m': 759.17, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 32.417, 'length_m': 2524.95, 'polarization': True, 'tsl_value': 9.0},
    {'frequency': 33.145, 'length_m': 3320.4, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 28.2345, 'length_m': 1791.58, 'polarization': True, 'tsl_value': 6.0},
    {'frequency': 32.361, 'length_m': 1261.66, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 22.134, 'length_m': 5818.45, 'polarization': True, 'tsl_value': 16.0},
    {'frequency': 29.1865, 'length_m': 1060.17, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 29.1865, 'length_m': 3933.76, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 29.2145, 'length_m': 759.17, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 29.2705, 'length_m': 947.0, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 37.324, 'length_m': 2359.74, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 29.2145, 'length_m': 3157.45, 'polarization': True, 'tsl_value': 17.0},
    {'frequency': 32.333, 'length_m': 3320.4, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 33.229, 'length_m': 2524.95, 'polarization': True, 'tsl_value': 9.0},
    {'frequency': 32.333, 'length_m': 3786.85, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 37.324, 'length_m': 885.3, 'polarization': True, 'tsl_value': 0.0},
    {'frequency': 28.1785, 'length_m': 2871.01, 'polarization': False, 'tsl_value': 12.0},
    {'frequency': 38.584, 'length_m': 2359.74, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 33.257, 'length_m': 2792.0, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 37.324, 'length_m': 1282.98, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 33.257, 'length_m': 3694.54, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 37.268, 'length_m': 793.11, 'polarization': True, 'tsl_value': -1.0},
    {'frequency': 29.2425, 'length_m': 3900.91, 'polarization': True, 'tsl_value': 15.0},
    {'frequency': 32.445, 'length_m': 1156.59, 'polarization': True, 'tsl_value': 1.0},
    {'frequency': 38.5, 'length_m': 1482.56, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 29.1865, 'length_m': 1068.83, 'polarization': True, 'tsl_value': 3.0},
    {'frequency': 38.528, 'length_m': 1780.87, 'polarization': True, 'tsl_value': 8.0},
    {'frequency': 38.584, 'length_m': 885.3, 'polarization': True, 'tsl_value': 0.0},
    {'frequency': 38.318, 'length_m': 1282.98, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 23.142, 'length_m': 5818.45, 'polarization': True, 'tsl_value': 16.0},
    {'frequency': 29.2705, 'length_m': 809.47, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 37.24, 'length_m': 1075.98, 'polarization': True, 'tsl_value': 3.0},
    {'frequency': 32.333, 'length_m': 2748.68, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 29.1865, 'length_m': 1087.46, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 33.145, 'length_m': 3786.85, 'polarization': True, 'tsl_value': 13.0},
    {'frequency': 28.2065, 'length_m': 3157.45, 'polarization': True, 'tsl_value': 17.0},
    {'frequency': 37.324, 'length_m': 1199.26, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 28.2625, 'length_m': 947.0, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 32.445, 'length_m': 3694.54, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 28.1785, 'length_m': 1060.17, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 38.528, 'length_m': 1741.62, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 38.528, 'length_m': 793.11, 'polarization': True, 'tsl_value': -1.0},
    {'frequency': 33.257, 'length_m': 1156.59, 'polarization': True, 'tsl_value': 1.0},
    {'frequency': 33.145, 'length_m': 2748.68, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 19.26, 'length_m': 5895.28, 'polarization': True, 'tsl_value': 16.0},
    {'frequency': 32.445, 'length_m': 2203.34, 'polarization': True, 'tsl_value': 10.0},
    {'frequency': 37.324, 'length_m': 1066.01, 'polarization': True, 'tsl_value': 11.0},
    {'frequency': 37.24, 'length_m': 884.26, 'polarization': True, 'tsl_value': -1.0},
    {'frequency': 33.173, 'length_m': 1261.66, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 29.2565, 'length_m': 1412.13, 'polarization': True, 'tsl_value': 0.0},
    {'frequency': 37.268, 'length_m': 1780.87, 'polarization': True, 'tsl_value': 8.0},
    {'frequency': 38.5, 'length_m': 1212.79, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 38.528, 'length_m': 1205.65, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 38.472, 'length_m': 1049.43, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 28.2625, 'length_m': 2903.12, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 37.268, 'length_m': 1205.65, 'polarization': True, 'tsl_value': 4.0},
    {'frequency': 37.268, 'length_m': 1741.62, 'polarization': True, 'tsl_value': 5.0},
    {'frequency': 32.445, 'length_m': 2792.0, 'polarization': True, 'tsl_value': 12.0},
    {'frequency': 38.5, 'length_m': 884.26, 'polarization': True, 'tsl_value': -1.0},
    {'frequency': 38.5, 'length_m': 1075.98, 'polarization': True, 'tsl_value': 3.0},
    {'frequency': 28.2345, 'length_m': 3900.91, 'polarization': True, 'tsl_value': 15.0},
    {'frequency': 37.212, 'length_m': 1049.43, 'polarization': True, 'tsl_value': 2.0},
    {'frequency': 38.584, 'length_m': 1182.27, 'polarization': True, 'tsl_value': 10.0}
]

attenuation_table = {
    18:  (0.07078, 1.0818, 0.07708, 1.0025),
    19:  (0.08084, 1.0691, 0.08642, 0.9930),
    20:  (0.09164, 1.0568, 0.09611, 0.9847),
    21:  (0.1032,  1.0447, 0.1063,  0.9771),
    22:  (0.1155,  1.0329, 0.1170,  0.9700),
    23:  (0.1286,  1.0214, 0.1284,  0.9630),
    24:  (0.1425,  1.0101, 0.1404,  0.9561),
    25:  (0.1571,  0.9991, 0.1533,  0.9491),
    26:  (0.1724,  0.9884, 0.1669,  0.9421),
    27:  (0.1884,  0.9780, 0.1813,  0.9349),
    28:  (0.2051,  0.9679, 0.1964,  0.9277),
    29:  (0.2224,  0.9580, 0.2124,  0.9203),
    30:  (0.2403,  0.9485, 0.2291,  0.9129),
    31:  (0.2588,  0.9392, 0.2465,  0.9055),
    32:  (0.2778,  0.9302, 0.2646,  0.8981),
    33:  (0.2972,  0.9214, 0.2833,  0.8907),
    34:  (0.3171,  0.9129, 0.3026,  0.8834),
    35:  (0.3374,  0.9047, 0.3224,  0.8761),
    36:  (0.3580,  0.8967, 0.3427,  0.8690),
    37:  (0.3789,  0.8890, 0.3633,  0.8621),
    38:  (0.4001,  0.8816, 0.3844,  0.8552),
    39:  (0.4215,  0.8743, 0.4058,  0.8486),
}

# Parameters
N_SAMPLES = 92 * 24 * 60 * 6  # 92 days × 8640 samples/day = 794880

# noinspection PyShadowingNames
class SyntheticLink:
    def __init__(self, tsl, meta_data):
        self.rsl = None
        self.tsl = tsl
        self.rain_rate = None        # R(t) - exponential
        self.attenuation = None      # A(t) = a * R(t)^b * L
        self.rain_rate_15min = None  # Down sampled to 15-minute averages
        self.meta_data = meta_data

def get_a_b(frequency, polarization):
    f_rounded = int(round(frequency))  # round to nearest integer
    if f_rounded not in attenuation_table:
        raise ValueError(f"Frequency {frequency} not in attenuation table.")

    a_H, b_H, a_V, b_V = attenuation_table[f_rounded]
    if polarization:  # vertical
        return a_V, b_V
    else:             # horizontal
        return a_H, b_H

from scipy.ndimage import gaussian_filter1d


def generate_synthetic_rain_rate(size,
                                 rain_probability=0.1025,
                                 target_mean=0.13,
                                 min_event_len=30,
                                 max_event_len=1080):

    rain_rate = np.zeros(size, dtype=np.float32)
    rng = np.random.default_rng()

    total_wet_samples = int(size * rain_probability)
    mean_rain = target_mean / rain_probability

    wet_samples = 0
    while wet_samples < total_wet_samples:
        event_len = rng.integers(min_event_len, max_event_len + 1)
        if wet_samples + event_len > total_wet_samples:
            event_len = total_wet_samples - wet_samples
        start = rng.integers(0, size - event_len)
        if np.any(rain_rate[start:start + event_len] > 0):
            continue
        scale = mean_rain * (10 ** rng.uniform(-0.85, 1.2))  # ≈ scale factor in [0.1, ~31.6]
        raw_vals = rng.exponential(scale=scale, size=event_len)

        # Apply EMA smoothing
        alpha = 0.95
        rain_vals = np.empty_like(raw_vals)
        rain_vals[0] = raw_vals[0]
        for i in range(1, event_len):
            rain_vals[i] = alpha * rain_vals[i - 1] + (1 - alpha) * raw_vals[i]

        # Smooth the start/end of the rain event with a window
        taper_ratio = 0.10  # 10% fade-in and fade-out
        taper_len = max(2, int(event_len * taper_ratio))

        if 2 * taper_len >= event_len:
            window = np.hanning(event_len)
        else:
            fade = np.hanning(taper_len * 2)
            fade_in = fade[:taper_len]
            fade_out = fade[taper_len:]
            flat = np.ones(event_len - 2 * taper_len)
            window = np.concatenate([fade_in, flat, fade_out])

        rain_vals *= window
        rain_vals = np.clip(rain_vals, 0, 50)

        rain_rate[start:start + event_len] = rain_vals
        wet_samples += event_len

    return rain_rate



def downsample_rain_rate_15min(rain_rate_10sec: np.ndarray, factor: int = 90) -> np.ndarray:
    """
    Downsample rain rate from 10-second resolution to 15-minute resolution (90 samples).
    """
    rain_rate_15min = rain_rate_10sec.reshape(-1, factor).mean(axis=1)
    return rain_rate_15min
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
# Generate all 80 synthetic links
synthetic_dataset = []
for metadata in link_metadata_list:
    tsl = np.full(N_SAMPLES, metadata['tsl_value'], dtype=np.float32)

    meta_data = SimpleNamespace(
        length=metadata['length_m'] / 1000.0,  # back to km
        frequency=metadata['frequency'],
        polarization=metadata['polarization']
    )

    link = SyntheticLink(tsl=tsl, meta_data=meta_data)
    synthetic_dataset.append(link)

for link in synthetic_dataset:
    link.rain_rate = generate_synthetic_rain_rate(794880)
    link.rain_rate_15min = downsample_rain_rate_15min(link.rain_rate)

for link in synthetic_dataset:
    a, b = get_a_b(link.meta_data.frequency, link.meta_data.polarization)
    L = link.meta_data.length  # in km

    # Calculate A(t) = a * R(t)^b * L
    link.attenuation = a * (link.rain_rate ** b) * L

for link in synthetic_dataset:
    # TSL is constant: use first sample or broadcast scalar
    tsl_value = link.tsl[0] if isinstance(link.tsl, np.ndarray) else float(link.tsl)
    link.rsl = tsl_value - link.attenuation

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

# Pick a random link and get its index
random_index = np.random.randint(0, len(synthetic_dataset))
link = synthetic_dataset[random_index]

# Plot sample points
plt.figure(figsize=(8, 4))
plt.plot(link.rain_rate, link.attenuation, '.', markersize=4, alpha=0.5, label='samples')

# Extract attenuation parameters
a, b = get_a_b(link.meta_data.frequency, link.meta_data.polarization)
L = link.meta_data.length

# Plot theoretical curve
R_theory = np.linspace(0, link.rain_rate.max(), 300)
A_theory = a * (R_theory ** b) * L
plt.plot(R_theory, A_theory, 'k--', linewidth=0.4, label=r'$A = aR^{b}L$')

# Labels and legend
plt.xlabel(r'Rain rate (mm h$^{-1}$)')
plt.ylabel('Attenuation (dB)')
plt.title(f'Link #{random_index} — freq={link.meta_data.frequency:.2f} GHz, pol={"V" if link.meta_data.polarization else "H"}')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()  # <-- this line keeps the plot open

# Second plot: RSL and TSL over time (every 6th sample)
plt.figure(figsize=(12, 5))
plt.plot(link.rsl[::], label='RSL', color='blue')
plt.plot(link.tsl[::], label='TSL', color='red')

plt.xlabel('Time [10 sec intervals]')
plt.ylabel('Signal Level [dB]')
plt.title(f'RSL and TSL measurements — Link #{random_index}, freq={link.meta_data.frequency:.2f} GHz, pol={"V" if link.meta_data.polarization else "H"}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print statistics
rain_rate_all = np.concatenate([link.rain_rate for link in synthetic_dataset])
param = scipy.stats.expon.fit(rain_rate_all, floc=0.0)

print("Rain Rate Statistics")
print("Mean[mm/hr]:", np.mean(rain_rate_all))
print("Std[mm/hr]:", np.std(rain_rate_all))
print("Percentage of wet samples:", 100 * np.sum(rain_rate_all > 0) / rain_rate_all.size)
print("Percentage of dry samples:", 100 * np.sum(rain_rate_all == 0) / rain_rate_all.size)
print("Exponential Distribution Parameters:", param)

x = np.linspace(0, np.max(rain_rate_all), 1000)
pdf = scipy.stats.expon.pdf(x, *param)

# Plot histogram and exponential PDF
plt.figure(figsize=(8, 6))
plt.hist(rain_rate_all, bins=300, density=True, alpha=0.6, label="Synthetic Rain Rate Histogram")
plt.plot(x, pdf, label=f"Fitted Exponential (λ = {1/param[1]:.2f})", linewidth=2)
plt.title("Rain Rate Histogram")
plt.xlabel("Rain Rate [mm/hr]")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()  # <-- this line keeps the plot open

import pickle

with open("synthetic_dataset.pkl", "wb") as f:
    pickle.dump(synthetic_dataset, f)


