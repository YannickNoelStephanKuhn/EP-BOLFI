import matplotlib.pyplot as plt

from utility.visualization import plot_measurement

from parameters.estimation.gitt_basf import (
    gitt_pulses_without_OCV, charge_discharge_cycles
)
from utility.read_csv_datasets import read_channels_from_measurement_system

fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
fig_gitt, ax_gitt = plt.subplots(
    figsize=(4 * 2**0.5, 4), constrained_layout=True
)
plot_measurement(fig, ax, charge_discharge_cycles, "BASF CC-CV data")
plot_measurement(fig_gitt, ax_gitt, gitt_pulses_without_OCV, "BASF GITT Test")
plt.show()
