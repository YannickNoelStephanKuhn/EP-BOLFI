"""!@file
Example usage of the general visualization tool.
"""

# import matplotlib
import matplotlib.pyplot as plt

from ep_bolfi.utility.visualization import plot_measurement

from parameters.estimation.gitt_basf import (
    charge_discharge_cycles, gitt_pulses, gitt_pulses_without_OCV
)

# "constrained_layout=True" is the better replacement for "tight_layout"
# here. It ensures that the colorbar and second y-axis don't overlap.
fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
fig_gitt, ax_gitt = plt.subplots(
    figsize=(4 * 2**0.5, 4), constrained_layout=True
)
fig_preprocessed, ax_preprocessed = plt.subplots(
    figsize=(4 * 2**0.5, 4), constrained_layout=True
)
fig
plot_measurement(fig, ax, charge_discharge_cycles, "Preceding CC-CV data")
plot_measurement(
    fig_gitt, ax_gitt, gitt_pulses, "GITT data"
)
plot_measurement(
    fig_preprocessed,
    ax_preprocessed,
    gitt_pulses_without_OCV,
    "GITT data with OCV curve removed",
)
ax_preprocessed.set_ylabel("Overpotential  /  V")

plt.show()
