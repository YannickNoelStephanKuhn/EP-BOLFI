"""!@file
Example usage of the OCV fitting tools.
"""

import matplotlib.pyplot as plt
import numpy as np

from ep_bolfi.utility.dataset_formatting import (
    read_csv_from_measurement_system
)
from ep_bolfi.utility.visualization import fit_and_plot_OCV
from ep_bolfi.utility.fitting_functions import a_fit

"""! Fitted plateau voltages of a graphite anode. """
E_0_g = np.array([0.35973, 0.17454, 0.12454, 0.081957])
"""! Fitted plateau inverse widths of a graphite anode. """
γUeminus1_g = np.array([-0.33144, 8.9434e-3, 7.2404e-2, 6.7789e-2])
"""! Fitted plateau inverse widths (transformed) of a graphite anode. """
a_g = a_fit(γUeminus1_g)
"""! Fitted plateau SOC fractions of a graphite anode. """
Δx_g = np.array([7.9842e-2, 0.230745, 0.29651, 0.39493])
"""! Fit parameters of a graphite anode OCV curve. """
graphite = [p[i] for i in range(4) for p in [E_0_g, a_g, Δx_g]]

"""! Fitted plateau voltages of a graphite anode. """
E_0_g_c = np.array([0.31443, 0.17271, 0.11786, 7.957e-2])
"""! Fitted plateau inverse widths of a graphite anode. """
γUeminus1_g_c = np.array([-0.29742, 1.3365e-2, 7.0613e-2, 6.9386e-2])
"""! Fitted plateau inverse widths (transformed) of a graphite anode. """
a_g_c = a_fit(γUeminus1_g_c)
"""! Fitted plateau SOC fractions of a graphite anode. """
Δx_g_c = np.array([7.8491e-2, 0.17804, 0.333, 0.41182])
"""! Fit parameters of a graphite anode OCV curve. """
graphite_charge = [p[i] for i in range(4) for p in [E_0_g_c, a_g_c, Δx_g_c]]

"""! Fitted plateau voltages of a graphite anode. """
E_0_g_dc = np.array([0.399, 0.17443, 0.12814, 8.357e-2])
"""! Fitted plateau inverse widths of a graphite anode. """
γUeminus1_g_dc = np.array([-0.36296, 5.8974e-3, 7.6006e-2, 6.8124e-2])
"""! Fitted plateau inverse widths (transformed) of a graphite anode. """
a_g_dc = a_fit(γUeminus1_g_dc)
"""! Fitted plateau SOC fractions of a graphite anode. """
Δx_g_dc = np.array([8.1194e-2, 0.28345, 0.26002, 0.37804])
"""! Fit parameters of a graphite anode OCV curve. """
graphite_discharge = [p[i] for i in range(4)
                      for p in [E_0_g_dc, a_g_dc, Δx_g_dc]]

measurement = read_csv_from_measurement_system(
    "./GITT data/L_ACB440_BP_2_OCV.csv", 'iso-8859-1', 0,
    headers={0: "SOC [-]", 2: "U [V]"}, delimiter=',', decimal='.'
    # headers={0: "SOC [-]", 1: "U [V]"}, delimiter=';', decimal=','
)
complete_SOC = np.array(measurement.other_columns["SOC [-]"][0][50:-50])
complete_OCV = np.array(measurement.voltages[0][50:-50])

fig, (ax0, ax1) = plt.subplots(figsize=(12.73, 9), ncols=2)
# fig, ax0 = plt.subplots(figsize=(12.73, 9))
# fit_and_plot_OCV(ax0, complete_SOC, complete_OCV, "graphite",
#                 spline_order=2, phases=8, spline_smoothing=1e-6,  # 2e-5,
#                 inverted=True, parameters_print=True, spline_print='python',
#                 spline_SOC_range=(0.02, 0.99))
# from ep_bolfi.utility.visualization import plot_ICA
# plot_ICA(fig, ax1, complete_SOC, complete_OCV, "graphite",
#         spline_order=5, spline_smoothing=8e-2, sign=1)  # 1e-5
ax0.set_xlabel("SOC  /  -")
ax1.set_xlabel("SOC  /  -")
ax0.set_ylabel("OCV  /  V")
fit_and_plot_OCV(ax0, np.linspace(0, 1, 10), np.linspace(0, 1, 10), "graphite",
                 fit=graphite_discharge, spline_SOC_range=(0.001, 0.995),
                 eval_points=500, spline_order=2, phases=4,
                 spline_smoothing=1e-4, inverted=False, spline_print='matlab',
                 parameters_print=True)
fig.tight_layout()
plt.show()
