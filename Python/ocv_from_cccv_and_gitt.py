import json
import matplotlib.pyplot as plt
import numpy as np

from utility.visualization import fit_and_plot_OCV, set_fontsize
from utility.fitting_functions import (
    smooth_fit, inverse_OCV_fit_function, inverse_d2_dSOC2_OCV_fit_function,
    a_fit
)

from parameters.estimation.gitt_basf import charge_discharge

# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
#     figsize=(8 * 2**0.5, 8), ncols=2, nrows=2, tight_layout=True
# )
fig0, ax0 = plt.subplots(figsize=(4 * 2**0.5, 4), tight_layout=True)
fig1, ax1 = plt.subplots(figsize=(4 * 2**0.5, 4), tight_layout=True)
fig2, ax2 = plt.subplots(figsize=(4 * 2**0.5, 4), tight_layout=True)
fig3, ax3 = plt.subplots(figsize=(4 * 2**0.5, 4), tight_layout=True)

ax0.plot(np.array(charge_discharge.timepoints[0]) / 3600,
         charge_discharge.voltages[0], label="CC charge")
ax0.plot(np.array(charge_discharge.timepoints[1]) / 3600,
         charge_discharge.voltages[1], label="CV charge")
ax0.plot(np.array(charge_discharge.timepoints[2]) / 3600,
         charge_discharge.voltages[2], label="CC discharge")
ax0.set_xlabel("Time since start of cycling  /  h")
ax0.set_ylabel("Cell voltage  /  V")
ax0.legend()

CC_charge_I = [0.5 * (I0 + I1) for (I0, I1)
               in zip(charge_discharge.currents[0][:-1],
                      charge_discharge.currents[0][1:])]
CC_charge_Δt = [t1 - t0 for (t0, t1)
                in zip(charge_discharge.timepoints[0][:-1],
                       charge_discharge.timepoints[0][1:])]
CC_charge_C = np.append(
    [0.0], np.cumsum(np.array(CC_charge_I) * np.array(CC_charge_Δt))
)
CC_charge_C = CC_charge_C - CC_charge_C[-1]
CC_charge_V = charge_discharge.voltages[0]

CV_I = [0.5 * (I0 + I1) for (I0, I1)
        in zip(charge_discharge.currents[1][:-1],
               charge_discharge.currents[1][1:])]
CV_Δt = [t1 - t0 for (t0, t1)
         in zip(charge_discharge.timepoints[1][:-1],
                charge_discharge.timepoints[1][1:])]
CV_C = np.abs(np.sum(np.array(CV_I) * np.array(CV_Δt)))

CC_discharge_I = [0.5 * (I0 + I1) for (I0, I1)
                  in zip(charge_discharge.currents[2][:-1],
                         charge_discharge.currents[2][1:])]
CC_discharge_Δt = [t1 - t0 for (t0, t1)
                   in zip(charge_discharge.timepoints[2][:-1],
                          charge_discharge.timepoints[2][1:])]
CC_discharge_C = np.append(
    [0.0], np.cumsum(np.array(CC_discharge_I) * np.array(CC_discharge_Δt))
)
CC_discharge_V = charge_discharge.voltages[2]

with open('../GITT estimation results/L_ACB440_BP_2_OCV.json') as f:
    gitt_terminal = json.load(f)
gitt_C = 31.75 + np.array(gitt_terminal["Cell SOC [C]"])
gitt_V = gitt_terminal["Cell OCV [V]"]
smoothing_factor = 1e-5
CC_charge_spline = smooth_fit(CC_charge_C + CV_C, CC_charge_V,
                              smoothing_factor=smoothing_factor)
gitt_spline = smooth_fit(gitt_C, gitt_V, smoothing_factor=smoothing_factor)
CC_discharge_spline = smooth_fit(CC_discharge_C - CV_C, CC_discharge_V,
                                 smoothing_factor=smoothing_factor)

C_eval = np.linspace(
    np.max([np.min(CC_charge_C + CV_C), np.min(gitt_C),
            np.min(CC_discharge_C - CV_C)]),
    np.min([np.max(CC_charge_C + CV_C), np.max(gitt_C),
            np.max(CC_discharge_C - CV_C)]),
    400
)
qd2OCV = (CC_charge_spline(C_eval) + CC_discharge_spline(C_eval)
          - 2 * gitt_spline(C_eval))

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

ax1.plot(CC_charge_C + CV_C, CC_charge_V,
         label="CC charge (moved right by CV charge)")
ax1.plot(gitt_C, gitt_V, label="GITT (aligned with CC curves)")
ax1.plot(C_eval, gitt_spline(C_eval), label="GITT (smoothing spline)", ls='--')
ax1.plot(CC_discharge_C - CV_C, CC_discharge_V,
         label="CC discharge (moved left by CV charge)")
ax1.set_xlabel("Capacity moved  /  C")
ax1.set_ylabel("Cell voltage  /  V")
ax1.legend()

SOC_range = (0.07, 0.67)
SOC_scale = 600
SOC_offset = 0.01

ax2.plot(C_eval, qd2OCV, label="CC charge + CC discharge - 2 * GITT")
ax2.plot(C_eval, inverse_d2_dSOC2_OCV_fit_function(
        np.linspace(*SOC_range, 400), *graphite
    )[::-1] / SOC_scale - SOC_offset,
    label="2nd derivative of graphite OCV (/ " + str(SOC_scale) + ", - "
    + str(SOC_offset) + ")")
ax2.set_xlabel("Capacity moved  /  C")
ax2.set_ylabel("Voltage  /  V")
ax2.legend()

OCV_anode = inverse_OCV_fit_function(
    np.linspace(*SOC_range, 400), *graphite
)[::-1]
OCV_cathode = gitt_spline(C_eval) + OCV_anode
phases = 6
fit_and_plot_OCV(
    ax3, C_eval, OCV_cathode, "NCM_851005", phases, z=1.0, T=298.15,
    eval_SOC=[0.15, 1.00], spline_smoothing=1e-5, spline_print='python',
    spline_order=0, parameters_print=True, info_accuracy=False,
    normalized_xaxis=True
)
ax3.set_xlabel(r"$\mathrm{{SOC}}_\mathrm{{p}}$  /  -")
ax3.set_ylabel("Positive electrode OCV $U_p$  /  V")
ax3.set_title("")
ax3.set_xlim([-0.05, 1.05])

fontsize = 13
set_fontsize(ax0, fontsize, fontsize, fontsize, fontsize, fontsize, fontsize)
set_fontsize(ax1, fontsize, fontsize, fontsize, fontsize, fontsize, fontsize)
set_fontsize(ax2, fontsize, fontsize, fontsize, fontsize, fontsize, fontsize)
set_fontsize(ax3, fontsize, fontsize, fontsize, fontsize, fontsize, fontsize)
fig1.savefig('../OCV_extraction_data.pdf', pad_inches=0.0)
fig2.savefig('../OCV_extraction_alignment.pdf', pad_inches=0.0)
fig3.savefig('../OCV_extraction_result.pdf', pad_inches=0.0)
plt.show()
