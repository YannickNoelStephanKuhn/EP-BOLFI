import numpy as np
from scipy.optimize import minimize
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import json

from utility.fitting_functions import (
    inverse_d_dSOC_OCV_fit_function
)
from utility.preprocessing import (
    calculate_both_SOC_from_OCV, calculate_means_and_standard_deviations
)
from utility.visualization import set_fontsize

from parameters.estimation.gitt_basf import (
    parameters, gitt_pulses,
    positive_SOC_from_cell_SOC, negative_SOC_from_cell_SOC
)
from parameters.models.basf_gitt_test_cell import (
    positive_electrode_OCV_fit
)

small_fig_fontsize = 15


def standard_butler_volmer(soc, i_00=1.0, alpha=0.5):
    return i_00 * 1000.0**0.5 * soc**(1 - alpha) * (1 - soc)**alpha


with open('../GITT estimation results/estimation_results.json', 'r') as f:
    estimation_results = json.load(f)
with open(
    '../GITT estimation verification/estimation_verification.json', 'r'
) as f:
    estimation_verification = json.load(f)
with open(
    '../GITT estimation results/experimental_features_at_each_pulse.json',
    'r'
) as f:
    experimental_features = json.load(f)
with open(
    '../GITT estimation results/simulated_features_at_each_pulse.json', 'r'
) as f:
    evaluations = json.load(f)
with open(
    '../GITT estimation results/boundaries_of_simulated_features.json', 'r'
) as f:
    sensitivities = json.load(f)
with open('../GITT estimation results/L_ACB440_BP_2_OCV.json', 'r') as f:
    terminal_OCVs = json.load(f)["Cell OCV [V]"][:85]

positive_SOCs = []
negative_SOCs = []
for ocv in terminal_OCVs:
    calculate_both_SOC_from_OCV(parameters, negative_SOC_from_cell_SOC,
                                positive_SOC_from_cell_SOC, ocv)
    positive_SOCs.append(
        parameters["Initial concentration in positive electrode [mol.m-3]"]
        / parameters["Maximum concentration in positive electrode [mol.m-3]"]
    )
    negative_SOCs.append(
        parameters["Initial concentration in negative electrode [mol.m-3]"]
        / parameters["Maximum concentration in negative electrode [mol.m-3]"]
    )
positive_SOCs = np.array(positive_SOCs)
positive_SOCs_for_ICI = positive_SOCs[1:]
positive_SOCs_for_GITT = positive_SOCs[:-1]

currents = []
for current in gitt_pulses.currents[::2]:
    currents.append(np.mean(current))

gitt_diffusivities = {"data": [], "sim": []}
for soc, I, sqrt_slope_exp_inv, sqrt_slope_sim_inv in zip(
        positive_SOCs_for_GITT,
        currents,
        experimental_features['GITT square root slope (pulse #0)'],
        evaluations['GITT square root slope (pulse #0)']):
    ocv_slope = inverse_d_dSOC_OCV_fit_function(
        soc, *positive_electrode_OCV_fit, z=1.0, T=298.15, inverted=True
    ) / parameters["Maximum concentration in positive electrode [mol.m-3]"]
    gitt_diffusivities["data"].append(4 / np.pi * (
        I * ocv_slope * sqrt_slope_exp_inv / (
            96485.33212
            * parameters["Current collector perpendicular area [m2]"]
        )
    )**2)
    gitt_diffusivities["sim"].append(4 / np.pi * (
        I * ocv_slope * sqrt_slope_sim_inv / (
            96485.33212
            * parameters["Current collector perpendicular area [m2]"]
        )
    )**2)

ici_diffusivities = {"data": [], "sim": []}
for soc, I, sqrt_slope_exp_inv, sqrt_slope_sim_inv in zip(
        positive_SOCs_for_ICI,
        currents,
        experimental_features['ICI square root slope (pulse #0)'],
        evaluations['ICI square root slope (pulse #0)']):
    ocv_slope = inverse_d_dSOC_OCV_fit_function(
        soc, *positive_electrode_OCV_fit, z=1.0, T=298.15, inverted=True
    ) / parameters["Maximum concentration in positive electrode [mol.m-3]"]
    ici_diffusivities["data"].append(4 / np.pi * (
        I * ocv_slope * sqrt_slope_exp_inv / (
            96485.33212
            * parameters["Current collector perpendicular area [m2]"]
        )
    )**2)
    ici_diffusivities["sim"].append(4 / np.pi * (
        I * ocv_slope * sqrt_slope_sim_inv / (
            96485.33212
            * parameters["Current collector perpendicular area [m2]"]
        )
    )**2)

"""
matplotlib.rcParams.update({'font.size': 13})
for i, (soc, diffusivities) in enumerate(zip(
        [positive_SOCs_for_GITT, positive_SOCs_for_ICI],
        [gitt_diffusivities, ici_diffusivities])):
    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
    ax.semilogy(soc, diffusivities["data"], marker='1', lw=0, ms=10
                label=["GITT", "ICI"][i] + " feature (data)")
    ax.semilogy(soc, diffusivities["sim"], marker='2', lw=0, ms=10,
                label=["GITT", "ICI"][i] + " feature (sim)")
    trans = matplotlib.transforms.ScaledTranslation(
        10/72, -5/72, fig.dpi_scale_trans
    )
    ax.text(0.0, 1.0, ['(a)', '(b)'][i], transform=ax.transAxes + trans,
            verticalalignment='top', fontsize=small_fig_fontsize,)
    ax.set_xlabel("Positive electrode SOCₚ  /  -")
    ax.set_ylabel("$D_p^*$  /  $m^2/s$")
    ax.legend(loc='lower center')
    fig.savefig('../analytic_' + ["GITT", "ICI"][i] + '.pdf',
                bbox_inches='tight', pad_inches=0.0)
"""

matplotlib.rcParams.update({'font.size': 13})
labels = {
    "discharge relaxation time": "τᵣ",
    "ohmic voltage drop": "R",
    "GITT square root slope": "GITT-√",
    "concentration overpotential": "η",
    "ICI square root slope": "ICI-√",
}
fig_dev, ax_dev = plt.subplots(figsize=(5 * 2**0.5, 5))
fig_comp, (ax_comp, ax_dev_comp) = plt.subplots(figsize=(8 * 2**0.5, 5),
                                                ncols=2)
fig_comp_cloud, (ax_comp_cloud, ax_dev_comp_cloud) = plt.subplots(
    figsize=(8 * 2**0.5, 5), ncols=2
)
list_of_dev_labels = []
for i, f_name in enumerate(experimental_features.keys()):
    deviation = (
        np.abs(np.array(experimental_features[f_name])
               - np.array(evaluations[f_name]))
        / np.abs(np.array(experimental_features[f_name]))
    )
    log_deviation = np.log(deviation)
    ax_dev.plot(range(len(experimental_features[f_name])),
                deviation, label=labels[f_name[:-11]], ls='None', marker='1')
    log_mean = np.mean(log_deviation)
    log_var = np.var(log_deviation)
    log_std = np.sqrt(log_var)
    ax_dev_comp.errorbar(
        [i], [np.exp(log_mean)],
        [[np.exp(log_mean) - np.exp(log_mean - log_std)],
         [np.exp(log_mean + log_std) - np.exp(log_mean)]],
        marker='o', capsize=8, lw=4, ms=15, capthick=4, ls='None',
        label=labels[f_name[:-11]]
    )
    ax_dev_comp_cloud.plot(
        [i] * len(deviation),
        deviation,
        ls='None',
        marker='1',
        label=labels[f_name[:-11]]
    )
    list_of_dev_labels.append(labels[f_name[:-11]])
    # ax_dev_comp.hist(np.log(deviation))
ax_dev_comp.set_xticks([0, 1, 2, 3, 4], list_of_dev_labels)
ax_dev_comp_cloud.set_xticks([0, 1, 2, 3, 4], list_of_dev_labels)
# Rotate the labels at the x-axis for better readability.
# plt.setp(ax_dev_comp.get_xticklabels(), rotation=45, ha='right',
#          rotation_mode='anchor')
ax_dev_comp.set_ylabel("Accuracy of data features  /  -")
ax_dev_comp_cloud.set_ylabel("Accuracy of data features  /  -")
ax_dev_comp.set_yscale('log')
ax_dev_comp_cloud.set_yscale('log')
ax_dev.legend()
ax_dev.set_xlabel("Pulse number")
ax_dev.set_ylabel("Accuracy of data features  /  -")
ax_dev.set_yscale('log')
fig_dev.tight_layout()
fig_dev.savefig('../feature_deviations.pdf',
                bbox_inches='tight', pad_inches=0.0)

matplotlib.rcParams.update({'font.size': 11})

datapoints = {
    name: []
    for name in estimation_results[0]['inferred parameters'].keys()
}
verification = {
    name: []
    for name in estimation_verification[0]['inferred parameters'].keys()
}
smoothed_estimate = {}
standard_butler_volmer_fits = {}
errorbars = {name: [[], []] for name in datapoints.keys()}

for estimation_result in estimation_results:
    estimate = estimation_result['inferred parameters']
    covariance = estimation_result['covariance']
    _, _, bounds = calculate_means_and_standard_deviations(
        estimate,
        covariance,
        list(estimate.keys()),
        transform_parameters={
            name: 'log' for name in estimate.keys()
        },
        bounds_in_standard_deviations=1,
        epsabs=1e-12, epsrel=1e-12
    )
    for (name, parameter), (_, bound) in zip(estimate.items(), bounds.items()):
        datapoints[name].append(parameter)
        errorbars[name][0].append(parameter - bound[0])
        errorbars[name][1].append(bound[1] - parameter)
for entry in estimation_verification:
    estimate = entry['inferred parameters']
    correlation = entry['correlation']
    bounds = entry['error bounds']
    for (name, parameter), (_, bound) in zip(estimate.items(), bounds.items()):
        verification[name].append(parameter)

positive_plot_SOC = np.linspace(positive_SOCs[0], positive_SOCs[-1], 100)
negative_plot_SOC = np.linspace(negative_SOCs[0], negative_SOCs[-1], 100)
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(8 * 2**0.5, 8),
                                             ncols=2, nrows=2)
fig_ver, ((ax0_ver, ax1_ver), (ax2_ver, ax3_ver)) = plt.subplots(
    figsize=(8 * 2**0.5, 8), ncols=2, nrows=2,
)
fig_hist, ((ax0_hist, ax1_hist), (ax2_hist, ax3_hist)) = plt.subplots(
    figsize=(8 * 2**0.5, 8), ncols=2, nrows=2
)
# Add an invisible axis for labelling.
ax_label = fig_ver.add_subplot(111, frameon=False)
ax_label.spines['top'].set_color('none')
ax_label.spines['bottom'].set_color('none')
ax_label.spines['left'].set_color('none')
ax_label.spines['right'].set_color('none')
ax_label.tick_params(
    labelcolor='w', top=False, bottom=False, left=False, right=False
)
ax_label.set_ylabel(
    "Estimation / Verification relative deviation in log-space  /  -\n"
)
list_of_labels = []
for i, (name, ax, ax_ver, ax_hist) in enumerate(zip(
        datapoints.keys(), [ax1, ax0, ax3, ax2],
        [ax1_ver, ax0_ver, ax3_ver, ax2_ver],
        [ax1_hist, ax0_hist, ax3_hist, ax2_hist])):
    trans = matplotlib.transforms.ScaledTranslation(
        10/72, -5/72, fig.dpi_scale_trans
    )
    ax.text(0.0, 1.0, ['(b)', '(a)', '(d)', '(c)'][i],
            fontsize=20,
            transform=ax.transAxes + trans, verticalalignment='top')
    trans_ver = matplotlib.transforms.ScaledTranslation(
        10/72, -5/72, fig_ver.dpi_scale_trans
    )
    trans_hist = matplotlib.transforms.ScaledTranslation(
        10/72, -5/72, fig_hist.dpi_scale_trans
    )
    datapoints[name] = np.array(datapoints[name])
    errorbars[name] = [
        np.array(errorbars[name][0]),
        np.array(errorbars[name][1])
    ]
    verification[name] = np.array(verification[name])
    if name.find("Positive") != -1:
        plot_SOC = positive_plot_SOC
        SOC = positive_SOCs
        ax.set_xlabel("Positive electrode SOCₚ  /  -")
        ax_ver.set_xlabel("Positive electrode SOCₚ  /  -")
        if name.find("diffusivity") != -1:
            title = "Positive electrode diffusivity $D_p^*$  /  $m^2/s$"
            ax.set_ylim([2.5e-14, 6.5e-11])
            short_title = r"$\frac{D_p^*}{m^2/s}$"
            smoothing = 6e+1
            """
            axOCV = ax.twinx()
            axOCV.plot(SOC, parameters[
                    "Positive electrode OCP [V]"
                ](np.array(SOC)),
                color='grey',
                # plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
                label="NMC-850510 OCV curve"
            )
            axOCV.set_ylabel("OCV  /  V")
            axOCV.legend(loc='upper right')
            """
        elif name.find("exchange") != -1:
            title = (
                "Positive electrode exchange-current density "
                "$i_{se,p,0}^*$  /  $A/m^2$"
            )
            ax.set_ylim([2.5, 105])
            short_title = r"$\frac{i_{se,p,0}^*}{A/m^2}$"
            smoothing = 1e+2
    elif name.find("Negative") != -1:
        ax.invert_xaxis()
        ax_ver.invert_xaxis()
        plot_SOC = negative_plot_SOC
        SOC = negative_SOCs
        ax.set_xlabel("Negative electrode SOCₙ  /  -")
        ax_ver.set_xlabel("Negative electrode SOCₙ  /  -")
        if name.find("diffusivity") != -1:
            title = "Negative electrode diffusivity $D_n^*$  /  $m^2/s$"
            ax.set_ylim([2.5e-14, 6.5e-11])
            short_title = r"$\frac{D_n^*}{m^2/s}$"
            smoothing = 2e+2
            """
            axOCV = ax.twinx()
            axOCV.plot(SOC, parameters[
                    "Negative electrode OCP [V]"
                ](np.array(SOC)),
                color='grey',
                # plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
                label="Graphite OCV curve"
            )
            axOCV.set_ylabel("OCV  /  V")
            axOCV.legend(loc='upper right')
            """
        elif name.find("exchange") != -1:
            title = (
                "Negative electrode exchange-current density "
                "$i_{se,n,0}^*$  /  $A/m^2$"
            )
            ax.set_ylim([2.5, 105])
            short_title = r"$\frac{i_{se,n,0}^*}{A/m^2}$"
            smoothing = 2e+2

    list_of_labels.append(short_title)
    ax.set_ylabel(title)
    ax_ver.text(0.0, 1.0, short_title,
                transform=ax_ver.transAxes + trans_ver,
                verticalalignment='top')
    ax_hist.text(0.0, 1.0, short_title,
                 transform=ax_hist.transAxes + trans_hist,
                 verticalalignment='top')
    ax.set_yscale('log')
    ax_ver.set_yscale('log')
    line_err = ax.errorbar(
        SOC, datapoints[name], errorbars[name], marker='_', ls='None',
        label="EP-BOLFI with errorbars"
    ).lines
    # smoothed_estimate[name] = smooth_fit(
    #     SOC, np.log(datapoints[name]), order=5, smoothing_factor=smoothing
    # )
    # line_smooth, = ax.plot(
    #     plot_SOC, np.exp(smoothed_estimate[name](plot_SOC)),
    #     label="Smoothing spline"
    # )
    # Judge the verification quality on the log of the variables, as
    # the estimation process effectively estimated log(variables).
    # line_comp, = ax.plot(
    #     SOC, verification[name], ls='None', marker='1',
    #     label='verification run'
    # )
    line_ver, = ax_ver.plot(
        SOC, np.abs(np.log(datapoints[name]) - np.log(verification[name]))
        / np.abs(np.log(datapoints[name])), ls='None', marker='1'
    )
    ax_hist.hist(
        np.log(np.abs(np.log(datapoints[name]) - np.log(verification[name]))
               / np.abs(np.log(datapoints[name]))) / np.log(10), bins=20
    )
    deviation = (
        np.abs(np.log(datapoints[name]) - np.log(verification[name]))
        / np.abs(np.log(datapoints[name]))
    )
    log_deviation = np.log(deviation)
    log_mean = np.mean(log_deviation)
    log_var = np.var(log_deviation)
    log_std = np.sqrt(log_var)
    ax_comp.errorbar(
        [i], [np.exp(log_mean)],
        [[np.exp(log_mean) - np.exp(log_mean - log_std)],
         [np.exp(log_mean + log_std) - np.exp(log_mean)]],
        marker='o', capsize=8, lw=4, ms=15, capthick=4, ls='None',
        label=ax.get_ylabel()
    )
    ax_comp_cloud.plot(
        [i] * len(deviation),
        deviation,
        ls='None',
        marker='1',
        label=ax.get_ylabel()
    )
    # Rotate the labels at the x-axis for better readability.
    # plt.setp(ax_comp.get_xticklabels(), rotation=45, ha='right',
    #          rotation_mode='anchor')

    if name.find("exchange") != -1:
        standard_butler_volmer_fits[name] = minimize(
            lambda x: np.sum((
                standard_butler_volmer(np.array(SOC), *x) - datapoints[name]
            )**2)**0.5,
            jac='cs', x0=[0.5 * np.mean(datapoints[name])],  # , 0.5],
            method='trust-constr', bounds=[(None, None)]  # , (0.0, 1.0)]
        ).x
        ax.plot(plot_SOC, standard_butler_volmer(
            plot_SOC, *standard_butler_volmer_fits[name]
        ),
            label="Standard Butler-Volmer fit")

    if name.find("Positive electrode diffusivity") != -1:
        line_g, = ax.plot(positive_SOCs_for_GITT, gitt_diffusivities["data"],
                          marker='1', lw=0, ms=10, label="GITT (analytic)")
        line_i, = ax.plot(positive_SOCs_for_ICI, ici_diffusivities["data"],
                          marker='2', lw=0, ms=10, label="ICI (analytic)")

    # Fix that the automatic legend always places the errorbars last.
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles[-1])
    handles.pop(-1)
    labels.insert(0, labels[-1])
    labels.pop(-1)
    ax.legend(handles, labels, loc='lower center')
ax_comp.set_xticks([0, 1, 2, 3], list_of_labels)
ax_comp_cloud.set_xticks([0, 1, 2, 3], list_of_labels)
ax_comp.set_ylim([0.00011, 4.9])
ax_comp_cloud.set_ylim([0.00011, 4.9])
ax_comp.set_yscale('log')
ax_comp_cloud.set_yscale('log')
ax_comp.set_ylabel("Accuracy of fit parameters in log-space  /  -")
ax_comp_cloud.set_ylabel("Accuracy of fit parameters  /  -")
ax_comp.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
)
ax_comp_cloud.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
)
ax_comp.set_yticks([0.001, 0.01, 0.1, 1.0])
ax_comp_cloud.set_yticks([0.001, 0.01, 0.1, 1.0])
ax_dev_comp.set_ylim([0.00011, 4.9])
ax_dev_comp_cloud.set_ylim([0.00011, 4.9])
ax_dev_comp.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, _: '{:g}'.format(y))
)
ax_dev_comp_cloud.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, _: '{:g}'.format(y))
)
ax_dev_comp.set_yticks([0.001, 0.01, 0.1, 1.0])
ax_dev_comp_cloud.set_yticks([0.001, 0.01, 0.1, 1.0])
trans_comp = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig_comp.dpi_scale_trans
)
trans_comp_cloud = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig_comp_cloud.dpi_scale_trans
)
ax_comp.text(
    0.0, 1.0, '(a)', transform=ax_comp.transAxes + trans_comp,
    verticalalignment='top', fontsize=20
)
ax_comp_cloud.text(
    0.0, 1.0, '(a)', transform=ax_comp_cloud.transAxes + trans_comp_cloud,
    verticalalignment='top', fontsize=20
)
ax_dev_comp.text(
    0.0, 1.0, '(b)', transform=ax_dev_comp.transAxes + trans_comp,
    verticalalignment='top', fontsize=20
)
ax_dev_comp_cloud.text(
    0.0, 1.0, '(b)', transform=ax_dev_comp_cloud.transAxes + trans_comp_cloud,
    verticalalignment='top', fontsize=20
)
set_fontsize(
    ax_comp, small_fig_fontsize, small_fig_fontsize, small_fig_fontsize,
    20, small_fig_fontsize, small_fig_fontsize
)
set_fontsize(
    ax_dev_comp, small_fig_fontsize, small_fig_fontsize, small_fig_fontsize,
    small_fig_fontsize, small_fig_fontsize, small_fig_fontsize
)
set_fontsize(
    ax_comp_cloud, small_fig_fontsize, small_fig_fontsize, small_fig_fontsize,
    20, small_fig_fontsize, small_fig_fontsize
)
set_fontsize(
    ax_dev_comp_cloud, small_fig_fontsize, small_fig_fontsize,
    small_fig_fontsize, small_fig_fontsize, small_fig_fontsize,
    small_fig_fontsize
)
fig.tight_layout()
fig_comp.tight_layout()
fig_comp_cloud.tight_layout()
fig_hist.tight_layout()
fig_comp_cloud.savefig('../verification_and_feature_deviation_summary.pdf',
                       bbox_inches='tight', pad_inches=0.0)
fig.savefig('../EP_BOLFI_fit.pdf',
            bbox_inches='tight', pad_inches=0.0)
fig_ver.savefig('../EP_BOLFI_verification.pdf',
                bbox_inches='tight', pad_inches=0.0)
plt.show()
