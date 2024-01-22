# Copyright (c): German Aerospace Center (DLR)
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import minimize

from utility.preprocessing import (
    calculate_both_SOC_from_OCV, calculate_means_and_standard_deviations
)
# from utility.read_csv_datasets import read_channels_from_measurement_system

from parameters.estimation.gitt_basf import (
    parameters, positive_SOC_from_cell_SOC, negative_SOC_from_cell_SOC
)

with open('../GITT estimation results/estimation_results.json', 'r') as f:
    estimation_results = json.load(f)
datapoints = {
    name: [] for name in estimation_results[0]['inferred parameters'].keys()
}
lower_bounds = {name: [] for name in datapoints.keys()}
upper_bounds = {name: [] for name in datapoints.keys()}
for estimation_result in estimation_results:
    estimate = estimation_result['inferred parameters']
    correlation = estimation_result['correlation']
    _, _, bounds = calculate_means_and_standard_deviations(
        estimate,
        estimation_result['covariance'],
        list(estimate.keys()),
        transform_parameters={
            name: 'log' for name in estimate.keys()
        },
        bounds_in_standard_deviations=1,
        epsabs=1e-12, epsrel=1e-12
    )
    for (name, parameter), (_, bound) in zip(estimate.items(), bounds.items()):
        datapoints[name].append(parameter)
        lower_bounds[name].append(bound[0])
        upper_bounds[name].append(bound[1])

with open('../GITT estimation results/L_ACB440_BP_2_OCV.json', 'r') as f:
    terminal_OCVs = json.load(f)["Cell OCV [V]"][:85]
cell_SOCs = []
for ocv in terminal_OCVs:
    cell_SOCs.append(calculate_both_SOC_from_OCV(
        parameters, negative_SOC_from_cell_SOC, positive_SOC_from_cell_SOC, ocv
    ))

# This is from the linearized overpotential term from the SPM.
joint_resistance = []
for data in (upper_bounds, datapoints, lower_bounds):
    joint_resistance.append((
        1 / (
            parameters["Negative electrode electrons in reaction"]
            * parameters[
                "Negative electrode surface area to volume ratio [m-1]"
            ]
            * parameters["Negative electrode thickness [m]"]
            * np.array(
                data["Negative electrode exchange-current density [A.m-2]"]
            )
        ) + 1 / (
            parameters["Positive electrode electrons in reaction"]
            * parameters[
                "Positive electrode surface area to volume ratio [m-1]"
            ]
            * parameters["Positive electrode thickness [m]"]
            * np.array(
                data["Positive electrode exchange-current density [A.m-2]"]
            )
        )
    ) * (
        constants.R * parameters["Reference temperature [K]"]
        / constants.physical_constants["Faraday constant"][0]
        / parameters["Current collector perpendicular area [m2]"]
    ))


def standard_butler_volmer_inv(soc, i_00=1.0, alpha=0.5):
    return 1 / (i_00 * 1000.0**0.5 * soc**(1 - alpha) * (1 - soc)**alpha)


bv_fit = minimize(
    lambda x: np.sum((
        standard_butler_volmer_inv(np.array(cell_SOCs), *x)
        - joint_resistance[1]
    )**2)**0.5,
    jac='cs', x0=[0.5 * np.mean(joint_resistance[1]), 0.5],
    method='trust-constr', bounds=[(None, None), (0.0, 1.0)]
).x
print("Butler-Volmer fit parameters: iₛₑ =", bv_fit[0], ", α =", bv_fit[1])

matplotlib.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
ax.errorbar(cell_SOCs, joint_resistance[1],
            [joint_resistance[1] - joint_resistance[0],
             joint_resistance[2] - joint_resistance[1]],
            marker='_', ls='None', label='joint exchange-current resistance')
ax.plot(cell_SOCs, standard_butler_volmer_inv(np.array(cell_SOCs), *bv_fit),
        label='Butler-Volmer-type resistance fit')
ax.set_xlabel("Cell SOC  /  -")
ax.set_xlim([0.19, 1.01])
ax.set_ylabel('Resistance  /  Ω')
ax.set_yscale('log')
ax.legend()
fig.tight_layout()
fig.savefig('../joint_resistance.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()
