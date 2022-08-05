"""!@file
Example usage of the visualization tools for GITT.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from runpy import run_module
import numpy as np

from utility.preprocessing import (
    SubstitutionDict, simulate_all_parameter_combinations
)
from utility.visualization import (
    plot_comparison, push_apart_text
)
from models.solversetup import spectral_mesh_pts_and_method

# from pybamm.models.full_battery_models.lithium_ion.dfn import DFN
from models.DFN import DFN
# from models.SPMe import SPMe
# from models.SPM import SPM

# Set the substitutions for dependent parameters and different names.
substitutions = {
    '1 + dlnf/dlnc':
        lambda base: 1.475 / (
            1 - np.array(base['Cation transference number'])
        ),
    'Negative electrode Bruggeman coefficient (electrode)':
        'Negative electrode Bruggeman coefficient',
    'Negative electrode Bruggeman coefficient (electrolyte)':
        'Negative electrode Bruggeman coefficient',
    'Positive electrode Bruggeman coefficient (electrode)':
        'Positive electrode Bruggeman coefficient',
    'Positive electrode Bruggeman coefficient (electrolyte)':
        'Positive electrode Bruggeman coefficient',
}

# From the estimation of all 7 parameters at pulses 66 and 67.
seed = 0
with open(
    '../GITT estimation results/seven_parameter_estimation_seed_'
    + str(seed) + '.json',
    'r'
) as f:
    result = json.load(f)
    free_parameters = result['inferred parameters']
    covariance = result['covariance']
    free_parameters_boundaries = result['error bounds']
    order_of_parameter_names = list(result['inferred parameters'].keys())

pulse_number = (66, 2)
# (66, 2) corresponds to cell SOC 0.765
data = run_module(
    'parameters.estimation.gitt_basf',
    init_globals={
        'optimize_simulation_speed': False,
        'soc_dependent_estimation': False,
        'white_noise': False,
        'parameter_noise': False,
        'pulse_number': pulse_number,
    }
)
if type(pulse_number) is tuple:
    pulse_number = pulse_number[0]
experiment = data['experiment']
input = data['input']
feature_visualizer = data['feature_visualizer']
parameters = data['parameters']
parameters.update(free_parameters)
parameters = SubstitutionDict(data['parameters'], substitutions)
transform_parameters = data['transform_parameters']

model = (
    DFN(halfcell=False, pybamm_control=True)
    # SPMe(halfcell=False, pybamm_control=True)
    # SPM(halfcell=False, pybamm_control=True)
)

# Doubling beyond 8-20-8 to 16-40-16 changes the features by about
# 2 promille. Halving from 8-20-8 to 4-10-4 changes them by 2 %.
solutions, errorbars = simulate_all_parameter_combinations(
    DFN(halfcell=False, pybamm_control=True), input,
    *spectral_mesh_pts_and_method(8, 20, 8, 2, 1, 1, halfcell=False),
    parameters, covariance=covariance,
    order_of_parameter_names=order_of_parameter_names,
    transform_parameters=transform_parameters,
    full_factorial=True, reltol=1e-9, abstol=1e-9, root_tol=1e-6,
    verbose=True
)
plt.style.use("default")
solutions["simulation"] = solutions.pop("DFN")
errorbars.pop("confidence semiaxes")
errorbars["95% confidence"] = errorbars.pop("all parameters")
fontsize = 13
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.loc': 'lower right'})
fig, ax = plt.subplots(figsize=(2**0.5 * 5, 5))
text_objects = plot_comparison(
    ax, solutions, errorbars, experiment,
    title="",
    xlabel="Experiment run-time  /  h", ylabel="Cell overpotential  /  mV",
    interactive_plot=False, feature_visualizer=feature_visualizer,
    voltage_scale=1e-3,
    output_variables={
        "Negative electrode SOC",
        "Positive electrode SOC",
        "Negative particle surface concentration",
        "Positive particle surface concentration",
        "Total overpotential [V]",
        "Terminal voltage [V]",
    },
    feature_fontsize=fontsize, overpotential=True, use_cycles=True
)
fig.tight_layout()
plt.show()

# These are for FIG. 7. Fontsize: 13.
bounding_box = Bbox([[-0.015, -6.25], [0.245, 0.25]])
for text in text_objects:
    if bounding_box.contains(*text.get_position()):
        text.set_visible(True)
    else:
        text.set_visible(False)
ax.set_xlim(sorted(bounding_box.intervalx))
ax.set_ylim(sorted(bounding_box.intervaly))
ax.yaxis.set_ticks([0 - 2 * i for i in range(4)])
ax.yaxis.set_ticklabels(["  " + str(0 - 2 * i) for i in range(4)])
trans = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig.dpi_scale_trans
)
label_text = ax.text(0.0, 1.0, '(a)', transform=ax.transAxes + trans,
                     verticalalignment='top', fontsize=20)
push_apart_text(fig, ax, text_objects)
fig.savefig('../GITT_7_parameters_fit_pulse_66.pdf',
            bbox_inches='tight', pad_inches=0.0)
label_text.set_visible(False)
bounding_box = Bbox([[0.348, -26], [0.392, 3.5]])
for text in text_objects:
    if bounding_box.contains(*text.get_position()):
        text.set_visible(True)
    else:
        text.set_visible(False)
ax.set_xlim(sorted(bounding_box.intervalx))
ax.set_ylim(sorted(bounding_box.intervaly))
ax.xaxis.set_ticks([0.35 + 0.01 * i for i in range(5)])
ax.yaxis.set_ticks([0 - 5 * i for i in range(6)])
ax.yaxis.set_ticklabels([str(0 - 5 * i) for i in range(6)])
trans = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig.dpi_scale_trans
)
ax.text(0.0, 1.0, '(b)', transform=ax.transAxes + trans,
        verticalalignment='top', fontsize=20)
push_apart_text(fig, ax, text_objects, lock_xaxis=True)
fig.savefig('../GITT_7_parameters_fit_pulse_67.pdf',
            bbox_inches='tight', pad_inches=0.0)
