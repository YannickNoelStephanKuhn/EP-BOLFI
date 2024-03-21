from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from pybamm.expression_tree.exceptions import SolverError
from scipy.stats import chi2

from ep_bolfi import EP_BOLFI
# from ep_bolfi.models.SPMe import SPMe
from pybamm.models.full_battery_models.lithium_ion.spme import SPMe
from ep_bolfi.models.solversetup import (
    solver_setup, spectral_mesh_pts_and_method
)

from parameters.models.spme_benchmark_cell import cₙ_max, cₚ_max, parameters

plt.style.use("default")

seed = 0
free_parameters = {
    "Electrolyte diffusivity [m2.s-1]":
        (np.log(2.8e-10),
         (np.log(100) - np.log(2.80))**2 / chi2(1).ppf(0.98)),
    "Cation transference number":
        (0.4, 0.2**2 / chi2(1).ppf(0.8)),
    "Negative electrode diffusivity [m2.s-1]":
        (np.log(3.9e-14),
         (np.log(100) - np.log(3.90))**2 / chi2(1).ppf(0.98)),
    "Positive electrode diffusivity [m2.s-1]":
        (np.log(1e-13),
         (np.log(100) - np.log(1))**2 / chi2(1).ppf(0.98)),
    "variance of the output noise":
        (np.log(1.6e-9), np.log(10)**2 / chi2(1).ppf(0.95)),
}
free_parameters_names = list(free_parameters.keys())
free_parameters_names.extend([
    "Initial concentration in negative electrode [mol.m-3]",
    "Initial concentration in positive electrode [mol.m-3]"
])
transform_parameters = {
    "Electrolyte diffusivity [m2.s-1]": "log",
    "Cation transference number": "none",
    "Negative electrode diffusivity [m2.s-1]": "log",
    "Positive electrode diffusivity [m2.s-1]": "log",
    "variance of the output noise": "log",
}
"""! The free parameters will have been subtracted from this deep copy. """
fixed_parameters = deepcopy(parameters)
# Now substract all parameters that are to be estimated.
for name in free_parameters_names:
    try:
        del fixed_parameters[name]
    except KeyError:
        pass

# Sinusoidal excitation plus discharge:
# C/24 sine at 1 mHz + 1C DC


def current(t):
    return 1.0 + 1 / 24 * pybamm.sin(0.001 * 2 * np.pi * t)


parameters["Current function [A]"] = current
t_eval = np.linspace(0, 5800, 4000)


model = SPMe()
solver = solver_setup(
    model, parameters, *spectral_mesh_pts_and_method(12, 6, 12, 1, 1, 1,
                                                     halfcell=False),
    free_parameters=free_parameters_names, verbose=False
)
white_noise_generator = np.random.default_rng(seed)


def simulator(trial_parameters, initial_socs={}):
    global white_noise_generator

    trial_parameters.update(initial_socs)
    param = {name: trial_parameters[name] for name in free_parameters_names}
    # Fail silently if the simulation did not work.
    try:
        solution = solver(t_eval, inputs=param)
    except SolverError:
        return 0.0 * t_eval
    if solution is None:
        return 0.0 * t_eval
    voltages = solution["Terminal voltage [V]"].entries
    # voltages = solution["Total overpotential [V]"].entries
    # Add the white noise. The paper states 1% two-sigma error in the
    # amplitude of the voltage response (the table showing the estimates
    # indicates σ²=1.6e-9).
    voltages = list(
        np.array(voltages) + param["variance of the output noise"]**0.5
        * white_noise_generator.standard_normal(len(voltages))
    )
    # Ensure that the solution is always the same length.
    voltages.extend([0.0 for i in range(len(voltages), len(t_eval))])
    return voltages


initial_socs = {
    "Initial concentration in negative electrode [mol.m-3]":
        0.97 * cₙ_max,
    "Initial concentration in positive electrode [mol.m-3]":
        0.41 * cₚ_max,
}
simulators = [lambda trial_parameters: simulator(
    trial_parameters, initial_socs=initial_socs
)]
experimental_data = [simulator(
    {
        "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
        "Cation transference number": 0.4,
        "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
        "Positive electrode diffusivity [m2.s-1]": 1e-13,
        "variance of the output noise": 1.6e-9,
    },
    initial_socs=initial_socs
)]

# Motivation for the slices:
# The electrolyte relaxation to the new equilibrium takes ~2 minutes.
# The next segment contains one period of the sine wave.
# The following segment contains all data until
# the final sine wave, which is in the last segment.


def features(dataset):
    return [
        np.array([dataset[segment[0]:segment[1]]])
        for segment in zip([0, 80, 864, 3288, None][:-1],
                           [0, 80, 864, 3288, None][1:])
    ]


features_list = [features]


def name_of_features(index):
    return {
        0: "Initial electrolyte relaxation",
        1: "First sine wave",
        2: "Data between first and last sine wave",
        3: "Last sine wave",
    }[index]


name_of_features_list = [name_of_features]

estimator = EP_BOLFI(
    simulators, experimental_data, features_list, fixed_parameters,
    free_parameters=free_parameters,
    transform_parameters=transform_parameters,
    display_current_feature=name_of_features_list
)
ep_iterations_counter = 0
"""
with open(
    '../spme_benchmark_results/unimodal/spme_benchmark_unimodal.log', 'w'
) as f:
    with redirect_stdout(f):
"""
estimator.run(
    bolfi_initial_evidence=65,
    bolfi_total_evidence=130,
    bolfi_posterior_samples=27,
    ep_iterations=4,
    final_dampening=0.5,
    gelman_rubin_threshold=1.2,
    ess_ratio_sampling_from_zero=10.0,
    ess_ratio_abort=20.0,
    show_trials=True,
    verbose=True,
    seed=seed
)
ep_iterations_counter = ep_iterations_counter + 4
with open(
    '../spme_benchmark_results/unimodal/'
    + str(ep_iterations_counter * 130 * 4)
    + '_samples.json', 'w'
) as f:
    json.dump({
        "inferred parameters": estimator.inferred_parameters,
        "covariance": [
            list(line) for line in estimator.final_covariance
        ],
        "correlation": [
            list(line) for line in estimator.final_correlation
        ],
        "error bounds": estimator.final_error_bounds,
    }, f)
# Run another 12 times without dampening.
# estimator stores the intermediate result.
for i in range(3):
    estimator.run(
        bolfi_initial_evidence=65,
        bolfi_total_evidence=130,
        bolfi_posterior_samples=27,
        ep_iterations=4,
        ep_dampener=0.0,
        gelman_rubin_threshold=1.2,
        ess_ratio_sampling_from_zero=10.0,
        ess_ratio_abort=20.0,
        show_trials=True,
        verbose=True,
        seed=seed
    )
    ep_iterations_counter = ep_iterations_counter + 4
    with open(
        '../spme_benchmark_results/unimodal/'
        + str(ep_iterations_counter * 130 * 4)
        + '_samples.json', 'w'
    ) as f:
        json.dump({
            "inferred parameters": estimator.inferred_parameters,
            "covariance": [
                list(line) for line in estimator.final_covariance
            ],
            "correlation": [
                list(line) for line in estimator.final_correlation
            ],
            "error bounds": estimator.final_error_bounds,
        }, f)

"""
optimized_parameters = deepcopy(parameters)

optimized_parameters.update({
    'Electrolyte diffusivity [m2.s-1]': 2.798437399019171e-10,
    'Cation transference number': 0.4001962531198564,
    'Negative electrode diffusivity [m2.s-1]': 3.9001132613880597e-14,
    'Positive electrode diffusivity [m2.s-1]': 1.0003063713144553e-13,
    'variance of the output noise': 1.3482856839508548e-09
})
free_parameters_boundaries = {
    'Electrolyte diffusivity [m2.s-1]':
        (2.792353942214955e-10, 2.804534109317555e-10),
    'Cation transference number':
        (0.39941944756997005, 0.40097305866974275),
    'Negative electrode diffusivity [m2.s-1]':
        (3.8993380258644485e-14, 3.9008886510378624e-14),
    'Positive electrode diffusivity [m2.s-1]':
        (9.981805216906142e-14, 1.0024367484125608e-13),
    'variance of the output noise':
        (1.3257511823795397e-09, 1.3712032164919453e-09),
}

solutions, errorbars = solve_all_parameter_combinations(
    SPMe(),
    t_eval, optimized_parameters,
    free_parameters_boundaries,
    *spectral_mesh_pts_and_method(12, 6, 12, 1, 1, 1, halfcell=False),
    full_factorial=True, reltol=1e-9, abstol=1e-9, root_tol=1e-6,
    verbose=False
)
solutions["simulation"] = solutions.pop("SPMe")
for name in free_parameters_boundaries.keys():
    errorbars.pop(name)
errorbars["95% confidence"] = errorbars.pop("all parameters")
fontsize = 13
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.loc': 'lower right'})

i_eval = 1.0 + 1 / 24 * np.sin(0.001 * 2 * np.pi * t_eval)

fig0, ax0 = plt.subplots(figsize=(5 * 2**0.5, 5))
ax0.plot(t_eval, experimental_data[0], label="Output voltage (simulated)")
ax0.legend(loc="lower right")
ax0I = ax0.twinx()
ax0I.plot(t_eval, i_eval,
          color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
          label="Input current")
ax0I.set_ylabel("Input current  /  A")
ax0I.legend(loc="upper right")
ax0.set_xlabel("Experiment duration  /  s")
ax0.set_ylabel("Terminal voltage  /  V")
fig0.tight_layout()
# fig0.savefig('../spme_benchmark_unimodal_synthetic_data.svg',
#              bbox_inches='tight', pad_inches=0.0)

fig1, ax1 = plt.subplots(figsize=(5 * 2**0.5, 5))
plot_comparison(
    ax1, solutions, errorbars, [t_eval, experimental_data[0]],
    title="",
    xlabel="Experiment run-time  /  h", ylabel="Terminal voltage  /  V",
    interactive_plot=False,
    feature_fontsize=fontsize, overpotential=False, use_cycles=False
)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)
fig1.tight_layout()
# fig1.savefig('../spme_benchmark_unimodal_result.svg',
#              bbox_inches='tight', pad_inches=0.0)
plt.show()
"""
