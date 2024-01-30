from contextlib import redirect_stdout
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from pybamm.expression_tree.exceptions import SolverError
from scipy.fft import rfft, rfftfreq
from scipy.stats import chi2

from ep_bolfi.EP_BOLFI import EP_BOLFI
# from ep_bolfi.models.SPMe import SPMe
from pybamm.models.full_battery_models.lithium_ion.spme import SPMe
from ep_bolfi.models.solversetup import (
    solver_setup, spectral_mesh_pts_and_method
)
from ep_bolfi.utility.preprocessing import find_occurrences

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

# Multimodal excitation (scaled to be 8 mV in each excitation):
# 15 mA sine at 100 mHz + 15 mA sine at 1 Hz + 15 mA sine at 10 Hz
# + 15 mA sine at 100 Hz, applied at 11 equispaced SOC levels,
# sampled at 4 kHz


def current(t):
    return 15e-3 * (
        pybamm.sin(0.1 * 2 * np.pi * t) + pybamm.sin(2 * np.pi * t)
        + pybamm.sin(10 * 2 * np.pi * t) + pybamm.sin(100 * 2 * np.pi * t)
    )


parameters["Current function [A]"] = current
t_eval = np.arange(0.0, 10.0, 1 / 4000)

# model = SPMe(halfcell=False, pybamm_control=False)
model = SPMe(name="SPMe")
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


simulators = []
experimental_data = []
# SOC scaling between negative and positive electrode: 0.590
# Initial conditions for the unimodal sinusoidal excitations.
soc_offsets = [0.17, 0.24, 0.30, 0.36, 0.42,
               0.48, 0.54, 0.60, 0.66, 0.72, 0.78]
offset_indices = [10]  # [0, 2, 5, 10]
for i in offset_indices:
    soc_offset = soc_offsets[i]
    initial_socs = {
        "Initial concentration in negative electrode [mol.m-3]":
            (0.97 - soc_offset) * cₙ_max,
        "Initial concentration in positive electrode [mol.m-3]":
            (0.41 + soc_offset * 0.590) * cₚ_max,
    }
    simulators.append(
        lambda trial_parameters, init_socs=initial_socs: simulator(
            trial_parameters, initial_socs=init_socs
        )
    )
    experimental_data.append(simulator({
        "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
        "Cation transference number": 0.4,
        "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
        "Positive electrode diffusivity [m2.s-1]": 1e-13,
        "variance of the output noise": 1.6e-9,
    }, initial_socs=initial_socs))

i_eval = 15e-3 * (
    np.sin(0.1 * 2 * np.pi * t_eval) + np.sin(2 * np.pi * t_eval)
    + np.sin(10 * 2 * np.pi * t_eval) + np.sin(100 * 2 * np.pi * t_eval)
)

dft_frequencies = rfftfreq(t_eval.shape[-1]) * 4000
dft_i_eval = rfft(i_eval, norm="ortho") / 100


def features(dataset):
    features = []
    dft_u_eval = rfft(dataset, norm="ortho") / 100
    for input_frequency in [0.1, 1.0, 10.0, 100.0]:
        index = find_occurrences(dft_frequencies, input_frequency)[0]
        features.append(dft_u_eval.imag[index] / dft_i_eval.imag[index])
    # features.append(np.array([dataset]))
    return features


features_list = [features] * len(offset_indices)


def name_of_features(index, number=0):
    if index < 4:
        return (
            "impedance for frequency "
            + str([0.1, 1.0, 10.0, 100.0][index])
            + " Hz at excitation point #"
            + str(number)
        )
    else:
        return "complete voltage curve"


# This names all features by the last offset index, for some reason.
name_of_features_list = [
    lambda index, number=point: name_of_features(index, number)
    for point in offset_indices
]

estimator = EP_BOLFI(
    simulators, experimental_data, features_list, fixed_parameters,
    free_parameters=free_parameters,
    transform_parameters=transform_parameters,
    display_current_feature=name_of_features_list
)
ep_iterations_counter = 0
with open(
    '../spme_benchmark_results/multimodal/individual_excitation_points/'
    + 'spme_benchmark_multimodal_'
    + '_'.join([str(index) for index in offset_indices])
    + '.log', 'w'
) as f:
    with redirect_stdout(f):
        estimator.run(
            bolfi_initial_evidence=65,
            bolfi_total_evidence=130,
            bolfi_posterior_samples=27,
            ep_iterations=4,
            final_dampening=0.5,
            gelman_rubin_threshold=1.2,
            ess_ratio_sampling_from_zero=10.0,
            ess_ratio_abort=20.0,
            show_trials=False,
            verbose=True,
            seed=seed
        )
        ep_iterations_counter = ep_iterations_counter + 4
        with open(
            '../spme_benchmark_results/multimodal/'
            + 'individual_excitation_points/'
            + str(ep_iterations_counter * 130 * 4)
            + '_samples_at_soc_point_'
            + '_'.join([str(index) for index in offset_indices])
            + '.json', 'w'
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
        # Run another 8 times without dampening.
        # estimator stores the intermediate result.
        for i in range(2):
            estimator.run(
                bolfi_initial_evidence=65,
                bolfi_total_evidence=130,
                bolfi_posterior_samples=27,
                ep_iterations=8,
                ep_dampener=0.0,
                gelman_rubin_threshold=1.2,
                ess_ratio_sampling_from_zero=10.0,
                ess_ratio_abort=20.0,
                show_trials=False,
                verbose=True,
                seed=seed
            )
            ep_iterations_counter = ep_iterations_counter + 4
            with open(
                '../spme_benchmark_results/multimodal/'
                + 'individual_excitation_points/'
                + str(ep_iterations_counter * 130 * 4)
                + '_samples_at_soc_point_'
                + '_'.join([str(index) for index in offset_indices])
                + '.json', 'w'
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
    'Electrolyte diffusivity [m2.s-1]': 2.5901876160129574e-10,
    'Cation transference number': 0.41608106271169887,
    'Negative electrode diffusivity [m2.s-1]': 4.2967532811788726e-14,
    'Positive electrode diffusivity [m2.s-1]': 1.0453662884662657e-13,
    'variance of the output noise': 1.296036528748575e-09,
    "Initial concentration in negative electrode [mol.m-3]":
        (0.97 - soc_offsets[10]) * cₙ_max,
    "Initial concentration in positive electrode [mol.m-3]":
        (0.41 + soc_offsets[10] * 0.590) * cₚ_max,
})
free_parameters_boundaries = {
    'Electrolyte diffusivity [m2.s-1]':
        (2.2127636228937993e-10, 3.0319876089489057e-10),
    'Cation transference number':
        (0.3980597556937854, 0.43410236972961236),
    'Negative electrode diffusivity [m2.s-1]':
        (3.46862608246917e-14, 5.3225941108587344e-14),
    'Positive electrode diffusivity [m2.s-1]':
        (8.366075687329022e-14, 1.3062165797959967e-13),
    'variance of the output noise':
        (1.0761718205801333e-09, 1.5608201699103888e-09),
}

solutions, errorbars = solve_all_parameter_combinations(
    # SPMe(halfcell=False, pybamm_control=False),
    SPMe(name="SPMe"),
    t_eval, optimized_parameters,
    free_parameters_boundaries,
    *spectral_mesh_pts_and_method(12, 6, 12, 1, 1, 1, halfcell=False),
    full_factorial=False, reltol=1e-9, abstol=1e-9, root_tol=1e-6,
    verbose=False
)
solutions["simulation"] = solutions.pop("SPMe")
for name in free_parameters_boundaries.keys():
    errorbars.pop(name)
errorbars["95% confidence"] = errorbars.pop("all parameters")
fontsize = 13
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.loc': 'lower right'})

i_eval = 15e-3 * (
    np.sin(0.1 * 2 * np.pi * t_eval) + np.sin(2 * np.pi * t_eval)
    + np.sin(10 * 2 * np.pi * t_eval) + np.sin(100 * 2 * np.pi * t_eval)
)
freq = rfftfreq(t_eval.shape[-1]) * 4000
i_dft = rfft(i_eval, norm="ortho") / 100
i_output = []
frequency_indices = []
for frequency in [0.1, 1.0, 10.0, 100.0]:
    index = find_occurrences(freq, frequency)[0]
    frequency_indices.append(index)
    # i_output.append(i_dft.imag[index])
    i_output.append(i_dft[index])
u_dft_exp = rfft(experimental_data[0], norm="ortho") / 100
u_output_exp = []
for index in frequency_indices:
    # u_output_exp.append(u_dft_exp.imag[index])
    u_output_exp.append(u_dft_exp[index])
dft_solutions = {}
dft_errorbars = {}
for k, v in solutions.items():
    # u_eval = simulators[0](optimized_parameters)
    # u_dft = rfft(u_eval, norm="ortho") / 100
    u_dft = rfft(v["Terminal voltage [V]"].entries, norm="ortho") / 100
    u_output = []
    for index in frequency_indices:
        # u_output.append(u_dft.imag[index])
        u_output.append(u_dft[index])
    dft_solutions[k] = np.abs(np.array(u_output) / np.array(i_output))
for k, v in errorbars.items():
    dft_errorbars[k] = [dft_solutions["simulation"]]
    for errorbar in v:
        u_dft = rfft(
            errorbar["Terminal voltage [V]"].entries, norm="ortho"
        ) / 100
        u_output = []
        for index in frequency_indices:
            # u_output.append(u_dft.imag[index])
            u_output.append(u_dft[index])
        dft_errorbars[k].append(
            np.abs(np.array(u_output) / np.array(i_output))
        )

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

fig1, ax1 = plt.subplots(figsize=(5 * 2**0.5, 5))
minimum_plot = (
    - np.min(dft_errorbars["95% confidence"], axis=0)
    + np.array(dft_solutions["simulation"])
)
maximum_plot = (
    np.max(dft_errorbars["95% confidence"], axis=0)
    - np.array(dft_solutions["simulation"])
)
ax1.plot(
    [0.1, 1.0, 10.0, 100.0],
    np.abs(np.array(u_output_exp) / np.array(i_output)), label='experiment',
    marker='_', ls='None'
)
ax1.errorbar(
    [0.1, 1.0, 10.0, 100.0], dft_solutions["simulation"],
    [minimum_plot, maximum_plot],
    marker='_', ls='None',
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
    label='simulation (with 95 % confidence)'
)
ax1.set_xlabel("Frequency  /  Hz")
ax1.set_ylabel("Impedance  /  Ω")
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1.tight_layout()
fig0.savefig('../spme_benchmark_multimodal_synthetic_data.svg',
             bbox_inches='tight', pad_inches=0.0)
fig1.savefig('../spme_benchmark_multimodal_result.svg',
             bbox_inches='tight', pad_inches=0.0)

plt.show()
"""
