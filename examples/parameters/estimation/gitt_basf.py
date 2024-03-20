"""!@file
Parameter file for the estimation of battery model parameters from the
GITT data provided by BASF.
"""

from copy import deepcopy
import json
import numpy as np
from pybamm.expression_tree.exceptions import SolverError

from ep_bolfi.models.solversetup import (
    simulation_setup, spectral_mesh_pts_and_method
)
from ep_bolfi.utility.dataset_formatting import (
    read_csv_from_measurement_system
)
from ep_bolfi.utility.fitting_functions import (
    fit_exponential_decay, fit_sqrt,
    # OCV_fit_function, inverse_OCV_fit_function
)
from ep_bolfi.utility.preprocessing import (
    subtract_both_OCV_curves_from_cycles, find_occurrences,
    calculate_both_SOC_from_OCV
)

from pybamm.models.full_battery_models.lithium_ion import DFN

from parameters.models.basf_gitt_test_cell import (
    parameters, positive_SOC_from_cell_SOC, negative_SOC_from_cell_SOC
)

if 'seed' not in globals():
    """!
    RNG seed.
    """
    seed = 0
    print("Default: RNG seed set to 0.")

if 'optimize_simulation_speed' not in globals():
    """!
    If True, the last input is assumed to be a rest phase and it is
    shortened to 30 seconds.
    """
    optimize_simulation_speed = False
    print("Default: simulates the whole timespan of the data.")

if 'soc_dependent_estimation' not in globals():
    """!
    If True, the simulator will be set up to fill-in the values of
    t₊, βₙ and βₚ from the estimation of all seven parameters.
    """
    soc_dependent_estimation = False
    print("Default: simulator does not provide values for "
          "t₊, βₙ and βₚ.")

if 'white_noise' not in globals():
    """! If True, white noise gets added to the simulated voltage. """
    white_noise = True
    print("Default: white noise added to simulated voltage.")

if 'parameter_noise' not in globals():
    """!
    Set to True when you want to incorporate the uncertainty in the
    estimation of the SOC-independent parameters t₊, βₙ and βₚ.
    """
    parameter_noise = False
    print("Default: no parameter uncertainty of t₊, βₙ and βₚ.")

try:
    with open(
        '../GITT estimation results/seven_parameter_estimation_seed_0.json',
        'r'
    ) as f:
        results = json.load(f)
        all_estimates = results['inferred parameters']
        """! The estimate of the SOC-independent parameters. """
        parameter_estimates = {
            name: all_estimates[name]
            for name in [
                "Cation transference number",
                "Negative electrode Bruggeman coefficient",
                "Positive electrode Bruggeman coefficient",
            ]
        }
        parameter_order = list(all_estimates.keys())
        all_covariances = results['covariance']
        """! The uncertainty in the estimation of the SOC-independent
        parameters. """
        parameter_deviations = {
            name: np.sqrt(all_covariances[parameter_order.index(
                name)][parameter_order.index(name)])
            for name in [
                "Cation transference number",
                "Negative electrode Bruggeman coefficient",
                "Positive electrode Bruggeman coefficient",
            ]
        }
except FileNotFoundError:
    parameter_estimates = {
        "Cation transference number": 0.3,
        "Negative electrode Bruggeman coefficient": 3.0,
        "Positive electrode Bruggeman coefficient": 3.0,
    }
    parameter_deviations = {
        "Cation transference number": 0.05,
        "Negative electrode Bruggeman coefficient": 0.6,
        "Positive electrode Bruggeman coefficient": 0.6,
    }


if 'pulse_number' not in globals():
    """! Pulse number to request the corresponding data. """
    pulse_number = 0
    print("Default: pulse #0.")

# Optionally set starting pulse and number of pulses.
if type(pulse_number) is tuple:
    pulse_amount = pulse_number[1]
    pulse_number = pulse_number[0]
else:
    pulse_amount = 1

if 'free_parameters' not in globals():
    if soc_dependent_estimation:
        free_parameters = {
            "Negative electrode exchange-current density [A.m-2]":
                np.log(10),
            "Positive electrode exchange-current density [A.m-2]":
                np.log(10),
            "Negative electrode diffusivity [m2.s-1]":
                np.log(1e-12),
            "Positive electrode diffusivity [m2.s-1]":
                np.log(1e-12),
        }
    else:
        """! Parameters that are to be estimated. """
        free_parameters = {
            "Negative electrode exchange-current density [A.m-2]":
                np.log(10),
            "Positive electrode exchange-current density [A.m-2]":
                np.log(10),
            "Negative electrode diffusivity [m2.s-1]":
                np.log(1e-12),
            "Positive electrode diffusivity [m2.s-1]":
                np.log(1e-12),
            "Cation transference number":
                0.3,
            "Negative electrode Bruggeman coefficient":
                3.0,
            "Positive electrode Bruggeman coefficient":
                3.0,
        }
    print("Default: the fit parameters will get estimated with "
          "pre-defined guesses (see associated paper).")

if 'free_parameters_boundaries' not in globals():
    if soc_dependent_estimation:
        free_parameters_boundaries = {
            "Negative electrode exchange-current density [A.m-2]":
                (1, 100),
            "Positive electrode exchange-current density [A.m-2]":
                (1, 100),
            "Negative electrode diffusivity [m2.s-1]":
                (1e-14, 1e-10),
            "Positive electrode diffusivity [m2.s-1]":
                (1e-14, 1e-10),
        }
    else:
        """! Bounds in which a parameter set is searched for. """
        free_parameters_boundaries = {
            "Negative electrode exchange-current density [A.m-2]":
                (1, 100),
            "Positive electrode exchange-current density [A.m-2]":
                (1, 100),
            "Negative electrode diffusivity [m2.s-1]":
                (1e-14, 1e-10),
            "Positive electrode diffusivity [m2.s-1]":
                (1e-14, 1e-10),
            "Cation transference number":
                (0.2, 0.4),
            "Negative electrode Bruggeman coefficient":
                (1.8, 4.2),
            "Positive electrode Bruggeman coefficient":
                (1.8, 4.2),
        }
    print("Default: the fit parameters will get estimated with "
          "pre-defined boundaries (see associated paper).")

try:
    """! Names of the free parameters. """
    free_parameters_names = list(free_parameters.keys())
except AttributeError:
    free_parameters_names = list(free_parameters_boundaries.keys())

if 'transform_parameters' not in globals():
    if soc_dependent_estimation:
        transform_parameters = {
            "Negative electrode exchange-current density [A.m-2]": "log",
            "Positive electrode exchange-current density [A.m-2]": "log",
            "Negative electrode diffusivity [m2.s-1]": "log",
            "Positive electrode diffusivity [m2.s-1]": "log",
        }
    else:
        """! Transformations/Scalings of the parameter ranges. """
        transform_parameters = {
            "Negative electrode exchange-current density [A.m-2]": "log",
            "Positive electrode exchange-current density [A.m-2]": "log",
            "Negative electrode diffusivity [m2.s-1]": "log",
            "Positive electrode diffusivity [m2.s-1]": "log",
            "Cation transference number": "none",
            "Negative electrode Bruggeman coefficient": "none",
            "Positive electrode Bruggeman coefficient": "none",
        }
    print("Default: sets exchange-current densities and diffusivities "
          "to be estimated in log-scale.")

"""! The RNG for the parameter noise. """
parameter_noise_rng = {
    p_name: np.random.default_rng(seed + i)
    for i, (p_name, limits) in enumerate(parameter_deviations.items())
}


def t_plus_noise_generator():
    """! The random generator for the transference number. """
    p_name = "Cation transference number"
    return (parameter_estimates[p_name]
            + parameter_deviations[p_name]
            * parameter_noise_rng[p_name].standard_normal(1)[0])


def beta_n_noise_generator():
    """! The random generator for the negative bruggeman coefficient. """
    p_name = "Negative electrode Bruggeman coefficient"
    return (parameter_estimates[p_name]
            + parameter_deviations[p_name]
            * parameter_noise_rng[p_name].standard_normal(1)[0])


def beta_p_noise_generator():
    """! The random generator for the positive bruggeman coefficient. """
    p_name = "Positive electrode Bruggeman coefficient"
    return (parameter_estimates[p_name]
            + parameter_deviations[p_name]
            * parameter_noise_rng[p_name].standard_normal(1)[0])


"""! The experimental data to be used for inference. """
complete_dataset = read_csv_from_measurement_system(
    '../GITT data/L_ACB440_BP_1.064', 'iso-8859-1', 1,
    headers={3: "t [s]", 7: "I [A]", 8: "U [V]"},
    delimiter='\t', decimal='.',
    segment_column=2,  # 9,
    current_sign_correction={
        "R": 1, "C": -1, "D": 1, "P": 1, "O": 1, "S": 1, "ACR": 1,
    },
    correction_column=9
)
# print(complete_dataset.asymptotic_voltages)

"""! Define the pulses that are to be plotted. """
dataset = complete_dataset  # .subslice(22, 25)
"""! Define the starting OCV before each used pulse. """
starting_OCV = dataset.voltages[0][-1]
# starting_OCV = complete_dataset.voltages[57][-1]#6

"""! The dataset with charge-discharge cycles. """
charge_discharge_cycles = read_csv_from_measurement_system(
    '../GITT data/L_ACB440_BP_1.064', 'iso-8859-1', 1,
    headers={3: "t [s]", 7: "I [A]", 8: "U [V]"},
    delimiter='\t', decimal='.',
    segment_column=2,  # 9,
    current_sign_correction={
        "R": 1, "C": -1, "D": 1, "P": 1, "O": 1, "S": 1, "ACR": 1,
    },
    correction_column=9
)
"""! One charge-discharge cycle at C/4. """
charge_discharge = charge_discharge_cycles.subslice(22, 25)
charge = charge_discharge.subslice(0, 1)
cv = charge_discharge.subslice(1, 2)
discharge = charge_discharge.subslice(2, 3)

gitt = read_csv_from_measurement_system(
    '../GITT data/L_ACB440_BP_2.064', 'iso-8859-1', 1,
    headers={3: "t [s]", 7: "I [A]", 8: "U [V]"},
    delimiter='\t', decimal='.',
    segment_column=9,
    current_sign_correction={
        "R": 1, "C": -1, "D": 1, "P": 1, "O": 1, "S": 1, "ACR": 1,
    },
    correction_column=9
)
gitt.indices = [index / 2 for index in gitt.indices]

gitt_pulses = gitt.subslice(4, -3)

starting_OCV = fit_exponential_decay(
    gitt.timepoints[4 + 2 * pulse_number - 1],
    gitt.voltages[4 + 2 * pulse_number - 1],
    threshold=0.95
)[0][2][0]
# gitt_for_estimation = gitt_pulses.subslice(2 * pulse_number,
#                                            2 * pulse_number + 4)
gitt_for_estimation = gitt_pulses.subslice(
    2 * pulse_number, 2 * pulse_number + 2 * pulse_amount
)

"""! The complete overpotential curve. """
gitt_pulses_without_OCV = deepcopy(gitt_pulses)
gitt_pulses_without_OCV.voltages = subtract_both_OCV_curves_from_cycles(
    gitt_pulses, parameters, negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC, starting_OCV=starting_OCV
)[0]
"""! The overpotential curve during the GITT experiment. """
voltages_without_OCV = subtract_both_OCV_curves_from_cycles(
    gitt_for_estimation, parameters, negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC, starting_OCV=starting_OCV
)[0]
# Add the initial SOCs to the parameters.
calculate_both_SOC_from_OCV(
    parameters, negative_SOC_from_cell_SOC, positive_SOC_from_cell_SOC,
    starting_OCV
)
"""! The experimental data. """
experimental_dataset = [gitt_for_estimation.timepoints,
                        voltages_without_OCV]
experiment = deepcopy(experimental_dataset)
experiment[0] = [[t - experiment[0][0][0] for t in t_segment]
                 for t_segment in experiment[0]]
# experiment = [[entry for segment in axis for entry in segment]
#               for axis in experimental_dataset]
# experiment.append([[t[0] - experiment[0][0], t[-1] - experiment[0][0]]
#                    for t in gitt_for_estimation.timepoints])
# experiment[0] = [t - experiment[0][0] for t in experiment[0]]

"""!
The current as a pybamm.Simulation input (gets set later).
Other example inputs: "Hold at 3.95 V for 180 s",
"Discharge at 1 C for 1 s".
"""
current_input = []
t0 = gitt_for_estimation.timepoints[0][0]
for t, I in zip(gitt_for_estimation.timepoints, gitt_for_estimation.currents):
    if np.mean(np.abs(np.atleast_1d(I))) < 1e-8:
        current_input.append(
            "Rest for " + str(t[-1] - t[0]) + " s (0.1 second period)"
        )
    else:
        current_input.append(
            "Discharge at " + str(np.mean(np.atleast_1d(I))) +
            " A for " + str(t[-1] - t[0]) + " s (0.1 second period)"
        )

# This is an optimization for the ignored relaxation at rest.
if optimize_simulation_speed:
    current_input[-1] = "Rest for 30.0 s (0.1 second period)"

"""
## Complete measured OCV curve.
#complete_OCV = np.array(gitt.asymptotic_voltages[5:-5:2])
## The SOCs for which that OCV was measured.
complete_SOC_dim = [0.0] + [C[-1] for C in calculate_SOC(
    gitt.timepoints[6:], gitt.currents[6:]
)][:-5:2]
complete_SOC = np.array(complete_SOC_dim) / complete_SOC_dim[-1]

for s, c, u in list(zip(list(complete_SOC), list(complete_SOC_dim),
                        list(complete_OCV))):
    print(s, ", ", c, ", ", u)
"""

"""! The free parameters will have been subtracted from this deep copy. """
fixed_parameters = deepcopy(parameters)
# Now substract all parameters that are to be estimated.
for name in free_parameters_names:
    try:
        del fixed_parameters[name]
    except KeyError:
        pass

solver_free_parameters = deepcopy(free_parameters_names)
solver_free_parameters.append("Thermodynamic factor")
for elec_sign in ["Negative ", "Positive "]:
    for part in [" (electrode)", " (electrolyte)"]:
        solver_free_parameters.append(
            elec_sign + "electrode Bruggeman coefficient" + part
        )

if soc_dependent_estimation:
    parameters["Cation transference number"] = (
        parameter_estimates["Cation transference number"]
    )
    for elec_sign in ["Negative ", "Positive "]:
        for part in [" (electrode)", " (electrolyte)"]:
            parameters[
                elec_sign + "electrode Bruggeman coefficient" + part
            ] = (parameter_estimates[
                elec_sign + "electrode Bruggeman coefficient"
            ])

model = DFN()
solver = simulation_setup(model, current_input, parameters,
                          *spectral_mesh_pts_and_method(8, 20, 8, 2, 1, 1,
                                                        halfcell=False),
                          free_parameters=solver_free_parameters, reltol=1e-9,
                          abstol=1e-9, root_tol=1e-6, verbose=False)[0]
white_noise_generator = np.random.default_rng(seed)


def simulator(trial_parameters):
    global parameter_noise, d_e_noise_generator, beta_n_noise_generator
    global beta_p_noise_generator, white_noise, white_noise_generator

    param = {name: trial_parameters[name] for name in solver_free_parameters}
    if parameter_noise:
        param.update({
            "Cation transference number":
                t_plus_noise_generator(),
            "Negative electrode Bruggeman coefficient":
                beta_n_noise_generator(),
            "Positive electrode Bruggeman coefficient":
                beta_p_noise_generator(),
        })
    elif soc_dependent_estimation:
        param.update(parameter_estimates)
    param["Thermodynamic factor"] = 1.475 / (
        1 - param["Cation transference number"]
    )
    for elec_sign in ["Negative ", "Positive "]:
        for part in [" (electrode)", " (electrolyte)"]:
            param[elec_sign + "electrode Bruggeman coefficient" + part] = (
                param[elec_sign + "electrode Bruggeman coefficient"]
            )
    # Fail silently if the simulation did not work.
    try:
        solution = solver(
            check_model=False, calc_esoh=False, inputs=param
        )
    except SolverError:
        return [[[0.0 for i in n] for n in e] for e in experimental_dataset]
    if solution is None:
        return [[[0.0 for i in n] for n in e] for e in experimental_dataset]
    if len(solution.cycles) < len(current_input):
        return [[[0.0 for i in n] for n in e] for e in experimental_dataset]

    sim_data = [[], []]
    t0 = gitt_for_estimation.timepoints[0][0]
    for cycle in solution.cycles:
        t_eval = t0 + cycle["Time [h]"].entries * 3600.0
        u_eval = cycle["Total overpotential [V]"].entries
        sim_data[0].append(t_eval)
        # Add some noise to imitate the measurement (1 sigma ̂= 0.1 mV).
        if white_noise:
            sim_data[1].append(
                u_eval + 0.1e-3 * white_noise_generator.standard_normal(
                    len(sim_data[0][-1])
                )
            )
        else:
            sim_data[1].append(u_eval)
    return sim_data


def feature_visualizer(t, U):
    """!
    May be used for plotting purposes. Fur further information, please
    refer to utility.visualization.plot_comparison.
    """

    visualizations = []
    for i, (t_cycle, U_cycle) in enumerate(zip(t, U)):
        if not i % 2:
            visualizations.extend(fit_exponential_decay(t_cycle, U_cycle,
                                                        threshold=0.95))
            visualizations[-1][2] = (
                "  τᵣ: {0:.4g}".format(
                    np.abs(1.0 / visualizations[-1][2][2]))
                + " s"
            )
    for i, (t_cycle, U_cycle) in enumerate(zip(t, U)):
        # Fit the square root slope to the first 30 seconds only.
        cutoff = find_occurrences(t_cycle, t_cycle[0] + 30.0)[0]
        visualizations.append(fit_sqrt(t_cycle[:cutoff], U_cycle[:cutoff],
                                       threshold=0.95))
        if not i % 2:
            visualizations[-1][2] = (
                "  IR: {0:.4g} V".format(visualizations[-1][2][0]) + "\n"
                + r"  $\left(\frac{dU}{d\sqrt{t}}\right)^{-1}$"
                + ": {0:.4g}".format(1.0 /
                                     visualizations[-1][2][1]) + " sqrt(s)/V"
            )
        else:
            visualizations[-1][2] = (
                r"  η: {0:.4g} V".format(visualizations[-1][2][0]) + "\n"
                + r"  $\left(\frac{dU}{d\sqrt{t}}\right)^{-1}$"
                + ": {0:.4g}".format(1.0 /
                                     visualizations[-1][2][1]) + " sqrt(s)/V"
            )
    return visualizations
