from ep_bolfi.utility.fitting_functions import fit_sqrt
from ep_bolfi.utility.preprocessing import capacity, find_occurrences
import matplotlib.pyplot as plt
import numpy as np
from pybamm import Experiment
from scipy.optimize import minimize

from automation_data_download import data, diffusivities

# F is Faraday's constant
F = 96485.33212
# z is the charge number, which is 1 for Li-ion batteries
z = 1
# The timescale in seconds that is considered short w.r.t. diffusion.
short_timescale = 90
# Resolution of the voltage measurement, as counted from the steps in
# the plot. It's just shy of 0.2 mV.
voltage_precision = 0.0021 / 11


def layered_sqrt_fit_function(t, b, c, τ, t_0):
    "Implements the √(t + τ) - √t function for improved GITT results."
    return b + c * (np.sqrt(t - t_0 + τ) - np.sqrt(t - t_0))


for electrode, direction in [
    ["negative", "lithiation"],
    ["negative", "delithiation"],
    ["positive", "lithiation"],
    ["positive", "delithiation"]
]:
    ############################################
    # Calculate the properties of the battery. #
    ############################################

    parameters = data["parameters"][electrode][direction]
    electrode_capacity = capacity(parameters, electrode=electrode)
    # V_m is the molar volume of the electrode in m³/mol; we calculate
    # it as 1 / c_max, since its purpose is to transform
    # between concentration and SOC
    V_M = 1 / parameters[
        "Maximum concentration in " + electrode + " electrode [mol.m-3]"
    ]
    # S is the "area of the sample-electrolyte interface" in m²;
    # since its purpose is to mirror the prefactor in the analytic
    # solution of the SPM, it is the cross-section area of the electrode
    # rather than the microporous surface area
    S = parameters["Current collector perpendicular area [m2]"]

    gitt_scaling = 4 / np.pi * (V_M / (z * F * S))**2

    ################################################
    # Calculate the properties of the measurement. #
    ################################################

    data["soc"][electrode][direction] = {}

    for type in ["voltage", "overpotential"]:
        measurement = data[type][electrode][direction]
        start_index_meas = 4 if direction == "lithiation" else 2
        end_index_meas = 63 if direction == "lithiation" else 67
        start_index_ocv = 12 if direction == "lithiation" else 8
        end_index_ocv = 71 if direction == "lithiation" else 73
        meas_indices = list(range(start_index_meas, end_index_meas, 2))
        ocv_indices = list(range(start_index_ocv, end_index_ocv, 2))
        tau = np.asarray([
            measurement.timepoints[i][-1] - measurement.timepoints[i][0]
            for i in meas_indices
        ])
        current = np.asarray([
            np.mean(measurement.currents[i]) for i in meas_indices
        ])
        delta_SOC = current * tau / electrode_capacity
        pulse_start_voltages = np.asarray([
            measurement.voltages[i][0] for i in meas_indices
        ])
        pulse_end_voltages = np.asarray([
            measurement.voltages[i][-1] for i in meas_indices
        ])
        relaxation_end_voltages = np.asarray([
            measurement.voltages[i][-1]
            for i in range(start_index_meas - 1, end_index_meas + 1, 2)
        ])
        delta_transient = pulse_end_voltages - pulse_start_voltages
        delta_static = np.diff(relaxation_end_voltages)
        # For higher precision, use the exponential asymptotes.
        delta_static_exponential = np.diff(
            data["ocv"][electrode][direction]["OCV [V]"]
        )

        ############################################################
        # Re-calculate properties for only the short-time effects. #
        ############################################################

        cutoff_indices = [
            find_occurrences(
                measurement.timepoints[i], measurement.timepoints[i][0] + 90
            )[0]
            for i in meas_indices
        ]
        tau_short = np.asarray([
            measurement.timepoints[i][ci] - measurement.timepoints[i][0]
            for i, ci in zip(meas_indices, cutoff_indices)
        ])
        delta_SOC_short = current * tau_short / electrode_capacity
        pulse_end_voltages_short = np.asarray([
            measurement.voltages[i][ci]
            for i, ci in zip(meas_indices, cutoff_indices)
        ])
        delta_transient_short = pulse_end_voltages_short - pulse_start_voltages
        delta_static_short = delta_static * tau_short / tau

        ####################################################
        # Calculate GITT derivatives via square-root fits. #
        ####################################################

        derivative_transient = np.asarray([
            fit_sqrt(
                measurement.timepoints[i][:ci], measurement.voltages[i][:ci]
            )[2][1]
            for i, ci in zip(meas_indices, cutoff_indices)
        ])
        ocv_data_indices = [
            data["ocv"][electrode][direction]["indices"].index(i - 1)
            for i in ocv_indices
        ]
        socs = np.asarray([
            data["ocv"][electrode][direction][
                electrode.capitalize() + " electrode SOC [-]"
            ][odi]
            for odi in ocv_data_indices
        ])
        data["soc"][electrode][direction][
            electrode.capitalize() + " electrode SOC [-]"
        ] = socs
        derivative_static = parameters[
            electrode.capitalize() + " electrode OCP derivative by SOC [V]"
        ](socs)

        #########################################################
        # Calculate layered square-root slopes (√(t + τ) - √t). #
        #########################################################

        # Since the protocol was stored in PyBaMM format, use PyBaMM.
        protocol = data["protocol"][electrode][direction]
        pulse_lengths = [
            Experiment(
                [protocol[i][protocol[i].find(":") + 2:]]
            ).unique_steps.pop().duration
            for i in meas_indices
        ]
        ts = [
            np.asarray(measurement.timepoints[i][:ci])
            for i, ci in zip(meas_indices, cutoff_indices)
        ]
        us = [
            np.asarray(measurement.voltages[i][:ci])
            for i, ci in zip(meas_indices, cutoff_indices)
        ]
        derivative_transient_layered = np.asarray([minimize(
            lambda x: np.sum((
                layered_sqrt_fit_function(t, *x, pulse_lengths[n // 2], t[0])
                - u
            )**2)**0.5,
            x0=[u[0], (u[-1] - u[0]) / np.sqrt(t[-1] - t[0])],
            method='trust-constr'
        ).x[1] for n, (t, u) in enumerate(zip(ts, us))])

        ############################################################
        # Calculate 1977 GITT with "Δ OCV" over "Δ overpotential". #
        ############################################################

        name = "ΔUₛ/ΔUₜ" if type == "voltage" else "Δηₛ/ΔUₜ"
        diffusivities[electrode][direction][name] = {}
        diffusivities[electrode][direction][name]["mode"] = (
            gitt_scaling * current**2 * tau / delta_SOC**2
            * (delta_static / delta_transient)**2
        )

        sign_static = ((delta_static > 0) - 0.5) * 2
        sign_transient = ((delta_transient > 0) - 0.5) * 2
        diffusivities[electrode][direction][name]["lower"] = (
            gitt_scaling * current**2 * tau / delta_SOC**2
            * (
                (delta_static - sign_static * voltage_precision)
                / (delta_transient + sign_transient * voltage_precision)
            )**2
        )
        diffusivities[electrode][direction][name]["upper"] = (
            gitt_scaling * current**2 * tau / delta_SOC**2
            * (
                (delta_static + sign_static * voltage_precision)
                / (delta_transient - sign_transient * voltage_precision)
            )**2
        )

        ###############################################################
        # Calculate 1977 GITT with "Δ OCV over "Δ overpotential",     #
        # but this time correct the Δs to the proper short timescale. #
        ###############################################################

        diffusivities[electrode][direction][name + " (Δt↓)"] = {}
        diffusivities[electrode][direction][name + " (Δt↓)"]["mode"] = (
            gitt_scaling * current**2 * tau_short / delta_SOC_short**2
            * (delta_static_short / delta_transient_short)**2
        )
        sign_static_short = ((delta_static > 0) - 0.5) * 2
        sign_transient_short = ((delta_transient > 0) - 0.5) * 2
        diffusivities[electrode][direction][name + " (Δt↓)"]["lower"] = (
            gitt_scaling * current**2 * tau_short / delta_SOC_short**2
            * (
                (delta_static_short
                 - sign_static_short * voltage_precision * tau_short / tau)
                / (delta_transient_short + sign_transient_short
                   * voltage_precision * tau_short / tau)
            )**2
        )
        diffusivities[electrode][direction][name + " (Δt↓)"]["upper"] = (
            gitt_scaling * current**2 * tau_short / delta_SOC_short**2
            * (
                (delta_static_short
                 + sign_static_short * voltage_precision * tau_short / tau)
                / (delta_transient_short - sign_transient_short
                   * voltage_precision * tau_short / tau)
            )**2
        )

        # cull zeros that come from limited voltage precision (relaxation_end_)
        for variant in [name, name + " (Δt↓)"]:
            for location in ["mode", "lower", "upper"]:
                diffusivities[electrode][direction][variant][location][
                    diffusivities[electrode][direction][variant][location] <= 0
                ] = float('nan')

        ############################################################
        # Calculate 1977 GITT with "∂ OCV" over "∂ overpotential". #
        ############################################################

        name = "∂Uₛ/∂√t" if type == "voltage" else "∂ηₛ/∂√t"
        diffusivities[electrode][direction][name] = {}
        diffusivities[electrode][direction][name]["mode"] = (
            gitt_scaling * current**2
            * (derivative_static / derivative_transient)**2
        )

        #######################################################################
        # Extra correction for overlap with pulse's own relaxation behaviour. #
        #######################################################################

        name = "∂Uₛ/∂(√(t+τ)-√t)" if type == "voltage" else "∂ηₛ/∂(√(t+τ)-√t)"
        diffusivities[electrode][direction][name] = {}
        diffusivities[electrode][direction][name]["mode"] = (
            gitt_scaling * current**2
            * (derivative_static / derivative_transient_layered)**2
        )


#####################
# Plot the results. #
#####################

for electrode in ["negative", "positive"]:
    for direction in ["lithiation", "delithiation"]:
        socs = data["soc"][electrode][direction][
            electrode.capitalize() + " electrode SOC [-]"
        ]
        for approach, types in [
            ["direct", [
                "ΔUₛ/ΔUₜ",
                "ΔUₛ/ΔUₜ (Δt↓)",
                "∂Uₛ/∂√t",
                "∂Uₛ/∂(√(t+τ)-√t)",
                "∂ηₛ/∂(√(t+τ)-√t)"
            ]],
            ["inverse", ["∂ηₛ/∂(√(t+τ)-√t)", "SPM", "SPMe", "DFN"]]
        ]:
            fig, ax = plt.subplots(
                figsize=(3 * 2**0.5, 3.1), layout='constrained'
            )
            for type in types:
                diffusivity = diffusivities[electrode][direction][type]
                if "lower" in diffusivity.keys():
                    ax.errorbar(
                        socs,
                        diffusivity["mode"],
                        yerr=np.array([
                            np.abs(diffusivity["mode"] - diffusivity["lower"]),
                            np.abs(diffusivity["upper"] - diffusivity["mode"])
                        ]),
                        fmt='o' if approach == "direct" else '^',
                        elinewidth=0.75,
                        label=type,
                        capthick=1.2,
                        capsize=3,
                        markersize=4,
                    )
                else:
                    ax.plot(
                        socs,
                        diffusivity["mode"],
                        'o'
                        if approach == "direct" or type == "∂ηₛ/∂(√(t+τ)-√t)"
                        else '^',
                        label=type,
                        markersize=4,
                        color='black' if type == "∂ηₛ/∂(√(t+τ)-√t)" else None,
                    )
            ax.set_yscale('log')
            ax.set_ylabel(electrode.capitalize() + " electrode diff.  /  m²/s")
            ax.set_xlabel(electrode.capitalize() + " electrode SOC  /  -")
            handles, labels = ax.get_legend_handles_labels()
            order = [3, 4, 0, 1, 2] if len(labels) == 5 else [0, 1, 2, 3]
            fig.legend(
                [handles[index] for index in order],
                [labels[index] for index in order],
                loc='outside lower center',
                ncol=3 if len(labels) == 5 else 4,
                borderpad=0.3,
                handlelength=0.5,
                handletextpad=0.3,
                borderaxespad=0.0,
                columnspacing=0.5,
            )
            fig.savefig(
                "gitt_"
                + electrode
                + "_"
                + direction
                + "_"
                + approach
                + ".pdf",
                bbox_inches='tight',
                pad_inches=0.0
            )
plt.show()
