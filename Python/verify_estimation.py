# Copyright (c): German Aerospace Center (DLR)
from contextlib import redirect_stdout
from runpy import run_module
import json
import numpy as np

from parameters.estimation.gitt import perform_gitt_estimation

# Limits the extent of each next prior.
prior_limits = {
    "Negative electrode exchange-current density [A.m-2]":
        (0.5, 80.0),
    "Positive electrode exchange-current density [A.m-2]":
        (0.5, 80.0),
    "Negative electrode diffusivity [m2.s-1]":
        (1e-14, 1e-10),
    "Positive electrode diffusivity [m2.s-1]":
        (1e-14, 1e-10),
}
magnification = 1 / 4

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
    for electrode in ['Negative ', 'Positive ']:
        for component in [' (electrolyte)', ' (electrode)']:
            parameter_estimates[
                electrode + 'electrode Bruggeman coefficient' + component
            ] = parameter_estimates[
                electrode + 'electrode Bruggeman coefficient'
            ]
    parameter_estimates['1 + dlnf/dlnc'] = 1.475 / (
        1 - parameter_estimates['Cation transference number']
    )
    # all_error_bounds = results['error bounds']
    # """! The uncertainty in the estimation of the SOC-independent parameters. """
    # parameter_errorbars = {
    #     name: all_error_bounds[name]
    #     for name in [
    #         "Cation transference number",
    #         "Negative electrode Bruggeman coefficient",
    #         "Positive electrode Bruggeman coefficient",
    #     ]
    # }

with open(
    '../GITT estimation results/seven_parameter_estimation_seed_0.json',
    'r'
) as f:
    all_error_bounds = json.load(f)['error bounds']
    free_parameters_boundaries = {
        name: all_error_bounds[name]
        for name in [
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
            "Negative electrode diffusivity [m2.s-1]",
            "Positive electrode diffusivity [m2.s-1]",
        ]
    }

free_parameters_boundaries = {
    name: (
        np.max([
            np.exp((np.log(prior_limits[name][0])
                    - np.log(prior_limits[name][1])) * magnification)
            * bounds[0],
            prior_limits[name][0]
        ]),
        np.min([
            np.exp((np.log(prior_limits[name][1])
                    - np.log(prior_limits[name][0])) * magnification)
            * bounds[1],
            prior_limits[name][1]
        ])
    )
    for name, bounds in free_parameters_boundaries.items()
}
for name, bounds in free_parameters_boundaries.items():
    limit = prior_limits[name]
    if (
        bounds[0] == limit[0]
        and
        bounds[1] < np.exp(
            (np.log(limit[1]) - np.log(limit[0])) * 2 * magnification
        ) * bounds[0]
    ):
        free_parameters_boundaries[name] = (
            bounds[0],
            np.exp((np.log(limit[1]) - np.log(limit[0])) * 2 * magnification)
            * bounds[0]
        )
    elif (
        bounds[1] == limit[1]
        and
        bounds[0] > np.exp(
            (np.log(limit[0]) - np.log(limit[1])) * 2 * magnification
        ) * bounds[1]
    ):
        free_parameters_boundaries[name] = (
            np.exp((np.log(limit[0]) - np.log(limit[1])) * 2 * magnification)
            * bounds[1],
            bounds[1]
        )

for pulse_number in range(84, 0 - 1, -1):
    with open(
        '../GITT estimation verification/pulses/pulse_' + str(pulse_number)
        + '.log', 'w'
    ) as f:
        with redirect_stdout(f):
            data = run_module(
                'parameters.estimation.gitt_basf',
                init_globals={
                    'optimize_simulation_speed': True,
                    'soc_dependent_estimation': True,
                    'white_noise': True,
                    'parameter_noise': False,
                    'pulse_number': pulse_number,
                    'free_parameters': None,
                    'free_parameters_boundaries': free_parameters_boundaries,
                    'transform_parameters': {
                        "Negative electrode exchange-current density [A.m-2]":
                            "log",
                        "Positive electrode exchange-current density [A.m-2]":
                            "log",
                        "Negative electrode diffusivity [m2.s-1]": "log",
                        "Positive electrode diffusivity [m2.s-1]": "log",
                    },
                    'seed': 0,
                }
            )
            with open(
                '../GITT estimation results/pulses/pulse_' + str(pulse_number)
                + '.json',
                'r'
            ) as g:
                estimation_result = json.load(g)['inferred parameters']
            estimation_result.update(parameter_estimates)
            experimental_dataset = data['simulator'](estimation_result)
            estimation_verification = perform_gitt_estimation(
                data['simulator'],
                experimental_dataset,
                data['fixed_parameters'],
                free_parameters=data['free_parameters'],
                free_parameters_boundaries=data['free_parameters_boundaries'],
                transform_parameters=data['transform_parameters'],
                bolfi_initial_evidence=65,
                bolfi_total_evidence=130,
                bolfi_posterior_samples=27,
                ep_iterations=4,
                final_dampening=0.5,
                verbose=True,
                seed=0
            )
    free_parameters_boundaries = {
        name: (
            np.max([
                np.exp((np.log(prior_limits[name][0])
                        - np.log(prior_limits[name][1])) * magnification)
                * bounds[0],
                prior_limits[name][0]
            ]),
            np.min([
                np.exp((np.log(prior_limits[name][1])
                        - np.log(prior_limits[name][0])) * magnification)
                * bounds[1],
                prior_limits[name][1]
            ])
        )
        for name, bounds in estimation_verification.final_error_bounds.items()
    }
    for name, bounds in free_parameters_boundaries.items():
        limit = prior_limits[name]
        if (
            bounds[0] == limit[0]
            and
            bounds[1] < np.exp(
                (np.log(limit[1]) - np.log(limit[0])) * 2 * magnification
            ) * bounds[0]
        ):
            free_parameters_boundaries[name] = (
                bounds[0],
                np.exp((np.log(limit[1]) - np.log(limit[0]))
                       * 2 * magnification)
                * bounds[0]
            )
        elif (
            bounds[1] == limit[1]
            and
            bounds[0] > np.exp(
                (np.log(limit[0]) - np.log(limit[1])) * 2 * magnification
            ) * bounds[1]
        ):
            free_parameters_boundaries[name] = (
                np.exp((np.log(limit[0]) - np.log(limit[1]))
                       * 2 * magnification)
                * bounds[1],
                bounds[1]
            )
    print()
    print('finished pulse #' + str(pulse_number))
    with open(
        '../GITT estimation verification/pulses/pulse_' + str(pulse_number)
        + '.json', 'w'
    ) as f:
        json.dump({
            "inferred parameters": estimation_verification.inferred_parameters,
            "covariance":
                [list(line)
                 for line in estimation_verification.final_covariance],
            "correlation":
                [list(line)
                 for line in estimation_verification.final_correlation],
            "error bounds": estimation_verification.final_error_bounds,
        }, f)
