from contextlib import redirect_stdout
import json
import numpy as np
from runpy import run_module

from parameters.estimation.gitt import build_gitt_estimator

seed = 0
# Estimates all 7 parameters of interest from pulses 66 and 67.
data = run_module(
    'parameters.estimation.gitt_basf',
    init_globals={
        'optimize_simulation_speed': True,
        'soc_dependent_estimation': False,
        'white_noise': True,
        'parameter_noise': False,
        'pulse_number': (66, 2),
        'free_parameters': None,
        'seed': seed,
    }
)
with open(
    '../GITT estimation results/seven_parameter_estimation_seed_'
    + str(seed)
    + '.log', 'w'
) as f:
    with redirect_stdout(f):
        estimator = build_gitt_estimator(
            data['simulator'],
            data['experimental_dataset'],
            data['fixed_parameters'],
            free_parameters=data['free_parameters'],
            free_parameters_boundaries=data['free_parameters_boundaries'],
            transform_parameters=data['transform_parameters'],
        )
        estimator.run(
            bolfi_initial_evidence=129,
            bolfi_total_evidence=258,
            bolfi_posterior_samples=35,
            ep_iterations=4,
            final_dampening=0.5,
            verbose=True,
            seed=seed,
        )
with open(
    '../GITT estimation results/seven_parameter_estimation_seed_'
    + str(seed)
    + '.json', 'w'
) as f:
    json.dump({
        "inferred parameters": estimator.inferred_parameters,
        "covariance":
            [list(line) for line in estimator.final_covariance],
        "correlation":
            [list(line) for line in estimator.final_correlation],
        "error bounds": estimator.final_error_bounds,
    }, f)


with open(
    '../GITT estimation results/seven_parameter_estimation_seed_'
    + str(seed)
    + '.json',
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

# Loop for estimating the 4 SOC-dependent parameters of interest.
for pulse_number in range(84, 0 - 1, -1):
    with open(
        '../GITT estimation results/pulses/pulse_' + str(pulse_number)
        + '.log', 'w'
    ) as f:
        with redirect_stdout(f):
            data = run_module(
                'parameters.estimation.gitt_basf',
                init_globals={
                    'optimize_simulation_speed': True,
                    'soc_dependent_estimation': True,
                    'white_noise': True,
                    'parameter_noise': True,
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
                    'seed': seed,
                }
            )
            estimator = build_gitt_estimator(
                data['simulator'],
                data['experimental_dataset'],
                data['fixed_parameters'],
                free_parameters=data['free_parameters'],
                free_parameters_boundaries=data['free_parameters_boundaries'],
                transform_parameters=data['transform_parameters'],

            )
            estimator.run(
                bolfi_initial_evidence=65,
                bolfi_total_evidence=130,
                bolfi_posterior_samples=20,
                ep_iterations=4,
                final_dampening=0.5,
                verbose=True,
                seed=seed,
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
        for name, bounds in estimator.final_error_bounds.items()
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
        '../GITT estimation results/pulses/pulse_' + str(pulse_number)
        + '.json', 'w'
    ) as f:
        json.dump({
            "inferred parameters": estimator.inferred_parameters,
            "covariance":
                [list(line) for line in estimator.final_covariance],
            "correlation":
                [list(line) for line in estimator.final_correlation],
            "error bounds": estimator.final_error_bounds,
        }, f)
