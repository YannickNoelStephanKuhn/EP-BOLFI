import json

from ep_bolfi.utility.preprocessing import (
    calculate_means_and_standard_deviations
)

with open(
    '../GITT estimation results/seven_parameter_estimation_seed_0.json', 'r'
) as f:
    result = json.load(f)
free_parameters = list(result['inferred parameters'].keys())
means, standard_deviations, error_bounds = (
    calculate_means_and_standard_deviations(
        result['inferred parameters'],
        result['covariance'],
        free_parameters,
        transform_parameters={
            "Negative electrode exchange-current density [A.m-2]": "log",
            "Positive electrode exchange-current density [A.m-2]": "log",
            "Negative electrode diffusivity [m2.s-1]": "log",
            "Positive electrode diffusivity [m2.s-1]": "log",
        },
        bounds_in_standard_deviations=1,
        epsabs=1e-12, epsrel=1e-12
    )
)
print("Results for the estimation of seven parameters:")
print("bounds of one standard deviation:")
for k, v in error_bounds.items():
    print(k, v)
print("means and standard deviations:")
for k in means.keys():
    print(k, means[k], standard_deviations[k])
print()
