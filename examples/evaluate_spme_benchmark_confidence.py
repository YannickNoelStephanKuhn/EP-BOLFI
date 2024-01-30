import json

from ep_bolfi.utility.preprocessing import (
    calculate_means_and_standard_deviations
)


def print_mean_and_standard_deviation(filename):
    with open(filename, 'r') as f:
        result = json.load(f)
    free_parameters = list(result['inferred parameters'].keys())
    """
    from ep_bolfi.utility.preprocessing import approximate_confidence_ellipsoid
    _, confidence_semiaxes = approximate_confidence_ellipsoid(
        parameters,
        free_parameters,
        result['covariance'],
        mean=result['inferred parameters'],
        transform_parameters={
            "Electrolyte diffusivity [m2.s-1]": "log",
            "Cation transference number": "none",
            "Negative electrode diffusivity [m2.s-1]": "log",
            "Positive electrode diffusivity [m2.s-1]": "log",
            "variance of the output noise": "log",
        },
        refinement=True,
        confidence=0.95
    )
    bounds = {
        name: uncertainty
        for name, uncertainty in zip(
            free_parameters,
            np.max(confidence_semiaxes, axis=0)
            - np.min(confidence_semiaxes, axis=0)
        )
    }
    """
    means, standard_deviations, error_bounds = (
        calculate_means_and_standard_deviations(
            result['inferred parameters'],
            result['covariance'],
            free_parameters,
            transform_parameters={
                "Electrolyte diffusivity [m2.s-1]": "log",
                "Cation transference number": "none",
                "Negative electrode diffusivity [m2.s-1]": "log",
                "Positive electrode diffusivity [m2.s-1]": "log",
                "variance of the output noise": "log",
            },
            bounds_in_standard_deviations=1,
            epsabs=1e-12, epsrel=1e-12
        )
    )
    print(filename)
    print("bounds of one standard deviation:")
    for k, v in error_bounds.items():
        print(k, v)
    print("means and standard deviations:")
    for k in means.keys():
        print(k, means[k], standard_deviations[k])
    print()


for sample_number in [2080, 4160, 6240, 8320]:
    print_mean_and_standard_deviation(
        './spme_benchmark_results/unimodal/'
        + str(sample_number)
        + '_samples.json'
    )

for soc_point in range(11):
    print_mean_and_standard_deviation(
        './spme_benchmark_results/multimodal/individual_excitation_points/'
        + str(6240)
        + '_samples_at_soc_point_'
        + str(soc_point)
        + '.json'
    )

for sample_number in [2080, 4160]:
    print_mean_and_standard_deviation(
        './spme_benchmark_results/multimodal/individual_excitation_points/'
        + str(sample_number)
        + '_samples_at_soc_point_'
        + str(10)
        + '.json'
    )

print_mean_and_standard_deviation(
    '../spme_benchmark_results/multimodal/'
    + '6240_samples_at_soc_point_0_2_5_10.json'
)
