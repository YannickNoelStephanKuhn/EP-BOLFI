"""!@file
GITT parameter estimation setup.
"""

from copy import deepcopy

from ep_bolfi import EP_BOLFI
from ep_bolfi.utility.fitting_functions import (
    fit_exponential_decay, fit_sqrt
)
from ep_bolfi.utility.preprocessing import (
    calculate_means_and_standard_deviations, find_occurrences
)


def exponential_decay_features(dataset):
    """!@brief Defines the features.
    @param dataset
        The dataset in the same form as the experimental_dataset and
        the output of the simulators.
    @return
        The exponential decay fit parameters.
    """

    features = []
    for t0, t1, U0, U1 in zip(dataset[0][:-1:2], dataset[0][1::2],
                              dataset[1][:-1:2], dataset[1][1::2]):
        exp_decay_0 = fit_exponential_decay(t0, U0, threshold=0.95)[0]
        # exp_decay_1 = fit_exponential_decay(t1, U1, threshold=0.95)[0]
        # Do not use the pause voltage limit 0V (no useful information).
        # All it states is an error in OCV, which is not fitted here.
        # Do not use the pause relaxation time (not fittable with DFN).
        # The slow relaxation may be fitted by a multi-particle model.
        features.append(1.0 / exp_decay_0[2][2])
    # print(features)
    return features


def sqrt_features(dataset):
    """!@brief Defines the features.
    @param dataset
        The dataset in the same form as the experimental_dataset and
        the output of the simulators.
    @return
        The extrapolations and square root slopes at the beginning of
        each segment.
    """

    features = []
    for i, (t, U) in enumerate(zip(dataset[0], dataset[1])):
        # Fit the square root slope to the first 30 seconds only.
        cutoff = find_occurrences(t, t[0] + 30.0)[0]
        sqrt_fit = fit_sqrt(t[:cutoff], U[:cutoff], threshold=0.95)[2]
        features.extend([sqrt_fit[0], 1.0 / sqrt_fit[1]])
    # print(features)
    return features


def name_of_exp_features(index):

    return {
        0: "discharge relaxation time",
    }[index % 1] + " (pulse #" + str(index // 1) + ")"


def name_of_sqrt_features(index):

    return {
        0: "ohmic voltage drop",
        1: "GITT square root slope",
        2: "concentration overpotential",
        3: "ICI square root slope"
    }[index % 4] + " (pulse #" + str(index // 4) + ")"


def build_gitt_estimator(
    simulator,
    experimental_dataset,
    fixed_parameters,
    free_parameters=None,
    free_parameters_boundaries=None,
    Q=None,
    r=None,
    Q_features=None,
    r_features=None,
    transform_parameters=None
):

    estimator = EP_BOLFI(
        [simulator] * 2, [experimental_dataset] * 2,
        [exponential_decay_features, sqrt_features],
        fixed_parameters, free_parameters=free_parameters,
        free_parameters_boundaries=free_parameters_boundaries,
        Q=Q, r=r, Q_features=Q_features, r_features=r_features,
        transform_parameters=transform_parameters,
        display_current_feature=[name_of_exp_features, name_of_sqrt_features]
    )

    return estimator


def calculate_experimental_and_simulated_features(
    simulators,
    experimental_datasets,
    parameter_sets,
    estimation_results,
    free_parameters,
    transform_parameters,
    bounds_in_standard_deviations=1,
):

    example_parameters = parameter_sets[0]
    example_simulator = simulators[0]
    example_parameters.update(estimation_results[0]['inferred parameters'])
    example_dataset = example_simulator(example_parameters)
    example_exp_features = exponential_decay_features(example_dataset)
    example_sqrt_features = sqrt_features(example_dataset)
    feature_names = [name_of_exp_features(i)
                     for i in range(len(example_exp_features))]
    feature_names.extend([name_of_sqrt_features(i)
                          for i in range(len(example_sqrt_features))])
    parameter_names = list(estimation_results[0]['inferred parameters'].keys())

    evaluations = {
        f_name: [0.0 for i in range(len(estimation_results))]
        for f_name in feature_names
    }
    sensitivities = {
        p_name: {
            f_name: [[0.0, 0.0] for i in range(len(estimation_results))]
            for f_name in feature_names
        }
        for p_name in parameter_names
    }

    for i, estimation_result in enumerate(estimation_results):
        estimate = estimation_result['inferred parameters']
        parameters = parameter_sets[i]
        simulator = simulators[i]
        input_parameters = deepcopy(parameters)
        input_parameters.update(estimate)
        output_dataset = simulator(input_parameters)
        output_exp_features = exponential_decay_features(output_dataset)
        output_sqrt_features = sqrt_features(output_dataset)
        for j, feature in enumerate(output_exp_features):
            evaluations[name_of_exp_features(j)][i] = feature
        for j, feature in enumerate(output_sqrt_features):
            evaluations[name_of_sqrt_features(j)][i] = feature

        _, _, errorbars = calculate_means_and_standard_deviations(
            estimate,
            estimation_result['covariance'],
            free_parameters,
            transform_parameters=transform_parameters,
            bounds_in_standard_deviations=bounds_in_standard_deviations,
            epsabs=1e-12, epsrel=1e-12
        )
        for p_name, limits in errorbars.items():
            for k, value in enumerate(limits):
                input_parameters = deepcopy(parameters)
                input_parameters.update(estimate)
                input_parameters.update({p_name: value})
                output_dataset = simulator(input_parameters)
                output_exp_features = exponential_decay_features(
                    output_dataset)
                output_sqrt_features = sqrt_features(output_dataset)
                for j, feature in enumerate(output_exp_features):
                    sensitivities[p_name][name_of_exp_features(
                        j)][i][k] = feature
                for j, feature in enumerate(output_sqrt_features):
                    sensitivities[p_name][name_of_sqrt_features(
                        j)][i][k] = feature

    experimental_features = {
        f_name: [0.0 for i in range(len(estimation_results))]
        for f_name in feature_names
    }

    for i in range(len(estimation_results)):
        experiment = experimental_datasets[i]
        experimental_exp_features = exponential_decay_features(experiment)
        experimental_sqrt_features = sqrt_features(experiment)
        for j, feature in enumerate(experimental_exp_features):
            experimental_features[name_of_exp_features(j)][i] = feature
        for j, feature in enumerate(experimental_sqrt_features):
            experimental_features[name_of_sqrt_features(j)][i] = feature

    return evaluations, sensitivities, experimental_features
