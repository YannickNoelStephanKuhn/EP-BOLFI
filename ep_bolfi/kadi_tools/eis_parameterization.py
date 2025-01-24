"""
Reads in an input file prepared by
``ep_bolfi.kadi_tools.eis_preprocessing`` and runs EP-BOLFI on it.
"""

from copy import deepcopy
from contextlib import redirect_stdout
import json
from os import linesep
from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.eis_parameterization',
    version='3.0'
)
@xmlhelpy.option(
    'input-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the optimizer input."
    )
)
@xmlhelpy.option(
    'input-file',
    char='f',
    param_type=xmlhelpy.String,
    required=True,
    description="File name of the optimizer input file."
)
@xmlhelpy.option(
    'parameters-file',
    char='m',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "File name of the model parameters. It must be a Python file and "
        "contain the following global variables:"
        + linesep + linesep
        + " - parameters: The dictionary of parameters to pass on to the solver."
        " May be a ep_bolfi.utility.preprocessing.SubstitutionDict."
        + linesep + linesep
        + " - unknowns: The dictionary of unknown parameters. Instead of single "
        "values as in 'parameters', input 2-tuples with their lower and upper "
        "bounds, e.g. from literature."
        + linesep + linesep
        + "It may contain the additional following global variables:"
        + linesep + linesep
        + " - transform_unknowns: Dictionary for transforming unknowns before "
        "inferring them via a normal distribution. Either give as 2-tuples "
        "with the first entry being the back-transformation and the second "
        "one being the transformation. For convenience, putting 'log' gives "
        "log-normal distributions."
        + linesep + linesep
        + " - negative_SOC_from_cell_SOC: A callable, used for OCV subtraction."
        + linesep + linesep
        + " - positive_SOC_from_cell_SOC: A callable, used for OCV subtraction."
        + linesep + linesep
        + " - uncertainties: The dictionary of parameter uncertainties. Used for"
        " scrambling them in the simulation samples. Give them as tuples: the "
        "first entry is the name of the distribution in scipy.stats, and the "
        "following are its parameters. Example: ('norm', mean, std)."
    )
)
@xmlhelpy.option(
    'output-record',
    char='o',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record to store the output of "
        "the optimizer in."
    )
)
@xmlhelpy.option(
    'seed',
    char='s',
    param_type=xmlhelpy.Integer,
    default=None,
    description="Seed for RNG. Set to a number for unvarying results."
)
@xmlhelpy.option(
    'overwrite',
    char='w',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Whether or not an already existing file by the same name in the "
        "record gets overwritten."
    )
)
def eis_parameterization(
    input_record,
    input_file,
    parameters_file,
    output_record,
    seed,
    overwrite
):
    """Please refer to the --help output of this file."""
    from ep_bolfi import EP_BOLFI
    from ep_bolfi.kadi_tools.eis_preprocessing import (
        eis_feature_names, eis_features
    )
    from ep_bolfi.models.analytic_impedance import AnalyticImpedance
    from kadi_apy.lib.core import KadiManager, Record
    from numpy import array, geomspace, ndarray
    from numpy.random import RandomState
    import scipy

    manager = KadiManager()

    file_prefix = parameters_file.split(".")[0]

    if not isfile(input_file) or not isfile("local_parameter_file.py"):
        input_record_handle = Record(manager, id=input_record, create=False)
    if not isfile(input_file):
        input_id = input_record_handle.get_file_id(input_file)
        input_record_handle.download_file(input_id, input_file)
    if not isfile("local_parameter_file.py"):
        parameters_id = input_record_handle.get_file_id(parameters_file)
        input_record_handle.download_file(
            parameters_id, "local_parameter_file.py"
        )
    from local_parameter_file import parameters, transform_unknowns, unknowns

    with open(input_file, 'r') as f:
        input_data = json.load(f)
    number_of_peaks = input_data['number_of_peaks']
    frequency_limits = input_data['frequency_limits']
    # experiment_features = input_data['experiment_features']
    initial_socs = input_data['initial_socs']
    three_electrode = input_data['three_electrode']
    dimensionless_reference_electrode_location = (
        input_data['dimensionless_reference_electrode_location']
    )
    lambda_values = input_data['lambda_values']
    white_noise = input_data['white_noise']
    uncertainties = input_data['uncertainties']
    discretization = input_data['discretization']
    optimizer_settings = input_data['optimizer_settings']
    experiment_data = input_data['experiment_data']

    # Make the separate real and imaginary part into complex again.
    experiment_data = [
        array(experiment_data[0]),
        array(experiment_data[1]) + 1j * array(experiment_data[2])
    ]

    # Note: geomspace is logspace with directly defined endpoints.
    f_eval = geomspace(*frequency_limits, discretization)
    s_eval = 1j * f_eval

    white_noise_generator_real = scipy.stats.norm(0, white_noise)
    white_noise_generator_real.random_state = RandomState(seed=seed + 1)
    white_noise_generator_imag = scipy.stats.norm(0, white_noise)
    white_noise_generator_imag.random_state = RandomState(seed=seed + 2)
    parameter_noise_rng = {}
    for i, (p_name, (s_name, *args)) in enumerate(uncertainties.items()):
        parameter_noise_rng[p_name] = getattr(scipy.stats, s_name)(*args)
        parameter_noise_rng[p_name].random_state = RandomState(
            seed=seed + i + 3
        )

    def eis_simulator(trial_parameters):
        # Trial parameters are served as arrays, since ELFI supports
        # batching. With BOLFI, it is always a 1-array though.
        # The symbolic calculations in PyBaMM do not support batching.
        for k, v in trial_parameters.items():
            if isinstance(v, ndarray):
                trial_parameters[k] = v[0]
        # Apply the correct initial SOC.
        trial_parameters_by_segment = []
        for initial_soc in initial_socs:
            trial_parameters_by_segment.append(deepcopy(trial_parameters))
            for electrode in ["negative", "positive"]:
                soc_value = initial_soc[
                    "Initial concentration in "
                    + electrode
                    + " electrode [mol.m-3]"
                ]
                if soc_value is not None:
                    trial_parameters_by_segment[-1][
                        "Initial concentration in "
                        + electrode
                        + " electrode [mol.m-3]"
                    ] = soc_value
            # Apply the parameter noise.
            for p_name, p_rng in parameter_noise_rng.items():
                trial_parameters_by_segment[-1][p_name] = p_rng.rvs(size=1)[0]
            # In case of a SubstitutionDict, make it into a dict.
            trial_parameters_by_segment[-1] = {
                k: v for k, v in trial_parameters_by_segment[-1].items()
            }
        solutions = [[], []]
        for tp in trial_parameters_by_segment:
            if three_electrode is None:
                solution = AnalyticImpedance(
                    tp, catch_warnings=False
                ).Z_SPMe(s_eval)
            else:
                solution = AnalyticImpedance(
                    tp, catch_warnings=False
                ).Z_SPMe_reference_electrode(
                    s_eval,
                    three_electrode,
                    dimensionless_reference_electrode_location
                )
            # Apply the measurement noise.
            solutions[0].append(f_eval)
            solutions[1].append(
                solution
                + white_noise_generator_real.rvs(size=len(solution))
                + 1j * white_noise_generator_imag.rvs(size=len(solution))
            )
        return solutions

    def features(dataset):
        try:
            return eis_features(dataset, number_of_peaks, lambda_values)
        except (AssertionError, ValueError):
            # scipy.optimize._nnls throws ValueError if the data
            # containts infs or NaNs.
            # pyimpspec throws AssertionError when the dataset has
            # length zero.
            # Note that this blanket except also returns zeroed features
            # if another than the current feature errored out.
            return array([[0.0, 0.0] for _ in range(len_features)])

    len_features = len(features(experiment_data))

    if not number_of_peaks:
        number_of_peaks = len_features

    def feature_names(index):
        segment_number = index // abs(number_of_peaks)
        return (
            eis_feature_names[0]
            + " and "
            + eis_feature_names[1]
            + " (segment #"
            + str(segment_number)
            + ", peak #"
            + str(index - segment_number * abs(number_of_peaks))
            + ")"
        )

    estimator = EP_BOLFI(
        [eis_simulator],
        [experiment_data],
        [features],
        parameters,
        free_parameters_boundaries=unknowns,
        transform_parameters=transform_unknowns,
        display_current_feature=[feature_names],
    )

    if 'seed' in optimizer_settings.keys():
        ep_bolfi_seed = optimizer_settings.pop('seed')
    else:
        ep_bolfi_seed = seed

    with open(file_prefix + '_optimization.log', 'w') as f:
        with redirect_stdout(f):
            estimator.run(seed=ep_bolfi_seed, **optimizer_settings)

    with open(file_prefix + '_parameterization.json', 'w') as f:
        f.write(estimator.result_to_json(seed=ep_bolfi_seed))

    with open(file_prefix + '_evaluations.json', 'w') as f:
        f.write(estimator.log_to_json())

    output_record_handle = Record(manager, id=output_record, create=False)
    # output_record_handle.upload_file(
    #     file_prefix + '_evaluations.log', force=overwrite
    # )
    output_record_handle.upload_file(
        file_prefix + '_optimization.log', force=overwrite
    )
    output_record_handle.upload_file(
        file_prefix + '_parameterization.json', force=overwrite
    )
    output_record_handle.upload_file(
        file_prefix + '_evaluations.json', force=overwrite
    )


if __name__ == '__main__':
    eis_parameterization()
