"""
Reads in an ``Impedance_Measurement`` json representation and
preprocesses it as EIS data, assuming each segment corresponds to an
equilibrium.
"""

from ast import literal_eval
import json
from os import linesep
from os.path import isfile
import sys
import xmlhelpy


eis_feature_names = [
    "logarithm of RC time constant",
    "resistance R of RC element"
]

eis_feature_shortnames = [
    "log(τ) [log(s)]", "R [Ω]"
]


def eis_features_by_segment(dataset, number_of_peaks, lambda_values):
    """
    Calculates DRT features for each EIS segment.

    :param dataset:
        An ``Impedance_Measurement`` object.
    :param number_of_peaks:
        DRT peaks get detected automatically. Default is to keep all.
        Set to a negative integer to get the first ``-number_of_peaks``,
        and to a positive one for the last ``number_of_peaks`` peaks.
    :param lambda_values:
        Please refer to the ``pyimpspec`` documentation for these.
        Either set one value for all segments, or a list of these for
        each segment. Defaults to an optimally calculated lambda value.
    :returns:
        A 2-tuple, with the list of DRT features for each segment, and
        the list of complete DRT result objects themselves.
    """
    from ep_bolfi.utility.fitting_functions import fit_drt
    import numpy as np
    drt_features = []
    drts = []
    if type(lambda_values) is not list:
        lambda_values = [lambda_values] * len(dataset)
    for f, Z, lv in zip(*dataset, lambda_values):
        drt_tau, drt_resistance, drt = fit_drt(f, Z, lv)
        drts.append(drt)
        drt_log_tau = [np.log(tau) for tau in drt_tau]
        all_drt_features = [
            np.array([dlt, dr])
            for dlt, dr in zip(drt_log_tau, drt_resistance)
        ]
        if number_of_peaks is None:
            drt_features.append(all_drt_features)
        elif number_of_peaks < 0:
            drt_features.append(all_drt_features[number_of_peaks:])
        else:
            drt_features.append(all_drt_features[:number_of_peaks])
    return drt_features, drts


def eis_features(dataset, number_of_peaks, lambda_values):
    """
    Collects DRT features from all segments into a contiguos array.

    :param dataset:
        An ``Impedance_Measurement`` object.
    :param number_of_peaks:
        DRT peaks get detected automatically. Default is to keep all.
        Set to a negative integer to get the first ``-number_of_peaks``,
        and to a positive one for the last ``number_of_peaks`` peaks.
    :param lambda_values:
        Please refer to the ``pyimpspec`` documentation for these.
        Either set one value for all segments, or a list of these for
        each segment. Defaults to an optimally calculated lambda value.
    :returns:
        A list that is the concatenation of all DRT features.
    """
    drt_features, _ = eis_features_by_segment(
        dataset, number_of_peaks, lambda_values
    )
    features = []
    for drt_f in drt_features:
        features.extend(drt_f)
    return features


def eis_feature_visualizer(dataset, number_of_peaks, lambda_values):
    """
    Visualizes DRT features for each EIS segment.

    :param dataset:
        An ``Impedance_Measurement`` object.
    :param number_of_peaks:
        DRT peaks get detected automatically. Default is to keep all.
        Set to a negative integer to get the first ``-number_of_peaks``,
        and to a positive one for the last ``number_of_peaks`` peaks.
    :param lambda_values:
        Please refer to the ``pyimpspec`` documentation for these.
        Either set one value for all segments, or a list of these for
        each segment. Defaults to an optimally calculated lambda value.
    :returns:
        A list of 3-tuples, where each of them represents one semicircle
        from the DRT approximation. First entry is the frequencies over
        which the semicircle in the second entry is plotted, and the
        third is a label describing the semicircle parameters.
    """
    from ep_bolfi.utility.fitting_functions import fit_drt
    from ep_bolfi.utility.preprocessing import find_occurrences
    import numpy as np
    visualization = []
    for f, Z, lv in zip(*dataset, lambda_values):
        drt_tau, drt_resistance, drt = fit_drt(f, Z, lv)
        drt_log_tau = [np.log(tau) for tau in drt_tau]
        drt_features = [
            np.array([dlt, dr])
            for dlt, dr in zip(drt_log_tau, drt_resistance)
        ]
        if number_of_peaks is None:
            features = drt_features
        elif number_of_peaks < 0:
            features = drt_features[number_of_peaks:]
        else:
            features = drt_features[:number_of_peaks]
        for log_tau, resistance in features:
            # Geometric interpretation: place the semicircle so that its
            # peak has the same real part as the original impedance at the
            # time constant.
            tau = np.exp(log_tau)
            time_constant_index = find_occurrences(f, 1 / tau)[0]
            res_0 = np.real(Z[time_constant_index]) - 0.5 * resistance
            label = (
                eis_feature_shortnames[0]
                + ': {0:.4g}'.format(log_tau)
                + linesep
                + eis_feature_shortnames[1]
                + ': {0:.4g}'.format(resistance)
            )
            visualization.append(
                (f, res_0 + resistance / (1 + 1j * f * tau), label)
            )
    return visualization


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.eis_preprocessing',
    version='3.0'
)
@xmlhelpy.option(
    'ocv-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the OCV data."
    )
)
@xmlhelpy.option(
    'ocv-file',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "The name of the OCV file with electrode SOC to cell SOC assignment. "
        "Prepare it like ep_bolfi.kadi_tools.extract_ocv_curve does. "
        "The 'indices' are what matches SOC to impedance measurements."
    )
)
@xmlhelpy.option(
    'parameters-record',
    char='k',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the model parameters."
    )
)
@xmlhelpy.option(
    'parameters-file',
    char='q',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "File name of the model parameters. It must be a Python file and "
        "contain the following global variables:"
        + linesep + linesep
        + " - parameters: The dictionary of parameters to pass on to the "
        "solver. May be a ep_bolfi.utility.preprocessing.SubstitutionDict."
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
        "Persistent record identifier of the record to store the input for "
        "the optimizer in."
    )
)
@xmlhelpy.option(
    'output-suffix',
    char='u',
    param_type=xmlhelpy.String,
    default=None,
    description=(
        "Optional string to append to the end of the name of the output file."
    ),
)
@xmlhelpy.option(
    'number-of-peaks',
    char='p',
    default=None,
    param_type=xmlhelpy.Integer,
    description=(
        "The number of peaks identified by Distribution of Relaxation Times "
        "that are to be used for fitting. Positive numbers refer to the first "
        "x peaks and negative numbers to the last -x peaks."
        "Default is to use all peaks."
    )
)
@xmlhelpy.option(
    'frequency-limits',
    char='f',
    default='[None, None]',
    param_type=xmlhelpy.String,
    description=(
        "Limits the frequencies that will be used for fitting. "
        "Default is the extent of the frequencies in the data."
    )
)
@xmlhelpy.option(
    'subsampling',
    char='s',
    default=1,
    param_type=xmlhelpy.IntRange(min=1),
    description=(
        "With this as x, every x-th datapoint is considered. This has "
        "a practical effect on the DRT, as a DRT with too many "
        "datapoints may degrade. Also speeds up simulations."
    )
)
@xmlhelpy.option(
    'three-electrode',
    char='t',
    default=None,
    param_type=xmlhelpy.Choice(
        [None, 'positive', 'negative'], case_sensitive=True
    ),
    description=(
        "With 'None', does nothing (i.e., cell potentials are used). If "
        "set to either 'positive' or 'negative', instead of cell "
        "potentials, the base for the displayed voltage will be the "
        "potential of the 'positive' or 'negative' electrode against a "
        "reference electrode. For placement of said reference electrode, "
        "please refer to 'dimensionless-reference-electrode-location'."
    )
)
@xmlhelpy.option(
    'dimensionless-reference-electrode-location',
    char='a',
    default=0.5,
    param_type=xmlhelpy.FloatRange(min=0.0, max=1.0),
    description=(
        "The location of the reference electrode, given as a scalar "
        "between 0 (placed at the point where negative electrode and "
        "separator meet) and 1 (placed at the point where positive "
        "electrode and separator meet). Defaults to 0.5 (in the middle)."
    )
)
@xmlhelpy.option(
    'lambda_value',
    char='l',
    default=-2.0,
    param_type=xmlhelpy.Float,
    description=(
        "Controls the smoothing of the impedance data via Tikhonov "
        "regularization. For details, consult the pyimpspec documentation. "
        "Defaults to a procedure that calculates this value optimally."
    )
)
@xmlhelpy.option(
    'white-noise',
    char='j',
    default=0.0,
    param_type=xmlhelpy.Float,
    description=(
        "Standard deviation of white noise superimposed on the measurement. "
        "This is used in the simulator later and has no impact on the "
        "experimental data. Set to the measurement noise in the experiment."
    )
)
@xmlhelpy.option(
    'discretization',
    char='d',
    default=200,
    param_type=xmlhelpy.Integer,
    description=(
        "Discretization points, i.e., amount of frequencies at which the "
        "simulated impedance gets evaluated. The frequencies will be "
        "log-spaced between minimum and maximum frequency as given above."
    )
)
@xmlhelpy.option(
    'optimizer-settings',
    char='e',
    param_type=xmlhelpy.String,
    default='{}',
    description=(
        "Settings that are passed on to ep_bolfi.ep_bolfi.EP_BOLFI.EP_BOLFI.r"
        "un. Please see there for the available options. Format as a Python d"
        "ictionary: e.g., '{\"ep_iterations\": 3, ...}'."
    )
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
def eis_preprocessing(
    ocv_record,
    ocv_file,
    parameters_record,
    parameters_file,
    output_record,
    output_suffix,
    number_of_peaks,
    frequency_limits,
    subsampling,
    three_electrode,
    dimensionless_reference_electrode_location,
    lambda_value,
    white_noise,
    discretization,
    optimizer_settings,
    overwrite
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.preprocessing import find_occurrences
    from ep_bolfi.utility.dataset_formatting import Impedance_Measurement
    from kadi_apy.lib.core import KadiManager, Record

    manager = KadiManager()

    file_prefix = parameters_file.split(".")[0]

    try:
        data = Impedance_Measurement.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    if frequency_limits == '[None, None]':
        try:
            f_min = min([min(f) for f in data.frequencies])
            f_max = max([max(f) for f in data.frequencies])
        except ValueError:
            raise ValueError(
                "The impedance dataset is empty. Maybe you selected indices "
                "in the data that it does not contain?"
            )
        frequency_limits = [f_min, f_max]
        dataset = [
            [f[::subsampling] for f in data.frequencies],
            [ci[::subsampling] for ci in data.complex_impedances]
        ]
    else:
        frequency_limits = literal_eval(frequency_limits)
        f_argmin = find_occurrences(data.frequencies, frequency_limits[0])[0]
        f_argmax = find_occurrences(data.frequencies, frequency_limits[1])[0]
        dataset = [
            [
                f[f_argmin:f_argmax + 1:subsampling]
                for f in data.frequencies
            ],
            [
                ci[f_argmin:f_argmax + 1:subsampling]
                for ci in data.complex_impedances
            ]
        ]
    features, drts = eis_features_by_segment(
        dataset, number_of_peaks, lambda_value
    )

    chosen_features = {
        eis_feature_names[k] + "(segment #" + str(i)
        + ", peak #" + str(j) + ")": log_tau_or_resistance
        for i, segment in enumerate(features)
        for j, feature in enumerate(segment)
        for k, log_tau_or_resistance in enumerate(feature)
    }
    lambda_values = [drt.lambda_value for drt in drts]

    if not isfile(ocv_file):
        ocv_record_handle = Record(manager, id=ocv_record, create=False)
        file_id = ocv_record_handle.get_file_id(ocv_file)
        ocv_record_handle.download_file(file_id, ocv_file)
    if not isfile("local_parameter_file.py"):
        parameters_record_handle = Record(
            manager, id=parameters_record, create=False
        )
        file_id = parameters_record_handle.get_file_id(parameters_file)
        parameters_record_handle.download_file(
            file_id, "local_parameter_file.py"
        )
    from local_parameter_file import parameters

    try:
        from local_parameter_file import uncertainties
    except ImportError:
        uncertainties = {}

    with open(ocv_file, 'r') as f:
        ocv_data = json.load(f)

    initial_socs = []
    # Each segment of data is its own impedance measurement.
    for i in range(len(data)):
        source_index = data.indices[i]

        initial_socs.append({
            "Initial concentration in " + electrode + " electrode [mol.m-3]":
            None
            for electrode in ["negative", "positive"]
        })
        for electrode in ["negative", "positive"]:
            if (
                electrode.capitalize() + " electrode SOC [-]"
                in ocv_data.keys()
            ):
                initial_socs[-1][
                    "Initial concentration in "
                    + electrode
                    + " electrode [mol.m-3]"
                ] = (
                    ocv_data[electrode.capitalize() + " electrode SOC [-]"][
                        ocv_data['indices'].index(source_index)
                    ]
                    * parameters[
                        "Maximum concentration in "
                        + electrode
                        + " electrode [mol.m-3]"
                    ]
                )

    optimizer_input_name = (
        file_prefix
        + '_optimizer_input'
        + (('_' + output_suffix) if output_suffix else '')
        + '.json'
    )

    # Since JSON can't handle complex numbers natively:
    dataset_json_compatible = [
        dataset[0],
        [[entry.real for entry in ci] for ci in dataset[1]],
        [[entry.imag for entry in ci] for ci in dataset[1]]
    ]
    with open(optimizer_input_name, 'w') as f:
        json.dump(
            {
                'number_of_peaks': number_of_peaks,
                'frequency_limits': frequency_limits,
                'experiment_features': chosen_features,
                'initial_socs': initial_socs,
                'three_electrode': three_electrode,
                'dimensionless_reference_electrode_location': (
                    dimensionless_reference_electrode_location
                ),
                'lambda_values': lambda_values,
                'white_noise': white_noise,
                'uncertainties': uncertainties,
                'discretization': discretization,
                'optimizer_settings': literal_eval(optimizer_settings),
                'experiment_data': dataset_json_compatible,
            },
            f
        )

    output_record_handle = Record(manager, id=output_record, create=False)
    output_record_handle.upload_file(
        "local_parameter_file.py",
        file_name=(
            file_prefix
            + (('_' + output_suffix) if output_suffix else '')
            + '.py'
        ),
        force=overwrite
    )
    output_record_handle.upload_file(optimizer_input_name, force=overwrite)


if __name__ == '__main__':
    eis_preprocessing()
