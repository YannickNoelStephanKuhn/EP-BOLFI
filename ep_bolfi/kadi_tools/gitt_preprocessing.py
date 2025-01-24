"""
Reads in a ``Cycling_Information`` json representation and preprocesses
it as GITT data, assuming each segment corresponds to a pulse or pause.
Additionally formats all necessary inputs for EP-BOLFI parameterization.
"""

from ast import literal_eval
import json
from os import linesep
from os.path import isfile
import sys
import xmlhelpy


gitt_feature_names = [
    "asymptotic concentration overpotential",
    "voltage loss attributed to exponential decay",
    "exponential relaxation time",
    "initial voltage by square root extrapolation",
    "square root short-time slope dU/dsqrt(t)"
]

gitt_feature_shortnames = [
    "η [V]", "exp_decay [V]", "τᵣ [s]", "U_0 [V]", "dU/dsqrt(t)^-1 [sqrt(s)/V]"
]


def gitt_features(
    dataset, list_of_feature_indices, sqrt_cutoff, sqrt_start, exp_cutoff
):
    """
    Calculates diffusion features for each GITT segment. They are
    flattened into one list, with each five entries referring to:
     - asymptotic concentration overpotential
     - exponential magnitude
     - exponential timescale
     - initial voltage
     - inverse square-root slope at beginning (d√t / dU)

    :param dataset:
        A ``Cycling_Information`` object.
    :param list_of_feature_indices:
        The indices from the whole list to retain for fitting.
    :param sqrt_cutoff:
        The upper time limit on data to use in each segment for the
        square-root fit. The time limit counts from the segment start.
    :param sqrt_start:
        The lower time limit on data to use in each segment for the
        square-root fit. The time limit counts from the segment start.
    :param exp_cutoff:
        The lower time limit on data to use in each segment for the
        exponential fit. The time limit counts from the segment end.
    :returns:
        The flattened list of selected diffusion features.
    """
    from ep_bolfi.utility.fitting_functions import (
        fit_exponential_decay, fit_sqrt
    )
    from ep_bolfi.utility.preprocessing import find_occurrences
    exp_features = []
    sqrt_features = []
    for i, (t, U) in enumerate(zip(*dataset)):
        exp_feature_indices = [5 * i + j for j in range(3)]
        sqrt_feature_indices = [5 * i + j for j in range(3, 5)]
        if sum([f in list_of_feature_indices for f in exp_feature_indices]):
            # Compute the exponential features.
            if exp_cutoff:
                cutoff_index = find_occurrences(t, t[-1] - exp_cutoff)[0]
            else:
                cutoff_index = 0
            exp_decay = fit_exponential_decay(
                t[cutoff_index:], U[cutoff_index:], threshold=0.95
            )[0][2]
            exp_features.append(
                [exp_decay[0], exp_decay[1], 1.0 / exp_decay[2]]
            )
        else:
            exp_features.append([0.0, 0.0, 0.0])
        if sum([f in list_of_feature_indices for f in sqrt_feature_indices]):
            # Compute the square-root features.
            cutoff_index = find_occurrences(t, t[0] + sqrt_cutoff)[0]
            start_index = find_occurrences(t, t[0] + sqrt_start)[0]
            try:
                sqrt_fit = fit_sqrt(
                    t[start_index:cutoff_index + 1],
                    U[start_index:cutoff_index + 1],
                    threshold=0.95
                )[2]
                sqrt_features.append([sqrt_fit[0], 1.0 / sqrt_fit[1]])
            except IndexError:
                sqrt_features.append([0.0, 0.0])

        else:
            sqrt_features.append([0.0, 0.0])
    features = []
    for exp_f, sqrt_f in zip(exp_features, sqrt_features):
        # Remove the initial overpotential (ohmic or concentration).
        exp_f[0] -= sqrt_f[0]
        features.extend(exp_f)
        features.extend(sqrt_f)
    return [features[index] for index in list_of_feature_indices]


def gitt_feature_visualizer(
    dataset,
    list_of_feature_indices,
    sqrt_cutoff,
    sqrt_start,
    exp_cutoff
):
    """
    Visualizes diffusion features for each GITT segment.

    :param dataset:
        A ``Cycling_Information`` object.
    :param list_of_feature_indices:
        The indices from the whole list to retain for fitting.
    :param sqrt_cutoff:
        The upper time limit on data to use in each segment for the
        square-root fit. The time limit counts from the segment start.
    :param sqrt_start:
        The lower time limit on data to use in each segment for the
        square-root fit. The time limit counts from the segment start.
    :param exp_cutoff:
        The lower time limit on data to use in each segment for the
        exponential fit. The time limit counts from the segment end.
    :returns:
        A list of 3-tuples, where each of them represents a square-root
        or exponential fit to a segment. First entry is the times over
        which the voltage in the second entry is plotted, and the third
        is a label describing the square-root or exponential parameters.
    """
    from ep_bolfi.utility.fitting_functions import (
        fit_exponential_decay, fit_sqrt
    )
    from ep_bolfi.utility.preprocessing import find_occurrences
    visualization = []
    for i, (t, U) in enumerate(zip(*dataset)):
        exp_feature_indices = [5 * i + j for j in range(3)]
        sqrt_feature_indices = [5 * i + j for j in range(3, 5)]
        if sum([f in list_of_feature_indices for f in exp_feature_indices]):
            # Compute the exponential features.
            if exp_cutoff:
                cutoff_index = find_occurrences(t, t[-1] - exp_cutoff)[0]
            else:
                cutoff_index = 0
            exp_decay = fit_exponential_decay(
                t[cutoff_index:], U[cutoff_index:], threshold=0.95
            )[0]
            exp_features = [
                exp_decay[2][0], exp_decay[2][1], 1.0 / exp_decay[2][2]
            ]
            label = ''
            for j in range(3):
                if 5 * i + j in list_of_feature_indices:
                    label = (
                        label
                        + gitt_feature_shortnames[j]
                        + ': {0:.4g}'.format(exp_features[j])
                        + linesep
                    )
            visualization.append((exp_decay[0], exp_decay[1], label))
        if sum([f in list_of_feature_indices for f in sqrt_feature_indices]):
            # Compute the square-root features.
            cutoff_index = find_occurrences(t, t[0] + sqrt_cutoff)[0]
            start_index = find_occurrences(t, t[0] + sqrt_start)[0]
            sqrt_fit = fit_sqrt(
                t[start_index:cutoff_index + 1],
                U[start_index:cutoff_index + 1],
                threshold=0.95
            )
            sqrt_features = [sqrt_fit[2][0], 1.0 / sqrt_fit[2][1]]
            label = ''
            for j in range(3, 5):
                if 5 * i + j in list_of_feature_indices:
                    label = (
                        label
                        + gitt_feature_shortnames[j]
                        + ': {0:.4g}'.format(sqrt_features[j - 3])
                        + linesep
                    )
            visualization.append((sqrt_fit[0], sqrt_fit[1], label))
    return visualization


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.gitt_preprocessing',
    version='3.0'
)
@xmlhelpy.option(
    'ocv-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the OCV data "
        "as prepared by ep_bolfi.kadi_tools.extract_ocv_curve."
    )
)
@xmlhelpy.option(
    'ocv-file',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "The name of the OCV file with electrode SOC to cell SOC assignment "
        "as prepared by ep_bolfi.kadi_tools.fit_and_plot_ocv."
    )
)
@xmlhelpy.option(
    'source-index',
    char='s',
    param_type=xmlhelpy.Float,
    default=float('inf'),
    description=(
        "Index of the data in the OCV file to use as the starting point of "
        "the GITT simulation, as written in its 'indices'. Defaults to the "
        "first 'indices' entry in the provided measurement data minus 1."
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
        + " - unknowns: The dictionary of unknown parameters. Instead of "
        "single values as in 'parameters', input 2-tuples with their lower "
        "and upper bounds, e.g. from literature."
        + linesep + linesep
        + "It may contain the additional following global variables:"
        + linesep + linesep
        + " - transform_unknowns: Dictionary for transforming unknowns before "
        "inferring them via a normal distribution. Either give as 2-tuples "
        "with the first entry being the back-transformation and the second "
        "one being the transformation. For convenience, putting 'log' gives "
        "log-normal distributions."
        + linesep + linesep
        + " - uncertainties: The dictionary of parameter uncertainties. Used "
        "for scrambling them in the simulation samples. Give them as tuples: "
        "the first entry is the name of the distribution in scipy.stats, and "
        "the following are its parameters. Example: ('norm', mean, std)."
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
    'feature-choice',
    char='f',
    default=None,
    param_type=xmlhelpy.String,
    description=(
        "List of features by their index to consider in the optimization. "
        "For each segment of data with index i, five features are computed, "
        "which results in the following numbering:"
        + linesep + linesep
        + " - 5 * i + 0: asymptotic long-time voltage"
        + linesep + linesep
        + " - 5 * i + 1: initial voltage for exponential decay"
        + linesep + linesep
        + " - 5 * i + 2: exponential relaxation time"
        + linesep + linesep
        + " - 5 * i + 3: actual initial voltage by square root extrapolation"
        + linesep + linesep
        + " - 5 * i + 4: square root short-time slope, i.e., dU/dsqrt(t)"
        + linesep + linesep
        + "Notation: e.g., '[0, 1, 3]'. Default is all but long-time voltage."
    )
)
@xmlhelpy.option(
    'current-threshold',
    char='b',
    default=1e-8,
    param_type=xmlhelpy.Float,
    description=(
        "If the mean current during a segment is below this value, it "
        "will be regarded as zero current for the whole segment for "
        "more sensible simulations."
    )
)
@xmlhelpy.option(
    'flip-current-sign',
    char='g',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Defaults to False, where positive current means discharge. "
        "Change to True if positive current shall mean charge."
    )
)
@xmlhelpy.option(
    'flip-voltage-sign',
    char='v',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Defaults to False, where measured voltage remains unaltered. "
        "Change to True if the voltage shall be multiplied by -1."
    )
)
@xmlhelpy.option(
    'overpotential',
    char='l',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Defaults to False, where the 'Terminal voltage [V]' gets fitted "
        "to the data. If True, it minus the OCV gets fitted instead."
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
    'sqrt-cutoff',
    char='c',
    default=30.0,
    param_type=xmlhelpy.Float,
    description=(
        "Controls the timespan (in s) that is used in each segment of data "
        "for the square-root fit. It's [start of segment, start of segment "
        "+ sqrt-cutoff]. Default is 30 seconds."
    )
)
@xmlhelpy.option(
    'sqrt-start',
    char='i',
    default=0.0,
    param_type=xmlhelpy.Float,
    description=(
        "Controls the timespan (in s) that is sued in each segment of data "
        "for the square-root fit. It's modifying the range set by "
        "'sqrt-cutoff' to start at 'start of segment + sqrt-start'. "
        "Default is 0 seconds."
    )
)
@xmlhelpy.option(
    'exp-cutoff',
    char='h',
    default=None,
    param_type=xmlhelpy.Float,
    description=(
        "Controls the timespan (in s) that is used in each segment of data "
        "for the exponential decay fit. It's [end of segment - exp-cutoff, "
        "end of segment]. Default is None, i.e., the whole segment gets used."
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
    'model',
    char='m',
    default='DFN()',
    param_type=xmlhelpy.String,
    description=(
        "The PyBaMM model to be used. The notation is the object instatiation "
        "of ,e.g., the 'SPM', the 'SPMe', the 'DFN', or the 'MPM'. So either "
        "write just that name or, e.g., "
        "'DFN(options={\"working_electrode\": \"positive\"}'."
    )
)
@xmlhelpy.option(
    'discretization',
    char='d',
    default=(
        "{'order_s_n': 10, 'order_s_p': 10, 'order_e': 10,"
        " 'volumes_e_n': 1, 'volumes_e_s': 1, 'volumes_e_p': 1,"
        " 'halfcell': False}"
    ),
    param_type=xmlhelpy.String,
    description=(
        "Discretization settings. Write as a dictionary: {"
        + linesep + linesep
        + "    order_s_n: 'Spectral Volume order in negative particles',"
        + linesep + linesep
        + "    order_s_p: 'Spectral Volume order in positive particles',"
        + linesep + linesep
        + "    order_e: 'Spectral Volume order in electrolyte',"
        + linesep + linesep
        + "    volumes_e_n: 'Spectral Volumes in negative electrode',"
        + linesep + linesep
        + "    volumes_e_s: 'Spectral Volumes in separator',"
        + linesep + linesep
        + "    volumes_e_p: 'Spectral Volumes in positive electrode',"
        + linesep + linesep
        + "    halfcell: 'True with a Li-metal anode, False for full-cell"
        + linesep + linesep
        + "}"
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
def gitt_preprocessing(
    ocv_record,
    ocv_file,
    source_index,
    parameters_record,
    parameters_file,
    output_record,
    output_suffix,
    feature_choice,
    current_threshold,
    flip_current_sign,
    flip_voltage_sign,
    overpotential,
    three_electrode,
    dimensionless_reference_electrode_location,
    sqrt_cutoff,
    sqrt_start,
    exp_cutoff,
    white_noise,
    model,
    discretization,
    optimizer_settings,
    overwrite
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.preprocessing import find_occurrences
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from kadi_apy.lib.core import KadiManager, Record
    import numpy as np

    manager = KadiManager()

    file_prefix = parameters_file.split(".")[0]

    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    if flip_voltage_sign:
        for i in range(len(data.voltages)):
            data.voltages[i] = [-entry for entry in data.voltages[i]]

    if feature_choice is None:
        list_of_feature_indices = [
            5 * i + j
            for i in range(len(data.timepoints))
            for j in range(1, 5)
        ]
    else:
        list_of_feature_indices = literal_eval(feature_choice)

    features = gitt_features(
        [data.timepoints, data.voltages],
        list_of_feature_indices,
        sqrt_cutoff,
        sqrt_start,
        exp_cutoff
    )

    chosen_features = {
        gitt_feature_names[index % 5] + "(segment #" + str(index // 5) + ")": (
            features[i]
        ) for i, index in enumerate(list_of_feature_indices)
    }

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

    if source_index == float('inf'):
        source_index = data.indices[0] - 1

    initial_socs = {
        "Initial concentration in " + electrode + " electrode [mol.m-3]": None
        for electrode in ["negative", "positive"]
    }
    for electrode in ["negative", "positive"]:
        if electrode.capitalize() + " electrode SOC [-]" in ocv_data.keys():
            initial_socs[
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

    # Normalize out the starting time.
    t_0 = data.timepoints[0][0]
    for i in range(len(data.timepoints)):
        data.timepoints[i] = [entry - t_0 for entry in data.timepoints[i]]

    current_input = []
    measurement = [[], []]
    for i, (t, I, U) in enumerate(zip(
        data.timepoints, data.currents, data.voltages
    )):
        if i * 5 > max(list_of_feature_indices):
            # No more simulation needed, since there are no more features.
            break
        timespan = t[-1] - t[0]
        # Add one extra 0.1 second period to the last segment.
        # This prevents NaN evaluations of timepoints slightly past it.
        if i == len(data.timepoints) - 1:
            timespan += 0.1
        if (i + 1) * 5 > max(list_of_feature_indices):
            # This is the last necessary part of the simulation.
            exp_feature_indices = [5 * i + j for j in range(3)]
            if not sum([
                f in list_of_feature_indices for f in exp_feature_indices
            ]):
                # The long exponential tail is not needed, trim it.
                timespan = sqrt_cutoff
        # Include only the experimental data that is within the timespan.
        cutoff_index = find_occurrences(t, t[0] + timespan)[0]
        measurement[0].append(t[:cutoff_index + 1])
        measurement[1].append(U[:cutoff_index + 1])
        # Write out the measurement protocol.
        if np.mean(np.abs(np.atleast_1d(I))) < current_threshold:
            current_input.append(
                "Rest for " + str(timespan) + " s (0.1 second period)"
            )
        else:
            if flip_current_sign:
                direction = "Discharge" if np.mean(I) <= 0 else "Charge"
            else:
                direction = "Discharge" if np.mean(I) > 0 else "Charge"
            current_input.append(
                direction + " at " + str(np.mean(np.abs(np.atleast_1d(I))))
                + " A for " + str(timespan) + " s (0.1 second period)"
            )

    optimizer_input_name = (
        file_prefix
        + '_optimizer_input'
        + (('_' + output_suffix) if output_suffix else '')
        + '.json'
    )
    with open(optimizer_input_name, 'w') as f:
        json.dump(
            {
                'list_of_feature_indices': list_of_feature_indices,
                'experiment_features': chosen_features,
                'initial_socs': initial_socs,
                'current_input': current_input,
                'overpotential': overpotential,
                'three_electrode': three_electrode,
                'dimensionless_reference_electrode_location': (
                    dimensionless_reference_electrode_location
                ),
                'sqrt_cutoff': sqrt_cutoff,
                'sqrt_start': sqrt_start,
                'exp_cutoff': exp_cutoff,
                'white_noise': white_noise,
                'uncertainties': uncertainties,
                'model_name': model,
                'discretization': literal_eval(discretization),
                'optimizer_settings': literal_eval(optimizer_settings),
                'experiment_data': measurement,
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
    gitt_preprocessing()
