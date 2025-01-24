"""
Reads in a ``Cycling_Information`` json representation and applies a
PyBaMM model parameter set to subtract the OCP curve(s) from it, leaving
us with only the overpotential.
"""

import json
from os import linesep
from os.path import isfile
import sys
import xmlhelpy


def ocv_mismatch(soc, ocv, parameters, electrode='both', voltage_sign=0):
    """
    Compares data OCV with model OCV(SOC) to determine their mismatch.

    :param soc:
        The SOC data points.
    :param ocv:
        The OCV data points.
    :param parameters:
        The PyBaMM model parameters containing the model OCP curves.
    :param electrode:
        Set to 'positive' or 'negative' for three-electrode setups or
        half-cell setups, or 'both' to subtract both electrode OCPs.
    :param voltage_sign:
        Optional prefactor to multiply the model OCV with before
        subtracting it from the data OCV when ``electrode != 'both'``.
        Has no effect otherwise. Defaults to usual sign conventions.
    :returns:
        The difference between data OCV and model OCV(SOC).
    """
    if electrode != 'both':
        if voltage_sign == 0:
            if electrode == "positive":
                voltage_sign = 1
            else:
                voltage_sign = -1
        return (
            ocv
            - voltage_sign * parameters[
                electrode.capitalize() + " electrode OCP [V]"
            ](soc)
        )
    else:
        return (
            ocv
            - parameters["Positive electrode OCP [V]"](soc)
            + parameters["Negative electrode OCP [V]"](soc)
        )


def transform_to_unity_interval(segment):
    """
    Transforms the *segment* list to start at 0 and end at 1.

    :param segment:
        A list of numbers.
    :returns:
        A linearly transformed *segment*, so that it lives on [0,1].
    """
    return [
        (s - segment[0]) / (segment[-1] - segment[0])
        for s in segment
    ]


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.extract_overpotential',
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
        "the GITT protocol, as written in its 'indices'. Defaults to the "
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
        + "If both OCPs shall be removed, it must contain additionally:"
        + linesep + linesep
        + " - negative_SOC_from_cell_SOC: A callable, for OCV subtraction."
        + linesep + linesep
        + " - positive_SOC_from_cell_SOC: A callable, for OCV subtraction."
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
    'adjust-for-ocv-mismatch',
    char='a',
    param_type=xmlhelpy.Bool,
    default=True,
    description=(
        "If True (default), each segment will be corrected by the OCV data in "
        "the OCV file in addition to the OCV fit function. In the case that "
        "the OCV fit function originated from the same OCV file, this just "
        "means that any imperfections in the fitted OCV curve get subtracted "
        "away. This is useful if absolute voltage values between model and "
        "data shall be compared, as the model will not have this mismatch. "
        "The 'indices' in the OCV file have to match those in the provided "
        "data and be monotonically increasing in both cases. The values that "
        "get subtracted are the linear interpolation of the imperfections. "
        "Note: the correction calculation assumes that there was a relaxation "
        "step right before the first given segment, that the last segment is a"
        "relaxation step again, and pulse and relaxation alternate each time."
    )
)
@xmlhelpy.option(
    'title',
    char='t',
    default=None,
    param_type=xmlhelpy.String,
    description="Title of the plot. Defaults to the template file name."
)
@xmlhelpy.option(
    'electrode',
    char='e',
    default='both',
    param_type=xmlhelpy.Choice(
        ['positive', 'negative', 'both'], case_sensitive=True
    ),
    description=(
        "Choose which OCPs to remove from the data."
    )
)
@xmlhelpy.option(
    'current-sign',
    char='c',
    default=0,
    param_type=xmlhelpy.Integer,
    description=(
        "Only applicable if 'electrode' is not 'both'. "
        "1 means the calculated SOC is added, -1 means it is subtracted. "
        "The default behaviour is 1 for 'positive' and -1 for 'negative'."
    )
)
@xmlhelpy.option(
    'voltage-sign',
    char='u',
    default=0,
    param_type=xmlhelpy.Integer,
    description=(
        "Only applicable if 'electrode' is not 'both'. "
        "1 means the given OCP is subtracted, -1 means it is added. "
        "The default behaviour is 1 for 'positive' and -1 for 'negative'."
    )
)
@xmlhelpy.option(
    'format',
    char='f',
    default='pdf',
    param_type=xmlhelpy.Choice(
        ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba',
         'svg', 'svgz', 'tif', 'tiff'],
        case_sensitive=True
    ),
    description="Format of generated image file."
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
@xmlhelpy.option(
    'display',
    char='v',
    is_flag=True,
    description=(
        "Toggle to display the plot on the machine this script runs on."
    )
)
def extract_overpotential(
    ocv_record,
    ocv_file,
    source_index,
    parameters_record,
    parameters_file,
    output_record,
    adjust_for_ocv_mismatch,
    title,
    electrode,
    current_sign,
    voltage_sign,
    format,
    overwrite,
    display,
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from ep_bolfi.utility.preprocessing import (
        subtract_OCV_curve_from_cycles, subtract_both_OCV_curves_from_cycles
    )
    from ep_bolfi.utility.visualization import plot_measurement
    from kadi_apy.lib.core import KadiManager, Record
    import matplotlib.pyplot as plt
    from scipy.optimize import root_scalar

    manager = KadiManager()

    file_prefix = ocv_file.split('.')[0]
    if title is None:
        title = file_prefix

    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    if not isfile(ocv_file):
        ocv_record_handle = Record(manager, id=ocv_record, create=False)
        file_id = ocv_record_handle.get_file_id(ocv_file)
        ocv_record_handle.download_file(file_id, ocv_file)
    with open(ocv_file, 'r') as f:
        ocv_data = json.load(f)

    if not isfile("local_parameter_file.py"):
        parameters_record_handle = Record(
            manager, id=parameters_record, create=False
        )
        file_id = parameters_record_handle.get_file_id(parameters_file)
        parameters_record_handle.download_file(
            file_id, "local_parameter_file.py"
        )
    from local_parameter_file import parameters

    if source_index == float('inf'):
        source_index = data.indices[0] - 1

    initial_socs = {
        "Initial concentration in " + e + " electrode": None
        for e in ["negative", "positive"]
    }
    for e in ["negative", "positive"]:
        if e.capitalize() + " electrode SOC [-]" in ocv_data.keys():
            initial_socs[
                "Initial concentration in " + e + " electrode"
            ] = (
                ocv_data[e.capitalize() + " electrode SOC [-]"][
                    ocv_data['indices'].index(source_index)
                ]
            )

    if electrode != 'both':
        data.voltages, returned_SOCs = subtract_OCV_curve_from_cycles(
            data,
            parameters,
            starting_SOC=initial_socs[
                "Initial concentration in " + electrode + " electrode"
            ],
            electrode=electrode,
            current_sign=current_sign,
            voltage_sign=voltage_sign
        )
    else:
        from local_parameter_file import (
            positive_SOC_from_cell_SOC, negative_SOC_from_cell_SOC
        )
        if initial_socs[
            "Initial concentration in positive electrode"
        ] is not None:
            starting_SOC = root_scalar(
                lambda s: positive_SOC_from_cell_SOC(s)
                - initial_socs["Initial concentration in positive electrode"],
                method='toms748',
                bracket=[0, 1],
                x0=0.5
            ).root
        elif initial_socs[
            "Initial concentration in negative electrode"
        ] is not None:
            starting_SOC = root_scalar(
                lambda s: negative_SOC_from_cell_SOC(s)
                - initial_socs["Initial concentration in negative electrode"],
                method='toms748',
                bracket=[0, 1],
                x0=0.5
            ).root
        else:
            starting_SOC = None
        data.voltages, returned_SOCs = subtract_both_OCV_curves_from_cycles(
            data,
            parameters,
            negative_SOC_from_cell_SOC,
            positive_SOC_from_cell_SOC,
            starting_SOC=starting_SOC,
        )

    if adjust_for_ocv_mismatch:
        ocv_file_index = ocv_data['indices'].index(data.indices[0] - 1)
        ocv_mismatch_start = ocv_mismatch(
            returned_SOCs[0][0],
            ocv_data['OCV [V]'][ocv_file_index],
            parameters,
            electrode,
            voltage_sign
        )
        for i, (pulse_index, relaxation_index) in enumerate(zip(
            data.indices[:-1:2], data.indices[1::2]
        )):
            ocv_file_index = ocv_data['indices'].index(relaxation_index)
            ocv_mismatch_end = ocv_mismatch(
                returned_SOCs[2 * i + 1][-1],
                ocv_data['OCV [V]'][ocv_file_index],
                parameters,
                electrode,
                voltage_sign
            )
            t_pulse = transform_to_unity_interval(data.timepoints[2 * i])
            data.voltages[2 * i] = [
                entry - ((1 - t) * ocv_mismatch_start + t * ocv_mismatch_end)
                for entry, t in zip(data.voltages[2 * i], t_pulse)
            ]
            data.voltages[2 * i + 1] = [
                entry - ocv_mismatch_end for entry in data.voltages[2 * i + 1]
            ]
            ocv_mismatch_start = ocv_mismatch_end

    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
    texts = plot_measurement(
        fig,
        ax,
        data,
        title,
        plot_current=False
    )
    ax.set_ylabel("Overpotential  /  V")
    for text in texts:
        text.set_visible(False)
    fig.savefig(
        file_prefix + '_ocv_alignment.' + format,
        bbox_inches='tight',
        pad_inches=0.0
    )
    for text in texts:
        text.set_visible(True)
    if display:
        plt.show()

    output_record_handle = Record(manager, id=output_record, create=False)
    output_record_handle.upload_file(
        file_prefix + '_ocv_alignment.' + format, force=overwrite
    )
    print(data.to_json())


if __name__ == '__main__':
    extract_overpotential()
