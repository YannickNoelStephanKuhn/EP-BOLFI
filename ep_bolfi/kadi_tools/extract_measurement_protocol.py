"""
Reads in a ``Cycling_Information`` json representation and outputs a
text-based representation of the different segments that corresponds to
``pybamm.Experiment`` inputs.
"""

import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.extract_measurement_protocol',
    version='3.0'
)
@xmlhelpy.option(
    'record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description="Persistent record identifier."
)
@xmlhelpy.option(
    'filename',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="File name as template for output files."
)
@xmlhelpy.option(
    'start',
    char='s',
    default=0,
    param_type=xmlhelpy.Integer,
    description="First segment to include in the plot."
)
@xmlhelpy.option(
    'stop',
    char='e',
    default=None,
    param_type=xmlhelpy.Integer,
    description="Segment after the last one to include in the plot."
)
@xmlhelpy.option(
    'step',
    char='d',
    default=1,
    param_type=xmlhelpy.Integer,
    description="Step between segments to include in the plot."
)
@xmlhelpy.option(
    'time-unit',
    char='t',
    default='h',
    param_type=xmlhelpy.Choice(['d', 'h', 'm', 's'], case_sensitive=False),
    description="Time unit to use for the timespans."
)
@xmlhelpy.option(
    'state-column',
    char='c',
    default=None,
    param_type=xmlhelpy.String,
    description=(
        "Optional name of a column stored in the 'other_columns' attribute. "
        "Will be used then to describe the individual segments."
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
    'voltage-deviation-threshold',
    char='u',
    default=1e-8,
    param_type=xmlhelpy.Float,
    description=(
        "If the standard deviation of the voltage during a segment is below "
        "this value, it will be regarded as zero deviation voltage for the "
        "whole segment, i.e., holding at this voltage."

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
    'overwrite',
    char='w',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Whether or not an already existing file by the same name in the "
        "record gets overwritten."
    )
)
def extract_measurement_protocol(
    record,
    filename,
    start,
    stop,
    step,
    time_unit,
    state_column,
    current_threshold,
    flip_current_sign,
    voltage_deviation_threshold,
    flip_voltage_sign,
    overwrite
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from kadi_apy.lib.core import KadiManager, Record
    import numpy as np

    manager = KadiManager()

    file_prefix = filename.split(".")[0]

    time_scale = {"d": 24 * 3600, "h": 3600, "m": 60, "s": 1}[time_unit]

    try:
        data = Cycling_Information.from_json(sys.stdin.read()).subslice(
            start, stop, step
        )
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    if flip_voltage_sign:
        for i in range(len(data.voltages)):
            data.voltages[i] = [-entry for entry in data.voltages[i]]

    protocol = []
    for i, (indices, t, I, U) in enumerate(zip(
        data.indices, data.timepoints, data.currents, data.voltages
    )):
        index = str(indices)
        timespan = (t[-1] - t[0]) / time_scale
        if state_column:
            descriptor = (
                index + " " + str(data.other_columns[state_column][i][0])
            )
        else:
            descriptor = index
        # Write out the measurement protocol.
        if np.mean(np.abs(np.atleast_1d(I))) < current_threshold:
            protocol.append(
                descriptor
                + ": Rest for "
                + "{0:.4g}".format(timespan)
                + " " + time_unit
            )
        elif np.std(np.atleast_1d(U)) < voltage_deviation_threshold:
            protocol.append(
                descriptor
                + ": Hold at "
                + "{0:.4g}".format(np.mean(np.atleast_1d(U)))
                + " V for "
                + "{0:.4g}".format(timespan)
                + " "
                + time_unit
            )
        else:
            if flip_current_sign:
                direction = "Discharge" if np.mean(I) <= 0 else "Charge"
            else:
                direction = "Discharge" if np.mean(I) > 0 else "Charge"
            protocol.append(
                descriptor
                + ": "
                + direction
                + " at "
                + "{0:.4g}".format(np.mean(np.abs(np.atleast_1d(I))))
                + " A for "
                + "{0:.4g}".format(timespan)
                + " "
                + time_unit
            )

    with open(file_prefix + '.json', 'w') as f:
        json.dump(protocol, f)

    output_record_handle = Record(manager, id=record, create=False)
    output_record_handle.upload_file(file_prefix + '.json', force=overwrite)


if __name__ == '__main__':
    extract_measurement_protocol()
