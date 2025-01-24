"""
Selects specified segments of the ``Measurement`` and returns them.
"""

from ast import literal_eval
import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.select_measurement_segments',
    version='3.0'
)
@xmlhelpy.argument(
    'segments',
    default="[(0, None)]",
    param_type=xmlhelpy.String,
    description=(
        "The segments that get selected from the dataset. Give as a list of "
        "2-tuples and 3-tuples, e.g., '[(0, 2), (3, 20, 2)]'. Each tuple "
        "denotes one range of segments of the dataset to return. The 2-tuples "
        "give the range as in Python's [x:y] slice notation, i.e., they refer "
        "to segments x through y - 1, counting from 0. Use 'None' to count to "
        "the end of the data including the last segment. The 3-tuples give "
        "the range as in Python's [x:y:z] slice notation, i.e., they refer to "
        "each zth segment from x through y - 1, counting from 0."
    )
)
@xmlhelpy.option(
    'type',
    char='t',
    param_type=xmlhelpy.Choice(["cycling", "static", "impedance"]),
    default="cycling",
    description=(
        "Type of the measurement, which determines the internal format for "
        "further processing. 'cycling' refers to Cycling_Information, 'static'"
        " to Static_Information' and 'impedance' to Impedance_Information. "
        "If you are unsure which to pick, consult the package documentation."
    )
)
def select_measurement_segments(segments, type):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import (
        Cycling_Information,
        Static_Information,
        Impedance_Measurement
    )

    segments = literal_eval(segments)
    try:
        if type == "cycling":
            data = Cycling_Information.from_json(sys.stdin.read())
        elif type == "static":
            data = Static_Information.from_json(sys.stdin.read())
        elif type == "impedance":
            data = Impedance_Measurement.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )
    for i, segment in enumerate(segments):
        if len(segment) == 2 or len(segment) == 3:
            if i == 0:
                selected_data = data.subslice(*segment)
            else:
                selected_data.extend(data.subslice(*segment))
        else:
            raise ValueError(
                "Segments have to be given as either 2- or 3-tuples."
            )
    print(selected_data.to_json())


if __name__ == '__main__':
    select_measurement_segments()
