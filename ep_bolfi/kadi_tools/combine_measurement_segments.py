"""
Reads in a JSON representation of a ``Measurement`` object, combines
segments of it as per the given rules, and outputs the resulting
``Measurement`` object in a JSON representation again.
"""

from ast import literal_eval
import json
import sys
import xmlhelpy


def perform_combination(data, labels, datatype, adapt_indices=False):
    """
    Concatenates segments according to *labels*.

    :param data:
        An object from ``ep_bolfi.utility.dataset_formatting`` derived
        from ``Measurement``.
    :param labels:
        A list of labels, corresponding to the segments in *data*.
        If identical labels are detected in neighbouring segments,
        they get concatenated, decreasing segment count by one.
    :param datatype:
        One of 'cycling', 'static', or 'impedance', relating to the
        classes defined in `dataset_formatting`.
    :param adapt_indices:
        Set to True if indices are numerical and you wish to keep their
        spacing the same as before.
    :returns:
        The *data* object, with segments concatenated per *labels*.
    """
    combination_counter = 0
    if datatype == "cycling":
        attr_list = ['timepoints', 'currents', 'voltages']
    elif datatype == "static":
        attr_list = [
            'timepoints',
            'currents',
            'voltages',
            'asymptotic_voltages',
            'ir_steps',
            'exp_I_decays',
            'exp_U_decays'
        ]
    elif datatype == "impedance":
        attr_list = [
            'frequencies',
            'real_impedances',
            'imaginary_impedances',
            'phases'
        ]
    for i in range(len(labels) - 1):
        if labels[i + 1] == labels[i]:
            for attr in attr_list:
                getattr(data, attr)[i - combination_counter].extend(
                    getattr(data, attr).pop(i - combination_counter + 1)
                )
            for name in data.other_columns.keys():
                data.other_columns[name][i - combination_counter].extend(
                    data.other_columns[name].pop(
                        i - combination_counter + 1
                    )
                )
            if adapt_indices:
                correction = (
                    data.indices.pop(i - combination_counter + 1)
                    - data.indices[i - combination_counter]
                )
                for j in range(
                    i - combination_counter + 1,
                    len(labels) - combination_counter - 1
                ):
                    data.indices[j] -= correction
            combination_counter += 1


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.combine_measurement_segments',
    version='3.0'
)
@xmlhelpy.option(
    'datatype',
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
@xmlhelpy.option(
    'group-by-list',
    char='l',
    param_type=xmlhelpy.String,
    default="{}",
    description=(
        "A dictionary in Python notation. The keys relate to attributes of "
        "the Measurement object (see dataset_formatting.py) or an extra "
        "column stored in its 'other_columns' attribute. "
        "The values are lists of lists, where each list contains the column "
        "contents that should be grouped together, when found in adjacent "
        "segments. Multiple lists will be worked through in order. "
        "Only the first entry of each segment is checked. "
        "All segments not attributed to a list form their own group."
    )
)
@xmlhelpy.option(
    'group-by-pivot',
    char='p',
    param_type=xmlhelpy.String,
    default="{}",
    description=(
        "A dictionary in Python notation. The keys have the same meaning as "
        "for 'group-by-list'. "
        "The values are lists of 2-tuples, where the first entry is the "
        "value checked for, and the second one of the following comparison "
        "operators: '=', '>=', '<=', '>', or '<'. Multiple comparisons will "
        "be worked through in order and are not cumulative."
        "Only the first entry of each segment is checked. "
        "All segments not attributed to a list form their own group."
    )
)
@xmlhelpy.option(
    'adapt-indices',
    char='a',
    param_type=xmlhelpy.Bool,
    default=False,
    description=(
        "Set to True if you wish to adapt the numbering found in 'indices' "
        "based on the combination of segments."
    )
)
def combine_measurement_segments(
    datatype,
    group_by_list,
    group_by_pivot,
    adapt_indices
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import (
        Cycling_Information,
        Static_Information,
        Impedance_Measurement
    )

    try:
        if datatype == "cycling":
            data = Cycling_Information.from_json(sys.stdin.read())
        elif datatype == "static":
            data = Static_Information.from_json(sys.stdin.read())
        elif datatype == "impedance":
            data = Impedance_Measurement.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    for name, rule in literal_eval(group_by_list).items():
        if hasattr(data, name):
            relevant_data = getattr(data, name)
        else:
            relevant_data = data.other_columns[name]
        group_assignment_by_list = [-1 for _ in range(len(relevant_data))]
        for i in range(len(relevant_data)):
            for j, group_list in enumerate(rule):
                if relevant_data[i][0] in group_list:
                    group_assignment_by_list[i] = j
        perform_combination(
            data, group_assignment_by_list, datatype, adapt_indices
        )

    for name, rule in literal_eval(group_by_pivot).items():
        if hasattr(data, name):
            relevant_data = getattr(data, name)
        else:
            relevant_data = data.other_columns[name]
        group_assignment_by_pivot = [-1 for _ in range(len(relevant_data))]
        for i in range(len(relevant_data)):
            for j, (value, comparison_operator) in enumerate(rule):
                if comparison_operator == '=':
                    if relevant_data[i][0] == value:
                        group_assignment_by_pivot[i] = j
                elif comparison_operator == '>=':
                    if relevant_data[i][0] >= value:
                        group_assignment_by_pivot[i] = j
                elif comparison_operator == '<=':
                    if relevant_data[i][0] <= value:
                        group_assignment_by_pivot[i] = j
                elif comparison_operator == '>':
                    if relevant_data[i][0] > value:
                        group_assignment_by_pivot[i] = j
                elif comparison_operator == '<':
                    if relevant_data[i][0] < value:
                        group_assignment_by_pivot[i] = j
                else:
                    raise ValueError(
                        "Comparison operator '"
                        + comparison_operator
                        + "' unknown. Has to be either '=', "
                        "'>=', '<=', '>', or '<'."
                    )
        perform_combination(
            data, group_assignment_by_pivot, datatype, adapt_indices
        )

    print(data.to_json())


if __name__ == '__main__':
    combine_measurement_segments()
