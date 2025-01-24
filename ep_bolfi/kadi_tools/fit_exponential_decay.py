"""
Reads in a ``Cycling_Information`` json representation and fits an
exponential fit function to its first segment.
"""

import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.fit_exponential_decay',
    version='3.0'
)
@xmlhelpy.option(
    'parameter-choice',
    char='p',
    param_type=xmlhelpy.Choice(
        ['asymptote', 'initial value', 'decay rate', 'all'],
        case_sensitive=True
    ),
    required=True,
    description=(
        "Choose the value that gets outputted to stdout. The fit function is "
        "f(t, a, b, c) = a + b * exp(-(t - t_0) / c). 'asymptote' is a, "
        "'initial value' is b and 'decay rate' is c. 'all' outputs all three "
        "in the format '[a, b, c]'."
    )
)
@xmlhelpy.option(
    'data-choice',
    char='d',
    param_type=xmlhelpy.Choice(
        ['currents', 'voltages'],
        case_sensitive=True
    ),
    default='voltages',
    description="Choose whether a decay is fitted to currents or voltages."
)
@xmlhelpy.option(
    'rsquared-threshold',
    char='r',
    param_type=xmlhelpy.Float,
    default=0.95,
    description=(
        "R²-value threshold. The beginning of the segment gets trimmed until "
        "it is reached. Gets calculated via the inversely transformed data."
    )
)
@xmlhelpy.option(
    'exclude',
    char='c',
    default=0.0,
    param_type=xmlhelpy.Float,
    description=(
        "Seconds at the start of each segment to exclude from the exponential "
        "fitting procedure to approximate the OCV."
    )
)
def fit_exponential_decay(
    parameter_choice, data_choice, rsquared_threshold, exclude
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    import ep_bolfi.utility.fitting_functions as ff
    from ep_bolfi.utility.preprocessing import find_occurrences

    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )
    exclude_index = find_occurrences(
        data.timepoints[0], data.timepoints[0][0] + exclude
    )[0]
    exp_fit = ff.fit_exponential_decay(
        data.timepoints[0][exclude_index:],
        getattr(data, data_choice)[0][exclude_index:],
        recursive_depth=1,
        threshold=rsquared_threshold
    )[0][2]

    result = {
        'asymptote': str(exp_fit[0]),
        'initial value': str(exp_fit[1]),
        'decay rate': str(exp_fit[2]),
        'all': str(exp_fit),
    }
    print(result[parameter_choice])


if __name__ == '__main__':
    fit_exponential_decay()
