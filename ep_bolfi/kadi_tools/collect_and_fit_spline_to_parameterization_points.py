"""
Reads in multiple parameterization result files representing individual
data points of a function dependent on some independent variable, and
fits a spline to data points given their independent variable.
"""

from ast import literal_eval
import json
import re
from os.path import isfile
import unicodedata
import xmlhelpy


def get_valid_filename(s):
    """
    Takes a string and transforms it into an ASCII-only filename.

    :param s:
        Any string.
    :returns:
        *s*, with diacritics stripped and non-letters removed except -_.
        Multiple adjacent instances of - or _ are shortened to one.
    """
    return re.sub(
        r'[-\s]+',
        '-',
        re.sub(
            r'[^\w\s-]',
            '',
            unicodedata.normalize('NFKD', s).encode(
                'ascii', 'ignore'
            ).decode('ascii').lower()
        )
    ).strip('-_')


@xmlhelpy.command(
    name='python -m '
    'ep_bolfi.kadi_tools.collect_and_fit_spline_to_parameterization_points',
    version='3.0'
)
@xmlhelpy.option(
    'input-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the data points."
    )
)
@xmlhelpy.option(
    'input-files',
    char='f',
    param_type=xmlhelpy.TokenList(separator='?'),
    required=True,
    description=(
        "File names of the parameterization result files, separated by '?'."
    )
)
@xmlhelpy.option(
    'input-names',
    char='e',
    param_type=xmlhelpy.TokenList(separator=','),
    required=True,
    description=(
        "List of the dependent variables to fit a spline to, separated by ','."
    )
)
@xmlhelpy.option(
    'input-transformation',
    char='t',
    param_type=xmlhelpy.Choice([None, 'log'], case_sensitive=False),
    default=None,
    description=(
        "Set to 'log' if you want to fit the spline to the logarithm of the "
        "data points and get the exponential of that spline as a result."
    )
)
@xmlhelpy.option(
    'spline-smoothing',
    char='m',
    param_type=xmlhelpy.FloatRange(min=0),
    default=None,
    description=(
        "Allows the fitting spline to miss datapoints to reduce overshoots "
        "and lessen the impact of outliers. Defaults to the automatic setting "
        "in scipy.interpolate.UnivariateSpline. Set to 0 to force strict "
        "interpolation through all datapoints. For more fine-grained control, "
        "please refer to the scipy.interpolate.UnivariateSpline documentation."
    )
)
@xmlhelpy.option(
    'independent-variable-record',
    char='v',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record containing the "
        "information about the independent variable."
    )
)
@xmlhelpy.option(
    'independent-variable-file',
    char='a',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "Filename of the file containing the independent variable points."
    )
)
@xmlhelpy.option(
    'independent-variable-name',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "Name to index to the independent variable points in the JSON."
    )
)
@xmlhelpy.option(
    'independent-variable-segments',
    char='s',
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
    'output-record',
    char='o',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record to store the output of "
        "the spline fit in."
    )
)
@xmlhelpy.option(
    'output-suffix',
    char='u',
    param_type=xmlhelpy.String,
    default='',
    description=(
        "Optional extra text to append to the end of the filename. "
        "Use this to have different filenames for the same variable."
    )
)
@xmlhelpy.option(
    'output-format',
    char='l',
    param_type=xmlhelpy.Choice(['python', 'matlab'], case_sensitive=False),
    default='python',
    description=(
        "Choose either 'python' (default) or 'matlab' to have the spline "
        "representation printed as Python or MatLab code."
    )
)
@xmlhelpy.option(
    'image-format',
    char='i',
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
    char='d',
    is_flag=True,
    description=(
        "Toggle to display the plot on the machine this script runs on."
    )
)
def collect_and_fit_spline_to_parameterization_points(
    input_record,
    input_files,
    input_names,
    input_transformation,
    spline_smoothing,
    independent_variable_record,
    independent_variable_file,
    independent_variable_name,
    independent_variable_segments,
    output_record,
    output_suffix,
    output_format,
    image_format,
    overwrite,
    display,
):
    """Please refer to the --help output of this file."""
    import matplotlib.pyplot as plt
    from ep_bolfi.utility.fitting_functions import (
        verbose_spline_parameterization
    )
    from kadi_apy.lib.core import KadiManager, Record
    from numpy import (
        argwhere, array, concatenate, diff, exp, isnan, linspace, log, sum
    )
    from scipy.interpolate import UnivariateSpline

    manager = KadiManager()

    if not isfile(independent_variable_file):
        independent_variable_rec = Record(
            manager, id=independent_variable_record, create=False
        )
        independent_variable_id = independent_variable_rec.get_file_id(
            independent_variable_file
        )
        independent_variable_rec.download_file(
            independent_variable_id, independent_variable_file
        )

    with open(independent_variable_file, 'r') as f:
        independent_variable_data = json.load(f)
    independent_variable_raw_points = independent_variable_data[
        independent_variable_name
    ]
    independent_variable_points = []
    for segment in literal_eval(independent_variable_segments):
        if len(segment) == 2:
            independent_variable_points.extend(
                independent_variable_raw_points[segment[0]:segment[1]]
            )
        elif len(segment) == 3:
            independent_variable_points.extend(
                independent_variable_raw_points[
                    segment[0]:segment[1]:segment[2]
                ]
            )
        else:
            raise ValueError(
                "Segments have to be given as either 2- or 3-tuples."
            )

    dependent_variables = {name: [] for name in input_names}
    dependent_variables_errorbounds = {name: [] for name in input_names}
    for filename in input_files:
        # Skip empty files, happens with trailing separators.
        if not filename:
            continue
        if not isfile(filename):
            input_record_handle = Record(
                manager, id=input_record, create=False
            )
            input_id = input_record_handle.get_file_id(filename)
            input_record_handle.download_file(input_id, filename)
        with open(filename, 'r') as f:
            json_data = json.load(f)
            fitted_variables = json_data['inferred parameters']
            fitting_errorbounds = json_data['error bounds']
        for dependent_name in input_names:
            dependent_variables[dependent_name].append(
                fitted_variables[dependent_name]
            )
            dependent_variables_errorbounds[dependent_name].append(
                fitting_errorbounds[dependent_name]
            )

    # Ensure that the filenames resemble the original variable name.
    output_filenames = [get_valid_filename(s) for s in input_names]

    for input_name, output_filename in zip(input_names, output_filenames):
        output_filename += output_suffix
        if input_transformation == 'log':
            y = log(array(dependent_variables[input_name]))
            spline_transformation = 'exp'
        else:
            y = array(dependent_variables[input_name])
            spline_transformation = ''
        # Warn if there are NaNs in the data (prevents spline fitting).
        nan_indices = argwhere(isnan(y)).T[0]
        if nan_indices.any():
            spline_label = (
                "spline fit failed due to NaNs at data indices:\n"
                + str(list(nan_indices))
            )
        else:
            spline_label = "interpolating spline"
        # The spline passes through all points with smoothing set to 0.
        # Check that the independent variable is increasing.
        if sum(diff(independent_variable_points)) < 0:
            x = independent_variable_points[::-1]
            y = y[::-1]
        else:
            x = independent_variable_points
        fit_spline = UnivariateSpline(x, y, k=2, s=spline_smoothing)
        spline_representation = verbose_spline_parameterization(
            fit_spline.get_coeffs(),
            fit_spline.get_knots(),
            order=2,
            format=output_format,
            function_name=input_name,
            function_args=independent_variable_name,
            derivatives=1,
            spline_transformation=spline_transformation,
        )
        lower = array([
            eb[0] for eb in dependent_variables_errorbounds[input_name]
        ])
        mean = array(dependent_variables[input_name])
        upper = array([
            eb[1] for eb in dependent_variables_errorbounds[input_name]
        ])
        fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
        ax.errorbar(
            independent_variable_points,
            mean,
            yerr=array([mean - lower, upper - mean]),
            fmt='o',
            ms=2,
            label="fitted parameters",
            capthick=1.5,
            capsize=4,
        )
        fine_x = concatenate([
            linspace(start, end, 11) for start, end in zip(x[:-1], x[1:])
        ])
        if input_transformation == 'log':
            ax.plot(
                fine_x, exp(fit_spline(fine_x)), label=spline_label
            )
        else:
            ax.plot(fine_x, fit_spline(fine_x), label=spline_label)
        if input_transformation == 'log':
            ax.set_yscale('log')
        ax.set_xlabel(independent_variable_name)
        ax.set_ylabel(input_name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_filename + '.' + image_format,
            bbox_inches='tight',
            pad_inches=0.0
        )
        with open(output_filename + '.py', 'w', newline='') as f:
            f.write(spline_representation)
        with open(output_filename + '.json', 'w') as f:
            json.dump({
                independent_variable_name: independent_variable_points,
                input_name: dependent_variables[input_name],
                "error bounds of " + input_name: (
                    dependent_variables_errorbounds[input_name]
                ),
            }, f)

    output_record_handle = Record(manager, id=output_record, create=False)
    for output_filename in output_filenames:
        output_filename += output_suffix
        output_record_handle.upload_file(
            output_filename + '.py', force=overwrite
        )
        output_record_handle.upload_file(
            output_filename + '.json', force=overwrite
        )
        output_record_handle.upload_file(
            output_filename + '.' + image_format, force=overwrite
        )

    if display:
        plt.show()


if __name__ == '__main__':
    collect_and_fit_spline_to_parameterization_points()
