"""
Reads in a json file containing OCV information and fits the model from
Birkl2015 (A Parametric OCV Model for Li-Ion Batteries,
DOI 10.1149/2.0331512jes) to it.
"""

from ast import literal_eval
import json
from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.fit_and_plot_ocv',
    version='3.0'
)
@xmlhelpy.option(
    'input-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description="Persistent record identifier where the input data is stored."
)
@xmlhelpy.option(
    'filename',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="File name."
)
@xmlhelpy.option(
    'output-record',
    char='e',
    param_type=xmlhelpy.Integer,
    required=True,
    description="Persistent record identifier where the results get stored."
)
@xmlhelpy.option(
    'title',
    char='t',
    default=None,
    param_type=xmlhelpy.String,
    description="Title of the plot. Defaults to the file name."
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
    'soc-key',
    char='s',
    default='SOC [C]',
    param_type=xmlhelpy.String,
    description="Key in the json dictionary for the SOC values."
)
@xmlhelpy.option(
    'ocv-key',
    char='o',
    default='OCV [V]',
    param_type=xmlhelpy.String,
    description="Key in the json dictionary for the OCV values."
)
@xmlhelpy.option(
    'phases',
    char='a',
    default=4,
    param_type=xmlhelpy.Integer,
    description="Number of phases of the OCV model. More are more accurate."
)
@xmlhelpy.option(
    'charge-number',
    char='z',
    default=1.0,
    param_type=xmlhelpy.Float,
    description="The charge number of the electrode interface reaction."
)
@xmlhelpy.option(
    'temperature',
    char='k',
    default=298.15,
    param_type=xmlhelpy.Float,
    description="Temperature at which the OCV got measured."
)
@xmlhelpy.option(
    'soc-range-bounds',
    char='u',
    default='(0.2, 0.8)',
    param_type=xmlhelpy.String,
    description=(
        "2-tuple giving the lower maximum and upper minimum SOC range to be "
        "considered in the automatic data SOC range determination."
    )
)
@xmlhelpy.option(
    'soc-range-limits',
    char='g',
    default='(0.0, 1.0)',
    param_type=xmlhelpy.String,
    description=(
        "Optional hard lower and upper bounds for the SOC correction from "
        "the left and the right side, respectively, as a 2-tuple. Use it "
        "if you know that your OCV data is incomplete and by how much. "
        "Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to allow the "
        "SOC range estimation to assign datapoints to the asymptotes."
    )
)
@xmlhelpy.option(
    'assign-fitted-soc-to',
    char='c',
    default=None,
    param_type=xmlhelpy.Choice(
        [None, 'Positive', 'Negative'],
        case_sensitive=False,
    ),
    description=(
        "Set this to 'Positive' or 'Negative' to apply the fitted SOC range "
        "on the data and assign the result to one of the two electrodes."
    )
)
@xmlhelpy.option(
    'flip-soc-convention',
    char='p',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Set to True if assigned SOCs shall go in the other direction. "
        "'soc-range' arguments always work as if this was set to False. "
        "Flips the SOCs by subtracting them from 1."
    )
)
@xmlhelpy.option(
    'spline-soc-range',
    char='b',
    default='(0.01, 0.99)',
    param_type=xmlhelpy.String,
    description=(
        "2-tuple giving the SOC range in which the SOC(OCV) model function "
        "gets inverted by a smoothing spline interpolation."
    )
)
@xmlhelpy.option(
    'spline-order',
    char='j',
    default=2,
    param_type=xmlhelpy.Integer,
    description=(
        "Order of the aforementioned smoothing spline. Setting it to 0 "
        "only fits and plots the OCV model."
    )
)
@xmlhelpy.option(
    'spline-print',
    char='l',
    default=None,
    param_type=xmlhelpy.String,
    description=(
        "If set to either 'python' or 'matlab', a string representation of "
        "the smoothing spline gets appended to the results file."
    )
)
@xmlhelpy.option(
    'normalized-xaxis',
    char='x',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "If True, the x-axis gets rescaled to match the asymptotes of the OCV "
        "fit function at 0 and 1."
    )
)
@xmlhelpy.option(
    'distance-order',
    char='d',
    default='2',
    param_type=xmlhelpy.String,
    description=(
        "The order of the norm of the vector of the distances between OCV "
        "data and OCV model. Default is 2, i.e., the Euclidean norm. "
        "1 sets it to absolute distance, and float('inf') sets it to "
        "maximum distance."
    )
)
@xmlhelpy.option(
    'minimize-options',
    char='m',
    default="{}",
    param_type=xmlhelpy.String,
    description=(
        "Dictionary that gets passed to scipy.optimize.minimize with the "
        "method 'trust-constr'. See scipy.optimize.show_options with the "
        "arguments 'minimize' and 'trust-constr' for details."
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
@xmlhelpy.option(
    'display',
    char='v',
    is_flag=True,
    description=(
        "Toggle to display the plot on the machine this script runs on."
    )
)
def fit_and_plot_ocv(
    input_record,
    filename,
    output_record,
    title,
    format,
    soc_key,
    ocv_key,
    phases,
    charge_number,
    temperature,
    soc_range_bounds,
    soc_range_limits,
    assign_fitted_soc_to,
    flip_soc_convention,
    spline_soc_range,
    spline_order,
    spline_print,
    normalized_xaxis,
    distance_order,
    minimize_options,
    overwrite,
    display
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.visualization import fit_and_plot_OCV
    from kadi_apy.lib.core import KadiManager, Record
    import matplotlib.pyplot as plt

    manager = KadiManager()

    file_prefix = filename.split('.')[0]
    if title is None:
        title = file_prefix

    if not isfile(filename):
        input_record_handle = Record(manager, id=input_record, create=False)
        file_id = input_record_handle.get_file_id(filename)
        input_record_handle.download_file(file_id, filename)

    with open(filename, 'r') as f:
        json_data = json.load(f)
        socs, ocvs = zip(*sorted(zip(json_data[soc_key], json_data[ocv_key])))

    if max(socs) < 0.0:
        socs = [-entry for entry in socs]
    if max(ocvs) < 0.0:
        ocvs = [-entry for entry in ocvs]
    # Electrochemical model convention is "inverted == True".
    # See fit_and_plot_OCV for why it is called "inverted".
    if not ocvs[0] > ocvs[-1]:
        socs, ocvs = zip(*reversed(list(zip(socs, ocvs))))

    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
    ocv_fit_result = fit_and_plot_OCV(
        ax,
        socs,
        ocvs,
        title,
        phases=phases,
        z=charge_number,
        T=temperature,
        SOC_range_bounds=literal_eval(soc_range_bounds),
        SOC_range_limits=literal_eval(soc_range_limits),
        spline_SOC_range=literal_eval(spline_soc_range),
        spline_order=spline_order,
        spline_print=spline_print,
        parameters_print=True,
        inverted=True,
        info_accuracy=True,
        normalized_xaxis=normalized_xaxis,
        distance_order=float(distance_order),
        minimize_options=literal_eval(minimize_options),
    )
    with open(file_prefix + '_fit_log.json', 'w') as f:
        f.write(ocv_fit_result.to_json())
    ax.set_xlabel(soc_key)
    ax.set_ylabel(ocv_key)
    fig.tight_layout()
    fig.savefig(
        file_prefix + '_fit_plot.' + format,
        bbox_inches='tight',
        pad_inches=0.0
    )

    if display:
        plt.show()

    if assign_fitted_soc_to is not None:
        true_SOCs = list(
            ocv_fit_result.SOC_range[0]
            + (ocv_fit_result.SOC - ocv_fit_result.SOC[0])
            / (ocv_fit_result.SOC[-1] - ocv_fit_result.SOC[0])
            * (ocv_fit_result.SOC_range[1] - ocv_fit_result.SOC_range[0])
        )
        true_SOCs = true_SOCs[::-1]
        if flip_soc_convention:
            true_SOCs = [1 - t for t in true_SOCs]
        json_data[
            assign_fitted_soc_to.capitalize() + " electrode SOC [-]"
        ] = true_SOCs
        with open(
            file_prefix + '_with_' + assign_fitted_soc_to.lower()
            + '_soc.json',
            'w'
        ) as f:
            json.dump(json_data, f)

    output_record_handle = Record(manager, id=output_record, create=False)
    output_record_handle.upload_file(
        file_prefix + '_fit_log.json', force=overwrite
    )
    output_record_handle.upload_file(
        file_prefix + '_fit_plot.' + format, force=overwrite
    )
    if assign_fitted_soc_to is not None:
        output_record_handle.upload_file(
            file_prefix + '_with_' + assign_fitted_soc_to.lower()
            + '_soc.json',
            force=overwrite
        )


if __name__ == '__main__':
    fit_and_plot_ocv()
