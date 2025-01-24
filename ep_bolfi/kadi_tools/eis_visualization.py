"""
Reads in an input file prepared by
``ep_bolfi.kadi_tools.eis_preprocessing`` and runs and compares the
simulations for the initial parameter guesses against the data.
"""

from copy import deepcopy
from itertools import cycle
import json
from os import linesep
from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.eis_visualization',
    version='3.0'
)
@xmlhelpy.option(
    'input-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record with the preprocessed "
        "data and model parameters."
    )
)
@xmlhelpy.option(
    'input-file',
    char='d',
    param_type=xmlhelpy.String,
    required=True,
    description="File name of the preprocessed data file."
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
        + " - parameters: The dictionary of parameters to pass on to the "
        "solver. May be a ep_bolfi.utility.preprocessing.SubstitutionDict."
        + linesep + linesep
        + "It may contain the additional following global variables:"
        + linesep + linesep
        + " - unknowns: The dictionary of unknown parameters. Instead of single "
        "values as in 'parameters', input 2-tuples with their lower and upper "
        "bounds, e.g. from literature."
    )
)
@xmlhelpy.option(
    'parameterization-result-record',
    char='u',
    param_type=xmlhelpy.Integer,
    default=None,
    description=(
        "Optional persistent record identifier of a record with "
        "parameterization results. When given, this result will be used for "
        "determining the 95% confidence bounds of the simulation. Else, if "
        "present, the 'unknowns' in the parameter file will be used."
    )
)
@xmlhelpy.option(
    'parameterization-result-file',
    char='z',
    param_type=xmlhelpy.String,
    default=None,
    description="Name of the file with the parameterization results."
)
@xmlhelpy.option(
    'output-record',
    char='o',
    param_type=xmlhelpy.Integer,
    required=True,
    description=(
        "Persistent record identifier of the record to store the plot and "
        "the log of the solver in."
    )
)
@xmlhelpy.option(
    'full-factorial',
    char='e',
    default=True,
    param_type=xmlhelpy.Bool,
    description=(
        "If 'unknowns' are present in the parameters file, or if a "
        "parameterization result is provided, this determines whether only "
        "each parameter is explored in its bounds (False), or if all "
        "combinations of them at their bounds are explored (True) for the "
        "95% confidence interval of the simulation."
    )
)
@xmlhelpy.option(
    'include-uncertainties',
    char='c',
    default=True,
    param_type=xmlhelpy.Bool,
    description=(
        "By default, the uncertainties get evaluated at their 95% intervals. "
        "Set to False to only evaluate the 95% confidence of the unknowns."
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
    'feature-fontsize',
    char='b',
    default=14,
    param_type=xmlhelpy.Integer,
    description="Fontsize used for the feature labels."
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
@xmlhelpy.option(
    'interactive-plot',
    char='i',
    is_flag=True,
    requires=['display'],
    description=(
        "Togggle to display an interactive plot to explore the search "
        "parameter space with. Only works with one EIS segment at a time."
    )
)
def eis_visualization(
    input_record,
    input_file,
    parameters_file,
    parameterization_result_record,
    parameterization_result_file,
    output_record,
    full_factorial,
    include_uncertainties,
    title,
    feature_fontsize,
    format,
    overwrite,
    display,
    interactive_plot,
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.kadi_tools.eis_preprocessing import eis_feature_visualizer
    from ep_bolfi.models.analytic_impedance import AnalyticImpedance
    from ep_bolfi.utility.preprocessing import prepare_parameter_combinations
    from ep_bolfi.utility.visualization import (
        interactive_impedance_model,
        nyquist_plot,
        push_apart_text,
        update_legend
    )
    from kadi_apy.lib.core import KadiManager, Record
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    manager = KadiManager()

    file_prefix = input_file.split(".")[0]
    if title is None:
        title = file_prefix

    if not isfile(input_file) or not isfile("local_parameter_file.py"):
        inp_rec = Record(manager, id=input_record, create=False)
    if not isfile(input_file):
        input_id = inp_rec.get_file_id(input_file)
        inp_rec.download_file(input_id, input_file)
    if not isfile("local_parameter_file.py"):
        parameters_id = inp_rec.get_file_id(parameters_file)
        inp_rec.download_file(parameters_id, "local_parameter_file.py")
    from local_parameter_file import parameters, transform_unknowns, unknowns

    with open(input_file, 'r') as f:
        input_data = json.load(f)
    number_of_peaks = input_data['number_of_peaks']
    frequency_limits = input_data['frequency_limits']
    initial_socs = input_data['initial_socs']
    three_electrode = input_data['three_electrode']
    dimensionless_reference_electrode_location = (
        input_data['dimensionless_reference_electrode_location']
    )
    lambda_values = input_data['lambda_values']
    uncertainties = input_data['uncertainties']
    discretization = input_data['discretization']
    experiment_data = input_data['experiment_data']

    # Make the separate real and imaginary part into complex again.
    experiment_data = [
        np.array(experiment_data[0]),
        np.array(experiment_data[1]) + 1j * np.array(experiment_data[2])
    ]

    # Note: geomspace is logspace with directly defined endpoints.
    f_eval = np.geomspace(*frequency_limits, discretization)
    s_eval = 1j * f_eval

    if parameterization_result_record:
        if not isfile(parameterization_result_file):
            opt_rec = Record(
                manager, id=parameterization_result_record, create=False
            )
            file_id = opt_rec.get_file_id(parameterization_result_file)
            opt_rec.download_file(file_id, parameterization_result_file)
        with open(parameterization_result_file, 'r') as f:
            optimization_result = json.load(f)
            parameters.update(optimization_result['inferred parameters'])
            parameters_to_try = None
            covariance = optimization_result['covariance']
            order_of_parameter_names = list(
                optimization_result['inferred parameters'].keys()
            )
    else:
        parameters_to_try = unknowns
        covariance = None
        order_of_parameter_names = list(parameters_to_try.keys())

    if include_uncertainties:
        for name, uncertainty in uncertainties.items():
            distribution = getattr(stats, uncertainty[0])
            interval = distribution.interval(
                0.95, *uncertainty[1:]
            )
            parameters_to_try[name] = interval

    individual_bounds, combinations = prepare_parameter_combinations(
        parameters,
        parameters_to_try,
        covariance,
        order_of_parameter_names,
        transform_unknowns,
        confidence=0.95
    )

    def eis_simulator(trial_parameters):
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
            solutions[1].append(solution)
        return solutions

    original_solution = eis_simulator(parameters)
    errorbars = {name: [] for name in individual_bounds.keys()}
    for name, trial_parameter_sets in individual_bounds.items():
        for trial_parameters in trial_parameter_sets:
            errorbars[name].append(eis_simulator(trial_parameters))
    if full_factorial:
        errorbars.update({"all permutations": []})
        for trial_parameters in combinations:
            errorbars["all permutations"].append(
                eis_simulator(trial_parameters)
            )
    plot_errorbars = [
        solution
        for errorbar in errorbars.values()
        for solution in errorbar
    ]

    def feature_visualizer(f, Z):
        return eis_feature_visualizer(
            [f, Z], number_of_peaks, lambda_values
        )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycler = cycle(prop_cycle.by_key()['color'])
    legend_handles = []
    legend_labels = []
    all_texts = []
    fig, ax = plt.subplots(figsize=(2**0.5 * 6, 6))
    ax.tick_params(axis="y", direction="in", left="off", labelleft="on")
    ax.tick_params(axis="x", direction="in", left="off", labelleft="on")
    nyquist_plot(
        fig,
        ax,
        *experiment_data,
        lw=2,
        title_text=title,
        legend_text="experiment",
        equal_aspect=False
    )
    feature_color = next(color_cycler)
    for f, Z, fit in feature_visualizer(*experiment_data):
        all_texts.append(
            ax.text(
                Z[0].real,
                Z[1].real,
                fit,
                color=feature_color,
                fontsize=feature_fontsize
            )
        )
    legend_handles.append(
        mpatches.Patch(color=feature_color, label="experiment features")
    )
    legend_labels.append("experiment features")
    nyquist_plot(
        fig,
        ax,
        *original_solution,
        lw=2,
        ls='-.',
        title_text=title,
        legend_text="simulation",
        equal_aspect=False,
        add_frequency_colorbar=False
    )
    feature_color = next(color_cycler)
    for f, Z, fit in feature_visualizer(*original_solution):
        all_texts.append(
            ax.text(
                Z[0].real,
                Z[1].real,
                fit,
                color=feature_color,
                fontsize=feature_fontsize
            )
        )
    legend_handles.append(
        mpatches.Patch(color=feature_color, label="simulation features")
    )
    legend_labels.append("simulation features")
    gray_colormap = ListedColormap(['gray'])
    for pe in plot_errorbars:
        for f, Z in zip(*pe):
            nyquist_plot(
                fig,
                ax,
                f,
                Z,
                cmap=gray_colormap,
                lw=0.8,
                title_text=title,
                legend_text=None,
                equal_aspect=False,
                add_frequency_colorbar=False
            )
    legend_handles.append(
        mpatches.Patch(color='gray', label="95% confidence")
    )
    legend_labels.append("95% confidence")

    update_legend(
        ax, additional_handles=legend_handles, additional_labels=legend_labels
    )

    fig.tight_layout()
    push_apart_text(fig, ax, all_texts, lock_xaxis=True)
    fig.savefig(
        file_prefix + '_plot.' + format, bbox_inches='tight', pad_inches=0.0
    )
    if interactive_plot and display:
        if len(initial_socs) > 1:
            raise ValueError(
                "Interactive EIS visualization works for single segments only."
            )
        trial_parameters = deepcopy(parameters)
        for electrode in ["negative", "positive"]:
            soc_value = initial_socs[0][
                "Initial concentration in "
                + electrode
                + " electrode [mol.m-3]"
            ]
            if soc_value is not None:
                trial_parameters[
                    "Initial concentration in "
                    + electrode
                    + " electrode [mol.m-3]"
                ] = soc_value
        interactive_impedance_model(
            experiment_data[0][0],
            experiment_data[1][0],
            trial_parameters,
            unknowns,
            transform_unknowns,
            three_electrode=three_electrode,
            dimensionless_reference_electrode_location=(
                dimensionless_reference_electrode_location
            )
        )
    if display:
        plt.show()

    out_rec = Record(manager, id=output_record, create=False)
    out_rec.upload_file(file_prefix + '_plot.' + format, force=overwrite)


if __name__ == '__main__':
    eis_visualization()
