"""
Reads in an input file prepared by
``ep_bolfi.kadi_tools.gitt_preprocessing`` and runs and compares the
simulations for the initial parameter guesses against the data.
"""

from ast import literal_eval
from copy import deepcopy
import json
from os import linesep
from os.path import isfile
import xmlhelpy


# Copied from pybamm.plotting.plot_voltage_components.
# Edited to allow for overpotential plots of individual electrodes.
def plot_voltage_components(
    input_data,
    parameters,
    fig,
    ax,
    show_legend=True,
    split_by_electrode=False,
    three_electrode=None,
    dimensionless_reference_electrode_location=0.5,
    show_plot=True,
    **kwargs_fill,
):
    """
    Generate a plot showing the component overpotentials that make up
    the voltage

    Parameters
    ----------
    input_data : :class:`pybamm.Solution` or :class:`pybamm.Simulation`
        Solution or Simulation object from which to extract voltage
        components.
    fig : matplotlib Figure
        The figure that contains the plot. Create with
        ``layout='constrained'`` for proper legend placement.
    ax : matplotlib Axis
        The axis on which to put the plot.
    show_legend : bool, optional
        Whether to display the legend. Default is True
    split_by_electrode : bool, optional
        Whether to show the overpotentials for the negative and positive
        electrodes
        separately. Default is False.
    three_electrode : str, optional
        With None, does nothing (i.e., cell potentials are used). If
        set to either 'positive' or 'negative', instead of cell
        potentials, the base for the displayed voltage will be the
        potential of the 'positive' or 'negative' electrode against a
        reference electrode. For placement of said reference electrode,
        please refer to "dimensionless_reference_electrode_location".
    dimensionless_reference_electrode_location : float, optional
        The location of the reference electrode, given as a scalar
        between 0 (placed at the point where negative electrode and
        separator meet) and 1 (placed at the point where positive
        electrode and separator meet). Defaults to 0.5 (in the middle).
    show_plot : bool, optional
        Whether to show the plots. Default is True. Set to False if you
        want to only display the plot after plt.show() has been called.
    kwargs_fill
        Keyword arguments: :obj:`matplotlib.axes.Axes.fill_between`
    """
    from ep_bolfi.utility.preprocessing import calculate_desired_voltage
    import matplotlib.pyplot as plt
    import numpy as np
    from pybamm.simulation import Simulation
    from pybamm.solvers.solution import Solution
    # Check if the input is a Simulation and extract Solution
    if isinstance(input_data, Simulation):
        solution = input_data.solution
    elif isinstance(input_data, Solution):
        solution = input_data

    # Set a default value for alpha, the opacity
    kwargs_fill = {"alpha": 0.6, **kwargs_fill}

    if split_by_electrode is False:
        overpotentials = [
            "Battery particle concentration overpotential [V]",
            "X-averaged battery reaction overpotential [V]",
            "X-averaged battery concentration overpotential [V]",
            "X-averaged battery electrolyte ohmic losses [V]",
            "X-averaged battery solid phase ohmic losses [V]",
        ]
        labels = [
            "Particle concentration overpotential",
            "Reaction overpotential",
            "Electrolyte concentration overpotential",
            "Ohmic electrolyte overpotential",
            "Ohmic electrode overpotential",
        ]
    else:
        if three_electrode == "negative":
            overpotentials = [
                "Battery negative particle concentration overpotential [V]",
                "X-averaged battery negative reaction overpotential [V]",
                "Electrode-electrolyte concentration overpotential [V]",
                "Separator-electrolyte concentration overpotential [V]",
                "Ohmic electrolyte overpotential [V]",
                "X-averaged battery negative solid phase ohmic losses [V]",
            ]
            labels = [
                "Negative particle concentration overpotential",
                "Negative reaction overpotential",
                "Electrode-electrolyte concentration overpotential",
                "Separator-electrolyte concentration overpotential",
                "Ohmic electrolyte overpotential",
                "Ohmic negative electrode overpotential",
            ]
        elif three_electrode == "positive":
            overpotentials = [
                "Battery positive particle concentration overpotential [V]",
                "X-averaged battery positive reaction overpotential [V]",
                "Electrode-electrolyte concentration overpotential [V]",
                "Separator-electrolyte concentration overpotential [V]",
                "Ohmic electrolyte overpotential [V]",
                "X-averaged battery positive solid phase ohmic losses [V]",
            ]
            labels = [
                "Positive particle concentration overpotential",
                "Positive reaction overpotential",
                "Electrode-electrolyte concentration overpotential",
                "Separator-electrolyte concentration overpotential",
                "Ohmic electrolyte overpotential",
                "Ohmic positive electrode overpotential",
            ]
        else:
            overpotentials = [
                "Battery negative particle concentration overpotential [V]",
                "Battery positive particle concentration overpotential [V]",
                "X-averaged battery negative reaction overpotential [V]",
                "X-averaged battery positive reaction overpotential [V]",
                "X-averaged battery concentration overpotential [V]",
                "X-averaged battery electrolyte ohmic losses [V]",
                "X-averaged battery negative solid phase ohmic losses [V]",
                "X-averaged battery positive solid phase ohmic losses [V]",
            ]
            labels = [
                "Negative particle concentration overpotential",
                "Positive particle concentration overpotential",
                "Negative reaction overpotential",
                "Positive reaction overpotential",
                "Electrolyte concentration overpotential",
                "Ohmic electrolyte overpotential",
                "Ohmic negative electrode overpotential",
                "Ohmic positive electrode overpotential",
            ]

    # Plot
    # Initialise
    time = solution["Time [h]"].entries
    if split_by_electrode is False:
        ocv = solution["Battery open-circuit voltage [V]"]
        initial_ocv = ocv(time[0])
        ocv = ocv.entries
        ax.fill_between(
            time, ocv, initial_ocv, **kwargs_fill, label="Open-circuit voltage"
        )
    else:
        ocp_n = solution[
            "Battery negative electrode bulk open-circuit potential [V]"
        ]
        ocp_p = solution[
            "Battery positive electrode bulk open-circuit potential [V]"
        ]
        initial_ocp_n = ocp_n(time[0])
        initial_ocp_p = ocp_p(time[0])
        delta_ocp_n = ocp_n.entries - initial_ocp_n
        delta_ocp_p = ocp_p.entries - initial_ocp_p
        if three_electrode:
            if three_electrode == "positive":
                initial_ocv = initial_ocp_p
                ax.fill_between(
                    time,
                    initial_ocv + delta_ocp_p,
                    initial_ocv,
                    **kwargs_fill,
                    label="Positive open-circuit potential",
                )
                ocv = initial_ocv + delta_ocp_p
            elif three_electrode == "negative":
                initial_ocv = initial_ocp_n
                ax.fill_between(
                    time,
                    initial_ocv + delta_ocp_n,
                    initial_ocv,
                    **kwargs_fill,
                    label="Negative open-circuit potential",
                )
                ocv = initial_ocv + delta_ocp_n
        else:
            initial_ocv = initial_ocp_p - initial_ocp_n
            ax.fill_between(
                time,
                initial_ocv - delta_ocp_n,
                initial_ocv,
                **kwargs_fill,
                label="Negative open-circuit potential",
            )
            ax.fill_between(
                time,
                initial_ocv - delta_ocp_n + delta_ocp_p,
                initial_ocv - delta_ocp_n,
                **kwargs_fill,
                label="Positive open-circuit potential",
            )
            ocv = initial_ocv - delta_ocp_n + delta_ocp_p
    top = ocv
    if three_electrode:
        geometry = solution.all_models[0].geometry
        x_n = geometry["negative electrode"]["x_n"]
        x_s = geometry["separator"]["x_s"]
        x_p = geometry["positive electrode"]["x_p"]
        L_n = x_n["max"] - x_n["min"]
        L_s = x_s["max"] - x_s["min"]
        L_p = x_p["max"] - x_p["min"]
        dimensional_location = (
            L_n + dimensionless_reference_electrode_location * L_s
        )
        # Directly calculate the electrolyte potentials for three-
        # electrode setups, as the X-averaged versions do not apply.
        # Use the SPMe expression, as it should apply.
        current = solution["Current [A]"].entries
        κ = parameters["Electrolyte conductivity [S.m-1]"]
        # Evaluate κ if it is a function.
        if callable(κ):
            c_e = parameters["Initial concentration in electrolyte [mol.m-3]"]
            T = parameters["Initial temperature [K]"]
            κ = κ(c_e, T).value
        A = (
            parameters["Electrode width [m]"]
            * parameters["Electrode height [m]"]
        )
        ε_s = parameters["Separator porosity"]
        β_s = parameters[
            "Separator Bruggeman coefficient (electrolyte)"
        ]
        if three_electrode == "positive":
            ε_p = parameters["Positive electrode porosity"]
            β_p = parameters[
                "Positive electrode Bruggeman coefficient (electrolyte)"
            ]
            ohmic_elde_elyte = current / (κ * A) * L_p / (3 * ε_p**β_p)
            ohmic_sep_elyte = current / (κ * A) * (
                (1 - dimensionless_reference_electrode_location) * L_s
                / (ε_s**β_s)
            )
            ohmic_elyte = ohmic_elde_elyte + ohmic_sep_elyte
        elif three_electrode == "negative":
            ε_n = parameters["Negative electrode porosity"]
            β_n = parameters[
                "Negative electrode Bruggeman coefficient (electrolyte)"
            ]
            ohmic_elde_elyte = current / (κ * A) * L_n / (3 * ε_n**β_n)
            ohmic_sep_elyte = current / (κ * A) * (
                dimensionless_reference_electrode_location * L_s
                / (ε_s**β_s)
            )
            ohmic_elyte = ohmic_elde_elyte + ohmic_sep_elyte
    # Plot components
    for overpotential, label in zip(overpotentials, labels):
        # negative overpotentials are positive for a discharge and negative
        # for a charge so we have to multiply by -1 to show them correctly
        sgn = -1 if "negative" in overpotential else 1
        if three_electrode == "negative":
            sgn *= -1
        three_electrode_sign = -1 if three_electrode == "negative" else 1
        # The labels checked for here are only used for three_electrode.
        if "lectrolyte concentration overpotential [V]" in overpotential:
            reference_electrode_potential = three_electrode_sign * (
                solution["Electrolyte potential [V]"](
                    x=dimensional_location
                )
            )
            if three_electrode == "positive":
                electrode_separator_interface_potential = (
                    solution["Electrolyte potential [V]"](x=L_n + L_s)
                )
                mean_elde_elyte_potential = solution[
                    "X-averaged positive electrolyte potential [V]"
                ].entries
            elif three_electrode == "negative":
                electrode_separator_interface_potential = -(
                    solution["Electrolyte potential [V]"](x=L_n)
                )
                mean_elde_elyte_potential = -solution[
                    "X-averaged negative electrolyte potential [V]"
                ].entries
            if overpotential[:9] == "Electrode":
                state = (
                    mean_elde_elyte_potential
                    - electrode_separator_interface_potential
                    + ohmic_elde_elyte
                )
            elif overpotential[:9] == "Separator":
                state = (
                    electrode_separator_interface_potential
                    - reference_electrode_potential
                    + ohmic_sep_elyte
                )
        elif overpotential == "Ohmic electrolyte overpotential [V]":
            state = -ohmic_elyte
        else:
            state = solution[overpotential].entries
        bottom = top + sgn * state
        ax.fill_between(time, bottom, top, **kwargs_fill, label=label)
        top = bottom

    if three_electrode:
        V = calculate_desired_voltage(
            solution,
            solution["Time [s]"].entries,
            1,  # voltage_scale
            False,  # overpotential
            three_electrode,
            dimensionless_reference_electrode_location,
            parameters={
                "Negative electrode thickness [m]": L_n,
                "Separator thickness [m]": L_s,
                "Positive electrode thickness [m]": L_p,
            }
        )
    else:
        V = solution["Battery voltage [V]"].entries
    ax.plot(time, V, "k--", label="Voltage")

    if show_legend:
        leg = fig.legend(loc='outside lower center', frameon=True)
        leg.get_frame().set_edgecolor("k")

    # Labels
    ax.set_xlim([time[0], time[-1]])
    ax.set_xlabel("Experiment run-time  /  h")

    y_min, y_max = (
        0.98 * min(np.nanmin(V), np.nanmin(ocv)),
        1.02 * (max(np.nanmax(V), np.nanmax(ocv))),
    )
    ax.set_ylim([y_min, y_max])

    if show_plot:
        plt.show()

    return fig, ax


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.gitt_visualization',
    version='${VERSION}'
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
        + " - unknowns: The dictionary of unknown parameters. Instead of "
        "single values as in 'parameters', input 2-tuples with their lower "
        "and upper bounds, e.g. from literature."
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
    'output-variables',
    char='a',
    default='None',
    param_type=xmlhelpy.String,
    description=(
        "List (in Python form) of the variable names that should be plotted "
        "in an interactive PyBaMM plot, if 'display' has been set."
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
    'xlabel',
    char='x',
    param_type=xmlhelpy.String,
    default="Experiment run-time  /  h",
    description="Label to be used for the x-axis."
)
@xmlhelpy.option(
    'voltage-scale',
    char='s',
    param_type=xmlhelpy.Choice(
        ['p', 'n', 'µ', 'm', '', 'k', 'M', 'G', 'T'],
        case_sensitive=True
    ),
    default='',
    description="Scale to plot the voltage in."
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
    'verbose',
    char='l',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Whether or not the simulation solver itself gets logged. "
        "Since PyBaMM's logger is bugged, it will only show up in "
        "the workflow log with [stderr]. See ~/.process_manager/workflows."
    )
)
def gitt_visualization(
    input_record,
    input_file,
    parameters_file,
    parameterization_result_record,
    parameterization_result_file,
    output_record,
    full_factorial,
    include_uncertainties,
    output_variables,
    title,
    xlabel,
    voltage_scale,
    feature_fontsize,
    format,
    overwrite,
    display,
    verbose,
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.kadi_tools.gitt_preprocessing import gitt_feature_visualizer
    from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method
    from ep_bolfi.utility.preprocessing import (
        simulate_all_parameter_combinations,
        SubstitutionDict
    )
    from ep_bolfi.utility.visualization import plot_comparison, push_apart_text
    from kadi_apy.lib.core import KadiManager, Record
    import matplotlib.pyplot as plt
    from pybamm.models.full_battery_models import lithium_ion
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
    reference_parameters = deepcopy(parameters)

    with open(input_file, 'r') as f:
        input_data = json.load(f)
    list_of_feature_indices = input_data['list_of_feature_indices']
    initial_socs = input_data['initial_socs']
    current_input = input_data['current_input']
    overpotential = input_data['overpotential']
    three_electrode = input_data['three_electrode']
    dimensionless_reference_electrode_location = (
        input_data['dimensionless_reference_electrode_location']
    )
    sqrt_cutoff = input_data['sqrt_cutoff']
    sqrt_start = input_data['sqrt_start']
    exp_cutoff = input_data['exp_cutoff']
    uncertainties = input_data['uncertainties']
    model_name = input_data['model_name']
    discretization = input_data['discretization']
    experiment_data = input_data['experiment_data']

    # If no extra arguments are given, this will be just the model name.
    # Else, the second entry will be the comma-denoted list of arguments.
    model_components = model_name.split("(")
    model_prefix = model_components[0]
    model_args = []
    model_kwargs = {}
    if len(model_components) > 1:
        for argument in model_components[1][:-1].split(","):
            # Remove whitespace.
            argument = argument.strip()
            if "=" in argument:  # Keyword argument.
                kwarg = argument.split("=")
                model_kwargs[kwarg[0]] = literal_eval(kwarg[1])
            elif argument == "":
                continue
            else:  # Positional argument.
                model_args.append(literal_eval(argument))
    model_instance = getattr(lithium_ion, model_prefix)(
        *model_args, **model_kwargs
    )

    # Apply the correct initial SOC.
    for electrode in ["negative", "positive"]:
        soc_value = initial_socs[
            "Initial concentration in "
            + electrode
            + " electrode [mol.m-3]"
        ]
        if soc_value is not None:
            parameters[
                "Initial concentration in "
                + electrode
                + " electrode [mol.m-3]"
            ] = soc_value

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
            interval = distribution.interval(0.95, *uncertainty[1:])
            parameters_to_try[name] = interval

    # When using SubstitutionDict, additional parameters may be variable.
    if isinstance(parameters, SubstitutionDict):
        additional_input_parameters = parameters.dependent_variables(
            order_of_parameter_names
        )
    else:
        additional_input_parameters = []

    solutions, errorbars = simulate_all_parameter_combinations(
        model_instance,
        current_input,
        *spectral_mesh_pts_and_method(**discretization),
        parameters,
        parameters_to_try=parameters_to_try,
        covariance=covariance,
        order_of_parameter_names=order_of_parameter_names,
        additional_input_parameters=additional_input_parameters,
        transform_parameters=transform_unknowns,
        full_factorial=full_factorial,
        verbose=verbose,
        # logging_file=file_prefix + '_solver.log' if verbose else None,
        voltage_scale={
            'p': 1e-12,
            'n': 1e-9,
            'µ': 1e-6,
            'm': 1e-3,
            '': 1,
            'k': 1e3,
            'M': 1e6,
            'G': 1e9,
            'T': 1e12,
        }[voltage_scale],
        overpotential=overpotential,
        three_electrode=three_electrode,
        dimensionless_reference_electrode_location=(
            dimensionless_reference_electrode_location
        ),
        t_eval=[t for t_segment in experiment_data[0] for t in t_segment],
    )
    errorbar_names = list(errorbars.keys())
    for p_name in errorbar_names:
        if p_name != "all parameters":
            errorbars.pop(p_name)
    errorbars["95% confidence"] = errorbars.pop("all parameters")

    def feature_visualizer(t, U):
        return gitt_feature_visualizer(
            [t, U],
            list_of_feature_indices,
            sqrt_cutoff,
            sqrt_start,
            exp_cutoff
        )

    output_variables = literal_eval(output_variables)
    fig, ax = plt.subplots(figsize=(2**0.5 * 3, 3.5), layout='constrained')
    text_objects = plot_comparison(
        fig,
        ax,
        solutions,
        errorbars,
        experiment_data,
        t_eval=[t for t_segment in experiment_data[0] for t in t_segment],
        title=title,
        xlabel=xlabel,
        ylabel=(
            "Cell overpotential  /  " + voltage_scale + "V"
            if overpotential else
            "Terminal voltage  /  " + voltage_scale + "V"
        ),
        interactive_plot=display and output_variables,
        feature_visualizer=feature_visualizer,
        voltage_scale={
            'p': 1e-12,
            'n': 1e-9,
            'µ': 1e-6,
            'm': 1e-3,
            '': 1,
            'k': 1e3,
            'M': 1e6,
            'G': 1e9,
            'T': 1e12,
        }[voltage_scale],
        output_variables=output_variables,
        feature_fontsize=feature_fontsize,
        overpotential=overpotential,
        three_electrode=three_electrode,
        dimensionless_reference_electrode_location=(
            dimensionless_reference_electrode_location
        ),
        parameters=reference_parameters,
        use_cycles=True,
    )
    push_apart_text(fig, ax, text_objects, lock_xaxis=True)
    fig.savefig(
        file_prefix + '_plot.' + format, bbox_inches='tight', pad_inches=0.0
    )
    fig_comp, ax_comp = plt.subplots(
        figsize=(4, 4), layout="constrained"
    )
    sol_comp = list(solutions.values())[0]
    plot_voltage_components(
        sol_comp,
        reference_parameters,
        fig_comp,
        ax_comp,
        split_by_electrode=True,
        three_electrode=three_electrode,
        dimensionless_reference_electrode_location=(
            dimensionless_reference_electrode_location
        ),
        show_plot=False,  # just disables plt.show()
    )
    ax_comp.set_ylabel("Cell voltage  /  V")
    ax_comp.autoscale(tight=True)
    fig_comp.savefig(
        file_prefix + '_voltage_components.' + format,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if display:
        plt.show()

    out_rec = Record(manager, id=output_record, create=False)
    out_rec.upload_file(file_prefix + '_plot.' + format, force=overwrite)
    out_rec.upload_file(
        file_prefix + '_voltage_components.' + format, force=overwrite
    )
    # if verbose:
    #     out_rec.upload_file(file_prefix + '_solver.log', force=overwrite)


if __name__ == '__main__':
    gitt_visualization()
