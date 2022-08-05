"""!@package utility.visualization
Various helper and plotting functions for common data visualizations.
"""

import pybamm
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
# import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons
import os
from itertools import cycle
from utility.fitting_functions import (
    smooth_fit, inverse_OCV_fit_function, d_dE_OCV_fit_function, fit_OCV,
    OCV_fit_result, verbose_spline_parameterization
)
from utility.preprocessing import OCV_from_CC_CV
# Reset the PyBaMM colour scheme.
plt.style.use("default")


def update_limits(
    ax,
    xmin=float('inf'),
    xmax=-float('inf'),
    ymin=float('inf'),
    ymax=-float('inf')
):
    """!@brief Convenience function for adjusting the view.

    @par ax
        The axis which viewport shall be adjusted.
    @par xmin
        The highest lower bound for the x-axis.
    @par xmax
        The lowest upper bound for the x-axis.
    @par ymin
        The highest lower bound for the y-axis.
    @par ymax
        The lowest upper bound for the y-axis.
    """

    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    # Check if the axis limits haven't been changed yet. If so, ignore.
    if old_xlim == (0, 1):
        old_xlim = (float('inf'), -float('inf'))
    if old_ylim == (0, 1):
        old_ylim = (float('inf'), -float('inf'))
    ax.set_xlim((np.min([old_xlim[0], xmin]), np.max([old_xlim[1], xmax])))
    ax.set_ylim((np.min([old_ylim[0], ymin]), np.max([old_ylim[1], ymax])))


def set_fontsize(
    ax, title=12, xaxis=12, yaxis=12, xticks=12, yticks=12, legend=12
):
    """!@brief Convenience function for fontsize changes.

    @par ax
        The axis which texts shall be adjusted.
    @par title
        The new fontsize for the title.
    @par xaxis
        The new fontsize for the x-axis label.
    @par yaxis
        The new fontsize for the y-axis label.
    @par xticks
        The new fontsize for the ticks/numbers at the x-axis.
    @par yticks
        The new fontsize for the ticks/numbers at the y-axis.
    @par legend
        The new fontsize for the legend entries.
    """

    for item in np.atleast_1d(ax.title):
        item.set_fontsize(title)
    for item in np.atleast_1d(ax.xaxis.label):
        item.set_fontsize(xaxis)
    for item in np.atleast_1d(ax.yaxis.label):
        item.set_fontsize(yaxis)
    for item in np.atleast_1d(ax.get_xticklabels()):
        item.set_fontsize(xticks)
    for item in np.atleast_1d(ax.get_yticklabels()):
        item.set_fontsize(yticks)
    try:
        for item in np.atleast_1d(ax.get_legend().get_texts()):
            item.set_fontsize(legend)
    except AttributeError:
        pass


def update_legend(
    ax,
    additional_handles=[],
    additional_labels=[],
    additional_handler_map={}
):
    """!@brief Makes sure that all items remain and all items show up.

    This basically replaces "ax.legend()" in a way that makes sure that
    new items can be added to the legend without losing old ones.
    Please note that only "handler_map"s with class keys work correctly.

    @par ax
        The axis which legend shall be updated.
    @par additional_handles
        The same input "ax.legend(...)" would expect. A list of artists.
    @par additional_labels
        The same input "ax.legend(...)" would expect. A list of strings.
    @par additonal_handler_map
        The same input "ax.legend(...)" would expect for "handler_map".
        Please note that, due to the internal structure of the Legend
        class, only entries with keys that represent classes work right.
        Entries that have _instances_ of classes (i.e., objects) for
        keys work exactly once, since the original handle of them is
        lost in the initialization of a Legend.
    """

    try:
        old_legend = ax.get_legend()
        handles = old_legend.legendHandles
        labels = [t._text for t in old_legend.texts]
        # Avoid duplicates; sadly, Legend does not carry the original
        # handles, so duplicate labels (which are a bad idea though)
        # are treated as representing duplicate handles.
        try:
            more_handles, more_labels = ax.get_legend_handles_labels()
            more_handles, more_labels = zip(*[
                (h, l)
                for h, l in zip(more_handles, more_labels)
                if l not in labels
            ])
            handles.extend(more_handles)
            labels.extend(more_labels)
        except ValueError:
            # Catch the case where ax.get_legend_handles_labels
            # returnd empty arrays.
            pass
    except AttributeError:
        handles, labels = ax.get_legend_handles_labels()
    handles.extend(additional_handles)
    labels.extend(additional_labels)
    try:
        handler_map = ax.get_legend().get_legend_handler_map()
        # Remove all class keys.
        # handler_map = {key: value for key, value in handler_map.items()
        #                if not isinstance(key, type)}
        handler_map.update(additional_handler_map)
    except AttributeError:
        handler_map = additional_handler_map
    ax.legend(handles, labels, handler_map=handler_map)


def push_apart_text(
    fig, ax, text_objects, lock_xaxis=False, temp_path="./temp_render.png"
):
    """!@brief Push apart overlapping texts until no overlaps remain.

    @par fig
        The figure which contains the text.
    @par ax
        The axis which contains the text.
    @par text_objects
        A list of the text objects that shall be pushed apart.
    @par lock_xaxis
        If True, texts will only be moved in the y-direction.
    @par temp_path
        The path to which a temporary image of the figure "fig" gets
        saved. This is necessary to establish the text bbox sizes.
    """

    fig.savefig(temp_path)
    os.remove(temp_path)
    overlaps = True
    while overlaps:
        overlaps = False
        for i in range(len(text_objects)):
            one_bbox = text_objects[i].get_window_extent()
            one_points = one_bbox.get_points()
            one_bounds = one_bbox.bounds
            for j in range(i + 1, len(text_objects)):
                other_bbox = text_objects[j].get_window_extent()
                other_points = other_bbox.get_points()
                other_bounds = other_bbox.bounds
                if (one_points[0][0] < other_points[1][0]
                        and other_points[0][0] < one_points[1][0]
                        and one_points[0][1] < other_points[1][1]
                        and other_points[0][1] < one_points[1][1]
                        and text_objects[i].get_visible()
                        and text_objects[j].get_visible()):
                    overlaps = True
                    distance = (
                        0.5 * (other_points[0] + other_points[1])
                        - 0.5 * (one_points[0] + one_points[1])
                    )
                    if not np.any(distance != 0.0):
                        distance = np.array([0.0, np.max([
                            one_bounds[3], other_bounds[3]
                        ])])
                    scaling = np.min(np.abs(
                        (np.abs(one_bounds[2:]) + np.abs(other_bounds[2:]))
                        / distance
                    )) - 1
                    translation = scaling * distance
                    if lock_xaxis:
                        translation[0] = 0.0
                    window_to_xy = ax.transData.inverted().transform
                    text_objects[i].set(position=window_to_xy(
                        one_points[0] - 0.25 * translation
                    ), horizontalalignment='left', verticalalignment='bottom')
                    text_objects[j].set(position=window_to_xy(
                        other_points[0] + 0.25 * translation
                    ), horizontalalignment='left', verticalalignment='bottom')


def make_segments(x, y):
    """!@brief Create a list of line segments from x and y coordinates.

    @par x
        The independent variable.
    @par y
        The dependent variable.
    @return
        An array of the form numlines x (points per line) times 2 (x and
        y) array. This is the correct format for LineCollection.
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap('viridis'),
    norm=matplotlib.colors.Normalize(0, 1),
    linewidth=1,
    linestyle='-',
    alpha=1.0
):
    """!@brief Generates a colored line using LineCollection.

    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/
    blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html

    @par x
        The independent variable.
    @par y
        The dependent variable.
    @par z
        Specify colors.
    @par cmap
        Specify a colormap for colors.
    @par norm
        Specify a normalization for mapping z to the colormap.
        Example: matplotlib.colors.LogNorm(10**(-2), 10**4).
    @par linewidth
        The linewidth of the generated LineCollection.
    @par linestyle
        The linestyle of the generated LineCollection. If the individual
        lines in there are too short, its effect might not be visible.
    @par alpha
        The transparency of the generated LineCollection.
    @return
        A matplotlib.collections.LineCollection object "lc". It can be
        plotted by a Matplotlib axis ax with "ax.add_collection(lc)".
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    z = np.array(z, subok=True, copy=False, ndmin=1)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha
    )
    return lc


def impedance_visualization(
    fig,
    ax,
    ω,
    Z,
    cmap=plt.get_cmap('tab20b'),
    ls='-',
    lw=3,
    legend_points=4,
    legend_text="impedance",
    colorbar_label="Frequency  /  Hz"
):
    """!@brief Plot an impedance measurement.

    @par fig
        The matplotlib.Figure for plotting
    @par ax
        The matplotlib.Axes for plotting.
    @par ω
        The frequencies at which the impedeance was measured.
    @par Z
        The impedances that were measured at those frequencies.
    @par cmap
        The colormap that is used to visualize the frequencies.
    @par ls
        The linestyle of the plot.
    @par lw
        The linewidth of the plot.
    @par legend_points
        The number of different colors for the legend entry.
    @par legend_text
        The text for the legend.
    @par colorbar_label
        The label that is displayed next to the colorbar.
    """

    real = np.real(Z)
    imag = np.imag(Z)
    lc = colorline(
        real, -imag, np.linspace(ω[0], ω[-1], len(np.atleast_1d(Z))),
        cmap=cmap, norm=matplotlib.colors.LogNorm(ω[0], ω[-1]), linestyle=ls,
        linewidth=lw
    )
    ax.add_collection(lc)
    # Add an item to the legend.
    update_legend(ax, [lc], [legend_text])
    fig.colorbar(matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.LogNorm(ω[0], ω[-1]), cmap=cmap
        ), ax=ax, label=colorbar_label
    )
    # Update the viewport.
    update_limits(ax, np.min(real), np.max(real), np.min(-imag), np.max(-imag))
    ax.set_title("Impedance Measurement")
    ax.set_xlabel("Real part  /  Ω")
    ax.set_ylabel("-Imaginary part  /  Ω")
    # This ensures that the colorbar is displayed nicely.
    fig.tight_layout()


def plot_comparison(
    ax,
    solutions,
    errorbars,
    experiment,
    solution_visualization=[],
    t_eval=None,
    title="",
    xlabel="",
    ylabel="",
    feature_visualizer=lambda *args: [],
    feature_fontsize=12,
    interactive_plot=False,
    output_variables=None,
    voltage_scale=1.0,
    overpotential=False,
    use_cycles=False
):
    """!@brief Tool for comparing simulation<->experiment with features.

    First, a pybamm.QuickPlot shows the contents of "solutions".
    Then, a plot for feature visualization is generated.

    @par ax
        The Axes onto which the comparison shall be plotted.
    @par solutions
        A dictionary of pybamm.Solution objects. The key goes into the
        figure legend and the value gets plotted as a line.
    @par errorbars
        A dictionary of lists of pybamm.Solution objects. The key goes
        into the figure legend and the value gets plotted as a shaded
        shaded area between the lists' minimum and maximum.
    @par experiment
        A list/tuple of at least length 2. The first two entries are the
        data timepoints in s and voltages in V. The entries after that
        are only relevant as optional arguments to "feature_visualizer".
    @par solution_visualization
        This list/tuple is passed on to "feature_visualizer" in place of
        the additional entries of "experiment" for the visualization of
        the simulated features.
    @par t_eval
        The timepoints at which the "solutions" and "errorbars" shall be
        evaluated in s. If None are given, the timepoints of the
        solutions will be chosen.
    @par title
        The optional title of the feature visualization plot.
    @par xlabel
        The optional label of the x-axis there. Please note that the
        time will be given in h.
    @par ylabel
        The optional label of the y-axis there.
    @par feature_visualizer
        This is an optional function that takes "experiment" and returns
        a list of 2- or 3-tuples. The first two entries in the tuples
        are x- and y-data to be plotted alongside the other curves.
        The third entry is a string that is plotted at the respective
        (x[0], y[0])-coordinates.
    @par interactive_plot
        Choose whether or not a browsable overview of the solution
        components shall be shown. Please note that this disrupts the
        execution of this function until that plot is closed, since it
        is plotted in a new figure rather than in ax.
    @par output_variables
        The variables of "solutions" that are to be plotted. When None
        are specified, some default variables get plotted. The full list
        of possible variables to plot are returned by PyBaMM models from
        their get_fundamental_variables and get_coupled_variables
        functions. Enter the keys from that as strings in a list here.
    @par voltage_scale
        The plotted voltage gets divided by this value. For example,
        1e-3 would produce a plot in [mV]. The voltage given to the
        "feature_visualizer" is not affected.
    @par overpotential
        If True, only the overpotential of "solutions" gets plotted.
        Otherwise, the cell voltage (OCV + overpotential) is plotted.
    @par use_cycles
        If True, the .cycles property of the "solutions" is used for the
        "feature_visualizer". Plotting is not affected.
    @return
        The text objects that were generated according to
        "feature_visualizer".
    """

    all_texts = []

    solution_voltage = ("Total overpotential [V]" if overpotential else
                        "Terminal voltage [V]")

    # Plot the comparison afterwards.
    ls_cycler = cycle(["-", "-.", "--", ":"])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycler = cycle(prop_cycle.by_key()['color'])
    legend_handles = []
    legend_labels = []

    ax.tick_params(axis="y", direction="in", left="off", labelleft="on")
    ax.tick_params(axis="x", direction="in", left="off", labelleft="on")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if experiment is not None:
        ax.plot([t / 3600 for t_segment in experiment[0]
                 for t in np.atleast_1d(t_segment)],
                [U / voltage_scale for U_segment in experiment[1]
                 for U in np.atleast_1d(U_segment)],
                label="experiment", lw=2, color=next(color_cycler))
        # print("Experimental feature fits:")
        color = next(color_cycler)
        for vis in feature_visualizer(*experiment):
            if len(vis) > 2:
                x, y, fit = vis
                # print(fit)
                all_texts.append(
                    ax.text(x[0] / 3600, y[0] / voltage_scale, fit,
                            color=color, fontsize=feature_fontsize)
                )
            else:
                x, y = vis
            if len(x) > 2:
                ax.plot(np.array(x) / 3600, np.array(y) / voltage_scale, lw=2,
                        ls="--", color=color, markevery=[0, -1], ms=10,
                        marker="1")
            else:
                ax.plot(np.array(x) / 3600, np.array(y) / voltage_scale, lw=0,
                        marker="1", ms=10, color=color)
        legend_handles.append(mpatches.Patch(color=color,
                                             label="experiment features"))
        legend_labels.append("experiment features")

    feature_color = next(color_cycler)
    for name, solution in solutions.items():
        if t_eval is None:
            t = solution.t * solution.timescale_eval / 3600
            U = solution[solution_voltage](t * 3600) / voltage_scale
        else:
            t = solution["Time [h]"](t_eval)
            U = solution[solution_voltage](t_eval) / voltage_scale
        # name = solution.model.name

        ax.plot(t, U, lw=2, label=name, ls=next(ls_cycler),
                color=next(color_cycler))
        # print("Simulation feature fits:")
        if use_cycles:
            feature_t = [cycle["Time [h]"].entries * 3600.0
                         for cycle in solution.cycles]
            feature_U = [cycle[solution_voltage].entries
                         for cycle in solution.cycles]
        else:
            feature_t = t * 3600
            feature_U = U
        for vis in feature_visualizer(feature_t, feature_U,
                                      *solution_visualization):
            if len(vis) > 2:
                x, y, fit = vis
                # print(fit)
                all_texts.append(
                    ax.text(x[0] / 3600, y[0] / voltage_scale, fit,
                            color=feature_color, fontsize=feature_fontsize)
                )
            else:
                x, y = vis
            if len(x) > 2:
                ax.plot(np.array(x) / 3600, np.array(y) / voltage_scale, lw=2,
                        ls="--", color=feature_color, markevery=[0, -1], ms=10,
                        marker="2")
            else:
                ax.plot(np.array(x) / 3600, np.array(y) / voltage_scale, lw=0,
                        marker="2", ms=10, color=feature_color)
    if solutions != {}:
        legend_handles.append(mpatches.Patch(color=feature_color,
                                             label="simulation features"))
        legend_labels.append("simulation features")

    if t_eval is None:
        t_eval = []
        for errorbar in errorbars.values():
            for e in errorbar:
                t_eval.extend(e.t * e.timescale_eval)
        if experiment is not None:
            t_eval.extend([t for t_segment in experiment[0]
                           for t in np.atleast_1d(t_segment)])
        t_eval = np.array(sorted(t_eval))

    for name, errorbar in errorbars.items():
        errorbar_plot = np.array([
            solution[solution_voltage](t_eval) / voltage_scale
            for solution in errorbar
        ])
        minimum_plot = np.min(errorbar_plot, axis=0)
        maximum_plot = np.max(errorbar_plot, axis=0)
        ax.fill_between(t_eval / 3600, minimum_plot, maximum_plot, alpha=1/3,
                        color=next(color_cycler), label=name)

    update_legend(ax, additional_handles=legend_handles,
                  additional_labels=legend_labels)

    if interactive_plot:
        if output_variables is not None:
            output_variables = list(output_variables) + [solution_voltage]
        # Plot the solution with a slider.
        plot_solutions = (
            [s for s in solutions.values()]
            # + [e for errorbar in errorbars.values() for e in errorbar]
        )
        plot = pybamm.QuickPlot(
            plot_solutions, linestyles=["-"],
            output_variables=output_variables or [
                "Negative electrode potential [V]",
                "Electrolyte concentration [mol.m-3]",
                "Interface current density [A.m-2]",
                "Positive electrode potential [V]",
                "Overpotential [V]",
                "Electrolyte potential [V]",
                "Current [A]",
                solution_voltage,
            ]
        )
        plot.dynamic_plot()

    return all_texts


def plot_OCV_from_CC_CV(
    ax_ICA_meas,
    ax_ICA_mean,
    ax_OCV_meas,
    ax_OCV_mean,
    charge,
    cv,
    discharge,
    name,
    phases,
    eval_points=200,
    spline_SOC_range=(0.01, 0.99),
    spline_order=2,
    spline_smoothing=2e-3,
    spline_print=None,
    parameters_print=False
):
    """!@brief Visualizes the OCV_fitting.OCV_from_CC_CV output.

    @par ax_ICA_meas
        The Axes where the Incremental Capacity Analysis of the
        measured charge and discharge cycle(s) shall be plotted.
    @par ax_ICA_mean
        The Axes where the Incremental Capacity Analysis of the mean
        voltages of charge and discharge cycle(s) shall be plotted.
    @par ax_OCV_meas
        The Axes where the measured voltage curves shall be plotted.
    @par ax_OCV_mean
        The Axes where the mean voltage curves shall be plotted.
    @par charge
        A Cycling_Information object containing the constant charge
        cycle(s). If more than one CC-CV-cycle shall be analyzed, please
        make sure that the order of this, cv and discharge align.
    @par cv
        A Cycling_Information object containing the constant voltage
        part between charge and discharge cycle(s).
    @par discharge
        A Cycling_Information object containing the constant discharge
        cycle(s). These occur after each cv cycle.
    @par name
        Name of the material for which the CC-CV-cycling was measured.
    @par phases
        Number of phases in the fitting_functions.OCV_fit_function as an
        int. The higher it is, the more (over-)fitted the model becomes.
    @par eval_points
        The number of points for plotting of the OCV curves.
    @par spline_SOC_range
        2-tuple giving the SOC range in which the inverted
        fitting_functions.OCV_fit_function will be interpolated by a
        smoothing spline. Outside of this range the spline is used for
        extrapolation. Use this to fit the SOC range of interest more
        precisely, since a fit of the whole range usually fails due to
        the singularities at SOC 0 and 1. Please note that this range
        considers the 0-1-range in which the given SOC lies and not the
        linear transformation of it from the fitting process.
    @par spline_order
        Order of this smoothing spline. If it is set to 0, only the
        fitting_functions.OCV_fit_function is calculated and plotted.
    @par spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @par spline_print
        If set to either 'python' or 'matlab', a string representation
        of the smoothing spline is printed in the respective format.
    @par parameters_print
        Set to True if the fit parameters should be printed to console.
    """

    (
        OCV_fits,
        I_mean,
        C_charge,
        U_charge,
        C_discharge,
        U_discharge,
        C_evals,
        U_means
    ) = OCV_from_CC_CV(
        charge, cv, discharge, name, phases,
        eval_points=200, spline_SOC_range=(0.01, 0.99), spline_order=2,
        spline_smoothing=2e-3, spline_print=spline_print,
        parameters_print=True
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ls = ['-', '-.', '--', ':']

    for i, (I, C, U) in enumerate(zip(I_mean, C_charge, U_charge)):
        ax_OCV_meas.plot(C, U, color=colors[i], ls=ls[0],
                         label='{:4.2f}'.format(I_mean[i]) + " A charge")
        dSOC_dV_charge_control = smooth_fit(U, C,
                                            smoothing_factor=spline_smoothing)
        ax_OCV_meas.plot(dSOC_dV_charge_control(U), U, color=colors[i],
                         ls=ls[2])
        ax_ICA_meas.plot(C, dSOC_dV_charge_control.derivative()(U),
                         color=colors[i], ls=ls[0],
                         label='{:4.2f}'.format(I_mean[i]) + " A charge")

    for i, (I, C, U) in enumerate(zip(I_mean, C_discharge, U_discharge)):
        ax_OCV_meas.plot(C, U, color=colors[i], ls=ls[1],
                         label='{:4.2f}'.format(I_mean[i]) + " A discharge")
        dSOC_dV_discharge_control = smooth_fit(
            U, C, smoothing_factor=spline_smoothing
        )
        ax_OCV_meas.plot(dSOC_dV_discharge_control(U), U, color=colors[i],
                         ls=ls[2])
        ax_ICA_meas.plot(C, dSOC_dV_discharge_control.derivative()(U),
                         color=colors[i], ls=ls[1],
                         label='{:4.2f}'.format(I_mean[i]) + " A discharge")

    update_legend(ax_OCV_meas,
                  [matplotlib.lines.Line2D([], [], color=colors[0], ls=ls[2])],
                  ["smoothed cycles"])

    for i, (OCV_model, C_eval, U_mean) in enumerate(zip(
            OCV_fits, C_evals, U_means)):
        dSOC_dV_mean = smooth_fit(
            U_mean, C_eval, smoothing_factor=spline_smoothing
        ).derivative()
        dummy_SOC = np.linspace(0.0, 1.0, eval_points)
        plot_SOC = np.linspace(*OCV_model.SOC_range, eval_points)
        OCV_plot = inverse_OCV_fit_function(dummy_SOC, *OCV_model.fit)
        ax_ICA_mean.plot(
            plot_SOC, dSOC_dV_mean(U_mean) / (np.max(C_eval) - np.min(C_eval)),
            color=colors[i], ls=ls[0],
            label='{:4.2f}'.format(I_mean[i]) + " A mean"
        )
        ax_ICA_mean.plot(
            dummy_SOC, -d_dE_OCV_fit_function(OCV_plot, *OCV_model.fit),
            color=colors[i], ls=ls[1],
            label='{:4.2f}'.format(I_mean[i]) + " A fit"
        )
        ax_OCV_mean.plot(plot_SOC, U_mean, color=colors[i], ls=ls[0],
                         label='{:4.2f}'.format(I_mean[i]) + " A mean")
        ax_OCV_mean.plot(dummy_SOC, OCV_plot, color=colors[i], ls=ls[1],
                         label='{:4.2f}'.format(I_mean[i]) + " A fit")
        # ax_OCV_mean.plot(plot_SOC, U_diff, color=colors[i], ls=ls[1],
        #         label='{:4.2f}'.format(I_mean[i]) + " A difference")

    ax_ICA_meas.set_title("Incremental Capacity Analysis")
    ax_ICA_meas.set_xlabel("Discharged capacity  /  Ah")
    ax_ICA_meas.set_ylabel("dSOC / dV  /  V⁻¹")
    update_legend(ax_ICA_meas)
    ax_ICA_mean.set_title("ICA extracted from averaging")
    ax_ICA_mean.set_xlabel("SOC  /  -")
    ax_ICA_mean.set_ylabel("dSOC / dV  /  V⁻¹")
    update_legend(ax_ICA_mean)
    ax_OCV_meas.set_title("Smoothed (dis-)charge curves used for ICA")
    ax_OCV_meas.set_xlabel("Charged capacity  /  Ah")
    ax_OCV_meas.set_ylabel("Cell voltage  /  V")
    update_legend(ax_OCV_meas)
    ax_OCV_mean.set_title("OCV extracted from averaging")
    ax_OCV_mean.set_xlabel("SOC  /  -")
    ax_OCV_mean.set_ylabel("Cell OCV  /  V")
    update_legend(ax_OCV_mean)


def plot_ICA(
    ax, SOC, OCV, name, spline_order=2, spline_smoothing=2e-3, sign=1
):
    """!@brief Show the derivative of charge by voltage.

    @par ax
        The matplotlib.Axes instance for plotting.
    @par SOC
        Presumed SOC points of the OCV measurement. They only need to be
        precise in respect to relative capacity between measurements.
    @par OCV
        OCV measurements as a list or np.array, matching SOC.
    @par name
        Name of the material for which the OCV curve was measured.
     @par spline_order
        Order of the smoothing spline used for derivation. Default: 2.
     @par spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @par sign
        Put -1 if the ICA comes out negative. Default: 1.
    """

    normalized_SOC = (np.array(SOC) - SOC[0]) / (SOC[-1] - SOC[0])
    ax.plot(SOC, sign * smooth_fit(
        OCV, normalized_SOC, order=spline_order,
        smoothing_factor=spline_smoothing
    ).derivative()(OCV), label=name, lw=2)
    ax.set_xlabel("SOC  /  -")
    ax.set_ylabel("∂SOC/∂OCV  /  V⁻¹")
    ax.set_title("ICA for identifying voltage plateaus")
    ax.legend()


def plot_measurement(fig, ax, dataset, title, cmap=plt.get_cmap('tab20c')):
    """!@brief Plots current and voltage curves in one diagram.

    Please don't use "fig.tight_layout()" with this, as it very well
    might mess up the placement of the colorbar and the second y-axis.
    Rather, use "plt.subplots(..., constrained_layout=True)".
    """

    norm = matplotlib.colors.Normalize(np.min(dataset.indices),
                                       np.max(dataset.indices))
    # Call "colorbar" before "twinx"; otherwise, the plot is bugged.
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label="Cycle number")
    axI = ax.twinx()

    t0 = dataset.timepoints[0][0]
    for i, t, I, U in zip(dataset.indices, dataset.timepoints,
                          dataset.currents, dataset.voltages):
        ax.plot((np.array(t) - t0) / 3600, U, color=cmap(norm(i)))
        axI.plot((np.array(t) - t0) / 3600, I, color=cmap(norm(i)), ls='--')
        update_limits(ax, np.min((np.array(t) - t0) / 3600),
                      np.max((np.array(t) - t0) / 3600), np.min(U), np.max(U))
        update_limits(axI, np.min((np.array(t) - t0) / 3600),
                      np.max((np.array(t) - t0) / 3600), np.min(I), np.max(I))

    ax.set_title(title)
    ax.set_xlabel("Elapsed time  /  h")
    ax.set_ylabel("Voltage  /  V")
    axI.set_ylabel("Current  /  A")


def fit_and_plot_OCV(
    ax,
    SOC,
    OCV,
    name,
    phases,
    initial=None,
    z=1.0,
    T=298.15,
    fit=None,
    eval_SOC=[0, 1],
    eval_points=200,
    spline_SOC_range=(0.01, 0.99),
    spline_order=2,
    spline_smoothing=2e-3,
    spline_print=None,
    parameters_print=False,
    inverted=True,
    info_accuracy=True,
    normalized_xaxis=False
):
    """!@brief Fits an SOC(OCV)-model and an OCV(SOC)-evaluable spline.

    # Exemplary fit parameters:

    # Fit parameters of a graphite anode.
    E_0_g = np.array([0.35973, 0.17454, 0.12454, 0.081957])
    γUeminus1_g = np.array([-0.33144, 8.9434e-3, 7.2404e-2, 6.7789e-2])
    a_g = a_fit(γUeminus1_g)
    Δx_g = np.array([8.041e-2, 0.23299, 0.29691, 0.39381])#0.22887
    graphite = [p[i] for i in range(4) for p in [E_0_g, a_g, Δx_g]]

    # Fit parameters of a NMC-622 cathode.
    E_0_NMC = np.array([4.2818, 3.9632, 3.9118, 3.6788])
    γUeminus1_NMC = np.array([-0.22022, -0.083146, 0.070787, -0.11461])
    a_NMC = a_fit(γUeminus1_NMC)
    Δx_NMC = np.array([0.38646, 0.28229, 0.15104, 0.26562])#0.30105
    NMC = [p[i] for i in range(4) for p in [E_0_NMC, a_NMC, Δx_NMC]]

    @par ax
        The matplotlib.Axes instance for plotting.
    @par SOC
        Presumed SOC points of the OCV measurement. They only need to be
        precise in respect to relative capacity between measurements.
        The SOC endpoints of the measurement will be fitted using the
        fitting_functions.OCV_fit_function. Type: list or np.array.
    @par OCV
        OCV measurements as a list or np.array.
    @par name
        Name of the material for which the OCV curve was measured.
    @par phases
        Number of phases in the fitting_functions.OCV_fit_function as an
        int. The higher it is, the more (over-)fitted the model becomes.
    @par initial
        An optional initial guess for the parameters of the model.
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par fit
        May provide the fit parameters if they are already known.
    @par eval_SOC
        Denotes the minimum and maximum SOC to plot the OCV curves at.
    @par eval_points
        The number of points for plotting of the OCV curves.
    @par spline_SOC_range
        2-tuple giving the SOC range in which the inverted
        fitting_functions.OCV_fit_function will be interpolated by a
        smoothing spline. Outside of this range the spline is used for
        extrapolation. Use this to fit the SOC range of interest more
        precisely, since a fit of the whole range usually fails due to
        the singularities at SOC 0 and 1. Please note that this range
        considers the 0-1-range in which the given SOC lies and not the
        linear transformation of it from the fitting process.
    @par spline_order
        Order of this smoothing spline. If it is set to 0, only the
        fitting_functions.OCV_fit_function is calculated and plotted.
    @par spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @par spline_print
        If set to either 'python' or 'matlab', a string representation
        of the smoothing spline is printed in the respective format.
    @par parameters_print
        Set to True if the fit parameters should be printed to console.
    @par inverted
        If True (default), the widely adopted SOC convention is assumed.
        If False, the formulation of "A parametric OCV model" is used.
    @par info_accuracy
        If True, some measures of fit accuracy are displayed in the
        figure legend: RMSE (root mean square error), MAE (mean absolute
        error) and ME (maximum error).
    @par normalized_xaxis
        If True, the x-axis gets rescaled to [0,1], where {0,1} matches
        the asymptotes of the OCV fit function.
    """

    OCV = np.array(OCV)

    # Allow for the use of capacity instead of non-dimensionalized SOC.

    def normalize(c):
        return (c - SOC[0]) / (SOC[-1] - SOC[0])

    def rescale(soc):
        return SOC[0] + soc * (SOC[-1] - SOC[0])

    if fit is None:
        OCV_model = fit_OCV(
            normalize(SOC), OCV, N=phases, initial=initial, z=z, T=T,
            inverted=inverted, fit_SOC_range=True
        )
    else:
        OCV_model = OCV_fit_result([0, 1] + fit, SOC, OCV)
    if parameters_print:
        print("Parameters of OCV fit function:")
        print("SOC range of data: " + repr(OCV_model.SOC_range))
        print("E₀: [" + ", ".join([str(x) for x in OCV_model.E_0]) + "]")
        print("a: [" + ", ".join([str(x) for x in OCV_model.a]) + "]")
        print("Δx: [" + ", ".join([str(x) for x in OCV_model.Δx]) + "]")

    # SOC_range → [0, 1], z ↦ (z - SOC_start) / (SOC_end - SOC_start)
    SOC_start, SOC_end = OCV_model.SOC_range

    def stretch(soc):
        return (soc - SOC_start) / (SOC_end - SOC_start)

    def compress(soc):
        return SOC_start + soc * (SOC_end - SOC_start)

    fit_diff = OCV - inverse_OCV_fit_function(
        compress(normalize(SOC)), *OCV_model.fit, inverted=inverted
    )
    fit_diff = fit_diff[~np.isnan(fit_diff)]
    fit_RMSE = np.sqrt(np.sum(fit_diff**2))
    fit_MAE = np.mean(np.abs(fit_diff))
    fit_ME = np.max(np.abs(fit_diff))
    label = (
        "OCV model fit of " + name if spline_order <= 0 else
        "OCV model fit (" + str(phases) + " phases) of " + name
        + info_accuracy * (
            "; RMSE " + "{:5.1e}".format(fit_RMSE)
            + ", MAE " + "{:5.1e}".format(fit_MAE)
            + ", ME " + "{:5.1e}".format(fit_ME)
        )
    )

    if normalized_xaxis:
        plot_SOC = np.linspace(*eval_SOC, eval_points)
        ax.plot(compress(normalize(SOC)), OCV,
                label="OCV measurement of " + name, marker='1', lw=0, ms=10)
    else:
        plot_SOC = rescale(stretch(np.linspace(*eval_SOC, eval_points)))
        ax.plot(SOC, OCV, label="OCV measurement of " + name, marker='1', lw=0,
                ms=10)
    ax.plot(
        plot_SOC, inverse_OCV_fit_function(
            np.linspace(*eval_SOC, eval_points), *OCV_model.fit,
            inverted=inverted
        ), lw=2, label=label
    )

    if spline_order > 0:
        spline_SOC = np.linspace(*spline_SOC_range, eval_points)
        spline_OCV = smooth_fit(spline_SOC, inverse_OCV_fit_function(
                spline_SOC, *OCV_model.fit, inverted=inverted
            ), order=spline_order, smoothing_factor=spline_smoothing)
        spline_diff = (OCV - spline_OCV(np.array(compress(normalize(SOC)))))
        spline_RMSE = np.sqrt(np.sum(spline_diff**2))
        spline_MAE = np.mean(np.abs(spline_diff))
        spline_ME = np.max(np.abs(spline_diff))
        ax.plot(
            plot_SOC, spline_OCV(np.linspace(*eval_SOC, eval_points)),
            label=(
                "Spline (order " + str(spline_order) + ") of " + name
                + info_accuracy * (
                    "; RMSE " + "{:5.1e}".format(spline_RMSE)
                    + ", MAE " + "{:5.1e}".format(spline_MAE)
                    + ", ME " + "{:5.1e}".format(spline_ME)
                )
            ),
            lw=2, ls="--"
        )

        if parameters_print:
            print("Knots of interpolating spline:")
            print(spline_OCV.get_knots())
            print("Coefficients of this spline:")
            print(spline_OCV.get_coeffs())
        if spline_print is not None:
            print(verbose_spline_parameterization(
                spline_OCV.get_coeffs(), spline_OCV.get_knots(), spline_order,
                function_name=name, format=spline_print
            ))

    ax.set_xlabel("SOC  /  -")
    ax.set_ylabel("OCV /  V")
    ax.set_title("Parametric OCV model fit")
    ax.legend()


def visualize_correlation(
    fig,
    ax,
    correlation,
    names=None,
    title=None,
    cmap=plt.get_cmap('BrBG'),
    entry_color='w'
):
    """!@brief Produces a heatmap of a correlation matrix.

    @par fig
        The matplotlib.Figure object for plotting.
    @par ax
        The matplotlib.Axes object for plotting.
    @par correlation
        A two-dimensional (numpy) array that is the correlation matrix.
    @par names
        A list of strings that are names of the variables corresponding
        to each row or column in the correlation matrix.
    @par title
        The title of the heatmap.
    @par cmap
        The matplotlib colormap for the heatmap.
    @par entry_color
        The colour of the correlation matrix entries.
    """

    # This one line produces the heatmap.
    ax.imshow(correlation, cmap=cmap,
              norm=matplotlib.colors.Normalize(-1, 1))
    # Define the coordinates of the ticks.
    ax.set_xticks(np.arange(len(correlation)))
    ax.set_yticks(np.arange(len(correlation)))
    # Display the names alongside the rows and columns.
    if names is not None:
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        # Rotate the labels at the x-axis for better readability.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor')

    # Plot the correlation matrix entries on the heatmap.
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            if i == j:
                color = 'w'
            else:
                color = entry_color
            ax.text(j, i, '{:3.2f}'.format(correlation[i][j]), ha='center',
                    va='center', color=color)

    ax.set_title(title or "Correlation matrix")
    fig.colorbar(matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-1, 1), cmap=cmap
        ), ax=ax, label="correlation"
    )
    fig.tight_layout()
