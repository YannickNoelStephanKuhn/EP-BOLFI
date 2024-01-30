"""!@package ep_bolfi.utility.visualization
Various helper and plotting functions for common data visualizations.
"""

from itertools import cycle
import pybamm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
# import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, Slider
import os
from ep_bolfi.models.analytic_impedance import AnalyticImpedance
from ep_bolfi.utility.fitting_functions import (
    smooth_fit, inverse_OCV_fit_function, d_dE_OCV_fit_function, fit_OCV,
    OCV_fit_result, verbose_spline_parameterization, fit_drt
)
from ep_bolfi.utility.preprocessing import (
    OCV_from_CC_CV, calculate_desired_voltage
)
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
    @param ax
        The axis which viewport shall be adjusted.
    @param xmin
        The highest lower bound for the x-axis.
    @param xmax
        The lowest upper bound for the x-axis.
    @param ymin
        The highest lower bound for the y-axis.
    @param ymax
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
    @param ax
        The axis which texts shall be adjusted.
    @param title
        The new fontsize for the title.
    @param xaxis
        The new fontsize for the x-axis label.
    @param yaxis
        The new fontsize for the y-axis label.
    @param xticks
        The new fontsize for the ticks/numbers at the x-axis.
    @param yticks
        The new fontsize for the ticks/numbers at the y-axis.
    @param legend
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
    @param ax
        The axis which legend shall be updated.
    @param additional_handles
        The same input "ax.legend(...)" would expect. A list of artists.
    @param additional_labels
        The same input "ax.legend(...)" would expect. A list of strings.
    @param additonal_handler_map
        The same input "ax.legend(...)" would expect for "handler_map".
        Please note that, due to the internal structure of the Legend
        class, only entries with keys that represent classes work right.
        Entries that have _instances_ of classes (i.e., objects) for
        keys work exactly once, since the original handle of them is
        lost in the initialization of a Legend.
    """

    try:
        old_legend = ax.get_legend()
        handles = old_legend.legend_handles
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
    @param fig
        The figure which contains the text.
    @param ax
        The axis which contains the text.
    @param text_objects
        A list of the text objects that shall be pushed apart.
    @param lock_xaxis
        If True, texts will only be moved in the y-direction.
    @param temp_path
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
    @param x
        The independent variable.
    @param y
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
    @param x
        The independent variable.
    @param y
        The dependent variable.
    @param z
        Specify colors.
    @param cmap
        Specify a colormap for colors.
    @param norm
        Specify a normalization for mapping z to the colormap.
        Example: matplotlib.colors.LogNorm(10**(-2), 10**4).
    @param linewidth
        The linewidth of the generated LineCollection.
    @param linestyle
        The linestyle of the generated LineCollection. If the individual
        lines in there are too short, its effect might not be visible.
    @param alpha
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


def nyquist_plot(
    fig,
    ax,
    ω,
    Z,
    cmap=plt.get_cmap('tab20b'),
    ls='-',
    lw=3,
    title_text="Impedance Measurement",
    legend_text="impedance",
    colorbar_label="Frequency  /  Hz",
    add_frequency_colorbar=True,
    equal_aspect=True,
):
    """!@brief Plot an impedance measurement.
    @param fig
        The matplotlib.Figure for plotting
    @param ax
        The matplotlib.Axes for plotting.
    @param ω
        The frequencies at which the impedeance was measured.
        May be a list of lists for multiple measurements.
    @param Z
        The impedances that were measured at those frequencies.
        May be a list of lists for multiple measurements.
    @param cmap
        The colormap that is used to visualize the frequencies.
    @param ls
        The linestyle of the plot.
    @param lw
        The linewidth of the plot.
    @param title_text
        The text for the title.
    @param legend_text
        The text for the legend. May be a list for multiple measurements.
    @param colorbar_label
        The label that is displayed next to the colorbar.
    @param add_frequency_colorbar
        Set to False if 'fig' was already decorated with a colorbar.
    @param equal_aspect
        Set to False in case the impedance is too vertical or horizontal.
    @return
        A list of the LineCollection objects of the impedance plots.
    """

    if hasattr(Z[0], '__len__'):
        if not hasattr(ω[0], '__len__'):
            ω = len(Z) * list(ω)
        if type(legend_text) is not list:
            legend_text = len(Z) * [legend_text]
        color_range = [ω[0][0], ω[0][-1]]
        for frequencies in ω[1:]:
            color_range[0] = min([color_range[0], frequencies[0]])
            color_range[1] = max([color_range[1], frequencies[-1]])
        alphas = np.linspace(1.0, 0.2, len(Z))
    else:
        color_range = [ω[0], ω[-1]]
        ω = [ω]
        Z = [Z]
        legend_text = [legend_text]
        alphas = [1.0]
    lcs = []
    for frequencies, impedances, alpha, lt in zip(ω, Z, alphas, legend_text):
        real = np.real(impedances)
        imag = np.imag(impedances)
        lc = colorline(
            real,
            -imag,
            np.linspace(
                frequencies[0], frequencies[-1], len(np.atleast_1d(impedances))
            ),
            cmap=cmap,
            norm=matplotlib.colors.Normalize(frequencies[0], frequencies[-1]),
            linestyle=ls,
            linewidth=lw,
            alpha=alpha
        )
        ax.add_collection(lc)
        # Add an item to the legend.
        if lt:
            update_legend(ax, [lc], [lt])
        # Update the viewport.
        update_limits(
            ax,
            np.nanmin(real, where=(real > -float('inf')), initial=0.0),
            np.nanmax(real, where=(real < float('inf')), initial=0.0),
            np.nanmin(-imag, where=(-imag > -float('inf')), initial=0.0),
            np.nanmax(-imag, where=(-imag < float('inf')), initial=0.0)
        )
        lcs.append(lc)
    if add_frequency_colorbar:
        fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.LogNorm(*color_range), cmap=cmap
            ),
            ax=ax,
            label=colorbar_label
        )
    ax.set_title(title_text)
    ax.set_xlabel("Real part  /  Ω")
    ax.set_ylabel("-Imaginary part  /  Ω")
    if equal_aspect:
        ax.set_aspect('equal')
    # This ensures that the colorbar is displayed nicely.
    fig.tight_layout()
    return lcs


def bode_plot(
    fig,
    ax_real,
    ax_imag,
    ω,
    Z,
    cmap=plt.get_cmap('tab20b'),
    ls_real='-',
    ls_imag='-.',
    lw=3,
    title_text="Impedance Measurement",
    legend_text="impedance",
):
    """!@brief Plot an impedance measurement.
    @param fig
        The matplotlib.Figure for plotting
    @param ax
        The matplotlib.Axes for plotting.
    @param ω
        The frequencies at which the impedeance was measured.
        May be a list of lists for multiple measurements.
    @param Z
        The impedances that were measured at those frequencies.
        May be a list of lists for multiple measurements.
    @param cmap
        The colormap that is used to differentiate multiple impedances.
    @param ls_real
        The linestyle of the plot of the real part of the impedance.
    @param ls_imag
        The linestyle of the plot of the imaginary part of the impedance.
    @param lw
        The linewidth of the plot.
    @param title_text
        The text for the title.
    @param legend_text
        The text for the legend. May be a list for multiple measurements.
    """

    if hasattr(Z[0], '__len__'):
        if not hasattr(ω[0], '__len__'):
            ω = len(Z) * list(ω)
        if type(legend_text) is not list:
            legend_text = len(Z) * [legend_text]
    else:
        ω = [ω]
        Z = [Z]
        legend_text = [legend_text]
    colors = cmap(np.linspace(0.0, 1.0, len(Z)))

    for frequencies, impedances, color, lt in zip(ω, Z, colors, legend_text):
        real = np.real(impedances)
        imag = np.imag(impedances)
        ax_real.semilogx(
            frequencies, real, color=color, ls=ls_real, lw=lw, label=lt
        )
        ax_imag.semilogx(
            frequencies, -imag, color=color, ls=ls_imag, lw=lw, label=lt
        )
    ax_real.set_title(title_text)
    ax_real.set_xlabel("Frequency  /  Hz")
    ax_real.set_ylabel("Real part  /  Ω")
    ax_imag.set_title(title_text)
    ax_imag.set_xlabel("Frequency  /  Hz")
    ax_imag.set_ylabel("-Imaginary part  /  Ω")
    ax_real.legend()
    ax_imag.legend()
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
    use_cycles=False,
    overpotential=False,
    three_electrode=None,
    dimensionless_reference_electrode_location=0.5,
    parameters=None,

):
    """!@brief Tool for comparing simulation<->experiment with features.
    First, a pybamm.QuickPlot shows the contents of "solutions".
    Then, a plot for feature visualization is generated.
    @param ax
        The Axes onto which the comparison shall be plotted.
    @param solutions
        A dictionary of pybamm.Solution objects. The key goes into the
        figure legend and the value gets plotted as a line.
    @param errorbars
        A dictionary of lists of either pybamm.Solution objects or
        lists of the desired variable at 't_eval' timepoints. The key
        goes into the figure legend and the values get plotted as a
        shaded area between the minimum and maximum.
    @param experiment
        A list/tuple of at least length 2. The first two entries are the
        data timepoints in s and voltages in V. The entries after that
        are only relevant as optional arguments to "feature_visualizer".
    @param solution_visualization
        This list/tuple is passed on to "feature_visualizer" in place of
        the additional entries of "experiment" for the visualization of
        the simulated features.
    @param t_eval
        The timepoints at which the "solutions" and "errorbars" shall be
        evaluated in s. If None are given, the timepoints of the
        solutions will be chosen.
    @param title
        The optional title of the feature visualization plot.
    @param xlabel
        The optional label of the x-axis there. Please note that the
        time will be given in h.
    @param ylabel
        The optional label of the y-axis there.
    @param feature_visualizer
        This is an optional function that takes "experiment" and returns
        a list of 2- or 3-tuples. The first two entries in the tuples
        are x- and y-data to be plotted alongside the other curves.
        The third entry is a string that is plotted at the respective
        (x[0], y[0])-coordinates.
    @param interactive_plot
        Choose whether or not a browsable overview of the solution
        components shall be shown. Please note that this disrupts the
        execution of this function until that plot is closed, since it
        is plotted in a new figure rather than in ax.
    @param output_variables
        The variables of "solutions" that are to be plotted. When None
        are specified, some default variables get plotted. The full list
        of possible variables to plot are returned by PyBaMM models from
        their get_fundamental_variables and get_coupled_variables
        functions. Enter the keys from that as strings in a list here.
    @param voltage_scale
        The plotted voltage gets divided by this value. For example,
        1e-3 would produce a plot in [mV]. The voltage given to the
        "feature_visualizer" is not affected.
    @param use_cycles
        If True, the .cycles property of the "solutions" is used for the
        "feature_visualizer". Plotting is not affected.
    @param overpotential
        If True, only the overpotential of "solutions" gets plotted.
        Otherwise, the cell voltage (OCV + overpotential) is plotted.
    @param three_electrode
        By default, does nothing (i.e., cell potentials are used). If
        set to either 'positive' or 'negative', instead of cell
        potentials, the base for the displayed voltage will be the
        potential of the 'positive' or 'negative' electrode against a
        reference electrode. For placement of said reference electrode,
        please refer to "dimensionless_reference_electrode_location".
    @param dimensionless_reference_electrode_location
        The location of the reference electrode, given as a scalar
        between 0 (placed at the point where negative electrode and
        separator meet) and 1 (placed at the point where positive
        electrode and separator meet). Defaults to 0.5 (in the middle).
    @param parameters
        The parameter dictionary that was used for the simulation. Only
        needed for a three-electrode output.
    @return
        The text objects that were generated according to
        "feature_visualizer".
    """

    all_texts = []

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
        color = next(color_cycler)
        for vis in feature_visualizer(*experiment):
            if len(vis) > 2:
                x, y, fit = vis
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
            t_eval = solution.t
            t = t_eval / 3600
        else:
            t = solution["Time [h]"](t_eval)

        U = calculate_desired_voltage(
            solution,
            t_eval,
            voltage_scale,
            overpotential,
            three_electrode,
            dimensionless_reference_electrode_location,
            parameters
        )

        ax.plot(t, U, lw=2, label=name, ls=next(ls_cycler),
                color=next(color_cycler))
        if use_cycles:
            feature_t = [cycle["Time [h]"].entries * 3600.0
                         for cycle in solution.cycles]
            feature_U = [
                calculate_desired_voltage(
                    cycle,
                    t_cycle,
                    1.0,
                    overpotential,
                    three_electrode,
                    dimensionless_reference_electrode_location,
                    parameters
                )
                for cycle, t_cycle in zip(solution.cycles, feature_t)
            ]
        else:
            feature_t = t_eval
            feature_U = U
        for vis in feature_visualizer(feature_t, feature_U,
                                      *solution_visualization):
            if len(vis) > 2:
                x, y, fit = vis
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
            if isinstance(errorbar[0], list):
                raise ValueError(
                    "If 't_eval' is not given, errorbar entries must be "
                    "lists of pybamm.Solution instances or 'solutions' "
                    "must have at least one entry."
                )
            for e in errorbar:
                t_eval.extend(e.t)
        if experiment is not None:
            t_eval.extend([t for t_segment in experiment[0]
                           for t in np.atleast_1d(t_segment)])
        t_eval = np.array(sorted(t_eval))

    for name, errorbar in errorbars.items():
        errorbar_plot = []
        for eb_entry in errorbar:
            if isinstance(eb_entry, pybamm.Solution):
                errorbar_plot.append(
                    calculate_desired_voltage(
                        eb_entry,
                        t_eval,
                        voltage_scale,
                        overpotential,
                        three_electrode,
                        dimensionless_reference_electrode_location,
                        parameters
                    )
                )
            else:
                errorbar_plot.append(eb_entry)
        errorbar_plot = np.array(errorbar_plot)
        minimum_plot = np.min(errorbar_plot, axis=0)
        maximum_plot = np.max(errorbar_plot, axis=0)
        ax.fill_between(t_eval / 3600, minimum_plot, maximum_plot, alpha=1/3,
                        color=next(color_cycler), label=name)

    update_legend(ax, additional_handles=legend_handles,
                  additional_labels=legend_labels)

    if interactive_plot:
        # Plot the solution with a slider.
        plot_solutions = ([s for s in solutions.values()])
        plot = pybamm.QuickPlot(
            plot_solutions,
            linestyles=["-"],
            output_variables=output_variables or [
                "Electrolyte concentration [mol.m-3]",
                "Electrolyte potential [V]",
                "Current [A]",
                "Voltage [V]",
            ]
        )
        # testing=True has the sole effect of not calling plt.show.
        plot.dynamic_plot(testing=True)

    return all_texts


def cc_cv_visualization(
    fig,
    ax,
    dataset,
    max_number_of_clusters=4,
    cmap=plt.get_cmap('tab20c'),
    check_location=[0.1, 0.7, 0.2, 0.225]
):
    """!@brief Automatically labels and displays a CC-CV dataset.
    A checkbutton list gets added for browsing through the labels.
    @param fig
        The Figure where the check boxes shall be drawn.
    @param ax
        The Axes where the measurements shall be drawn.
    @param dataset
        An instance of Cycling_Information. Please refer to
        utility.dataset_formatting for further information.
    @param max_number_of_clusters
        The maximum number of different labels that shall be tried in
        the automatic labelling of the dataset.
    @param cmap
        The Colormap that is used for colorcoding the cycles.
    @param check_location
        The (x,y)-coordinates (first two entries) and the (width,height)
        (last two entries) of the checkbutton list canvas.
    @return
        The CheckButtons instance. The only thing that must be done with
        this is to keep it in memory. Otherwise, it gets garbage
        collected ("weak reference") and the checkbuttons don't work.
    """

    norm = matplotlib.colors.Normalize(np.min(dataset.indices),
                                       np.max(dataset.indices))
    I_means = []
    I_vars = []
    U_means = []
    U_vars = []
    plots = []
    for i, t, I, U in zip(dataset.indices, dataset.timepoints,
                          dataset.currents, dataset.voltages):
        I_means.append(np.mean(I))
        I_vars.append(np.var(I))
        U_means.append(np.mean(U))
        U_vars.append(np.var(U))

        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        I_int = np.array([0.5 * (I1 + I0) for (I0, I1) in zip(I[:-1], I[1:])])
        # Give the capacities in Ah.
        C = [0.0] + list(np.cumsum(Δt * I_int) / 3600)
        plots.append((C, U, cmap(norm(i))))
        update_limits(ax, np.min(C), np.max(C), np.min(U), np.max(U))

    # Search for the "best" data clustering.
    info = [[x0, x1, x2, x3]
            for (x0, x1, x2, x3) in zip(I_means, I_vars, U_means, U_vars)]
    scores = []
    for i in range(2, max_number_of_clusters + 1):
        clustering_i = KMeans(n_clusters=i)
        labels_i = clustering_i.fit_predict(info)
        scores.append(silhouette_score(info, labels_i))
    clustering = KMeans(n_clusters=np.argmax(scores) + 2)
    labels = clustering.fit_predict(info)

    # The order of the cluster centres gives the indices for the labels.
    names = ['{:4.2f}'.format(center[0]) + ' ± '
             + '{:3.1e}'.format(center[1]) + ' A'
             if np.abs(center[1]) < np.abs(center[3]) else
             '{:4.2f}'.format(center[2]) + ' ± '
             + '{:3.1e}'.format(center[3]) + ' V'
             for center in clustering.cluster_centers_]
    # Organize the plots in a way that allows for a toggle switch.
    organized_plots = {name: [] for name in names}
    for label, (C, U, color) in zip(labels, plots):
        organized_plots[names[label]].append(ax.plot(C, U, color=color)[0])

    ax.set_title("CC-CV cycles")
    ax.set_xlabel("Moved capacity  /  Ah")
    ax.set_ylabel("Voltage  /  V")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label="Cycle number")
    fig.tight_layout()

    # Add the toggle switches to the plot.
    check = CheckButtons(fig.add_axes(check_location), names,
                         [True for name in names])

    def apply_check(name):
        for line in organized_plots[name]:
            line.set_visible(not line.get_visible())
        plt.draw()
    check.on_clicked(apply_check)

    return check


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
    @param ax_ICA_meas
        The Axes where the Incremental Capacity Analysis of the
        measured charge and discharge cycle(s) shall be plotted.
    @param ax_ICA_mean
        The Axes where the Incremental Capacity Analysis of the mean
        voltages of charge and discharge cycle(s) shall be plotted.
    @param ax_OCV_meas
        The Axes where the measured voltage curves shall be plotted.
    @param ax_OCV_mean
        The Axes where the mean voltage curves shall be plotted.
    @param charge
        A Cycling_Information object containing the constant charge
        cycle(s). If more than one CC-CV-cycle shall be analyzed, please
        make sure that the order of this, cv and discharge align.
    @param cv
        A Cycling_Information object containing the constant voltage
        part between charge and discharge cycle(s).
    @param discharge
        A Cycling_Information object containing the constant discharge
        cycle(s). These occur after each cv cycle.
    @param name
        Name of the material for which the CC-CV-cycling was measured.
    @param phases
        Number of phases in the fitting_functions.OCV_fit_function as an
        int. The higher it is, the more (over-)fitted the model becomes.
    @param eval_points
        The number of points for plotting of the OCV curves.
    @param spline_SOC_range
        2-tuple giving the SOC range in which the inverted
        fitting_functions.OCV_fit_function will be interpolated by a
        smoothing spline. Outside of this range the spline is used for
        extrapolation. Use this to fit the SOC range of interest more
        precisely, since a fit of the whole range usually fails due to
        the singularities at SOC 0 and 1. Please note that this range
        considers the 0-1-range in which the given SOC lies and not the
        linear transformation of it from the fitting process.
    @param spline_order
        Order of this smoothing spline. If it is set to 0, only the
        fitting_functions.OCV_fit_function is calculated and plotted.
    @param spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @param spline_print
        If set to either 'python' or 'matlab', a string representation
        of the smoothing spline is printed in the respective format.
    @param parameters_print
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
                                            s=spline_smoothing)
        ax_OCV_meas.plot(dSOC_dV_charge_control(U), U, color=colors[i],
                         ls=ls[2])
        ax_ICA_meas.plot(C, dSOC_dV_charge_control.derivative()(U),
                         color=colors[i], ls=ls[0],
                         label='{:4.2f}'.format(I_mean[i]) + " A charge")

    for i, (I, C, U) in enumerate(zip(I_mean, C_discharge, U_discharge)):
        ax_OCV_meas.plot(C, U, color=colors[i], ls=ls[1],
                         label='{:4.2f}'.format(I_mean[i]) + " A discharge")
        dSOC_dV_discharge_control = smooth_fit(
            U, C, s=spline_smoothing
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
            U_mean, C_eval, s=spline_smoothing
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
    @param ax
        The matplotlib.Axes instance for plotting.
    @param SOC
        Presumed SOC points of the OCV measurement. They only need to be
        precise in respect to relative capacity between measurements.
    @param OCV
        OCV measurements as a list or np.array, matching SOC.
    @param name
        Name of the material for which the OCV curve was measured.
     @param spline_order
        Order of the smoothing spline used for derivation. Default: 2.
     @param spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @param sign
        Put -1 if the ICA comes out negative. Default: 1.
    """

    normalized_SOC = (np.array(SOC) - SOC[0]) / (SOC[-1] - SOC[0])
    ax.plot(SOC, sign * smooth_fit(
        OCV, normalized_SOC, order=spline_order,
        s=spline_smoothing
    ).derivative()(OCV), label=name, lw=2)
    ax.set_xlabel("SOC  /  -")
    ax.set_ylabel("∂SOC/∂OCV  /  V⁻¹")
    ax.set_title("ICA for identifying voltage plateaus")
    ax.legend()


def plot_measurement(
    fig, ax, dataset, title, cmap=plt.get_cmap('tab20c'), plot_current=True
):
    """!@brief Plots current and voltage curves in one diagram.
    Please don't use "fig.tight_layout()" with this, as it very well
    might mess up the placement of the colorbar and the second y-axis.
    Rather, use "plt.subplots(..., constrained_layout=True)".
    @return
        The list of text objects for the numbers.
    """

    norm = matplotlib.colors.Normalize(np.min(dataset.indices),
                                       np.max(dataset.indices))
    # Call "colorbar" before "twinx"; otherwise, the plot is bugged.
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label="Cycle number")
    if plot_current:
        axI = ax.twinx()
    texts = []

    t0 = dataset.timepoints[0][0]
    for i, t, I, U in zip(dataset.indices, dataset.timepoints,
                          dataset.currents, dataset.voltages):
        ax.plot((np.array(t) - t0) / 3600, U, color=cmap(norm(i)))
        texts.append(ax.text(
            (t[0] - t0) / 3600,
            U[0],
            str(i),
            color=cmap(norm(i))
        ))
        if plot_current:
            axI.plot(
                (np.array(t) - t0) / 3600, I, color=cmap(norm(i)), ls='--'
            )
        update_limits(
            ax,
            np.nanmin((np.array(t) - t0) / 3600),
            np.nanmax((np.array(t) - t0) / 3600),
            np.nanmin(U),
            np.nanmax(U)
        )
        if plot_current:
            update_limits(
                axI,
                np.nanmin((np.array(t) - t0) / 3600),
                np.nanmax((np.array(t) - t0) / 3600),
                np.nanmin(I),
                np.nanmax(I)
            )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time  /  h")
    ax.set_ylabel("Voltage  /  V")
    if plot_current:
        axI.set_ylabel("Current  /  A")

    return texts


def fit_and_plot_OCV(
    ax,
    SOC,
    OCV,
    name,
    phases,
    SOC_range_bounds=(0.2, 0.8),
    SOC_range_limits=(0.0, 1.0),
    z=1.0,
    T=298.15,
    fit=None,
    eval_SOC=[0, 1],
    eval_points=200,
    spline_SOC_range=(0.01, 0.99),
    spline_order=2,
    spline_print=None,
    parameters_print=False,
    inverted=True,
    info_accuracy=True,
    normalized_xaxis=False,
    distance_order=2,
    weights=None,
    initial_parameters=None,
    minimize_options=None,
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
    @param ax
        The matplotlib.Axes instance for plotting.
    @param SOC
        Presumed SOC points of the OCV measurement. They only need to be
        precise in respect to relative capacity between measurements.
        The SOC endpoints of the measurement will be fitted using the
        fitting_functions.OCV_fit_function. Type: list or np.array.
    @param OCV
        OCV measurements as a list or np.array.
    @param name
        Name of the material for which the OCV curve was measured.
    @param phases
        Number of phases in the fitting_functions.OCV_fit_function as an
        int. The higher it is, the more (over-)fitted the model becomes.
    @param SOC_range_bounds
        Optional hard upper and lower bounds for the SOC correction from
        the left and the right side, respectively, as a 2-tuple. Use it
        as a limiting guess for the actual SOC range represented in the
        measurement. Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to
        effectively disable SOC range estimation.
    @param SOC_range_limits
        Optional hard lower and upper bounds for the SOC correction from
        the left and the right side, respectively, as a 2-tuple. Use it
        if you know that your OCV data is incomplete and by how much.
        Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to allow the
        SOC range estimation to assign datapoints to the asymptotes.
    @param z
        The charge number of the electrode interface reaction.
    @param T
        The temperature of the electrode.
    @param fit
        May provide the fit parameters if they are already known.
    @param eval_SOC
        Denotes the minimum and maximum SOC to plot the OCV curves at.
    @param eval_points
        The number of points for plotting of the OCV curves.
    @param spline_SOC_range
        2-tuple giving the SOC range in which the inverted
        fitting_functions.OCV_fit_function will be interpolated by a
        smoothing spline. Outside of this range the spline is used for
        extrapolation. Use this to fit the SOC range of interest more
        precisely, since a fit of the whole range usually fails due to
        the singularities at SOC 0 and 1. Please note that this range
        considers the 0-1-range in which the given SOC lies and not the
        linear transformation of it from the fitting process.
    @param spline_order
        Order of this smoothing spline. If it is set to 0, only the
        fitting_functions.OCV_fit_function is calculated and plotted.
    @param spline_print
        If set to either 'python' or 'matlab', a string representation
        of the smoothing spline is printed in the respective format.
    @param parameters_print
        Set to True if the fit parameters should be printed to console.
    @param inverted
        If True (default), the widely adopted SOC convention is assumed.
        If False, the formulation of "A parametric OCV model" is used.
    @param info_accuracy
        If True, some measures of fit accuracy are displayed in the
        figure legend: RMSE (root mean square error), MAE (mean absolute
        error) and ME (maximum error).
    @param normalized_xaxis
        If True, the x-axis gets rescaled to [0,1], where {0,1} matches
        the asymptotes of the OCV fit function.
    @param distance_order
        The argument passed to the numpy.linalg.norm of the vector of
        distances between OCV data and OCV model. Default is 2, i.e.,
        the Euclidean norm. 1 sets it to absolute distance.
    @param weights
        Optional weights to apply to the vector of the distances between
        OCV data and OCV model. Defaults to equal weights.
    @param initial_parameters
        Optional initial guess for the model parameters. If left as-is,
        this will be automatically gleaned from the data. Use only if
        you have another fit to data of the same electrode material.
    @param minimize_options
        Dictionary that gets passed to scipy.optimize.minimize with the
        method 'trust-constr'. See scipy.optimize.show_options with the
        arguments 'minimize' and 'trust-constr' for details.
    """

    SOC = np.array(SOC)
    OCV = np.array(OCV)

    # Allow for the use of capacity instead of non-dimensionalized SOC.

    def normalize(c):
        return (c - SOC[0]) / (SOC[-1] - SOC[0])

    def rescale(soc):
        return SOC[0] + soc * (SOC[-1] - SOC[0])

    if fit is None:
        OCV_model = fit_OCV(
            normalize(SOC),
            OCV,
            N=phases,
            SOC_range_bounds=SOC_range_bounds,
            SOC_range_limits=SOC_range_limits,
            z=z,
            T=T,
            inverted=inverted,
            fit_SOC_range=True,
            distance_order=distance_order,
            weights=weights,
            initial_parameters=initial_parameters,
            minimize_options=minimize_options,
        )
    else:
        OCV_model = OCV_fit_result(fit, SOC, OCV)
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

    OCV_fit = inverse_OCV_fit_function(
        compress(normalize(SOC)), *OCV_model.fit, inverted=inverted
    )
    fit_diff = OCV - OCV_fit
    fit_diff = fit_diff[~np.isnan(fit_diff)]
    fit_RMSE = np.sqrt(np.sum(fit_diff**2))
    fit_MAE = np.mean(np.abs(fit_diff))
    fit_ME = np.max(np.abs(fit_diff))
    label = (
        "OCV model fit of " + name if spline_order <= 0 else
        "OCV model fit (" + str(phases) + " phases) of " + name
        + info_accuracy * (
            ";" + os.linesep + "RMSE " + "{: 5.1e}".format(fit_RMSE)
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
        spline_OCV_data = inverse_OCV_fit_function(
            spline_SOC, *OCV_model.fit, inverted=inverted
        )
        spline_OCV = smooth_fit(
            spline_SOC,
            spline_OCV_data,
            order=spline_order,
            w=np.ones_like(spline_SOC),
            s=fit_MAE**2
        )
        spline_diff = (OCV - spline_OCV(np.array(compress(normalize(SOC)))))
        spline_RMSE = np.sqrt(np.sum(spline_diff**2))
        spline_MAE = np.mean(np.abs(spline_diff))
        spline_ME = np.max(np.abs(spline_diff))
        ax.plot(
            plot_SOC, spline_OCV(np.linspace(*eval_SOC, eval_points)),
            label=(
                "Spline (order " + str(spline_order) + ") of " + name
                + info_accuracy * (
                    ";" + os.linesep + "RMSE " + "{: 5.1e}".format(spline_RMSE)
                    + ", MAE " + "{:5.1e}".format(spline_MAE)
                    + ", ME " + "{:5.1e}".format(spline_ME)
                )
            ),
            lw=2, ls="--"
        )

        OCV_model.spline_interpolation_knots = spline_OCV.get_knots()
        OCV_model.spline_interpolation_coefficients = spline_OCV.get_coeffs()

        if parameters_print:
            print("Knots of interpolating spline:")
            print(spline_OCV.get_knots())
            print("Coefficients of this spline:")
            print(spline_OCV.get_coeffs())
        if spline_print is not None:
            OCV_model.function_string = verbose_spline_parameterization(
                spline_OCV.get_coeffs(),
                spline_OCV.get_knots(),
                spline_order,
                function_name=name,
                format=spline_print,
                derivatives=1,
            )
            print(OCV_model.function_string)

    ax.set_xlabel("SOC  /  -")
    ax.set_ylabel("OCV /  V")
    ax.set_title("Parametric OCV model fit")
    ax.legend()

    return OCV_model


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
    @param fig
        The matplotlib.Figure object for plotting.
    @param ax
        The matplotlib.Axes object for plotting.
    @param correlation
        A two-dimensional (numpy) array that is the correlation matrix.
    @param names
        A list of strings that are names of the variables corresponding
        to each row or column in the correlation matrix.
    @param title
        The title of the heatmap.
    @param cmap
        The matplotlib colormap for the heatmap.
    @param entry_color
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
