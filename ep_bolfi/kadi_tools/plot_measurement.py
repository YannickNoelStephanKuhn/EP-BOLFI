"""
Reads in a ``Cycling_Information`` json representation and plots it.
"""

import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.plot_measurement',
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
    'datatype',
    char='y',
    default="cycling",
    param_type=xmlhelpy.Choice(["cycling", "static", "impedance"]),
    description=(
        "Type of the measurement, which determines the internal format for "
        "further processing. 'cycling' refers to Cycling_Information, 'static'"
        " to Static_Information' and 'impedance' to Impedance_Information. "
        "If you are unsure which to pick, consult the package documentation."
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
    'plot_current',
    char='c',
    default=True,
    param_type=xmlhelpy.Bool,
    description="Set to False if the current shall not be plotted."
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
def plot_measurement(
    record,
    filename,
    datatype,
    title,
    format,
    plot_current,
    start,
    stop,
    step,
    overwrite,
    display
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility import visualization
    from ep_bolfi.utility.dataset_formatting import (
        Cycling_Information,
        Static_Information,
        Impedance_Measurement
    )
    import matplotlib.pyplot as plt
    from kadi_apy.lib.core import KadiManager, Record

    manager = KadiManager()

    file_prefix = filename.split(".")[0]
    if title is None:
        title = file_prefix

    try:
        if datatype == "cycling":
            data = Cycling_Information.from_json(sys.stdin.read()).subslice(
                start, stop, step
            )
        elif datatype == "static":
            data = Static_Information.from_json(sys.stdin.read()).subslice(
                start, stop, step
            )
        elif datatype == "impedance":
            data = Impedance_Measurement.from_json(sys.stdin.read()).subslice(
                start, stop, step
            )
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    # ToDo: implement actual plot of the static part of the information.
    if datatype in ["cycling", "static"]:
        # "constrained_layout=True" is the better replacement for
        # "tight_layout" here. It ensures that the colorbar and second
        # y-axis don't overlap.
        fig, ax = plt.subplots(
            figsize=(4 * 2**0.5, 4), constrained_layout=True
        )
        texts = visualization.plot_measurement(
            fig, ax, data, title, plot_current=plot_current
        )
        for text in texts:
            text.set_visible(False)
        fig.savefig(
            file_prefix + '_plot.' + format,
            bbox_inches='tight',
            pad_inches=0.0
        )
        for text in texts:
            text.set_visible(True)
        if display:
            plt.show()
        record_handle = Record(manager, id=record, create=False)
        record_handle.upload_file(
            file_prefix + '_plot.' + format, force=overwrite
        )
    elif datatype == "impedance":
        ω = data.frequencies
        Z = data.complex_impedances
        legend_texts = [str(index) for index in data.indices]
        x_min = min([min(real_Z) for real_Z in data.real_impedances])
        x_max = max([max(real_Z) for real_Z in data.real_impedances])
        y_max = -min([min(imag_Z) for imag_Z in data.imaginary_impedances])
        y_min = -max([max(imag_Z) for imag_Z in data.imaginary_impedances])
        data_ratio = (y_max - y_min) / (x_max - x_min)
        # Make the plot roughly the right aspect ratio to later match
        # the 'equal' ratio Nyquist plot.
        fig, ax = plt.subplots(
            figsize=(
                (6 / data_ratio, 6) if data_ratio < 1 else (6, 6 * data_ratio)
            )
        )
        fig_bode, ax_real = plt.subplots(figsize=(6 * 2**0.5, 6))
        ax_imag = ax_real.twinx()
        fig_bode_2, ax_abs = plt.subplots(figsize=(6 * 2**0.5, 6))
        ax_phase = ax_abs.twinx()
        visualization.nyquist_plot(
            fig,
            ax,
            ω,
            Z,
            lw=2,
            title_text=title,
            legend_text=legend_texts,
            equal_aspect=True
        )
        visualization.bode_plot(
            fig_bode,
            ax_real,
            ax_imag,
            ω,
            Z,
            lw=1,
            ls_real='-',
            ls_imag='-.',
            title_text=title,
            legend_text=legend_texts
        )
        ax_real.set_ylabel("Real part (solid)  /  Ω")
        ax_imag.set_xlabel("")
        ax_imag.set_ylabel("-Imaginary part (dash-dotted)  /  Ω")
        phase_plot = [
            [abs(c) - 1j * p for c, p in zip(complex_impedance, phase)]
            for complex_impedance, phase in zip(
                data.complex_impedances, data.phases
            )
        ]
        visualization.bode_plot(
            fig_bode_2,
            ax_abs,
            ax_phase,
            ω,
            phase_plot,
            lw=1,
            ls_real='-',
            ls_imag='-.',
            title_text=title,
            legend_text=legend_texts
        )
        ax_abs.set_ylabel("Magnitude (solid)  /  Ω")
        ax_phase.set_xlabel("")
        ax_phase.set_ylabel("Phase (dash-dotted)  /  rad")
        fig.savefig(
            file_prefix + '_nyquist_plot.' + format,
            bbox_inches='tight',
            pad_inches=0.0
        )
        fig_bode.savefig(
            file_prefix + '_bode_impedance_plot.' + format,
            bbox_inches='tight',
            pad_inches=0.0
        )
        fig_bode_2.savefig(
            file_prefix + '_bode_phase_plot.' + format,
            bbox_inches='tight',
            pad_inches=0.0
        )
        if display:
            plt.show()
        record_handle = Record(manager, id=record, create=False)
        record_handle.upload_file(
            file_prefix + '_nyquist_plot.' + format, force=overwrite
        )
        record_handle.upload_file(
            file_prefix + '_bode_impedance_plot.' + format, force=overwrite
        )
        record_handle.upload_file(
            file_prefix + '_bode_phase_plot.' + format, force=overwrite
        )


if __name__ == '__main__':
    plot_measurement()
