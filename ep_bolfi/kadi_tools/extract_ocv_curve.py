"""
Reads in a ``Cycling_Information`` json representation and extracts OCV
information from it, assuming it represents a GITT measurement.
"""

from ast import literal_eval
import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.extract_ocv_curve',
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
    'segments',
    char='s',
    default="[(1, None, 2)]",
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
    'exclude',
    char='c',
    default=0.0,
    param_type=xmlhelpy.Float,
    description=(
        "Seconds at the start of each segment to exclude from the exponential "
        "fitting procedure to approximate the OCV."
    )
)
@xmlhelpy.option(
    'split-on-current-direction',
    char='l',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Set to True if two OCV curves shall be produced, one for each "
        "direction the current had before each datapoint."
    )
)
@xmlhelpy.option(
    'positive-current-is-lithiation',
    char='u',
    default=True,
    param_type=xmlhelpy.Bool,
    description=(
        "Set to True if positive current in the measurement corresponds to "
        "lithiation of the material, and False otherwise."
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
def extract_ocv_curve(
    record,
    filename,
    title,
    format,
    segments,
    exclude,
    split_on_current_direction,
    positive_current_is_lithiation,
    overwrite,
    display
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from ep_bolfi.utility.fitting_functions import fit_exponential_decay
    from ep_bolfi.utility.preprocessing import calculate_SOC, find_occurrences
    from ep_bolfi.utility.visualization import plot_measurement
    from kadi_apy.lib.core import KadiManager, Record
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from numpy import mean

    manager = KadiManager()

    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )
    segments = literal_eval(segments)
    rest_indices = []
    for segment in segments:
        if len(segment) == 2:
            rest_indices.extend(range(
                len(data.indices) if segment[0] is None else segment[0],
                len(data.indices) if segment[1] is None else segment[1]
            ))
        elif len(segment) == 3:
            rest_indices.extend(range(
                len(data.indices) if segment[0] is None else segment[0],
                len(data.indices) if segment[1] is None else segment[1],
                segment[2]
            ))
        else:
            raise ValueError(
                "Segments have to be given as either 2- or 3-tuples."
            )

    source_indices = [
        data.indices[rest_index] for rest_index in rest_indices
    ]
    soc_in_coulomb = calculate_SOC(data.timepoints, data.currents)
    socs = [soc_in_coulomb[i][-1] for i in rest_indices]

    exclude_indices = {
        i: find_occurrences(
            data.timepoints[i], data.timepoints[i][0] + exclude
        )[0]
        for i in rest_indices
    }
    parallel_arguments = [
        (
            data.timepoints[i][exclude_indices[i]:],
            data.voltages[i][exclude_indices[i]:],
        )
        for i in rest_indices
    ]
    with Pool() as p:
        exp_decays = p.starmap(fit_exponential_decay, parallel_arguments)
    ocvs = []
    marker_locations = [
        (data.timepoints[i][-1] - data.timepoints[0][0]) / 3600
        for i in rest_indices
    ]
    pop_counter = 0
    for i, soc_result in enumerate(exp_decays):
        if len(soc_result) == 0:
            rest_indices.pop(i - pop_counter)
            source_indices.pop(i - pop_counter)
            socs.pop(i - pop_counter)
            marker_locations.pop(i - pop_counter)
            pop_counter = pop_counter + 1
        else:
            ocvs.append(soc_result[0][2][0])

    file_prefix = filename.split('.')[0]

    if split_on_current_direction:
        pulse_indices = [(ri - 1 if ri > 0 else 0) for ri in rest_indices]
        lithiation_flip = 1 if positive_current_is_lithiation else -1
        socs_lithiation = []
        socs_delithiation = []
        ocvs_lithiation = []
        ocvs_delithiation = []
        source_indices_lithiation = []
        source_indices_delithiation = []
        marker_locations_lithiation = []
        marker_locations_delithiation = []
        for pi, s, o, si, ml in zip(
            pulse_indices, socs, ocvs, source_indices, marker_locations
        ):
            if mean(data.currents[pi]) * lithiation_flip >= 0.0:
                socs_lithiation.append(s)
                ocvs_lithiation.append(o)
                source_indices_lithiation.append(si)
                marker_locations_lithiation.append(ml)
            else:
                socs_delithiation.append(s)
                ocvs_delithiation.append(o)
                source_indices_delithiation.append(si)
                marker_locations_delithiation.append(ml)
        with open(file_prefix + '_lithiation_extraction.json', 'w') as f:
            json.dump({
                "SOC [C]": socs_lithiation,
                "OCV [V]": ocvs_lithiation,
                "indices": source_indices_lithiation,
            }, f)
        with open(file_prefix + '_delithiation_extraction.json', 'w') as f:
            json.dump({
                "SOC [C]": socs_delithiation,
                "OCV [V]": ocvs_delithiation,
                "indices": source_indices_delithiation,
            }, f)
    else:
        with open(file_prefix + '_extraction.json', 'w') as f:
            json.dump(
                {"SOC [C]": socs, "OCV [V]": ocvs, "indices": source_indices},
                f
            )

    # "constrained_layout=True" is the better replacement for "tight_layout"
    # here. It ensures that the colorbar and second y-axis don't overlap.
    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
    texts = plot_measurement(fig, ax, data, title)
    if split_on_current_direction:
        marker_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.plot(
            marker_locations_lithiation,
            ocvs_lithiation,
            marker='1',
            lw=0,
            ms=10,
            color=marker_colours[0],
            label="lithiation"
        )
        ax.plot(
            marker_locations_delithiation,
            ocvs_delithiation,
            marker='1',
            lw=0,
            ms=10,
            color=marker_colours[1],
            label="delithiation"
        )
        ax.legend()
    else:
        ax.plot(
            marker_locations,
            ocvs,
            marker='1',
            lw=0,
            ms=10,
            color='gray'
        )
    for text in texts:
        text.set_visible(False)
    fig.savefig(
        file_prefix + '_extraction.' + format,
        bbox_inches='tight',
        pad_inches=0.0
    )
    for text in texts:
        text.set_visible(True)
    if display:
        plt.show()

    record_handle = Record(manager, id=record, create=False)
    if split_on_current_direction:
        record_handle.upload_file(
            file_prefix + '_lithiation_extraction.json', force=overwrite
        )
        record_handle.upload_file(
            file_prefix + '_delithiation_extraction.json', force=overwrite
        )
    else:
        record_handle.upload_file(
            file_prefix + '_extraction.json', force=overwrite
        )
    record_handle.upload_file(
        file_prefix + '_extraction.' + format, force=overwrite
    )


if __name__ == '__main__':
    extract_ocv_curve()
