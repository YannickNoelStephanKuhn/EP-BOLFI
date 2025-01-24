"""
Provides preprocessing from HDF5 contents to the datatypes defined in
``ep_bolfi.utility.dataset_formatting``.
"""

from ast import literal_eval
from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.read_hdf5_datasets',
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
    description="File name."
)
@xmlhelpy.option(
    'data-location',
    char='d',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "A list (give as string). Gives the location in the HDF5 file where "
        "the data table is stored. Each entry goes one level deeper into the "
        "HDF structure. "
        "Each entry can either be the index to go into next itself, or a "
        "2-tuple or a 2-list. In the latter case, (None, x) denotes "
        "slicing[:, x] and (x, None) denotes slicing[x, :]."
    )
)
@xmlhelpy.option(
    'headers',
    char='h',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "Dictionary (give as string). Its keys are the indices of the columns "
        "which are to be read in. These can also be 2-tuples for slicing; "
        "'(x, None)' or '(None, x)' get translated to slicing with [x, :] or "
        "[:, x]. The default behaviour for integer indices x is the same as "
        "'(None, x)', slicing the column x. The corresponding values are "
        "there to tell this function which kind of data is in which column. "
        "The following format has to be used: '<name> [<unit>]' where 'name' "
        "is 'U' (voltage), 'I' (current) or 't' (time) and 'unit' is "
        "'V', 'A', 'h', 'm' or 's' with the optional prefixes 'k', 'm', "
        "'µ' or 'n'. This converts the data to prefix-less SI units. "
        "For impedance data, 'name' is 'f' (frequency), 'real_Z' (real part "
        "of impedance), 'imag_Z' (imaginary part of impedance), or 'ϕ' "
        "(phase), and 'unit' is 'Hz', 'Ω', 'rad', or 'deg'. The same prefixes "
        "as before can be used to convert the data to prefix-less SI units. "
        "Additional columns may be read in with keys not in this format."
    )
)
@xmlhelpy.option(
    'datatype',
    char='y',
    default="cycling",
    param_type=xmlhelpy.String,
    description=(
        "Type of the measurement, which determines the internal format for "
        "further processing. 'cycling' refers to Cycling_Information, 'static'"
        " to Static_Information' and 'impedance' to Impedance_Information. "
        "If you are unsure which to pick, consult the package documentation."
    )
)
@xmlhelpy.option(
    'segment-location',
    char='s',
    param_type=xmlhelpy.String,
    default=None,
    description=(
        "A list (give as string), with the same format as 'data_location'. "
        "It points to the part of the data that stores the index of the "
        "current segment. If it changes from one data point to the next, that "
        "is used as the dividing line between two segments."
    )
)
@xmlhelpy.option(
    'segments-to-process',
    char='m',
    param_type=xmlhelpy.Integer,
    default=None,
    description=(
        "A list of indices which give the segments that shall be "
        "processed. Default is None, i.e., the whole file gets processed."
    )
)
@xmlhelpy.option(
    'current-sign-correction',
    char='c',
    param_type=xmlhelpy.String,
    default='{}',
    description=(
        "Dictionary (give as string). Its keys are the "
        "strings used in the file to indicate a state. The column from "
        "which this state is retrieved is given by #correction-location. "
        "The dictionaries' values are used to correct/normalize the "
        "current value in the file."
    )
)
@xmlhelpy.option(
    'correction-location',
    char='o',
    param_type=xmlhelpy.String,
    default=None,
    description=(
        "A list, with the same format as 'data_location'. "
        "For its use, see 'current_sign_correction'. Default: None."
    )
)
@xmlhelpy.option(
    'flip-voltage-sign',
    char='v',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Defaults to False, where measured voltage remains unaltered. "
        "Change to True if the voltage shall be multiplied by -1. "
        "Also applies to impedances; real and imaginary parts."
    )
)
@xmlhelpy.option(
    'flip-imaginary-impedance-sign',
    char='f',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Defaults to False, where measured impedance remains unaltered. "
        "Change to True if the imaginary part of the impedance shall be "
        "multiplied by -1. Cancels out with 'flip-voltage-sign'."
    )
)
def read_hdf5_datasets(
    record,
    filename,
    data_location,
    headers,
    datatype,
    segment_location,
    segments_to_process,
    current_sign_correction,
    correction_location,
    flip_voltage_sign,
    flip_imaginary_impedance_sign
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import read_hdf5_table
    from kadi_apy.lib.core import KadiManager, Record

    headers = {
        k: v if isinstance(v, tuple) else (None, v)
        for k, v in literal_eval(headers)
    }

    manager = KadiManager()
    if not isfile(filename):
        record_handle = Record(manager, id=record, create=False)
        file_id = record_handle.get_file_id(filename)
        record_handle.download_file(file_id, filename)
    print(read_hdf5_table(
        filename,
        data_location,
        headers,
        datatype=datatype,
        segment_location=segment_location,
        segments_to_process=segments_to_process,
        current_sign_correction=current_sign_correction,
        correction_location=correction_location,
        flip_voltage_sign=flip_voltage_sign,
        flip_imaginary_impedance_sign=flip_imaginary_impedance_sign
    ).to_json())


if __name__ == "__main__":
    read_hdf5_datasets()
