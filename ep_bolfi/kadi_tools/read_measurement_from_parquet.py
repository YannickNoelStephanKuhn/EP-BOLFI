"""
Reads a ``Measurement`` object that got stored in an Apache Parquet
file with ``ep_bolfi.kadi_tools.store_measurement_in_parquet``.
"""

from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.read_measurement_from_parquet',
    version='3.0'
)
@xmlhelpy.option(
    'record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description="Persistent record identifier to download the file from."
)
@xmlhelpy.option(
    'filename',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="File name of the Parquet file to download."
)
@xmlhelpy.option(
    'datatype',
    char='t',
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
    'arg-indices',
    char='a',
    is_flag=True,
    description=(
        "Toggle to replace the indices attribute with (0, ..., len(data) - 1)."
    )
)
def read_measurement_from_parquet(
    record,
    filename,
    datatype,
    arg_indices,
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import read_parquet_table
    from kadi_apy.lib.core import KadiManager, Record

    manager = KadiManager()
    if not isfile(filename):
        record_handle = Record(manager, id=record, create=False)
        file_id = record_handle.get_file_id(filename)
        record_handle.download_file(file_id, filename)

    data = read_parquet_table(filename, datatype)
    if arg_indices:
        data.indices = [i for i in range(len(data))]
    print(data.to_json())


if __name__ == '__main__':
    read_measurement_from_parquet()
