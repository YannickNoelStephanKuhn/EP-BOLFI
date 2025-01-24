"""
Stores a ``Measurement`` object in an Apache Parquet file. This
optimally compresses the data and makes it faster to read.
"""

import json
import sys
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.store_measurement_as_parquet',
    version='3.0'
)
@xmlhelpy.option(
    'record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=True,
    description="Persistent record identifier to upload the file to."
)
@xmlhelpy.option(
    'filename',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="File name of the Parquet file to upload."
)
@xmlhelpy.option(
    'datatype',
    char='t',
    param_type=xmlhelpy.Choice(["cycling", "static", "impedance"]),
    default="cycling",
    description=(
        "Type of the measurement, which determines the internal format for "
        "further processing. 'cycling' refers to Cycling_Information, 'static'"
        " to Static_Information' and 'impedance' to Impedance_Information. "
        "If you are unsure which to pick, consult the package documentation."
    )
)
@xmlhelpy.option(
    'compression-level',
    char='c',
    param_type=xmlhelpy.IntRange(min=-7, max=22),
    default=22,
    description=(
        "Compression level setting for the compression algorithm Zstandard. "
        "-7 is fastest and largest, while 22 ist slowest and smallest. "
        "Note that decompression has the same (fast) speed at any level."
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
def store_measurement_as_parquet(
    record,
    filename,
    datatype,
    compression_level,
    overwrite,
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import (
        Cycling_Information,
        Static_Information,
        Impedance_Measurement,
        store_parquet_table
    )
    from kadi_apy.lib.core import KadiManager, Record

    file_prefix = filename.split('.')[0]

    manager = KadiManager()

    try:
        if datatype == "cycling":
            data = Cycling_Information.from_json(sys.stdin.read())
        elif datatype == "static":
            data = Static_Information.from_json(sys.stdin.read())
        elif datatype == "impedance":
            data = Impedance_Measurement.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped "
            "to this tool."
        )

    store_parquet_table(data, file_prefix, compression_level)

    record = Record(manager, id=record, create=False)
    record.upload_file(file_prefix + '.parquet', force=overwrite)


if __name__ == '__main__':
    store_measurement_as_parquet()
