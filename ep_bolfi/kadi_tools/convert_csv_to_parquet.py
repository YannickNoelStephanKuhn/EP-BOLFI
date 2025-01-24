"""
Stores CSV file contents in an Apache Parquet file. This optimally
compresses the data and makes it faster to read.
"""

from os.path import isfile
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.convert_csv_to_parquet',
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
    'encoding',
    char='f',
    default='iso-8859-1',
    param_type=xmlhelpy.String,
    description="Encoding of the file."
)
@xmlhelpy.option(
    'comment-lines',
    char='t',
    default=0,
    param_type=xmlhelpy.Integer,
    description=(
        "Number of comment lines to trim at the start of the file. This "
        "excludes an optional column header line, in contrast to "
        "'ep_bolfi.kadi_tools.read_csv_datasets'. The contents will be dumped "
        "in a file separate from the Parquet file."
    )
)
@xmlhelpy.option(
    'delimiter',
    char='l',
    default='\t',
    param_type=xmlhelpy.String,
    description="Column delimiter used in the file."
)
@xmlhelpy.option(
    'decimal',
    char='d',
    default='.',
    param_type=xmlhelpy.String,
    description="Decimal symbol used in the file."
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
def convert_csv_to_parquet(
    input_record,
    filename,
    output_record,
    compression_level,
    encoding,
    comment_lines,
    delimiter,
    decimal,
    overwrite,
):
    """Please refer to the --help output of this file."""
    from kadi_apy.lib.core import KadiManager, Record
    from pyarrow import csv, parquet

    manager = KadiManager()
    file_prefix = filename.split('.')[0]
    if not isfile(filename):
        input_record_handle = Record(manager, id=input_record, create=False)
        file_id = input_record_handle.get_file_id(filename)
        input_record_handle.download_file(file_id, filename)
    pyarrow_table = csv.read_csv(
        filename,
        read_options=csv.ReadOptions(
            skip_rows=comment_lines,
            encoding=encoding,
        ),
        parse_options=csv.ParseOptions(
            delimiter=delimiter,
        ),
        convert_options=csv.ConvertOptions(
            decimal_point=decimal,
        ),
    )
    parquet.write_table(
        pyarrow_table,
        file_prefix + '.parquet',
        compression='zstd',
        compression_level=compression_level,
    )
    comments = []
    with open(filename, 'r') as f:
        for _ in range(comment_lines):
            comments.append(f.readline())
    with open(file_prefix + '_comments.txt', 'w') as f:
        f.writelines(comments)

    output_record_handle = Record(manager, id=output_record, create=False)
    output_record_handle.upload_file(file_prefix + '.parquet', force=overwrite)
    output_record_handle.upload_file(
        file_prefix + '_comments.txt', force=overwrite
    )


if __name__ == '__main__':
    convert_csv_to_parquet()
