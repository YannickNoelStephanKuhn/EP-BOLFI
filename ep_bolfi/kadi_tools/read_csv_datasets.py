"""
Preprocesses commonly encountered CSV measurement files to the datatypes
defined in ``ep_bolfi.utility.dataset_formatting``.
"""

from ast import literal_eval
from os.path import isfile
import re
import xmlhelpy


@xmlhelpy.command(
    name='python -m ep_bolfi.kadi_tools.read_csv_datasets',
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
    'filenames',
    char='n',
    param_type=xmlhelpy.TokenList(separator='?'),
    required=True,
    description=(
        "File name. If you want to give more than one, separate them by '?'. "
        "In that case, the file contents get concatenated."
    )
)
@xmlhelpy.option(
    'headers',
    char='h',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "Dictionary (enclose in quotation marks). Its keys are the indices of "
        "the columns which are to be read in . The corresponding values are "
        "there to tell this function which kind of data is in which column. "
        "The following format has to be used: '<name> [<unit>]'. "
        "For cycling data, 'name' is 'U' (voltage), 'I' (current), or "
        "'t' (time), and 'unit' is 'V', 'A', 'h', 'm', or 's' with the "
        "optional prefixes 'M', 'k', 'm', 'µ', or 'n'. This converts the "
        "data to prefix-less SI units. "
        "For impedance data, 'name' is 'f' (frequency), 'real_Z' (real part "
        "of impedance), 'imag_Z' (imaginary part of impedance), or 'ϕ' "
        "(phase), and 'unit' is 'Hz', 'Ω', 'rad', or 'deg'. The same prefixes "
        "as before can be used to convert the data to prefix-less SI units. "
        "Additional columns may be read in, if their 'name' is different."
    )
)
@xmlhelpy.option(
    'automatic-numbering-digits',
    char='a',
    param_type=xmlhelpy.Integer,
    default=-1,
    description=(
        "Optional max number of digits for the automatic reindexing of data "
        "from multiple files. Default is all digits found in the filenames. "
        "Setting this to 0 deactivates the number extraction from filenames."
    )
)
@xmlhelpy.option(
    'encoding',
    char='e',
    default='iso-8859-1',
    param_type=xmlhelpy.String,
    description="Encoding of the file."
)
@xmlhelpy.option(
    'comment-lines',
    char='t',
    default=0,
    param_type=xmlhelpy.Integer,
    description="Number of comment lines to trim at the start of the file."
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
    'segment-column',
    char='s',
    default=-1,
    param_type=xmlhelpy.Integer,
    description=(
        "The index of the column that stores the index of the current segment."
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
    default='{}',
    param_type=xmlhelpy.String,
    description=(
        "Dictionary (give as Python-formatted string). Its keys are the "
        "strings used in the file to indicate a state. The column from "
        "which this state is retrieved is given by #correction-column. "
        "The dictionaries' values are used to correct/normalize the "
        "current value in the file."
    )
)
@xmlhelpy.option(
    'correction-column',
    char='o',
    default=-1,
    param_type=xmlhelpy.Integer,
    description="See #current-sign-correction."
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
@xmlhelpy.option(
    'max-number-of-lines',
    char='b',
    default=-1,
    param_type=xmlhelpy.Integer,
    description="The maximum number of dataset lines that are to be read in."
)
def read_csv_datasets(
    record,
    filenames,
    automatic_numbering_digits,
    headers,
    encoding,
    comment_lines,
    delimiter,
    decimal,
    datatype,
    segment_column,
    segments_to_process,
    current_sign_correction,
    correction_column,
    flip_voltage_sign,
    flip_imaginary_impedance_sign,
    max_number_of_lines
):
    """Please refer to the --help output of this file."""
    from ep_bolfi.utility.dataset_formatting import (
        read_csv_from_measurement_system
    )
    from kadi_apy.lib.core import KadiManager, Record

    manager = KadiManager()
    if not isfile(filenames[0]):
        record_handle = Record(manager, id=record, create=False)
        file_id = record_handle.get_file_id(filenames[0])
        record_handle.download_file(file_id, filenames[0])
    cycl_info = read_csv_from_measurement_system(
        filenames[0],
        encoding,
        comment_lines,
        literal_eval(headers),
        delimiter=delimiter,
        decimal=decimal,
        datatype=datatype,
        segment_column=segment_column,
        segments_to_process=segments_to_process,
        current_sign_correction=literal_eval(current_sign_correction),
        correction_column=correction_column,
        flip_voltage_sign=flip_voltage_sign,
        flip_imaginary_impedance_sign=flip_imaginary_impedance_sign,
        max_number_of_lines=max_number_of_lines
    )
    if len(filenames) > 1:
        # Try to closely mimic the file numbering.
        filenumbers = []
        backup_numbering = 0
        for filename in filenames:
            # Skip empty filenames.
            if filename == '':
                continue
            extracted_number = ''.join(re.findall("[0-9]+", filename))
            if automatic_numbering_digits > -1:
                extracted_number = (
                    extracted_number[-automatic_numbering_digits:]
                )
                filenumbers.append(int(extracted_number))
            elif extracted_number == '':
                filenumbers.append(backup_numbering)
                backup_numbering = backup_numbering + 1
            else:
                filenumbers.append(int(extracted_number))
                backup_numbering = filenumbers[-1] + 1
        if automatic_numbering_digits != 0:
            cycl_info.indices = [filenumbers[0]]
    for j, filename in enumerate(filenames[1:]):
        # Skip empty filenames.
        if filename == '':
            continue
        if not isfile(filename):
            record_handle = Record(manager, id=record, create=False)
            file_id = record_handle.get_file_id(filename)
            record_handle.download_file(file_id, filename)
        new_cycl_info = read_csv_from_measurement_system(
            filename,
            encoding,
            comment_lines,
            literal_eval(headers),
            delimiter=delimiter,
            decimal=decimal,
            datatype=datatype,
            segment_column=segment_column,
            segments_to_process=segments_to_process,
            current_sign_correction=literal_eval(current_sign_correction),
            correction_column=correction_column,
            flip_voltage_sign=flip_voltage_sign,
            flip_imaginary_impedance_sign=flip_imaginary_impedance_sign,
            max_number_of_lines=max_number_of_lines
        )
        cycl_info.extend(new_cycl_info)
        if automatic_numbering_digits != 0:
            cycl_info.indices.append(filenumbers[1 + j])
    print(cycl_info.to_json())


if __name__ == '__main__':
    read_csv_datasets()
