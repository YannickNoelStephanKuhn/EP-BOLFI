"""
Defines datatypes for further processing.
"""

import csv
import h5py
import json
import pyarrow
import re

from ep_bolfi.utility.fitting_functions import fit_exponential_decay
from math import pi
from os import linesep
from pyarrow import parquet


class Measurement:
    """Defines common methods for measurement objects."""

    def __len__(self):
        """
        Gives the number of data segments.

        :returns:
            # of data segments.
        """
        raise NotImplementedError

    def to_json(self):
        """
        Losslessly transforms this object into JSON.

        :returns:
            The JSON representation, given as a string.
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_string):
        """
        Reads this object from JSON that was prepared with ``to_json``.

        :param json_string:
            The JSON, fed in as a string.
        """
        raise NotImplementedError

    def table_descriptors(self):
        """
        Gives the headings for table formatting.

        :returns:
            Table headings in Apache Parquet syntax.
        """
        raise NotImplementedError

    @classmethod
    def table_mapping(cls):
        """
        Gives the mapping from attributes to headings.

        :returns:
            A dictionary: keys are attribute names, values are headings.
        """
        raise NotImplementedError

    def segment_tables(self, start=0, stop=None, step=1):
        """
        Iterator that gives data segments in Apache Parquet syntax.

        :param start:
            First data segment.
        :param stop:
            Last data segment.
        :param step:
            Step size in each iteration.
        """
        raise NotImplementedError

    def example_table_row(self):
        """
        Makes singular *self.indices* entries into lists and returns an
        exempular line for table formatting.

        :returns:
            A list with the entries for each column.
        """
        indices_into_columns = False
        try:
            self.indices[0][0]
        except (TypeError, IndexError):
            indices_into_columns = True
        if type(self.indices[0]) is str:
            indices_into_columns = True
        if indices_into_columns:
            print("Transforming indices into columns for tabular view.")
            lengths_of_segments = []
            for i in range(len(self)):
                try:
                    length = len(self.timepoints[i])
                    if length > 0:
                        lengths_of_segments.append(length)
                        continue
                except AttributeError:
                    try:
                        length = len(self.frequencies[i])
                        if length > 0:
                            lengths_of_segments.append(length)
                            continue
                    except IndexError:
                        pass
                except IndexError:
                    pass
                key = list(self.other_columns.keys())[0]
                lengths_of_segments.append((len(self.other_columns[key][i])))
            self.indices = [
                [index] * length
                for index, length in zip(self.indices, lengths_of_segments)
            ]

    def subslice(self, start, stop, step=1):
        """
        Returns a ``Measurement`` object containing the requested slice.

        :param start:
            First data segment.
        :param stop:
            Last data segment.
        :param step:
            Step size in each iteration.
        :returns:
            A ``Measurement[start:stop:step]`` object.
        """
        raise NotImplementedError

    def subarray(self, array):
        """
        Returns a ``Measurement`` object containing *array* segments.

        :param array:
            A list of integers denoting the segments to collect.
            Does not correspond to *self.indices*.
        :returns:
            A ``Measurement[array]`` object.
        """
        raise NotImplementedError

    def extend(self, other):
        """
        Extends *self* with the *other* ``Measurement`` object.

        :param other:
            Another ``Measurement object``.
        """
        raise NotImplementedError


class Cycling_Information(Measurement):
    """
    Contains basic cycling informations. Each member variable is a list
    and has the same length as the other ones.
    """

    def __init__(
        self, timepoints, currents, voltages, other_columns={}, indices=None
    ):
        self.timepoints = timepoints
        """
        The times at which measurements were taken. Usually a list of
        lists where each list corresponds to a segment of the
        measurement.
        """
        self.currents = currents
        """
        The measured current at those times. A list which usually
        contains lists for segments with variable current and floats
        for segments with constant current.
        """
        self.voltages = voltages
        """
        The measured voltage at those times. A list which usually
        contains lists for segments with variable voltage and floats
        for segments with constant voltage.
        """
        self.other_columns = other_columns
        """
        The contents of any other columns. A dictionary ("columns")
        which values are lists which contain lists for segments.
        The keys should match user input for the columns.
        """
        if indices is None or indices is []:
            self.indices = [i for i in range(len(self))]
            """
            The indices of the individual segments. Defaults to a simple
            numbering of the segments present. May be used for plotting
            purposes, e.g., for colourcoding the segments by cycle.
            """
        else:
            self.indices = indices
        # This syntax doesn't work when arrays are involved, apparently.
        # self.indices = indices or [i for i in range(len(self))]

    def __len__(self):
        return max([
            len(self.timepoints),
            len(self.currents),
            len(self.voltages),
            *[len(column) for column in self.other_columns.values()]
        ])

    def to_json(self):
        return json.dumps({
            "timepoints [s]": self.timepoints,
            "currents [A]": self.currents,
            "voltages [V]": self.voltages,
            "other_columns": self.other_columns,
            "indices": self.indices,
        })

    @classmethod
    def from_json(cls, json_string):
        return cls(*(json.loads(json_string).values()))

    def table_descriptors(self):
        descriptors = ["indices"]
        if self.timepoints:
            descriptors.append("timepoints [s]")
        if self.currents:
            descriptors.append("currents [A]")
        if self.voltages:
            descriptors.append("voltages [V]")
        descriptors.extend([name for name in self.other_columns.keys()])
        return descriptors

    @classmethod
    def table_mapping(cls):
        return {
            'indices': "indices",
            'timepoints': "timepoints [s]",
            'currents': "currents [A]",
            'voltages': "voltages [V]",
        }

    def segment_tables(self, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self)
        for i in range(start, stop, step):
            segment = [self.indices[i]]
            if self.timepoints:
                segment.append(self.timepoints[i])
            if self.currents:
                segment.append(self.currents[i])
            if self.voltages:
                segment.append(self.voltages[i])
            segment.extend([
                column[i] for column in self.other_columns.values()
            ])
            yield segment

    def example_table_row(self):
        super().example_table_row()
        example = []
        example.append([self.indices[0][0]])
        if self.timepoints:
            example.append([self.timepoints[0][0]])
        if self.currents:
            example.append([self.currents[0][0]])
        if self.voltages:
            example.append([self.voltages[0][0]])
        example.extend([
            [column[0][0]]
            for column in self.other_columns.values()
        ])
        return example

    def subslice(self, start, stop, step=1):
        return Cycling_Information(
            self.timepoints[start:stop:step],
            self.currents[start:stop:step],
            self.voltages[start:stop:step],
            {name: column[start:stop:step]
             for name, column in self.other_columns.items()},
            self.indices[start:stop:step]
        )

    def subarray(self, array):
        return Cycling_Information(
            [self.timepoints[index] for index in array],
            [self.currents[index] for index in array],
            [self.voltages[index] for index in array],
            {name: [column[index] for index in array]
             for name, column in self.other_columns.items()},
            [self.indices[index] for index in array]
        )

    def extend(self, other):
        other_column_names = set(self.other_columns.keys()).intersection(
            set(other.other_columns.keys())
        )
        self.timepoints.extend(other.timepoints)
        self.currents.extend(other.currents)
        self.voltages.extend(other.voltages)
        for name in other_column_names:
            self.other_columns[name].extend(other.other_columns[name])
        self.indices.extend(other.indices)


class Static_Information(Cycling_Information):
    """
    Contains additional informations, e.g. for GITT. Each member
    variable is a list and has the same length as the other ones.
    """

    def __init__(
        self,
        timepoints,
        currents,
        voltages,
        other_columns={},
        indices=None
    ):
        super().__init__(timepoints, currents, voltages, other_columns,
                         indices)
        asymptotic_voltages = [-1.0] * len(timepoints)
        ir_steps = [0.0] * len(timepoints)
        last_U = 0
        exp_U_decays = []
        exp_I_decays = []

        for i, (t, I, U) in enumerate(zip(
                timepoints, currents, voltages)):
            for curve in fit_exponential_decay(t, I):
                I_0, ΔI, I_decay = curve[2]
                exp_I_decays.append((I_0, ΔI, I_decay))
            for j, curve in enumerate(fit_exponential_decay(t, U)):
                U_0, ΔU, U_decay = curve[2]
                exp_U_decays.append((U_0, ΔU, U_decay))
                if j == 0:
                    asymptotic_voltages[i] = U_0
            if asymptotic_voltages[i] == -1.0:
                asymptotic_voltages[i] = U[-1]
            ir_steps[i] = U[0] - last_U
            last_U = U[-1]
        ir_steps[0] = 0.0

        self.asymptotic_voltages = asymptotic_voltages
        """
        The voltages that the voltage curve seems to converge to in a
        segment. Only makes sense for those segments that are rest
        periods or when the OCV was subtracted.
        """
        self.ir_steps = ir_steps
        """
        The instantaneous IR drops before each segment. Positive
        values are voltage rises and negative values voltage drops.
        """
        self.exp_I_decays = exp_I_decays
        """Same as exp_U_decays for current decays (PITT)."""
        self.exp_U_decays = exp_U_decays
        """
        The fit parameters of the exponential voltage decays in each
        segment. Each set of fit parameters is a 3-tuple (a,b,c) where
        the fit function has the following form:
            a + b * exp(-c * (t - t_end_of_segment)).
        Failed or missing fits are best indicated by (NaN, NaN, NaN).
        """

    def to_json(self):
        return json.dumps({
            "timepoints [s]": self.timepoints,
            "currents [A]": self.currents,
            "voltages [V]": self.voltages,
            "other_columns": self.other_columns,
            "indices": self.indices,
            "asymptotic voltages [V]": self.asymptotic_voltages,
            "IR steps [V]": self.ir_steps,
            "exponential current decays (fit parameters)": self.exp_I_decays,
            "exponential voltage decays (fit parameters)": self.exp_U_decays
        })

    def subslice(self, start, stop, step=1):
        return Static_Information(
            self.timepoints[start:stop:step],
            self.currents[start:stop:step],
            self.voltages[start:stop:step],
            self.asymptotic_voltages[start:stop:step],
            self.ir_steps[start:stop:step],
            self.exp_I_decays[start:stop:step],
            self.exp_U_decays[start:stop:step],
            {name: column[start:stop:step]
             for name, column in self.other_columns.items()},
            self.indices[start:stop:step]
        )

    def subarray(self, array):
        return Static_Information(
            [self.timepoints[index] for index in array],
            [self.currents[index] for index in array],
            [self.voltages[index] for index in array],
            [self.asymptotic_voltages[index] for index in array],
            [self.ir_steps[index] for index in array],
            [self.exp_I_decays[index] for index in array],
            [self.exp_U_decays[index] for index in array],
            {name: [column[index] for index in array]
             for name, column in self.other_columns.items()},
            [self.indices[index] for index in array]
        )

    def extend(self, other):
        other_column_names = set(self.other_columns.keys()).intersection(
            set(other.other_columns.keys())
        )
        self.timepoints.extend(other.timepoints)
        self.currents.extend(other.currents)
        self.voltages.extend(other.voltages)
        self.asymptotic_voltages.extend(other.asymptotic_voltages)
        self.ir_steps.extend(other.ir_steps)
        self.exp_I_decays.extend(other.exp_I_decays)
        self.exp_U_decays.extend(other.exp_U_decays)
        for name in other_column_names:
            self.other_columns[name].extend(other.other_columns[name])
        self.indices.extend(other.indices)


class Impedance_Measurement(Measurement):
    """
    Contains basic impedance data. Each member variable is a list and
    has the same length as the other ones.
    """

    def __init__(
        self,
        frequencies,
        real_impedances,
        imaginary_impedances,
        phases,
        other_columns={},
        indices=None
    ):
        self.frequencies = frequencies
        """
        The frequencies at which impedances were measured. Usually a
        list of lists where each list corresponds to a different
        equilibrium.
        """
        self.real_impedances = real_impedances
        """
        The real part of the impedances measured at those frequencies.
        """
        self.imaginary_impedances = imaginary_impedances
        """
        The imaginary part of the impedances measured at those
        frequencies.
        """
        self.phases = phases
        """The phases of the impedance measured at those frequencies."""
        self.other_columns = other_columns
        """
        The contents of any other columns. A dictionary ("columns")
        which values are lists which contain lists for segments.
        The keys should match user input for the columns.
        """
        if indices is None:
            self.indices = [i for i in range(len(self))]
            """
            The indices of the individual segments. Defaults to a simple
            numbering of the segments present. May be used for plotting
            purposes, e.g., for colourcoding the segments by cycle.
            """
        else:
            self.indices = indices

    def __len__(self):
        return max([
            len(self.frequencies),
            len(self.real_impedances),
            len(self.imaginary_impedances),
            len(self.phases),
            *[len(column) for column in self.other_columns.values()]
        ])

    @property
    def complex_impedances(self):
        return [
            [r + 1j * i for r, i in zip(real_segment, imag_segment)]
            for real_segment, imag_segment
            in zip(self.real_impedances, self.imaginary_impedances)
        ]

    def to_json(self):
        return json.dumps({
            "frequencies [Hz]": self.frequencies,
            "real_impedances [Ω]": self.real_impedances,
            "imaginary_impedances [Ω]": self.imaginary_impedances,
            "phases [rad]": self.phases,
            "other_columns": self.other_columns,
            "indices": self.indices,
        })

    @classmethod
    def from_json(cls, json_string):
        return cls(*(json.loads(json_string).values()))

    def table_descriptors(self):
        descriptors = ["indices"]
        if self.frequencies:
            descriptors.append("frequencies [Hz]")
        if self.real_impedances:
            descriptors.append("real_impedances [Ω]")
        if self.imaginary_impedances:
            descriptors.append("imaginary_impedances [Ω]")
        if self.phases:
            descriptors.append("phases [rad]")
        descriptors.extend([name for name in self.other_columns.keys()])
        return descriptors

    @classmethod
    def table_mapping(cls):
        return {
            'indices': "indices",
            'frequencies': "frequencies [Hz]",
            'real_impedances': "real_impedances [Ω]",
            'imaginary_impedances': "imaginary_impedances [Ω]",
            'phases': "phases [rad]",
        }

    def segment_tables(self, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self)
        for i in range(start, stop, step):
            segment = [self.indices[i]]
            if self.frequencies:
                segment.append(self.frequencies[i])
            if self.real_impedances:
                segment.append(self.real_impedances[i])
            if self.imaginary_impedances:
                segment.append(self.imaginary_impedances[i])
            if self.phases:
                segment.append(self.phases[i])
            segment.extend([
                column[i] for column in self.other_columns.values()
            ])
            yield segment

    def example_table_row(self):
        super().example_table_row()
        example = []
        example.append([self.indices[0][0]])
        if self.frequencies:
            example.append([self.frequencies[0][0]])
        if self.real_impedances:
            example.append([self.real_impedances[0][0]])
        if self.imaginary_impedances:
            example.append([self.imaginary_impedances[0][0]])
        if self.phases:
            example.append([self.phases[0][0]])
        example.extend([
            [column[0][0]]
            for column in self.other_columns.values()
        ])
        return example

    def subslice(self, start, stop, step=1):
        return Impedance_Measurement(
            self.frequencies[start:stop:step],
            self.real_impedances[start:stop:step],
            self.imaginary_impedances[start:stop:step],
            self.phases[start:stop:step],
            {name: column[start:stop:step]
             for name, column in self.other_columns.items()},
            self.indices[start:stop:step]
        )

    def subarray(self, array):
        return Impedance_Measurement(
            [self.frequencies[index] for index in array],
            [self.real_impedances[index] for index in array],
            [self.imaginary_impedances[index] for index in array],
            [self.phases[index] for index in array],
            {name: [column[index] for index in array]
             for name, column in self.other_columns.items()},
            [self.indices[index] for index in array]
        )

    def extend(self, other):
        other_column_names = set(self.other_columns.keys()).intersection(
            set(other.other_columns.keys())
        )
        self.frequencies.extend(other.frequencies)
        self.real_impedances.extend(other.real_impedances)
        self.imaginary_impedances.extend(other.imaginary_impedances)
        self.phases.extend(other.phases)
        for name in other_column_names:
            self.other_columns[name].extend(other.other_columns[name])
        self.indices.extend(other.indices)


def read_csv_from_measurement_system(
    path,
    encoding,
    number_of_comment_lines,
    headers,
    delimiter='\t',
    decimal='.',
    datatype="cycling",
    segment_column=-1,
    segments_to_process=None,
    current_sign_correction={},
    correction_column=-1,
    flip_voltage_sign=False,
    flip_imaginary_impedance_sign=False,
    max_number_of_lines=-1
):
    """
    Read the measurements as returned by common instruments.

    Example: cycling measurements from Basytec devices. Their format
    resembles a csv file with one title and one header comment line.
    So the first line will be ignored and the second used for headers.

    :param path:
        The full or relative path to the measurement file.
    :param encoding:
        The encoding of that file, e.g. "iso-8859-1".
    :param number_of_comment_lines:
        The number of lines that have to be skipped over in order to
        arrive at the first dataset line.
    :param headers:
        A dictionary. Its keys are the indices of the columns
        which are to be read in. The corresponding values are there to
        tell this function which kind of data is in which column. The
        following format has to be used: "<name> [<unit>]" where "name"
        is "U" (voltage), "I" (current), or "t" (time) and "unit" is
        "V", "A", "h", "m" ,or "s" with the optional prefixes "k", "m",
        "µ", or "n". This converts the data to prefix-less SI units.
        Additional columns may be read in with keys not in this format.
        The columns for segments and sign correction are only given by
        *segment_column* and *correction_column*.
    :param delimiter:
        The delimiter string between datapoints. The default is "\t".
    :param decimal:
        The string used for the decimal point. Default: ".".
    :param datatype:
        Default is "cycling", where cycling information is assumed in
        the file. "static" will trigger the additional extraction of
        exponential decays that are relevant to e.g. GITT.
        "impedance" will treat the file as an impedance measurement
        with frequencies and impedances instead of time and voltage.
    :param segment_column:
        The index of the column that stores the index
        of the current segment. If it changes from one data point to the
        next, that is used as the dividing line between two segments.
        Default is -1, which returns the dataset in one segment.
    :param segments_to_process:
        A list of indices which give the segments that shall be
        processed. Default is None, i.e., the whole file gets processed.
    :param current_sign_correction:
        A dictionary. Its keys are the
        strings used in the file to indicate a state. The column from
        which this state is retrieved is given by *correction_column*.
        The dictionaries' values are used to correct/normalize the
        current value in the file. For example, if discharge currents
        have the same positive sign as charge currents in the file, use
        -1 to correct that, or if the values are to be scaled by weight,
        use the scaling factor. The default is the empty dictionary.
    :param correction_column:
        See #current_sign_correction. Default: -1.
    :param max_number_of_lines:
        The maximum number of dataset lines that are to be read in.
        Default: -1 (no limit).
    :param flip_voltage_sign:
        Defaults to False, where measured voltage remains unaltered.
        Change to True if the voltage shall be multiplied by -1.
        Also applies to impedances; real and imaginary parts.
    :param flip_imaginary_impedance_sign:
        Defaults to False, where measured impedance remains unaltered.
        Change to True if the imaginary part of the impedance shall be
        multiplied by -1. Cancels out with *flip_voltage_sign*.
    :returns:
        The measurement, packaged in a Measurement subclass. It depends
        on "datatype" which one it is:
         - "cycling": ``Cycling_Information``
         - "static": ``Static_Information``
         - "impedance": ``Impedance_Measurement``
    """

    file = open(path, encoding=encoding)
    voltage_sign = -1.0 if flip_voltage_sign else 1.0
    imaginary_impedance_sign = -1.0 if flip_imaginary_impedance_sign else 1.0

    # Truncate the commented lines at the start.
    length_of_comments = 0
    for i in range(number_of_comment_lines):
        line = file.readline()
        length_of_comments = length_of_comments + len(line) + 1
    file.seek(length_of_comments)

    csv_file = csv.reader(file, delimiter=delimiter)
    extra_sign_column = max(headers.keys()) + 1
    data = [] if segment_column != -1 else [
        {index: [] for index in list(headers.keys()) + [extra_sign_column]}
    ]
    indices = []
    last_segment_id = ""
    for i, row in enumerate(csv_file):
        if max_number_of_lines != -1 and i > max_number_of_lines:
            break
        try:
            if segment_column != -1 and row[segment_column] != last_segment_id:
                data.append({
                    index: []
                    for index in list(headers.keys()) + [extra_sign_column]
                })
                last_segment_id = row[segment_column]
                stored_index = last_segment_id
                try:
                    stored_index_str = str(stored_index)
                    if '.' in stored_index_str:
                        stored_index = float(stored_index)
                    else:
                        stored_index = int(stored_index)
                except (TypeError, ValueError):
                    ...
                indices.append(stored_index)
        except IndexError:
            raise IndexError((
                "correction_column or segment_column are set to values outside"
                + " of the row range. Affected row: #{0}. Length of row: {1}. "
                + "Values for correction_column and segment_column: {2}, {3}. "
            ).format(i, len(row), correction_column, segment_column)
                + "Contents of affected row:" + linesep
                + linesep.join(
                    [str(j) + ' ' + str(entry) for j, entry in enumerate(row)]
            )
            )
        try:
            if correction_column != -1:
                data[-1][extra_sign_column].append(
                    current_sign_correction[row[correction_column]]
                )
        except IndexError:
            raise IndexError((
                "correction_column is set to a value outside "
                + "of the row range. Affected row: #{0}. Length of row: {1}. "
                + "Value for correction_column: {2}. "
            ).format(i, len(row), correction_column)
                + "Contents of affected row:" + linesep
                + linesep.join(
                    [str(j) + ' ' + str(entry) for j, entry in enumerate(row)]
            ))
        except ValueError:
            raise ValueError((
                "current_sign_correction is missing an entry for the content "
                "of row #{0} at correction_column {1}. Content: {2}."
            ).format(i, correction_column, row[correction_column]))
        for index in headers.keys():
            try:
                data[-1][index].append(float(
                    row[index].replace(decimal, '.')
                ))
            except ValueError:  # Entry is not a number.
                data[-1][index].append(row[index])
            except IndexError:
                raise IndexError(
                    "Index " + str(index) + " out of bounds for row "
                    + str(row) + "."
                )

    file.close()

    header_info = {}
    for column_index, header in headers.items():
        # The following regex extracts the identifier, the unit prefix
        # and the unit. The "filter" removes the empty strings returned
        # from the regex, which are there for reversibility reasons.
        info = list(filter(None, re.split(
            r"([UItfϕ]?)(real_Z)?(imag_Z)?\s"
            r"\[([Mkmµn]??)([ΩVAhms]?)(Hz)?(rad)?(deg)?\]",
            header
        )))
        if len(info) == 2:
            id, unit = info
            prefix = ""
        elif len(info) == 3:
            id, prefix, unit = info
        else:
            id = header
            prefix = ""
            unit = ""
        header_info[id] = (column_index, prefix, unit)

    prefixes = {"M": 1e6, "k": 1e3, "": 1.0, "m": 1e-3, "µ": 1e-6, "n": 1e-9}

    if datatype in ["cycling", "static"]:
        try:
            timescale = {"h": 3600.0, "m": 60.0, "s": 1.0}[header_info["t"][2]]
        except KeyError:
            timescale = 1

        timepoints = []
        # timesteps = []
        currents = []
        voltages = []
        for index, segment in enumerate(data):
            if segments_to_process is not None:
                if index not in segments_to_process:
                    continue
            try:
                timepoints.append([
                    timescale * prefixes[header_info["t"][1]] * t
                    for t in segment[header_info["t"][0]]
                ])
            except KeyError:
                pass
            try:
                if correction_column != -1:
                    currents.append([
                        prefixes[header_info["I"][1]] * sign * current
                        for (current, sign) in zip(
                            segment[header_info["I"][0]],
                            segment[extra_sign_column]
                        )
                    ])
                else:
                    currents.append([
                        prefixes[header_info["I"][1]] * current
                        for (current) in segment[header_info["I"][0]]
                    ])
            except KeyError:
                pass
            try:
                voltages.append([
                    prefixes[header_info["U"][1]] * voltage_sign * voltage
                    for voltage in segment[header_info["U"][0]]
                ])
            except KeyError:
                pass

        for h in ('t', 'I', 'U'):
            try:
                del header_info[h]
            except KeyError:
                pass
        other_columns = {}
        for id, column_index in header_info.items():
            other_columns[id] = []
            for segment in data:
                other_columns[id].append(segment[column_index[0]])

        if datatype == "cycling":
            return Cycling_Information(
                timepoints, currents, voltages, other_columns, indices
            )
        elif datatype == "static":
            return Static_Information(
                timepoints, currents, voltages, other_columns, indices
            )
    elif datatype == "impedance":
        frequencies = []
        real_impedances = []
        imaginary_impedances = []
        phases = []
        for index, segment in enumerate(data):
            if segments_to_process is not None:
                if index not in segments_to_process:
                    continue
            try:
                frequencies.append([
                    prefixes[header_info["f"][1]] * f
                    for f in segment[header_info["f"][0]]
                ])
            except KeyError:
                pass
            try:
                real_impedances.append([
                    prefixes[header_info["real_Z"][1]] * voltage_sign * imp
                    for imp in segment[header_info["real_Z"][0]]
                ])
            except KeyError:
                pass
            try:
                imaginary_impedances.append([
                    prefixes[header_info["imag_Z"][1]]
                    * voltage_sign
                    * imaginary_impedance_sign
                    * imp
                    for imp in segment[header_info["imag_Z"][0]]
                ])
            except KeyError:
                pass
            try:
                if header_info["ϕ"][2] == "deg":
                    phase_prefix = 2 * pi / 360
                elif header_info["ϕ"][2] == "rad":
                    phase_prefix = 1
                else:
                    raise ValueError(
                        "Phase angles have to be given in 'deg' or 'rad'."
                    )
                phases.append([
                    prefixes[header_info["ϕ"][1]] * phase_prefix * phase
                    for phase in segment[header_info["ϕ"][0]]
                ])
            except KeyError:
                pass

        for h in ('f', 'real_Z', 'imag_Z', 'ϕ'):
            try:
                del header_info[h]
            except KeyError:
                pass
        other_columns = {}
        for id, column_index in header_info.items():
            other_columns[id] = []
            for segment in data:
                other_columns[id].append(segment[column_index[0]])

        return Impedance_Measurement(
            frequencies,
            real_impedances,
            imaginary_impedances,
            phases,
            other_columns,
            indices
        )
    else:
        raise ValueError(
            "Unsupported type of measurement: "
            + str(datatype)
            + ". Must be one of 'cycling', 'static', or 'impedance'."
        )


def print_hdf5_structure(
    h5py_object,
    depth=1,
    table_limit=4,
    verbose_limit=64
):
    """
    Simple HDF5 structure viewer.

    :param h5py_object:
        The HDF5 object wrapped by H5Py, for example
        ``h5py.File(filename, 'r')``.
    :param depth:
        For pretty printing the recursive depth. Do not change.
    :param table_limit:
        h5py.Dataset object tables will be truncated prior to printing
        up to this number in the higher dimension.
    :param verbose_limit:
        h5py.Dataset objects will be truncated to at most this number
        in any dimension.
    """

    tree_visualizer = (depth - 1) * "  " + "|_"
    if isinstance(h5py_object, h5py.Dataset):
        print(tree_visualizer, h5py_object.shape, h5py_object.dtype)
        if len(h5py_object.shape) < 2:
            print(tree_visualizer, list(h5py_object[:verbose_limit]))
        else:
            if h5py_object.shape[0] < h5py_object.shape[1]:
                for column in h5py_object:
                    print(tree_visualizer, list(column[:table_limit]))
            else:
                for line in h5py_object[:table_limit]:
                    print(tree_visualizer, line)
        return
    elif (
        isinstance(h5py_object, h5py.Group)
        or isinstance(h5py_object, h5py.File)
    ):
        for key in h5py_object.keys():
            print(tree_visualizer[:-1])
            print(tree_visualizer, key)
            print_hdf5_structure(h5py_object[key], depth + 1)


def convert_none_notation_to_slicing(h5py_object, index):
    """
    Access a slice of an HDF5 object by transferable notation.

    :param h5py_object:
        The HDF5 object wrapped by H5Py, for example
        ``h5py.File(filename, 'r')``.
    :param index:
        A 2-tuple or a 2-list. (None, x) denotes slicing [:, x] and
        (x, None) denotes slicing [x, :].
    :returns:
        The None-notation sliced h5py object.
    """

    if len(index) != 2:
        raise ValueError(
            "Location in HDF5 structure can only contain indices "
            "and 2-tuples or 2-lists denoting a [x ,y]-type slice."
        )
    if index[0] is None:
        return h5py_object[:, index[1]]
    elif index[1] is None:
        return h5py_object[index[0], :]
    else:
        raise ValueError(
            "2-tuple or 2-list in HDF5 structure location has to "
            "contain one None entry to denote the dimension of "
            "the slice that gets read completely."
        )


def get_hdf5_dataset_by_path(h5py_object, path):
    """
    Follow the structure of a HDF object to get a certain part.

    :param h5py_object:
        The HDF5 object wrapped by H5Py, for example
        ``h5py.File(filename, 'r')``.
    :param path:
        A list. Each entry goes one level deeper into the HDF structure.
        Each entry can either be the index to go into next itself, or a
        2-tuple or a 2-list. In the latter case, (None, x) denotes
        slicing [:, x] and (x, None) denotes slicing [x, :].
    :returns:
        Returns the HDF5 object found at the end of *path*.
    """

    target_dataset = h5py_object
    for index in path:
        if isinstance(index, tuple) or isinstance(index, list):
            target_dataset = convert_none_notation_to_slicing(
                target_dataset, index
            )
        else:
            target_dataset = target_dataset[index]
    return target_dataset


def read_hdf5_table(
    path,
    data_location,
    headers,
    datatype="cycling",
    segment_location=None,
    segments_to_process=None,
    current_sign_correction={},
    correction_location=None,
    flip_voltage_sign=False,
    flip_imaginary_impedance_sign=False
):
    """
    Read the measurements as stored in a HDF5 file.

    :param path:
        The full or relative path to the measurement file.
    :param data_location:
        A list. Gives the location in the HDF5 file where the data table
        is stored. Set to None if everything is stored at the top level.
        Each entry goes one level deeper into the HDF structure. Each
        entry can either be the index to go into next itself, or a
        2-tuple or a 2-list. In the latter case, (None, x) denotes
        slicing [:, x] and (x, None) denotes slicing [x, :].
    :param headers:
        A dictionary. Its keys are 2-tuples, slicing the data which are
        to be read in. Use the format "(x, None)" or "(None, x)" to
        slice the dimension with "None". The corresponding values are
        there to tell this function which kind of data is in which
        column. If necessary, the keys may also be tuples like
        *data_location*.
        The following format has to be used: "<name> [<unit>]" where
        "name" is "U" (voltage), "I" (current), or "t" (time) and "unit"
        is "V", "A", "h", "m", or "s" with the optional prefixes "k",
        "m", "µ", or "n". This converts the data to prefix-less SI
        units. Additional columns may be read in with keys not in this
        format. The columns for segments and sign correction are only
        given by *segment_column* and *correction_column*.
    :param datatype:
        Default is "cycling", where cycling information is assumed in
        the file. "static" will trigger the additional extraction of
        exponential decays that are relevant to e.g. GITT.
        "impedance" will treat the file as an impedance measurement
        with frequencies and impedances instead of time and voltage.
    :segment_location:
        A list, with the same format as *data_location*. It points to
        the part of the data that stores the index of the current
        segment. If it changes from one data point to the next, that is
        used as the dividing line between two segments.
        Default is None, which returns the dataset in one segment.
    :param segments_to_process:
        A list of indices which give the segments that shall be
        processed. Default is None, i.e., the whole file gets processed.
    :param current_sign_correction:
        A dictionary. Its keys are the strings used in the file to
        indicate a state. The column from which this state is retrieved
        is given by *correction_column*.
        The dictionaries' values are used to correct/normalize the
        current value in the file. For example, if discharge currents
        have the same positive sign as charge currents in the file, use
        -1 to correct that, or if the values are to be scaled by weight,
        use the scaling factor. The default is the empty dictionary.
    :param correction_location:
        A list, with the same format as *data_location*.
        For its use, see *current_sign_correction*. Default: None.
    :param flip_voltage_sign:
        Defaults to False, where measured voltage remains unaltered.
        Change to True if the voltage shall be multiplied by -1.
        Also applies to impedances; real and imaginary parts.
    :param flip_imaginary_impedance_sign:
        Defaults to False, where measured impedance remains unaltered.
        Change to True if the imaginary part of the impedance shall be
        multiplied by -1. Cancels out with *flip_voltage_sign*.
    :returns:
        The measurement, packaged in a Measurement subclass. It depends
        on "datatype" which one it is:
         - "cycling": ``Cycling_Information``
         - "static": ``Static_Information``
         - "impedance": ``Impedance_Measurement``
    """

    file = h5py.File(path, 'r')
    voltage_sign = -1.0 if flip_voltage_sign else 1.0
    imaginary_impedance_sign = -1.0 if flip_imaginary_impedance_sign else 1.0
    data = [{}]

    header_info = {}
    for column_index, header in headers.items():
        # The following regex extracts the identifier, the unit prefix
        # and the unit. The "filter" removes the empty strings returned
        # from the regex, which are there for reversibility reasons.
        info = list(filter(None, re.split(
            r"([UItfϕ]?)(real_Z)?(imag_Z)?\s"
            r"\[([Mkmµn]??)([ΩVAhms]?)(Hz)?(rad)?(deg)?\]",
            header
        )))
        if len(info) == 2:
            id, unit = info
            prefix = ""
        elif len(info) == 3:
            id, prefix, unit = info
        else:
            id = header
            prefix = ""
            unit = ""
        header_info[id] = (column_index, prefix, unit)

    split_indices = [0]
    if segment_location:
        segment_data = get_hdf5_dataset_by_path(file, segment_location)
        last_segment_value = segment_data[0]
        for i, segment_value in enumerate(segment_data):
            if segment_value != last_segment_value:
                split_indices.append(i)
                data.append({})
                last_segment_value = segment_value
    split_indices.append(None)

    if data_location is None:
        data_table = file
    else:
        data_table = get_hdf5_dataset_by_path(file, data_location)
    for index in headers.keys():
        try:
            data_column = convert_none_notation_to_slicing(data_table, index)
        except ValueError:
            data_column = get_hdf5_dataset_by_path(data_table, index)
        for i, (start, stop) in enumerate(
            zip(split_indices[:-1], split_indices[1:])
        ):
            data[i][index] = data_column[start:stop]

    prefixes = {"M": 1e6, "k": 1e3, "": 1.0, "m": 1e-3, "µ": 1e-6, "n": 1e-9}

    if datatype in ["cycling", "static"]:
        try:
            timescale = {"h": 3600.0, "m": 60.0, "s": 1.0}[header_info["t"][2]]
        except KeyError:
            timescale = 1

        timepoints = []
        # timesteps = []
        currents = []
        voltages = []
        if correction_location:
            correction_data = get_hdf5_dataset_by_path(
                file, correction_location
            )
        for index, segment in enumerate(data):
            if segments_to_process is not None:
                if index not in segments_to_process:
                    continue
            try:
                timepoints.append([
                    timescale * prefixes[header_info["t"][1]] * t
                    for t in segment[header_info["t"][0]]
                ])
            except KeyError:
                pass
            # timesteps.append([
            #     t1 - t0 for (t0, t1) in zip(
            #         timepoints[-1][:-1], timepoints[-1][1:]
            #     )
            # ])
            try:
                if correction_location:
                    for i, (start, stop) in enumerate(
                        zip(split_indices[:-1], split_indices[1:])
                    ):
                        currents.append([
                            prefixes[header_info["I"][1]]
                            * current_sign_correction[sign]
                            * current
                            for (current, sign) in zip(
                                segment[header_info["I"][0]],
                                correction_data[start:stop]
                            )
                        ])
                else:
                    currents.append([
                        prefixes[header_info["I"][1]] * current
                        for (current) in segment[header_info["I"][0]]
                    ])
            except KeyError:
                pass
            try:
                voltages.append([
                    prefixes[header_info["U"][1]] * voltage_sign * voltage
                    for voltage in segment[header_info["U"][0]]
                ])
            except KeyError:
                pass

        for h in ('t', 'I', 'U'):
            try:
                del header_info[h]
            except KeyError:
                pass
        other_columns = {}
        for id, column_index in header_info.items():
            other_columns[id] = []
            for segment in data:
                other_columns[id].append(segment[column_index[0]])

        if datatype == "cycling":
            return Cycling_Information(
                timepoints, currents, voltages, other_columns
            )
        elif datatype == "static":
            return Static_Information(
                timepoints, currents, voltages, other_columns
            )
    elif datatype == "impedance":
        frequencies = []
        real_impedances = []
        imaginary_impedances = []
        phases = []
        for index, segment in enumerate(data):
            if segments_to_process is not None:
                if index not in segments_to_process:
                    continue
            try:
                frequencies.append([
                    prefixes[header_info["f"][1]] * f
                    for f in segment[header_info["f"][0]]
                ])
            except KeyError:
                pass
            try:
                real_impedances.append([
                    prefixes[header_info["real_Z"][1]] * voltage_sign * imp
                    for imp in segment[header_info["real_Z"][0]]
                ])
            except KeyError:
                pass
            try:
                imaginary_impedances.append([
                    prefixes[header_info["imag_Z"][1]]
                    * voltage_sign
                    * imaginary_impedance_sign
                    * imp
                    for imp in segment[header_info["imag_Z"][0]]
                ])
            except KeyError:
                pass
            try:
                if header_info["ϕ"][2] == "deg":
                    phase_prefix = 2 * pi / 360
                elif header_info["ϕ"][2] == "rad":
                    phase_prefix = 1
                else:
                    raise ValueError(
                        "Phase angles have to be given in 'deg' or 'rad'."
                    )
                phases.append([
                    prefixes[header_info["ϕ"][1]] * phase_prefix * phase
                    for phase in segment[header_info["ϕ"][0]]
                ])
            except KeyError:
                pass

        for h in ('f', 'real_Z', 'imag_Z', 'ϕ'):
            try:
                del header_info[h]
            except KeyError:
                pass
        other_columns = {}
        for id, column_index in header_info.items():
            other_columns[id] = []
            for segment in data:
                other_columns[id].append(segment[column_index[0]])

        return Impedance_Measurement(
            frequencies,
            real_impedances,
            imaginary_impedances,
            phases,
            other_columns
        )
    else:
        raise ValueError(
            "Unsupported type of measurement: "
            + str(datatype)
            + ". Must be one of 'cycling', 'static', or 'impedance'."
        )


def read_parquet_table(file_name, datatype):
    """
    Read an Apache Parquet serialization of a ``Measurement`` object.
    For storing such a serialization, refer to ``store_parquet_table``.

    :param file_name:
        The full name of the file to read from.
    :param datatype:
        One of the following, denoting what ``Measurement`` was stored:
         - 'cycling': ``Cycling_Information``
         - 'static': ``Static_Information``
         - 'impedance': ``Impedance_Information``
    :returns:
        The ``Measurement`` object of choice.
    """

    datafile = parquet.ParquetFile(file_name)

    indices = []
    other_columns = {}

    if datatype == "cycling":
        table_mapping = Cycling_Information.table_mapping()
    elif datatype == "static":
        table_mapping = Static_Information.table_mapping()
    elif datatype == "impedance":
        table_mapping = Impedance_Measurement.table_mapping()

    if datatype in ["cycling", "static"]:
        timepoints = []
        currents = []
        voltages = []
        for row_group_number in range(datafile.num_row_groups):
            row_group = datafile.read_row_group(row_group_number)
            # Each segment can only have one index; assume all are the same.
            indices.append(
                row_group.column(table_mapping['indices'])[0].as_py()
            )
            try:
                timepoints.append(row_group.column(
                    table_mapping['timepoints']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            try:
                currents.append(row_group.column(
                    table_mapping['currents']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            try:
                voltages.append(row_group.column(
                    table_mapping['voltages']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            other_names = (
                set(row_group.column_names) - set(table_mapping.values())
            )
            for name in other_names:
                if name not in other_columns.keys():
                    other_columns[name] = []
                # Pray and hope that to_numpy has enough coverage.
                other_columns[name].append(
                    row_group.column(name).to_numpy().tolist()
                )
        if datatype == "cycling":
            data = Cycling_Information(
                timepoints, currents, voltages, other_columns, indices
            )
        elif datatype == "static":
            data = Static_Information(
                timepoints, currents, voltages, other_columns, indices
            )
    elif datatype == "impedance":
        frequencies = []
        real_impedances = []
        imaginary_impedances = []
        phases = []
        for row_group_number in range(datafile.num_row_groups):
            row_group = datafile.read_row_group(row_group_number)
            # Each segment can only have one index; assume all are the same.
            indices.append(
                row_group.column(table_mapping['indices'])[0].as_py()
            )
            try:
                frequencies.append(row_group.column(
                    table_mapping['frequencies']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            try:
                real_impedances.append(row_group.column(
                    table_mapping['real_impedances']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            try:
                imaginary_impedances.append(row_group.column(
                    table_mapping['imaginary_impedances']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            try:
                phases.append(row_group.column(
                    table_mapping['phases']
                ).combine_chunks().to_numpy().tolist())
            except KeyError:
                pass
            other_names = (
                set(row_group.column_names) - set(table_mapping.values())
            )
            for name in other_names:
                if name not in other_columns.keys():
                    other_columns[name] = []
                # Assume that to_numpy has enough coverage.
                other_columns[name].append(
                    row_group.column(name).to_numpy().tolist()
                )
        data = Impedance_Measurement(
            frequencies,
            real_impedances,
            imaginary_impedances,
            phases,
            other_columns,
            indices
        )
    return data


def store_parquet_table(measurement, file_prefix, compression_level=22):
    """
    Store an Apache Parquet serialization of a ``Measurement`` object.
    For reading such a serialization, refer to ``read_parquet_table``.

    :param measurement:
        The ``Measurement`` object to serialize.
    :param file_prefix:
        The filename to write into. A '.parquet' ending is appended.
    :param compression_level:
        Compression level of the compression algorithm Zstandard.
        -7 gives the largest files with highest compression speed.
        22 gives the smallest files with slowest compression speed.
        Note that decompression has the same (fast) speed at any level.
    """

    table_schema = pyarrow.table(
        measurement.example_table_row(), names=measurement.table_descriptors(),
    ).schema

    # Write each segment in a separate ParquetWriter call. This sets up
    # the row groups to correspond to segments for faster reading.
    with parquet.ParquetWriter(
        file_prefix + '.parquet',
        table_schema,
        compression='zstd',
        compression_level=compression_level,
    ) as writer:
        for segment in measurement.segment_tables():
            segment_table = pyarrow.table(
                segment, names=measurement.table_descriptors(),
            )
            writer.write_table(segment_table)
