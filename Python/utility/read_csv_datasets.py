"""!@package utility.read_csv_datasets
Defines datatypes for further processing and containtes the means to
convert commonly encountered measurement files into them.
"""

import csv
import re
from utility.fitting_functions import fit_exponential_decay


class Cycling_Information(object):
    """!@brief Contains basic cycling informations.

    Each member variable is a list and has the same length as the other
    ones.
    """

    def __init__(
        self, timepoints, currents, voltages, other_columns={}, indices=None
    ):
        """! The times at which measurements were taken. Usually a list of
        lists where each list corresponds to a segment of the
        measurement. """
        self.timepoints = timepoints
        """! The measured current at those times. A list which usually
        contains lists for segments with variable current and floats
        for segments with constant current. """
        self.currents = currents
        """! The measured voltage at those times. A list which usually
        contains lists for segments with variable voltage and floats
        for segments with constant voltage. """
        self.voltages = voltages
        """! The contents of any other columns. A dictionary ("columns")
        which values are lists which contain lists for segments.
        The keys should match user input for the columns. """
        self.other_columns = other_columns
        """! The indices of the individual segments. Defaults to a simple
        numbering of the segments present. May be used for plotting
        purposes, e.g. for colorcoding the segments by cycle. """
        if indices is None:
            self.indices = [i for i in range(len(timepoints))]
        else:
            self.indices = indices
        # This syntax doesn't work when arrays are involved, apparently.
        # self.indices = indices or [i for i in range(len(timepoints))]

    def subslice(self, start, stop, step=1):
        """!@brief Selects a subslice of the data segments.

        The arguments exactly match the slice(...) notation.

        @par start
            The index of the first segment to be included.
        @par stop
            The index of the first segment to not be included.
        @par step
            Steps between selected segments. Default: 1.
        @return
            A new Cycling_Information object containing the slice.
        """

        return Cycling_Information(
            self.timepoints[start:stop:step],
            self.currents[start:stop:step],
            self.voltages[start:stop:step],
            {name: column[start:stop:step]
             for name, column in self.other_columns.items()},
            self.indices[start:stop:step]
        )

    def subarray(self, array):
        """!@brief Selects the data segments with the given indices.

        @par array
            The indices of the segments that are to be returned.
        @return
            A new Cycling_Information object containing the subset.
        """

        return Cycling_Information(
            [self.timepoints[index] for index in array],
            [self.currents[index] for index in array],
            [self.voltages[index] for index in array],
            {name: [column[index] for index in array]
             for name, column in self.other_columns.items()},
            [self.indices[index] for index in array]
        )


class Static_Information(Cycling_Information):
    """!@brief Contains additional informations, e.g. for GITT.

    Each member variable is a list and has the same length as the other
    ones.
    """

    def __init__(
        self,
        timepoints,
        currents,
        voltages,
        asymptotic_voltages,
        ir_steps,
        exp_I_decays,
        exp_U_decays,
        other_columns={},
        indices=None
    ):
        super().__init__(timepoints, currents, voltages, other_columns,
                         indices)
        """! The voltages that the voltage curve seems to converge to in a
        segment. Only makes sense for those segments that are rest
        periods or when the OCV was subtracted. """
        self.asymptotic_voltages = asymptotic_voltages
        """! The instantaneous IR drops before each segment. Positive
        values are voltage rises and negative values voltage drops. """
        self.ir_steps = ir_steps
        """! Same as exp_U_decays for current decays (PITT). """
        self.exp_I_decays = exp_I_decays
        """! The fit parameters of the exponential voltage decays in each
        segment. Each set of fit parameters is a 3-tuple (a,b,c) where
        the fit function has the following form:
            a + b * exp(-c * (t - t_end_of_segment)).
        Failed or missing fits are best indicated by (NaN, NaN, NaN). """
        self.exp_U_decays = exp_U_decays

    def subslice(self, start, stop, step=1):
        """!@brief Selects a subslice of the data segments.

        The arguments exactly match the slice(...) notation.

        @par start
            The index of the first segment to be included.
        @par stop
            The index of the first segment to not be included.
        @par step
            Steps between selected segments. Default: 1.
        @return
            A new Static_Information object containing the slice.
        """

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
        """!@brief Selects the data segments with the given indices.

        @par array
            The indices of the segments that are to be returned.
        @return
            A new Static_Information object containing the subset.
        """

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


def read_channels_from_measurement_system(
    path,
    encoding,
    number_of_comment_lines,
    headers,
    delimiter='\t',
    decimal='.',
    type="",
    segment_column=-1,
    segments_to_process=None,
    current_sign_correction={},
    correction_column=-1,
    max_number_of_lines=-1
):
    """!@brief Read the measurements as returned by common instruments.

    Example: cycling measurements from Basytec devices. Their format
    resembles a csv file with one title and one header comment line.
    So the first line will be ignored and the second used for headers.

    @par path
        The full or relative path to the measurement file.
    @par encoding
        The encoding of that file, e.g. "iso-8859-1".
    @par number_of_comment_lines
        The number of lines that have to be skipped over in order to
        arrive at the first dataset line.
    @par headers
        A dictionary. Its keys are the indices of the columns
        which are to be read in. The corresponding values are there to
        tell this function which kind of data is in which column. The
        following format has to be used: "<name> [<unit>]" where "name"
        is "U" (voltage), "I" (current) or "t" (time) and "unit" is
        "V", "A", "h", "m" or "s" with the optional prefixes "k", "m",
        "µ" or "n". This converts the data to prefix-less SI units.
        Additional columns may be read in with keys not in this format.
        The columns for segments and sign correction are only given by
        #segment_column and #correction_column.
    @par delimiter
        The delimiter string between datapoints. The default is "\t".
    @par decimal
        The string used for the decimal point. Default: ".".
    @par type
        Default is "", where only basic information is extracted from
        the file. "static" will trigger the additional extraction of
        exponential decays that are relevant to e.g. GITT.
    @par segment_column
        The index of the column that stores the index
        of the current segment. If it changes from one data point to the
        next, that is used as the dividing line between two segments.
        Default is -1, which returns the dataset in one segment.
    @par segments_to_process
        A list of indices which give the segments that shall be
        processed. Default is None, i.e., the whole file gets processed.
    @par current_sign_correction
        A dictionary. Its keys are the
        strings used in the file to indicate a state. The column from
        which this state is retrieved is given by #correction_column.
        The dictionaries' values are used to correct/normalize the
        current value in the file. For example, if discharge currents
        have the same positive sign as charge currents in the file, use
        -1 to correct that, or if the values are to be scaled by weight,
        use the scaling factor. The default is the empty dictionary.
    @par correction_column
        See #current_sign_correction. Default: -1.
    @par max_number_of_lines
        The maximum number of dataset lines that are to be read in.
        Default: -1 (no limit).
    @return
        A Cycling_Information or Static_Information object, depending on
        the value of "type".
    """

    file = open(path, encoding=encoding)

    # Truncate the commented lines at the start.
    length_of_comments = 0
    for i in range(number_of_comment_lines):
        line = file.readline()
        length_of_comments = length_of_comments + len(line) + 1
    file.seek(length_of_comments)

    csv_file = csv.reader(file, delimiter=delimiter)
    data = [] if segment_column != -1 else [
        {index: [] for index in headers.keys()}
    ]
    last_segment_id = ""

    for i, row in enumerate(csv_file):
        if max_number_of_lines != -1 and i > max_number_of_lines:
            break
        if segment_column != -1 and row[segment_column] != last_segment_id:
            data.append({index: [] for index in headers.keys()})
            data[-1][correction_column] = []
            last_segment_id = row[segment_column]
        if correction_column != -1:
            data[-1][correction_column].append(
                current_sign_correction[row[correction_column]]
            )
        for index, name in headers.items():
            data[-1][index].append(float(row[index].replace(decimal, '.')))

    file.close()

    header_info = {}
    for column_index, header in headers.items():
        # The following regex extracts the identifier, the unit prefix
        # and the unit. The "filter" removes the empty strings returned
        # from the regex, which are there for reversibility reasons.
        info = list(filter(None, re.split(
            r"([UIt])?\s\[([kmµn]??)([VAhms]?)\]", header
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
    try:
        timescale = {"h": 3600.0, "m": 60.0, "s": 1.0}[header_info["t"][2]]
    except KeyError:
        timescale = 1
    prefixes = {"k": 1e3, "": 1.0, "m": 1e-3, "µ": 1e-6, "n": 1e-9}

    timepoints = []
    # timesteps = []
    currents = []
    voltages = []
    for index, segment in enumerate(data):
        if segments_to_process is not None:
            if index not in segments_to_process:
                continue
        try:
            timepoints.append([timescale * prefixes[header_info["t"][1]] * t
                               for t in segment[header_info["t"][0]]])
        except KeyError:
            pass
        # timesteps.append([t1 - t0
        #                   for (t0, t1) in zip(timepoints[-1][:-1],
        #                                       timepoints[-1][1:])])
        try:
            if correction_column != -1:
                currents.append([
                    prefixes[header_info["I"][1]] * sign * current
                    for (current, sign) in zip(segment[header_info["I"][0]],
                                               segment[correction_column])
                ])
            else:
                currents.append([
                    prefixes[header_info["I"][1]] * current
                    for (current) in segment[header_info["I"][0]]
                ])
        except KeyError:
            pass
        voltages.append([prefixes[header_info["U"][1]] * voltage
                         for voltage in segment[header_info["U"][0]]])

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

    if type == "":
        return Cycling_Information(timepoints, currents, voltages,
                                   other_columns)
    elif type == "static":
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

        return Static_Information(
            timepoints, currents, voltages, asymptotic_voltages, ir_steps,
            exp_I_decays, exp_U_decays, other_columns
        )
