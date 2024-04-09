"""!@package ep_bolfi.utility.preprocessing
Contains frequently used workflows in dataset preprocessing.

The functions herein are a collection of simple, but frequent,
transformations of arrays of raw measurement data.
"""

import numpy as np
import copy
from collections.abc import MutableMapping
from contextlib import contextmanager
from itertools import chain
from multiprocessing import Pool
from numbers import Number
from pybamm import Scalar
from scipy.optimize import root_scalar
from scipy.stats import chi2, norm
from ep_bolfi.models.solversetup import solver_setup, simulation_setup
from ep_bolfi.utility.fitting_functions import (
    fit_OCV, inverse_OCV_fit_function, smooth_fit,
    verbose_spline_parameterization
)


class SubstitutionDict(MutableMapping):
    """!@brief A dictionary with some automatic substitutions.
    "substitutions" is a dictionary that extends "storage" with
    automatic substitution rules depending on its value types:
     - string, which serves the value of "storage" at that value.
     - callable which takes one parameter, which will get passed its
       SubstitutionDict instance and serves its return value.
     - any other type, which serves the value as-is.
    Assigning values to keys afterwards will overwrite substitutions.
    """

    def __init__(self, storage, substitutions={}):
        self._storage = storage
        self._substitutions = substitutions
        self._log = []
        self._log_switch = False

    def __delitem__(self, key):
        if key in self._storage.keys():
            self._storage.__delitem__(key)
        if key in self._substitutions.keys():
            self._substitutions.__delitem__(key)
        if (
            key not in self._storage.keys()
            and
            key not in self._substitutions.keys()
        ):
            raise KeyError(key)

    def __getitem__(self, key):
        if self._log_switch:
            self._log.append(key)
        if key in self._substitutions.keys():
            storage_value = self._substitutions[key]
            if isinstance(storage_value, Number):
                return storage_value
            elif isinstance(storage_value, str):
                return self[storage_value]
            elif callable(storage_value):
                return storage_value(self)
            else:
                # fallback
                return storage_value
        elif key in self._storage.keys():
            return self._storage[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        return chain(self._storage, self._substitutions)

    def __len__(self):
        return len(set(
            list(self._storage.keys()) + list(self._substitutions.keys())
        ))

    def __setitem__(self, key, value):
        if key in self._substitutions.keys():
            del self._substitutions[key]
        self._storage[key] = value

    def __str__(self):
        return '{' + ', '.join(
            [str(k) + ': ' + str(v) for k, v in self.items()]
        ) + '}'

    def __repr__(self):
        return self.__str__()

    @contextmanager
    def log_lock(self):
        self._log_switch = True
        self._log = []
        try:
            yield
        finally:
            self._log_switch = False
            self._log = []

    def dependent_variables(self, parameters):
        with self.log_lock():
            for p in parameters:
                self[p]
            log = copy.deepcopy(self._log)
        for s in self._substitutions.keys():
            with self.log_lock():
                self[s]
                for p in parameters:
                    if p in self._log:
                        log.append(s)
                        break
        return log


def fix_parameters(parameters_to_be_fixed):
    """!@brief Returns a function which sets some parameters in advance.
    @param parameters_to_be_fixed
        These parameters will at least be a
        part of the dictionary that the returned function returns.
    @return
        The function which adds additional parameters to a
        dictionary or replaces existing parameters with the new ones.
    """

    def return_all_parameters(free_parameters):
        """!@brief Adds the 'free_parameters' to a pool of parameters.
        @param free_parameters
            A dictionary which gets added to the fixed pool of
            parameters (see the function 'fix_parameters').
        @return
            A dictionary containing free_parameters and some other
            key-value-pairs as defined by 'fix_parameters'. If a key is
            present in both, the value from 'free_parameters' is used.
        """

        return_dict = copy.deepcopy(parameters_to_be_fixed)
        return_dict.update(free_parameters)
        return return_dict

    return return_all_parameters


def combine_parameters_to_try(parameters, parameters_to_try_dict):
    """!@brief Give every combination as full parameter sets.
    Compatible with SubstitutionDict, if "parameters" is one.
    @param parameters
        The base full parameter set as a dictionary.
    @param parameters_to_try_dict
        The keys of this dictionary correspond to the "parameters"' keys
        where different values are to be inserted. These are given by
        the tuples which are the values of this dictionary.
    @return
        A 2-tuple where the first item is the list of all parameter set
        combinations and the second the list of the combinations only.
    """

    # Fetch the parameters to try.
    parameters_to_try = []
    for key, value in parameters_to_try_dict.items():
        parameters_to_try.append((key, value))

    def recursive_combination(last_index):
        """! Recursively get every combination of parameters to try. """
        return_list = []
        if last_index == 0:
            key = parameters_to_try[0][0]
            for value in parameters_to_try[0][1]:
                return_list.append({key: value})
        elif last_index > 0:
            key = parameters_to_try[last_index][0]
            list_of_combinations = recursive_combination(last_index-1)
            for value in parameters_to_try[last_index][1]:
                for combination in list_of_combinations:
                    return_dict = {key: value}
                    return_dict.update(combination)
                    return_list.append(return_dict)
        return return_list

    combinations = recursive_combination(len(parameters_to_try)-1)
    parameters_list = []
    for combination in combinations:
        parameters_list.append(fix_parameters(parameters)(combination))
    return (parameters_list, combinations)


def calculate_means_and_standard_deviations(
    mean,
    covariance,
    free_parameters_names,
    transform_parameters={},
    bounds_in_standard_deviations=1,
    **kwargs
):
    """!@brief Calculate means and standard deviations.
    Please note that standard deviations translate differently into
    confidence regions in different dimensions. For the confidence
    region, use "approximate_confidence_ellipsoid".
    @param mean
        The mean of the uncertain parameters as a dictionary.
    @param covariance
        The covariance of the uncertain parameters as a two-dimensional
        numpy array.
    @param free_parameters_names
        The names of the parameters that are uncertain as a list. This
        parameter maps the order of parameters in "covariance".
    @param transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @param bounds_in_standard_deviations
        Sets how many standard deviations in each direction the returned
        error bounds are. These are first applied and then transformed.
    @param **kwargs
        Keyword arguments for scipy.integrate.quad, which is used to
        numerically calculate mean and variance.
    @return
        A 3-tuple with three dictionaries. Their keys are the free
        parameters' names as keys and their values are those parameters'
        means, standard deviations, and error bounds.
    """

    # Substitute transformations given by name.
    if transform_parameters is not {}:
        for name, function in transform_parameters.items():
            if type(function) is str:
                if function == 'none':
                    transform_parameters[name] = (
                        lambda s: s, lambda b: b
                    )
                elif function == 'log':
                    transform_parameters[name] = (
                        lambda s: np.exp(s), lambda b: np.log(b)
                    )
    # Fill-in 'none' transformations.
    for name in free_parameters_names:
        if name not in transform_parameters.keys():
            transform_parameters[name] = (
                lambda s: s, lambda b: b
            )

    means = {}
    standard_deviations = {}
    error_bounds = {}

    for i, name in enumerate(free_parameters_names):
        mean_internal = transform_parameters[name][1](mean[name])
        variance_internal = covariance[i][i]
        mean_actual = norm.expect(
            lambda x: (transform_parameters[name][0](
                mean_internal + x * np.sqrt(variance_internal)
            )),
            **kwargs
        )
        variance_actual = norm.expect(
            lambda x: (transform_parameters[name][0](
                mean_internal + x * np.sqrt(variance_internal)
            ))**2,
            **kwargs
        ) - mean_actual**2
        means[name] = mean_actual
        standard_deviations[name] = np.sqrt(variance_actual)
        error_bounds[name] = (
            transform_parameters[name][0](
                mean_internal
                - bounds_in_standard_deviations * np.sqrt(variance_internal)
            ),
            transform_parameters[name][0](
                mean_internal
                + bounds_in_standard_deviations * np.sqrt(variance_internal)
            ),
        )

    return (means, standard_deviations, error_bounds)


def approximate_confidence_ellipsoid(
    parameters,
    free_parameters_names,
    covariance,
    mean=None,
    transform_parameters={},
    refinement=True,
    confidence=0.95
):
    """!@brief Approximate a confidence ellipsoid.
    Compatible with SubstitutionDict, if "parameters" is one.
    The geometric approximation is a refinement of the polytope with
    nodes on the semiaxes of the confidence ellipsoid. The refinement
    step adds a node for each face, i.e., each sub-polytope with
    dimension smaller by 1. This node is centered on that face and
    projected onto the confidence ellipsoid.
    @param parameters
        The base full parameter set as a dictionary.
    @param free_parameters_names
        The names of the parameters that are uncertain as a list. This
        parameter has to match the order of parameters in "covariance".
    @param covariance
        The covariance of the uncertain parameters as a two-dimensional
        numpy array.
    @param mean
        The mean of the uncertain parameters as a dictionary. If not
        set, the values from 'parameters' will be used.
    @param transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @param confidence
        The confidence within the ellipsoid. Defaults to 0.95, i.e., the
        95% confidence ellipsoid.
    @param refinement
        If False, only the nodes on the semiaxes get returned. If True,
        the nodes centered on the faces get returned as well.
    @return
        A 2-tuple where the first item is the list of all parameter set
        combinations and the second the ellipsoid nodes only as a
        two-dimensional numpy array with each node in on row.
    """

    if mean is None:
        mean = parameters
    # Substitute transformations given by name.
    if transform_parameters is not {}:
        for name, function in transform_parameters.items():
            if type(function) is str:
                if function == 'none':
                    transform_parameters[name] = (
                        lambda s: s, lambda b: b
                    )
                elif function == 'log':
                    transform_parameters[name] = (
                        lambda s: np.exp(s), lambda b: np.log(b)
                    )
    # Fill-in 'none' transformations.
    for name in free_parameters_names:
        if name not in transform_parameters.keys():
            transform_parameters[name] = (
                lambda s: s, lambda b: b
            )

    mahalanobis_squared = chi2(len(free_parameters_names))
    standard_deviations = np.sqrt(mahalanobis_squared.ppf(confidence))

    semiaxes_length_squared, semiaxes_normed = np.linalg.eigh(covariance)
    semiaxes_length = np.sqrt(semiaxes_length_squared)
    confidence_semiaxes_length = semiaxes_length * standard_deviations

    # These calculations happen in the eigenspace of the covariance.
    semiaxis_nodes = []
    for dim, csl in enumerate(confidence_semiaxes_length):
        semiaxis_nodes.append(
            [0.0] * dim
            + [-csl]
            + [0.0] * (len(free_parameters_names) - 1 - dim)
        )
        semiaxis_nodes.append(
            [0.0] * dim
            + [csl]
            + [0.0] * (len(free_parameters_names) - 1 - dim)
        )

    if refinement:
        # [aᵢ / √n]ᵢⁿ fulfills ∑ᵢⁿ (aᵢ / √n)² / aᵢ² = 1.
        diagonal = np.array(
            [-confidence_semiaxes_length,
             confidence_semiaxes_length]
        ) / np.sqrt(len(free_parameters_names))

        def recursive_combination(length, index=0):
            """! Recursively get every combination of nodes. """
            return_list = []
            if index == length - 1:
                for entry in diagonal.T[index]:
                    return_list.append([entry])
            elif index < length - 1:
                list_of_combinations = recursive_combination(length, index + 1)
                for entry in diagonal.T[index]:
                    for combination in list_of_combinations:
                        node = [entry]
                        node.extend(combination)
                        return_list.append(node)
            return return_list

        combinations = recursive_combination(len(free_parameters_names))
    else:
        combinations = []

    transformed_mean = np.array([
        transform_parameters[name][1](mean[name])
        for name in free_parameters_names
    ])
    transformed_ellipsoid = transformed_mean + np.array([
        (semiaxes_normed @ node)
        for node in [*semiaxis_nodes, *combinations]
    ])
    ellipsoid = np.array([
        [
            transform_parameters[name][0](v[i])
            for i, name in enumerate(free_parameters_names)
        ]
        for v in transformed_ellipsoid
    ])
    parameters_list = []
    for node in ellipsoid:
        node_parameters = {
            name: node[i] for i, name in enumerate(free_parameters_names)
        }
        parameters_list.append(fix_parameters(parameters)(node_parameters))
    return (parameters_list, ellipsoid)


def capacity(parameters, electrode="positive"):
    """!@brief Convenience function for calculating the capacity.
    @param parameters
        A parameter file as defined by models.standard_parameters.
    @param electrode
        The prefix of the electrode to use for capacity calculation.
        Change to "negative" to use the one with the lower OCP.
    @return
        The capacity of the parameterized battery in C.
    """

    return (
        parameters[
            electrode.capitalize()
            + " electrode active material volume fraction"
        ]
        * parameters[electrode.capitalize() + " electrode thickness [m]"]
        * parameters["Current collector perpendicular area [m2]"]
        * parameters[
            "Maximum concentration in " + electrode + " electrode [mol.m-3]"
        ]
        * 96485.33212
    )


def calculate_SOC(timepoints, currents, initial_SOC=0, sign=1, capacity=1):
    """!@brief Transforms applied current over time into SOC.
    @param timepoints
        Array of the timepoint segments.
    @param currents
        Array of the current segments.
    @param initial_SOC
        The SOC value to start accumulating from.
    @param sign
        The value by which to multiply the current.
    @param capacity
        A scaling by which to convert from C to dimensionless SOC.
    @return
        An array of the same shape describing SOC in C.
    """

    complete_SOC = []
    for t, I in zip(timepoints, currents):
        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        try:
            I_int = np.array([0.5 * (I1 + I0)
                             for (I0, I1) in zip(I[:-1], I[1:])])
        except TypeError:
            I_int = np.atleast_1d(I)
        complete_SOC.append(np.append(
            initial_SOC,
            initial_SOC + sign * np.cumsum(I_int * Δt) / capacity
        ))
        initial_SOC = complete_SOC[-1][-1]
    return complete_SOC


def calculate_both_SOC_from_OCV(
    parameters,
    negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC,
    OCV
):
    """!@brief Calculates the SOC of both electrodes from their OCV.
    The SOCs are substitued in the given "parameters". The SOC of the
    cell as a whole gets returned in case it is needed.
    @param parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @param negative_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the negative electrode.
    @param positive_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the positive electrode.
    @param OCV
        The OCV for which the SOCs shall be calculated.
    @return
        The SOC of the cell as a whole.
    """

    positive_OCV = parameters["Positive electrode OCP [V]"]
    negative_OCV = parameters["Negative electrode OCP [V]"]
    cell_SOC = root_scalar(
        lambda cell_SOC: positive_OCV(positive_SOC_from_cell_SOC(cell_SOC))
        - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC)) - OCV,
        method='toms748', bracket=[0, 1], x0=0.5
    ).root
    positive_SOC = positive_SOC_from_cell_SOC(cell_SOC)
    negative_SOC = negative_SOC_from_cell_SOC(cell_SOC)
    parameters["Initial concentration in positive electrode [mol.m-3]"] = (
        positive_SOC
        * parameters["Maximum concentration in positive electrode [mol.m-3]"]
    )
    parameters["Initial concentration in negative electrode [mol.m-3]"] = (
        negative_SOC
        * parameters["Maximum concentration in negative electrode [mol.m-3]"]
    )

    return cell_SOC


def subtract_OCV_curve_from_cycles(
    dataset,
    parameters,
    starting_SOC=None,
    starting_OCV=None,
    electrode="positive",
    current_sign=0,
    voltage_sign=0,
):
    """!@brief Removes the OCV curve from a cycling measurement.
    @param dataset
        A Cycling_Information object of the measurement.
    @param parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @param starting_SOC
        The SOC at the beginning of the measurement. If not given, the
        OCV curve will be inverted to determine the initial SOC.
    @param starting_OCV
        The OCV at the beginning of the measurement. If not given and
        starting_SOC is also not given, the first entry of voltages is
        used for this. If not given, but starting_SOC is, the OCP
        function will be evaluated at starting_SOC to get the OCV.
    @param electrode
        "positive" (default) or "negative" for current sign correction and
        capacity calculation. "positive" adds SOC with positive current and
        vice versa. The sign corrections can be overwritten with '*_sign'.
    @param current_sign
        1 adds SOC, -1 subtracts it, 0 follows the default behaviour above.
    @param voltage_sign
        1 subtracts the OCP, -1 adds it, 0 follows the default behaviour above.
    @return
        2-tuple. First entry are the voltages minus the OCV as estimated for
        each data point. These are structured in exactly the same way as in the
        "dataset". Second entry are the electrode SOCs as counted in the data.
    """

    if current_sign == 0:
        if electrode == "positive":
            current_sign = 1
        else:
            current_sign = -1
    if voltage_sign == 0:
        if electrode == "positive":
            voltage_sign = 1
        else:
            voltage_sign = -1

    OCV_function = parameters[electrode.capitalize() + " electrode OCP [V]"]
    if type(OCV_function(0.5)) is Scalar:
        def OCV_function(s): return (
            parameters[electrode.capitalize() + " electrode OCP [V]"](s).value
        )
    if starting_SOC is None:
        starting_OCV = starting_OCV or dataset.voltages[0][0]
        initial_SOC = root_scalar(
            lambda s: OCV_function(s) - starting_OCV,
            method='toms748',
            bracket=[0, 1],
            x0=0.5
        ).root
    else:
        initial_SOC = starting_SOC
        starting_OCV = starting_OCV or OCV_function(initial_SOC)
    C = capacity(parameters, electrode)
    SOCs = calculate_SOC(
        dataset.timepoints,
        dataset.currents,
        initial_SOC=initial_SOC,
        sign=current_sign,
        capacity=C,
    )

    returned_SOCs = copy.deepcopy(SOCs)

    # Skip over segments where nothing happens anyway.
    SOC_skip_indices = []
    for i, current in enumerate(dataset.currents):
        if not any(current):  # Read "if current is perfectly 0".
            SOC_skip_indices.append(i)
    for i in reversed(SOC_skip_indices):
        SOCs.pop(i)
    open_circuit_voltages = []
    for soc in SOCs:
        open_circuit_voltages.append(list(OCV_function(soc)))
    if 0 in SOC_skip_indices:
        open_circuit_voltages.insert(
            0, [starting_OCV] * len(dataset.timepoints[0])
        )
        SOC_skip_indices.pop(0)
    for i in SOC_skip_indices:
        open_circuit_voltages.insert(
            i, [open_circuit_voltages[i - 1][-1]] * len(dataset.timepoints[i])
        )
    overpotentials = [
        [
            U_entry - voltage_sign * OCV_entry
            for U_entry, OCV_entry in zip(U, OCV)
        ]
        for U, OCV in zip(dataset.voltages, open_circuit_voltages)
    ]
    return overpotentials, returned_SOCs


def subtract_both_OCV_curves_from_cycles(
    dataset,
    parameters,
    negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC,
    starting_SOC=None,
    starting_OCV=None,
):
    """!@brief Removes the OCV curve from a single cycle.
    @param dataset
        A Cycling_Information object of the measurement.
    @param parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @param negative_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the negative electrode.
    @param positive_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the positive electrode.
    @param starting_SOC
        The SOC at the beginning of the measurement. If not given, the
        OCV curves will be inverted to determine the initial SOC.
    @param starting_OCV
        The OCV at the beginning of the measurement. If not given, the
        first entry of voltages is used for this.
    @return
        2-tuple. First entry are the voltages minus the OCV as estimated for
        each data point. These are structured in exactly the same way as in the
        "dataset". Second entry are the electrode SOCs as counted in the data.
    """

    positive_OCV = parameters["Positive electrode OCP [V]"]
    if type(positive_OCV(0.5)) is Scalar:
        def positive_OCV(s): return (
            parameters["Positive electrode OCP [V]"](s).value
        )
    negative_OCV = parameters["Negative electrode OCP [V]"]
    if type(negative_OCV(0.5)) is Scalar:
        def negative_OCV(s): return (
            parameters["Negative electrode OCP [V]"](s).value
        )
    if starting_SOC is None:
        starting_OCV = starting_OCV or dataset.voltages[0][0]
        cell_SOC_tracker = root_scalar(
            lambda cell_SOC: (
                positive_OCV(positive_SOC_from_cell_SOC(cell_SOC))
                - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC))
                - starting_OCV
            ),
            method='toms748',
            bracket=[0, 1],
            x0=0.5
        ).root
    else:
        cell_SOC_tracker = starting_SOC
        starting_OCV = starting_OCV or (
            positive_OCV(positive_SOC_from_cell_SOC(cell_SOC_tracker))
            - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC_tracker))
        )
    C = capacity(parameters)
    returned_SOCs = []
    voltages = []
    for t, I, U in zip(dataset.timepoints, dataset.currents, dataset.voltages):
        Δt = [t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])]
        try:
            I_int = [0.5 * (I0 + I1) for (I0, I1) in zip(I[:-1], I[1:])]
        except TypeError:
            I_int = np.atleast_1d(I)
        cell_SOC = cell_SOC_tracker + np.cumsum(
            np.array(I_int) * np.array(Δt)
        ) / C
        voltages.append([U[0] - starting_OCV] + list(
            np.array(U[1:])
            - np.array(positive_OCV(positive_SOC_from_cell_SOC(cell_SOC)))
            + np.array(negative_OCV(negative_SOC_from_cell_SOC(cell_SOC)))
        ))
        returned_SOCs.append(cell_SOC)
        cell_SOC_tracker = cell_SOC[-1]
        starting_OCV = (
            positive_OCV(positive_SOC_from_cell_SOC(cell_SOC_tracker))
            - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC_tracker))
        )
    return voltages, returned_SOCs


def laplace_transform(x, y, s):
    """!@brief Performs a basic laplace transformation.
    @param x
        The independent variable.
    @param y
        The dependent variable.
    @param s
        The (possibly complex) frequencies for which to perform the
        transform.
    @return
        The evaluation of the laplace transform at s.
    """

    x = np.array(x)
    y = np.array(y)
    Δx = x[1:] - x[:-1]
    x_int = 0.5 * (x[1:] + x[:-1])
    y_int = 0.5 * (y[1:] + y[:-1])
    return np.sum(y_int * np.exp(-(s[:, None] + 0.0j) * x_int) * Δx, axis=1)


def find_occurrences(sequence, value):
    """!@brief Gives indices in sequence where it is closest to value.
    @param sequence
        A list that represents a differentiable function.
    @param value
        The value that is searched for in "sequence". Also, crossings of
        consecutive values in "sequence" with "value" are searched for.
    @return
        A list of indices in "sequence" in ascending order where "value"
        or a close match for "value" was found.
    """
    root_finder = np.array(sequence) - value
    crossings = (
        np.sign(root_finder[1:]) - np.sign(root_finder[:-1])
    ).nonzero()[0]
    nearest_indices = np.where(
        np.abs(root_finder[crossings]) < np.abs(root_finder[crossings + 1]),
        crossings, crossings + 1
    )
    if len(nearest_indices) == 0:
        nearest_indices = (
            [0]
            if np.abs(root_finder[0]) < np.abs(root_finder[-1]) else
            [len(sequence) - 1]
        )
    return nearest_indices


def OCV_from_CC_CV(
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
    """!@brief Tries to extract the OCV curve from CC-CV cycling data.
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
    @return
        A 8-tuple consisting of the following:
        0: OCV_fits
            The fitted OCV curve parameters for each CC-CV cycle as
            returned by fitting_functions.fit_OCV.
        1: I_mean
            The currents assigned to each CC-CV cycle (without CV).
        2: C_charge
            The moved capacities during the charge segment(s). This is
            a list of the same length as charge, cv or discharge.
        3: U_charge
            The voltages during the charge segment(s). Length: same.
        4: C_discharge
            The moved capacities during the discharge segment(s).
            Length: same.
        5: U_discharge
            The voltages during the discharge segment(s). Length: same.
        6: C_evals
            Structurally the same as C_charge or C_discharge, this
            contains the moved capacities that were assigned to the mean
            voltages of charge and discharge cycle(s).
        7: U_means
            The mean voltages of each charge and discharge cycle.
    """

    I_mean = [0.0] * len(charge.timepoints)
    C_min = [0.0] * len(charge.timepoints)
    C_max = [0.0] * len(charge.timepoints)
    C_charge = []
    U_charge = []
    charge_splines = []
    corr = []
    C_discharge = []
    U_discharge = []
    discharge_splines = []
    OCV_fits = []
    C_evals = []
    U_means = []

    for i, (t, I, U) in enumerate(zip(cv.timepoints, cv.currents,
                                      cv.voltages)):
        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        I_int = np.array([0.5 * (I0 + I1) for (I0, I1) in zip(I[:-1], I[1:])])
        C = [0.0] + list(np.cumsum(Δt * I_int) / 3600.0)
        corr.append(np.abs(C[-1]))

    for i, (t, I, U) in enumerate(zip(charge.timepoints, charge.currents,
                                      charge.voltages)):
        I_mean[i] += 0.5 * np.mean(I)
        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        C = [0.0] + list(np.cumsum(Δt * np.array(I[1:])) / 3600.0)
        C = np.array(C) - np.min(C) + corr[i]
        C_charge.append(C)
        U_charge.append(U)
        C_min[i] = np.min(C)
        C_max[i] = np.max(C)
        charge_splines.append(smooth_fit(C, U,
                                         s=spline_smoothing))

    for i, (t, I, U) in enumerate(zip(discharge.timepoints, discharge.currents,
                                      discharge.voltages)):
        I_mean[i] -= 0.5 * np.mean(I)
        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        C = [0.0] + list(np.cumsum(Δt * np.array(I[1:])) / 3600.0)
        C = np.array(C) - np.min(C) - corr[i]
        U = np.array(U)[C >= 0]
        C = C[C >= 0]
        C_discharge.append(C)
        U_discharge.append(U)
        C_min[i] = np.max([C_min[i], np.min(C)])
        C_max[i] = np.min([C_max[i], np.max(C)])
        discharge_splines.append(smooth_fit(C, U,
                                            s=spline_smoothing))

    for i, (min, max, c, d) in enumerate(zip(C_min, C_max, charge_splines,
                                             discharge_splines)):
        C_eval = np.linspace(min, max, eval_points)
        C_evals.append(C_eval)
        U_mean = 0.5 * (c(C_eval) + d(C_eval))
        U_means.append(U_mean)
        # diff = 0.5 * (c(C_eval) - d(C_eval))
        dummy_SOC = np.linspace(0.0, 1.0, eval_points)
        OCV_model = fit_OCV(dummy_SOC, U_mean, N=phases)
        OCV_fits.append(OCV_model)
        fit_SOC = np.linspace(*spline_SOC_range, eval_points)
        spline_OCV = smooth_fit(fit_SOC, inverse_OCV_fit_function(fit_SOC,
                                *OCV_model.fit, inverted=True), spline_order,
                                s=spline_smoothing)
        if parameters_print:
            print("Parameters of OCV fit function (" + str(I_mean[i]) + " A):")
            print("SOC range of data: " + repr(OCV_model.SOC_range))
            print("E₀: [" + ", ".join([str(x) for x in OCV_model.E_0]) + "]")
            print("a: [" + ", ".join([str(x) for x in OCV_model.a]) + "]")
            print("Δx: [" + ", ".join([str(x) for x in OCV_model.Δx]) + "]")
        if spline_order > 0:
            if parameters_print:
                print("Knots of interpolating spline:")
                print(spline_OCV.get_knots())
                print("Coefficients of this spline:")
                print(spline_OCV.get_coeffs())
            if spline_print is not None:
                print(verbose_spline_parameterization(
                    spline_OCV.get_coeffs(), spline_OCV.get_knots(),
                    spline_order, function_name=name, format=spline_print,
                    derivatives=1
                ))

    return (OCV_fits, I_mean, C_charge, U_charge, C_discharge, U_discharge,
            C_evals, U_means)


def calculate_desired_voltage(
        solution,
        t_eval,
        voltage_scale,
        overpotential,
        three_electrode=None,
        dimensionless_reference_electrode_location=0.5,
        parameters={},
):
    """!
    @param solution
        The pybamm.Solution object from which to calculate the voltage.
    @param t_eval
        The times at which to evaluate the "solution".
    @param voltage_scale
        The returned voltage gets divided by this value. For example,
        1e-3 would produce a plot in [mV].
    @param overpotential
        If True, only the overpotential of "solutions" gets plotted.
        Otherwise, the cell voltage (OCV + overpotential) is plotted.
    @param three_electrode
        With None, does nothing (i.e., cell potentials are used). If
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
        The array of the specified voltages over time.
    """

    if three_electrode:
        if parameters is None:
            raise ValueError(
                "To calculate the potentials for a three-electrode setup, "
                "the parameter dictionary of the simulation is needed. "
                "Give it as the keyword argument 'parameters'."
            )
        if three_electrode not in ["positive", "negative"]:
            raise ValueError(
                "'three_electrode' has to be either None, 'positive', or "
                "'negative'."
            )
        # When this issue gets completed, use it instead:
        # https://github.com/pybamm-team/PyBaMM/issues/2188
        L_n = parameters["Negative electrode thickness [m]"]
        L_s = parameters["Separator thickness [m]"]
        L_p = parameters["Positive electrode thickness [m]"]
        dimensional_location = (
            L_n + dimensionless_reference_electrode_location * L_s
        )
        x_working_electrode = (
            0 if three_electrode == "negative" else L_n + L_s + L_p
        )
        reference_electrode_potential = (
            solution["Electrolyte potential [V]"](
                t_eval, x=dimensional_location
            )
        )
        if overpotential:
            ocp = np.array([
                parameters[
                    three_electrode.capitalize() + " electrode OCP [V]"
                ](soc) for soc in solution[
                    "Average "
                    + three_electrode
                    + " particle concentration"
                ](t_eval)
            ])
            U = (
                solution[
                    three_electrode.capitalize()
                    + " electrode potential [V]"
                ](t_eval, x=x_working_electrode)
                - ocp
                - reference_electrode_potential
            ) / voltage_scale
        else:
            U = (
                solution[
                    three_electrode.capitalize()
                    + " electrode potential [V]"
                ](t_eval, x=x_working_electrode)
                - reference_electrode_potential
            ) / voltage_scale
    else:
        if overpotential:
            U = (
                solution["Voltage [V]"](t_eval)
                - solution["Bulk open-circuit voltage [V]"](t_eval)
            ) / voltage_scale
        else:
            U = solution["Voltage [V]"](t_eval) / voltage_scale
    return U


def solve_all_parameter_combinations(
    model,
    t_eval,
    parameters,
    parameters_to_try,
    submesh_types,
    var_pts,
    spatial_methods,
    full_factorial=True,
    **kwargs
):
    """!
    @param model
        The PyBaMM battery model that is to be solved.
    @param t_eval
        The timepoints in s at which this model is to be solved.
    @param parameters
        The model parameters as a dictionary.
    @param parameters_to_try
        A dictionary with the names of the model parameters as keys and
        lists of the values that are to be tried out for them as values.
    @param submesh_types
        The submeshes for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @param var_pts
        The number of discretization points. See
        solversetup.spectral_mesh_pts_and_method.
    @param spatial_methods
        The spatial methods for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @param full_factorial
        If True, all parameter combinations are tried out. If False,
        only each parameter is varied with the others staying fixed.
    @param kwargs
        The optional parameters for solversetup.solver_setup. See there.
    @return
        A 2-tuple with the model solution for parameters as first entry.
        The second entry mimics parameters_to_try with each entry in
        their lists replaced by the model solution for the corresponding
        parameter substitution. The second entry has one additional key
        "all parameters", where all parameters_to_try combinations are
        the value.
    """

    solutions = {}
    errorbars = {}
    free_parameters = list(parameters_to_try.keys())
    solver = solver_setup(
        model, parameters, submesh_types, var_pts, spatial_methods,
        free_parameters=free_parameters, **kwargs
    )
    input_parameters = {name: parameters[name] for name in free_parameters}
    solutions[model.name] = solver(t_eval, inputs=input_parameters)

    errorbars = {name: [
        solver(t_eval, inputs=p)
        for p in combine_parameters_to_try(input_parameters, {name: limits})[0]
    ] for name, limits in parameters_to_try.items()}

    errorbars["all parameters"] = [solutions[model.name]]
    for e in errorbars.values():
        errorbars["all parameters"].extend(e)
    if full_factorial:
        errorbars["all parameters"].extend([
            solver(t_eval, inputs=p)
            for p in combine_parameters_to_try(
                input_parameters, parameters_to_try
            )[0]
        ])

    return (solutions, errorbars)


def prepare_parameter_combinations(
    parameters,
    parameters_to_try,
    covariance,
    order_of_parameter_names,
    transform_parameters,
    confidence
):
    """!@brief Calculates all permutations of the parameter boundaries.
    @param parameters
        The model parameters as a dictionary.
    @param parameters_to_try
        A dictionary with the names of the model parameters as keys and
        lists of the values that are to be tried out for them as values.
        Mutually exclusive to "covariance".
    @param covariance
        A covariance matrix describing an estimation result of model
        parameters. Will be used to calculate parameters to try that
        together approximate the confidence ellipsoid. This confidence
        ellipsoid will be centered on "parameters".
        Mutually exclusive to "parameters_to_try".
    @param order_of_parameter_names
        A list of names from "parameters" that correspond to the order
        these parameters appear in the rows and columns of "covariance".
        Only needed when "covariance" is set.
    @param transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @param confidence
        The confidence within the ellipsoid. Defaults to 0.95, i.e., the
        95% confidence ellipsoid.
    @return
        A 2-tuple with the individual parameter variations and then all
        permutations of them.
    """

    if not ((parameters_to_try is None) ^ (covariance is None)):
        raise ValueError(
            "Please set either 'parameters_to_try' or 'covariance'."
        )

    if parameters_to_try is not None:
        individual_bounds = {
            name: combine_parameters_to_try(
                parameters, {name: limits}
            )[0]
            for name, limits in parameters_to_try.items()
        }
        combinations = combine_parameters_to_try(
            parameters, parameters_to_try
        )[0]
    elif covariance is not None:
        individual_bounds = {
            "confidence semiaxes": approximate_confidence_ellipsoid(
                parameters,
                list(order_of_parameter_names),
                covariance,
                transform_parameters=transform_parameters,
                refinement=False,
                confidence=confidence
            )[0]
        }
        combinations = approximate_confidence_ellipsoid(
            parameters,
            list(order_of_parameter_names),
            covariance,
            transform_parameters=transform_parameters,
            refinement=True,
            confidence=confidence
        )[0][len(individual_bounds):]

    return individual_bounds, combinations


def parallel_simulator_with_setup(
    model,
    current_input,
    parameters,
    submesh_types,
    var_pts,
    spatial_methods,
    calc_esoh,
    inputs,
    t_eval,
    voltage_scale,
    overpotential,
    three_electrode,
    dimensionless_reference_electrode_location,
    kwargs
):
    par_var = copy.deepcopy(parameters)
    par_var.update(inputs)
    solver, callback = simulation_setup(
        model, current_input, par_var,
        submesh_types, var_pts, spatial_methods,
        **kwargs
    )
    solution = solver(
        check_model=False, calc_esoh=calc_esoh, callbacks=callback
    )
    variable = calculate_desired_voltage(
        solution,
        t_eval,
        voltage_scale,
        overpotential,
        three_electrode,
        dimensionless_reference_electrode_location,
        parameters
    )
    return variable


def simulate_all_parameter_combinations(
    model,
    current_input,
    submesh_types,
    var_pts,
    spatial_methods,
    parameters,
    parameters_to_try=None,
    covariance=None,
    order_of_parameter_names=None,
    additional_input_parameters=[],
    transform_parameters={},
    confidence=0.95,
    full_factorial=True,
    calc_esoh=False,
    voltage_scale=1.0,
    overpotential=False,
    three_electrode=None,
    dimensionless_reference_electrode_location=0.5,
    **kwargs
):
    """!
    @param model
        The PyBaMM battery model that is to be solved.
    @param current_input
        The list of battery operation conditions. See pybamm.Simulation.
    @param submesh_types
        The submeshes for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @param var_pts
        The number of discretization points. See
        solversetup.spectral_mesh_pts_and_method.
    @param spatial_methods
        The spatial methods for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @param parameters
        The model parameters as a dictionary.
    @param parameters_to_try
        A dictionary with the names of the model parameters as keys and
        lists of the values that are to be tried out for them as values.
        Mutually exclusive to "covariance".
    @param covariance
        A covariance matrix describing an estimation result of model
        parameters. Will be used to calculate parameters to try that
        together approximate the confidence ellipsoid. This confidence
        ellipsoid will be centered on "parameters".
        Mutually exclusive to "parameters_to_try".
    @param order_of_parameter_names
        A list of names from "parameters" that correspond to the order
        these parameters appear in the rows and columns of "covariance".
        Only needed when "covariance" is set.
    @param additional_input_parameters
        A list of the parameter names that are changed by any of the
        variable parameters, if "parameters" is a SubstitutionDict.
    @param transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @param confidence
        The confidence within the ellipsoid. Defaults to 0.95, i.e., the
        95% confidence ellipsoid.
    @param full_factorial
        When "parameters_to_try" is set:
         - If True, all parameter combinations are tried out. If False,
           only each parameter is varied with the others staying fixed.
        When "covariance" is set:
         - If False, only the points on the semiaxes of the confidence
           ellipsoid constitute the parameters to try. If True, the
           centres of the faces of the polytope of these points get
           added to the parameters to try, projected onto the surface
           of the confidence ellipsoid.
    @param calc_esoh
        Passed on to pybamm.Simulator, see there.
    @param kwargs
        The optional parameters for solversetup.solver_setup. See there.
    @return
        A 2-tuple with the model solution for parameters as first entry.
        The second entry mimics parameters_to_try with each entry in
        their lists replaced by the model solution for the corresponding
        parameter substitution. The second entry has one additional key
        "all parameters", where all parameters_to_try combinations are
        the value.
    """

    individual_bounds, combinations = prepare_parameter_combinations(
        parameters,
        parameters_to_try,
        covariance,
        order_of_parameter_names,
        transform_parameters,
        confidence
    )

    if parameters_to_try is not None:
        if isinstance(parameters, SubstitutionDict):
            free_parameters = (
                list(parameters_to_try.keys())
                + list(additional_input_parameters)
            )
        else:
            free_parameters = list(parameters_to_try.keys())
    elif covariance is not None:
        if isinstance(parameters, SubstitutionDict):
            free_parameters = (
                list(order_of_parameter_names)
                + list(additional_input_parameters)
            )
        else:
            free_parameters = list(order_of_parameter_names)

    individual_bounds = {
        name: [
            {
                free_name: parameters_variation[free_name]
                for free_name in free_parameters
            }
            for parameters_variation in bounds
        ]
        for name, bounds in individual_bounds.items()
    }
    combinations = [
        {
            free_name: parameters_variation[free_name]
            for free_name in free_parameters
        }
        for parameters_variation in combinations
    ]

    solutions = {}
    errorbars = {}

    solver, callback = simulation_setup(
        model, current_input, parameters,
        submesh_types, var_pts, spatial_methods,
        **kwargs
    )
    solutions[model.name] = solver(
        check_model=False, calc_esoh=calc_esoh, callbacks=callback
    )
    t_eval = solutions[model.name].t

    parallel_arguments = []
    indices_of_individual_bounds = {}
    index_counter = 0
    for name in individual_bounds:
        new_parallel_arguments = [
            (
                copy.deepcopy(model),
                current_input,
                copy.deepcopy(parameters),
                submesh_types,
                var_pts,
                spatial_methods,
                calc_esoh,
                ib,
                t_eval,
                voltage_scale,
                overpotential,
                three_electrode,
                dimensionless_reference_electrode_location,
                kwargs,
            )
            for ib in individual_bounds[name]
        ]
        indices_of_individual_bounds[name] = [i for i in range(
            index_counter, index_counter + len(new_parallel_arguments)
        )]
        index_counter += len(new_parallel_arguments)
        parallel_arguments.extend(new_parallel_arguments)
    with Pool() as p:
        all_errorbars = p.starmap(
            parallel_simulator_with_setup, parallel_arguments
        )
    for name in individual_bounds:
        errorbars[name] = [
            all_errorbars[i] for i in indices_of_individual_bounds[name]
        ]

    errorbars["all parameters"] = [solutions[model.name]]
    for e in errorbars.values():
        errorbars["all parameters"].extend(e)
    if full_factorial:
        parallel_arguments = [
            (
                copy.deepcopy(model),
                current_input,
                copy.deepcopy(parameters),
                submesh_types,
                var_pts,
                spatial_methods,
                calc_esoh,
                c,
                t_eval,
                voltage_scale,
                overpotential,
                three_electrode,
                dimensionless_reference_electrode_location,
                kwargs,
            )
            for c in combinations
        ]
        with Pool() as p:
            errorbars["all parameters"].extend(p.starmap(
                parallel_simulator_with_setup, parallel_arguments
            ))

    return (solutions, errorbars)
