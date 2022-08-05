"""!@package utility.preprocessing
Contains frequently used workflows in dataset preprocessing.

The functions herein are a collection of simple, but frequent,
transformations of arrays of raw measurement data.
"""

import numpy as np
import copy
from collections.abc import MutableMapping
from itertools import chain
from numbers import Number
from os import linesep
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import chi2, norm
from models.solversetup import solver_setup, simulation_setup
from utility.fitting_functions import (
    fit_OCV, OCV_fit_function, inverse_OCV_fit_function, smooth_fit,
    verbose_spline_parameterization
)


class SubstitutionDict(MutableMapping):
    """!@brief A dictionary with some automatic substitutions.

    "substitutions" is a dictionary that extends "storage" with
    automatic substitution rules depending on its key types:
     - string, which serves the value of "storage" at that value.
     - callable which takes one parameter, which will get passed its
       SubstitutionDict instance and serves its return value.
     - any other type, which serves the value as-is.
    """

    def __init__(self, storage, substitutions={}):
        self._storage = storage
        self._substitutions = substitutions

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
        self._storage[key] = value

    def __str__(self):
        return '{' + ', '.join(
            [str(k) + ': ' + str(v) for k, v in self.items()]
        ) + '}'

    def __repr__(self):
        # return {' + linesep + ('\t' + linesep).join(
        #     [repr(k) + ': ' + repr(v) for k, v in self._storage.items()]
        # ) + linesep + '}'
        return self.__str__()


def fix_parameters(parameters_to_be_fixed):
    """!@brief Returns a function which sets some parameters in advance.

    @par parameters_to_be_fixed
        These parameters will at least be a
        part of the dictionary that the returned function returns.
    @return
        The function which adds additional parameters to a
        dictionary or replaces existing parameters with the new ones.
    """

    def return_all_parameters(free_parameters):
        """!@brief Adds the 'free_parameters' to a pool of parameters.

        @par free_parameters
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

    @par parameters
        The base full parameter set as a dictionary.
    @par parameters_to_try_dict
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
    **kwargs
):
    """!@brief Calculate means and standard deviations.

    Please note that standard deviations translate differently into
    confidence regions in different dimensions. For the confidence
    region, use "approximate_confidence_ellipsoid".

    @par mean
        The mean of the uncertain parameters as a dictionary.
    @par covariance
        The covariance of the uncertain parameters as a two-dimensional
        numpy array.
    @par free_parameters_names
        The names of the parameters that are uncertain as a list. This
        parameter maps the order of parameters in "covariance".
    @par transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @par **kwargs
        Keyword arguments for scipy.quad, which is used to numerically
        calculate mean and variance.
    @return
        A 2-tuple with two dictionaries. Their keys are the free
        parameters' names as keys and their values are those parameters'
        means and standard deviations.
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
        print(mean_actual, variance_actual, mean_internal, variance_internal)

    return (means, standard_deviations)


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

    @par parameters
        The base full parameter set as a dictionary.
    @par free_parameters_names
        The names of the parameters that are uncertain as a list. This
        parameter has to match the order of parameters in "covariance".
    @par covariance
        The covariance of the uncertain parameters as a two-dimensional
        numpy array.
    @par mean
        The mean of the uncertain parameters as a dictionary. If not
        set, the values from 'parameters' will be used.
    @par transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @par confidence
        The confidence within the ellipsoid. Defaults to 0.95, i.e., the
        95% confidence ellipsoid.
    @par refinement
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
        # print(fix_parameters(parameters)(node_parameters))
    return (parameters_list, ellipsoid)


def capacity(parameters):
    """!@brief Convenience function for calculating the capacity.

    @par parameters
        A parameter file as defined by models.standard_parameters.
    @return
        The capacity of the parameterized battery in C.
    """

    return (
        (1 - parameters["Positive electrode porosity"])
        * parameters["Positive electrode thickness [m]"]
        * parameters["Current collector perpendicular area [m2]"]
        * parameters["Maximum concentration in positive electrode [mol.m-3]"]
        * 96485.33212
    )


def calculate_SOC(timepoints, currents):
    """!@brief Transforms applied current over time into SOC.

    @par timepoints
        Array of the timepoint segments.
    @par currents
        Array of the current segments.
    @return
        An array of the same shape describing SOC in C.
    """

    complete_SOC = []
    SOC_tracker = 0.0
    for t, I in zip(timepoints, currents):
        Δt = np.array([t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])])
        I_int = np.array([0.5 * (I1 + I0) for (I0, I1) in zip(I[:-1], I[1:])])
        SOC = np.cumsum(Δt * I_int)
        complete_SOC.append([SOC_tracker] + list(SOC_tracker + SOC))
        if len(t) > 1:
            SOC_tracker = SOC_tracker + SOC[-1]
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

    @par parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @par negative_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the negative electrode.
    @par positive_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the positive electrode.
    @par OCV
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


def subtract_OCV_curve(
    voltages,
    timepoints,
    currents,
    parameters,
    OCV_fit,
    starting_OCV=-1,
    electrode="cathode"
):
    """!@brief Removes the OCV curve from a single cycle.

    @par voltages
        An array of the voltage measurement points.
    @par timepoints
        An array of the timepoints at which voltages were measured.
    @par currents
        An array of the corresponding currents. A number works as well.
    @par parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @par OCV_fit
        The fit parameters of an OCV curve as returned by
        fitting_functions.fit_OCV(). If fit_OCV_and_SOC_range() was
        used, crop the first two entries for SOC start and end.
    @par starting_OCV
        The OCV at the beginning of the measurement. If not given, the
        first entry of voltages is used for this.
    @par electrode
        "cathode" (default) or "anode" for sign correction.
    @return
        voltages minus the OCV as estimated for each data point.
    """

    starting_OCV = starting_OCV or voltages[0]
    SOC_tracker = 1 - OCV_fit_function(starting_OCV, *OCV_fit)
    C = capacity(parameters)
    sign = 1.0 if electrode == "cathode" else -1.0
    Δt = [t1 - t0 for (t0, t1) in zip(timepoints[:-1], timepoints[1:])]
    try:
        I_int = [
            0.5 * (I0 + I1) for (I0, I1) in zip(currents[:-1], currents[1:])
        ]
    except TypeError:
        I_int = np.atleast_1d(currents)
    SOC = SOC_tracker + sign * np.cumsum(np.array(I_int) * np.array(Δt)) / C
    return [voltages[0] - starting_OCV] + list(
        np.array(voltages[1:])
        - np.array(inverse_OCV_fit_function(SOC, *OCV_fit))
    )


def subtract_both_OCV_curves(
    voltages,
    timepoints,
    currents,
    parameters,
    negative_OCV,
    positive_OCV,
    negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC,
    starting_OCV=None
):
    """!@brief Removes the OCV curve from a single cycle.

    @par voltages
        An array of the voltage measurement points.
    @par timepoints
        An array of the timepoints at which voltages were measured.
    @par currents
        An array of the corresponding currents. A number works as well.
    @par parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @par negative_OCV
        A function that takes the SOC of the negative electrode and
        returns its OCV.
    @par positive_OCV
        A function that takes the SOC of the positive electrode and
        returns its OCV.
    @par negative_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the negative electrode.
    @par positive_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the positive electrode.
    @par starting_OCV
        The OCV at the beginning of the measurement. If not given, the
        first entry of voltages is used for this.
    @return
        voltages minus the OCVs as estimated for each data point.
    """

    starting_OCV = starting_OCV or voltages[0]
    cell_SOC_tracker = root_scalar(
        lambda cell_SOC: positive_OCV(positive_SOC_from_cell_SOC(cell_SOC))
        - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC)) - starting_OCV,
        method='toms748', bracket=[0, 1], x0=0.5
    ).root
    C = capacity(parameters)
    Δt = [t1 - t0 for (t0, t1) in zip(timepoints[:-1], timepoints[1:])]
    try:
        I_int = [
            0.5 * (I0 + I1) for (I0, I1) in zip(currents[:-1], currents[1:])
        ]
    except TypeError:
        I_int = np.atleast_1d(currents)
    cell_SOC = cell_SOC_tracker + np.cumsum(np.array(I_int) * np.array(Δt)) / C
    return [voltages[0] - starting_OCV] + list(
        np.array(voltages[1:])
        - np.array(positive_OCV(positive_SOC_from_cell_SOC(cell_SOC)))
        + np.array(negative_OCV(negative_SOC_from_cell_SOC(cell_SOC)))
    )


def subtract_OCV_curve_from_cycles(
    dataset,
    parameters,
    OCV_fit,
    starting_OCV=None,
    electrode="cathode"
):
    """!@brief Removes the OCV curve from a cycling measurement.

    @par dataset
        A Cycling_Information object of the measurement.
    @par parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @par OCV_fit
        The fit parameters of an OCV curve as returned by
        fitting_functions.fit_OCV(). If fit_OCV_and_SOC_range() was
        used, crop the first two entries for SOC start and end.
    @par starting_OCV
        The OCV at the beginning of the measurement. If not given, the
        first entry of voltages is used for this.
    @par electrode
        "cathode" (default) or "anode" for sign correction.
    @return
        voltages minus the OCV as estimated for each data point. These
        are structured in exactly the same way as in the "dataset".
    """

    starting_OCV = starting_OCV or dataset.voltages[0][0]
    SOC_tracker = 1 - OCV_fit_function(starting_OCV, *OCV_fit)
    C = capacity(parameters)
    sign = 1.0 if electrode == "cathode" else -1.0
    voltages = []
    for t, I, U in zip(dataset.timepoints, dataset.currents, dataset.voltages):
        Δt = [t1 - t0 for (t0, t1) in zip(t[:-1], t[1:])]
        try:
            I_int = [0.5 * (I0 + I1) for (I0, I1) in zip(I[:-1], I[1:])]
        except TypeError:
            I_int = np.atleast_1d(I)
        SOC = SOC_tracker + sign * np.cumsum(
            np.array(I_int) * np.array(Δt)
        ) / C
        voltages.append([U[0] - starting_OCV] + list(np.array(U[1:])
                        - np.array(inverse_OCV_fit_function(SOC, *OCV_fit))))
        SOC_tracker = SOC[-1]
        starting_OCV = inverse_OCV_fit_function(SOC_tracker, *OCV_fit)
    return voltages


def subtract_both_OCV_curves_from_cycles(
    dataset,
    parameters,
    negative_SOC_from_cell_SOC,
    positive_SOC_from_cell_SOC,
    starting_OCV=None
):
    """!@brief Removes the OCV curve from a single cycle.

    @par dataset
        A Cycling_Information object of the measurement.
    @par parameters
        The parameters of the battery as used for the PyBaMM simulations
        (see models.standard_parameters).
    @par negative_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the negative electrode.
    @par positive_SOC_from_cell_SOC
        A function that takes the SOC of the cell and returns the SOC of
        the positive electrode.
    @par starting_OCV
        The OCV at the beginning of the measurement. If not given, the
        first entry of voltages is used for this.
    @return
        voltages minus the OCVs as estimated for each data point. These
        are structured in exactly the same way as in the "dataset".
    """

    positive_OCV = parameters["Positive electrode OCP [V]"]
    negative_OCV = parameters["Negative electrode OCP [V]"]
    starting_OCV = starting_OCV or dataset.voltages[0][0]
    cell_SOC_tracker = root_scalar(
        lambda cell_SOC: positive_OCV(positive_SOC_from_cell_SOC(cell_SOC))
        - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC)) - starting_OCV,
        method='toms748', bracket=[0, 1], x0=0.5
    ).root
    C = capacity(parameters)
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
        cell_SOC_tracker = cell_SOC[-1]
        starting_OCV = (
            positive_OCV(positive_SOC_from_cell_SOC(cell_SOC_tracker))
            - negative_OCV(negative_SOC_from_cell_SOC(cell_SOC_tracker))
        )
    return voltages


def laplace_transform(x, y, s):
    """!@brief Performs a basic laplace transformation.

    @par x
        The independent variable.
    @par y
        The dependent variable.
    @par s
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

    @par sequence
        A list that represents a differentiable function.
    @par value
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

    @par charge
        A Cycling_Information object containing the constant charge
        cycle(s). If more than one CC-CV-cycle shall be analyzed, please
        make sure that the order of this, cv and discharge align.
    @par cv
        A Cycling_Information object containing the constant voltage
        part between charge and discharge cycle(s).
    @par discharge
        A Cycling_Information object containing the constant discharge
        cycle(s). These occur after each cv cycle.
    @par name
        Name of the material for which the CC-CV-cycling was measured.
    @par phases
        Number of phases in the fitting_functions.OCV_fit_function as an
        int. The higher it is, the more (over-)fitted the model becomes.
    @par eval_points
        The number of points for plotting of the OCV curves.
    @par spline_SOC_range
        2-tuple giving the SOC range in which the inverted
        fitting_functions.OCV_fit_function will be interpolated by a
        smoothing spline. Outside of this range the spline is used for
        extrapolation. Use this to fit the SOC range of interest more
        precisely, since a fit of the whole range usually fails due to
        the singularities at SOC 0 and 1. Please note that this range
        considers the 0-1-range in which the given SOC lies and not the
        linear transformation of it from the fitting process.
    @par spline_order
        Order of this smoothing spline. If it is set to 0, only the
        fitting_functions.OCV_fit_function is calculated and plotted.
    @par spline_smoothing
        Smoothing factor for this smoothing spline. Default: 2e-3. Lower
        numbers give more precision, while higher numbers give a simpler
        spline that smoothes over steep steps in the fitted OCV curve.
    @par spline_print
        If set to either 'python' or 'matlab', a string representation
        of the smoothing spline is printed in the respective format.
    @par parameters_print
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
                                         smoothing_factor=spline_smoothing))

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
                                            smoothing_factor=spline_smoothing))

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
                                smoothing_factor=spline_smoothing)
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
                    spline_order, function_name=name, format=spline_print
                ))

    return (OCV_fits, I_mean, C_charge, U_charge, C_discharge, U_discharge,
            C_evals, U_means)


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
    """
    @par model
        The PyBaMM battery model that is to be solved.
    @par t_eval
        The timepoints in s at which this model is to be solved.
    @par parameters
        The model parameters as a dictionary.
    @par parameters_to_try
        A dictionary with the names of the model parameters as keys and
        lists of the values that are to be tried out for them as values.
    @par submesh_types
        The submeshes for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @par var_pts
        The number of discretization points. See
        solversetup.spectral_mesh_pts_and_method.
    @par spatial_methods
        The spatial methods for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @par full_factorial
        If True, all parameter combinations are tried out. If False,
        only each parameter is varied with the others staying fixed.
    @par kwargs
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


def simulate_all_parameter_combinations(
    model,
    input,
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
    **kwargs
):
    """
    @par model
        The PyBaMM battery model that is to be solved.
    @par input
        The list of battery operation conditions. See pybamm.Simulation.
    @par submesh_types
        The submeshes for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @par var_pts
        The number of discretization points. See
        solversetup.spectral_mesh_pts_and_method.
    @par spatial_methods
        The spatial methods for discretization. See
        solversetup.spectral_mesh_pts_and_method.
    @par parameters
        The model parameters as a dictionary.
    @par parameters_to_try
        A dictionary with the names of the model parameters as keys and
        lists of the values that are to be tried out for them as values.
        Mutually exclusive to "covariance".
    @par covariance
        A covariance matrix describing an estimation result of model
        parameters. Will be used to calculate parameters to try that
        together approximate the confidence ellipsoid. This confidence
        ellipsoid will be centered on "parameters".
        Mutually exclusive to "parameters_to_try".
    @par order_of_parameter_names
        A list of names from "parameters" that correspond to the order
        these parameters appear in the rows and columns of "covariance".
        Only needed when "covariance" is set.
    @par additional_input_parameters
        A list of the parameter names that are changed by any of the
        variable parameters, if "parameters" is a SubstitutionDict.
    @par transform_parameters
        Optional transformations between the parameter space that is
        used for searching for optimal parameters and the model
        parameters. Any missing free parameter is not transformed.
        The values are 2-tuples. The first entry is a function taking
        the search space parameter and returning the model parameter.
        The second entry is the inverse function.
        For convenience, any value may also be one of the following:
         - 'none' => (identity, identity)
         - 'log' => (exp, log)
    @par confidence
        The confidence within the ellipsoid. Defaults to 0.95, i.e., the
        95% confidence ellipsoid.
    @par full_factorial
        When "parameters_to_try" is set:
         - If True, all parameter combinations are tried out. If False,
           only each parameter is varied with the others staying fixed.
        When "covariance" is set:
         - If False, only the points on the semiaxes of the confidence
           ellipsoid constitute the parameters to try. If True, the
           centres of the faces of the polytope of these points get
           added to the parameters to try, projected onto the surface
           of the confidence ellipsoid.
    @par calc_esoh
        Passed on to pybamm.Simulator, see there.
    @par kwargs
        The optional parameters for solversetup.solver_setup. See there.
    @return
        A 2-tuple with the model solution for parameters as first entry.
        The second entry mimics parameters_to_try with each entry in
        their lists replaced by the model solution for the corresponding
        parameter substitution. The second entry has one additional key
        "all parameters", where all parameters_to_try combinations are
        the value.
    """

    if not ((parameters_to_try is None) ^ (covariance is None)):
        raise ValueError(
            "Please set either 'parameters_to_try' or 'covariance'."
        )

    if parameters_to_try is not None:
        free_parameters = list(parameters_to_try.keys())
        input_parameters = {
            name: parameters[name]
            for name in free_parameters + additional_input_parameters
        }
        if isinstance(parameters, SubstitutionDict):
            substituted_inputs = {
                name: parameters._substitutions[name]
                for name in additional_input_parameters
            }
            input_parameters = SubstitutionDict(
                input_parameters, substituted_inputs
            )
        individual_bounds = {
            name: combine_parameters_to_try(
                input_parameters, {name: limits}
            )[0]
            for name, limits in parameters_to_try.items()
        }
        combinations = combine_parameters_to_try(
            input_parameters, parameters_to_try
        )[0]
    elif covariance is not None:
        free_parameters = order_of_parameter_names
        input_parameters = {
            name: parameters[name]
            for name in free_parameters + additional_input_parameters
        }
        if isinstance(parameters, SubstitutionDict):
            substituted_inputs = {
                name: parameters._substitutions[name]
                for name in additional_input_parameters
            }
            input_parameters = SubstitutionDict(
                input_parameters, substituted_inputs
            )
        individual_bounds = approximate_confidence_ellipsoid(
            input_parameters,
            free_parameters,
            covariance,
            transform_parameters=transform_parameters,
            refinement=False,
            confidence=confidence
        )[0]
        combinations = approximate_confidence_ellipsoid(
            input_parameters,
            free_parameters,
            covariance,
            transform_parameters=transform_parameters,
            refinement=True,
            confidence=confidence
        )[0][len(individual_bounds):]

    solutions = {}
    errorbars = {}

    solver = simulation_setup(
        model, input, parameters,
        submesh_types, var_pts, spatial_methods,
        free_parameters=free_parameters + additional_input_parameters, **kwargs
    )

    solutions[model.name] = solver.solve(
        check_model=False, calc_esoh=calc_esoh, inputs=input_parameters
    )

    if parameters_to_try is not None:
        errorbars = {name: [
            solver.solve(check_model=False, calc_esoh=calc_esoh, inputs=p)
            for p in individual_bounds[name]
        ] for name in individual_bounds}
    elif covariance is not None:
        errorbars = {"confidence semiaxes": [
            solver.solve(check_model=False, calc_esoh=calc_esoh, inputs=p)
            for p in individual_bounds
        ]}

    errorbars["all parameters"] = [solutions[model.name]]
    for e in errorbars.values():
        errorbars["all parameters"].extend(e)
    if full_factorial:
        errorbars["all parameters"].extend([
            solver.solve(check_model=False, calc_esoh=calc_esoh, inputs=p)
            for p in combinations
        ])

    return (solutions, errorbars)
