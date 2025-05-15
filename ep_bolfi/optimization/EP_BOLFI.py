"""
This file contains functions to perform Expectation Propagation on
simulator models using BOLFI (Bayesian Optimization for Likelihood-
Free Inference).
"""

from copy import deepcopy
import elfi
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


class NDArrayEncoder(json.JSONEncoder):
    """Use with the JSON library to store NumPy arrays."""

    def default(self, item):
        """
        Unpacks all NumPy arrays it finds into nested lists.

        Since this is called recursively, it only needs to check one
        level.

        :param item:
            Any object, which gets treated with the JSON library default
            if it is not a NumPy array. Else, it gets turned to a list.
        :returns:
            A JSON encoded version of *item*.
        """
        if isinstance(item, np.ndarray):
            return item.tolist()
        return json.JSONEncoder.default(self, item)


def combine_parameters_to_try(parameters, parameters_to_try_dict):
    """
    Give every combination as full parameter sets.

    :param parameters:
        The base full parameter set as a dictionary.
    :param parameters_to_try_dict:
        The keys of this dictionary correspond to the *parameters*' keys
        where different values are to be inserted. These are given by
        the tuples which are the values of this dictionary.
    :returns:
        A 2-tuple where the first item is the list of all parameter set
        combinations and the second the list of the combinations only.
    """

    # Fetch the parameters to try.
    parameters_to_try = []
    for key, value in parameters_to_try_dict.items():
        parameters_to_try.append((key, value))

    def recursive_combination(last_index):
        """Recursively get every combination of parameters to try."""
        return_list = []
        if last_index == 0:
            key = parameters_to_try[0][0]
            for value in parameters_to_try[0][1]:
                return_list.append({key: value})
        elif last_index > 0:
            key = parameters_to_try[last_index][0]
            list_of_combinations = recursive_combination(last_index - 1)
            for value in parameters_to_try[last_index][1]:
                for combination in list_of_combinations:
                    return_dict = {key: value}
                    return_dict.update(combination)
                    return_list.append(return_dict)
        return return_list

    combinations = recursive_combination(len(parameters_to_try) - 1)
    parameters_list = []
    for combination in combinations:
        parameters_list.append(fix_parameters(parameters)(combination))
    return (parameters_list, combinations)


def fix_parameters(parameters_to_be_fixed):
    """
    Returns a function which sets some parameters in advance.

    :param parameters_to_be_fixed:
        These parameters will at least be a part of the dictionary that
        the returned function returns.
    :returns:
        The function which adds additional parameters to a dictionary or
        replaces existing parameters with the new ones.
    """

    def return_all_parameters(free_parameters):
        """
        Adds the *free_parameters* to a pool of parameters.

        :param free_parameters:
            A dictionary which gets added to the fixed pool of
            parameters (see the function ``fix_parameters``).
        :returns:
            A dictionary containing free_parameters and some other
            key-value-pairs as defined by ``fix_parameters``. If a key
            is present in both, the value in *free_parameters* is used.
        """

        return_dict = deepcopy(parameters_to_be_fixed)
        return_dict.update(free_parameters)
        return return_dict

    return return_all_parameters


class Preprocessed_Simulator:
    """
    Normalizes sampling to a standard normal distribution.

    In order to help BOLFI to work efficiently with the least amount of
    setup required, this class mediates between the model parameters
    and a standard normal distribution for sampling. In a sense, the
    simulator output gets transformed into covariance eigenvectors.
    """

    def __init__(
        self,
        simulator,
        fixed_parameters,
        free_parameters_names,
        r,
        Q,
        experimental_data,
        feature_extractor,
        logged_feature_names=None,
        transform_parameters={},
        fixed_parameter_order=None,
    ):
        """
        :param simulator:
            The function that returns results given parameters.
        :param fixed_parameters:
            Dictionary of parameters that stay fixed and their values.
        :param free_parameters_names:
            List of the names of parameters which shall be inferred.
        :param r:
            *Q* times the mean of the distribution of free parameters.
        :param Q:
            Inverse covariance matrix of free parameters, the precision.
            It is used to transform the free parameters given to the
            'simulator' into the ones used in the model. Most notably,
            these univariate standard normal distributions get
            transformed into a multivariate normal distribution
            corresponding to *Q* and *r*.
        :param experimental_data:
            The experimental data that the model will be fitted to.
            It has to have the same structure as the *simulator* output.
        :param feature_extractor:
            A function that takes the output of *simulator* or the
            *experimental_data* and returns a list of numbers.
        :param logged_feature_names:
            Optional list assigning feature names to the array elements
            of the `feature_extractor` output.
        :param transform_parameters:
            Optional transformations between the parameter space that is
            used for searching for optimal parameters and the battery
            model parameters. *Q* and *r* define a normal distribution in
            that search space. The keys are the names of the free
            parameters. The values are 2-tuples. The first entry is a
            function taking the search space parameter and returning the
            model parameter. The second entry is the inverse function.
        :param fixed_parameter_order:
            Optional fixed parameter order. Prevents erroneous behaviour
            when the parameter dictionaries get reshuffled.
            Highly recommended.
        """

        self.simulator = simulator
        self.fixed_parameters = fixed_parameters
        self.free_parameters_names = free_parameters_names
        if fixed_parameter_order:
            self.fixed_parameter_order = fixed_parameter_order
        else:
            self.fixed_parameter_order = list(
                enumerate(self.free_parameters_names)
            )
        self.r = r
        self.Q = Q
        self.experimental_data = experimental_data
        self.feature_extractor = feature_extractor
        self.transform_parameters = transform_parameters
        self.logged_feature_names = logged_feature_names

        self.log_of_tried_parameters = {
            name: [] for name in self.free_parameters_names
        }
        """Stores all parameter combinations that have been tried."""

        self.experimental_features = feature_extractor(experimental_data)
        """Extract the features from the experimental data."""

        if self.logged_feature_names is None:
            self.logged_features_names = [
                str(i) for i in range(len(self.experimental_features))
            ]
        self.log_of_calculated_features = {
            name: [] for name in self.logged_feature_names
        }
        """Stores all features calculated on the tried parameters."""

        self.input_dim = len(self.free_parameters_names)
        """Input dimension of the estimation task."""
        self.output_dim = len(self.experimental_features)
        """
        Output dimension of the estimation task (number of features).
        """

        self.add_parameters = fix_parameters(self.fixed_parameters)
        """
        Create a function to combine the free and fixed parameters.
        """

        self.inv_variances, self.back_transform_matrix = np.linalg.eigh(self.Q)
        """
        Compute the linear transformation of parameters for which the
        covariance of the underlying multivariate normal distribution
        is a diagonal matrix. That is, compute the eigenvectors of *Q*.
        It is more stable since *Q* has growing eigenvectors in
        convergence.
        """
        self.variances = self.inv_variances**(-1)
        """Variances of the model parameters."""
        self.transform_matrix = self.back_transform_matrix.T
        """Inverse of *back_transform_matrix*."""

        self.transformed_means = np.linalg.multi_dot(
            [np.diag(self.variances), self.transform_matrix, self.r]
        )
        """
        ``transform_matrix @ Q @ back_transform_matrix`` is diagonal.
        The correct transformation for vectors ``v`` is then
        ``transform_matrix @ v``. The product below corresponds to
        ``Q⁻¹ @ r``. It is just expressed in the eigenvector space of
        *Q* for efficiency.
        """

        self.norm_factor = np.diag(np.sqrt(self.inv_variances))
        """
        Now that the multivariate normal distribution is decomposed into
        various univariate ones, norm them to have equal variance 1.
        """
        self.un_norm_factor = np.diag(np.sqrt(self.variances))
        """Inverse of norm_factor."""
        self.normed_means = np.matmul(self.norm_factor, self.transformed_means)
        """
        Expectation value of the normed univariate normal distributions.
        """

    def search_to_transformed_trial(self, search_space_parameters):
        """
        Transforms search space parameters to model ones.

        :param search_space_parameters:
            A list of lists which each contain a single search space
            parameter sample as it is returned by the sample functions
            of ELFI. In the case of only sample, a list also works.
        :returns:
            A dictionary with its keys as the names of the parameters.
            Their order in the *search_space_parameters* is given by the
            order of *self.free_parameters_names*. The values yield the
            model parameters when passed through the functions in
            *self.transform_parameters*.
        """

        # Get the normed parameter sample.
        normed_parameters = np.apply_along_axis(
            lambda slice: slice + self.normed_means,
            0,
            search_space_parameters
        )

        # Un-norm it into the eigenvector space of *Q* represenation.
        transformed_parameters = self.un_norm_factor @ normed_parameters

        # Transform it backwards to the transformed model space.
        transformed_trial_parameters = (
            self.back_transform_matrix @ transformed_parameters
        )

        # Convert this into a dictionary. The ordering is still given
        # by *self.free_parameters_names*.
        return {
            name: np.array(transformed_trial_parameters[i])
            for i, name in self.fixed_parameter_order
        }

    def transformed_trial_to_search(self, model_space_parameters):
        """
        Transforms model space parameters to search ones.

        :param model_space_parameters:
            A dictionary. The keys are the *self.free_parameters_names*
            and the values are the model parameters after applying the
            transformations given in *self.transform_parameters*.
        :returns:
            A list (of lists) which each contain a single search space
            parameter sample as it is returned by the sample functions
            of ELFI. If the *model_space_parameters* dictionary values
            are numbers, the returned value is a list. If they are
            lists, the returned value is a list of corresponding lists.
            In that case, each and every list must have the same length.
        """

        length = len(model_space_parameters[self.free_parameters_names[0]])
        transformed_trial_parameters = np.matrix([
            [model_space_parameters[name][j]
                for _, name in self.fixed_parameter_order]
            for j in range(length)
        ]).T

        # Transform it forwards to the eigenvector space of Q.
        transformed_parameters = (
            self.transform_matrix @ transformed_trial_parameters
        )

        # Get the normed parameter sample.
        normed_parameters = self.norm_factor @ transformed_parameters

        # Get the search space parameters.
        search_parameters = np.apply_along_axis(
            lambda slice: slice - self.normed_means,
            0,
            normed_parameters
        )

        return search_parameters

    def undo_transformation(self, transformed_trial_parameters):
        """
        Undo the transforms in *self.transform_parameters*.

        :param transformed_trial_parameters:
            A dictionary. The keys are the *free_parameters_names* and
            the values are the model parameters after they have been
            transformed as specified in *self.transform_parameters*.
        :returns:
            The given dictionary with the values transformed back to the
            actual model parameter values.
        """

        trial_parameters = {
            name: transform[0](transformed_trial_parameters[name])
            for name, transform in self.transform_parameters.items()
        }

        return trial_parameters

    def apply_transformation(self, trial_parameters):
        """
        Apply the transforms in *self.transform_parameters*.

        :param trial_parameters:
            A dictionary. The keys are the *free_parameters_names*
            and the values are the actual model parameters.
        :returns:
            The given dictionary with the vales transformed to the
            modified parameter space as specified in
            *self.transform_parameters*.
        """

        try:
            transformed_trial_parameters = {
                name: transform[1](trial_parameters[name])
                for name, transform in self.transform_parameters.items()
            }
        except KeyError as e:
            raise KeyError(
                "A transformation was given for a parameter that "
                + "isn't to be estimated. Original error message: " + repr(e)
            )

        return transformed_trial_parameters

    def elfi_simulator(self, *args, **kwargs):
        """
        A model simulator that can be used with ELFI.

        :param *args:
            The parameters as given by the prior nodes. Their
            order has to correspond to that of the parameter
            'free_parameters' given to 'return_simulator'.
        :param **kwargs:
            Keyword parameters batch_size and random_state,
            but both are unused (they just get passed by BOLFI).
        :returns:
            Simulated features for the given free parameters.
        """

        if kwargs["batch_size"] > 1:
            print("EP_BOLFI: batch_size > 1 is not implemented.")
            raise NotImplementedError

        trial_parameters = self.undo_transformation(
            self.search_to_transformed_trial(np.array(args))
        )
        trial_parameters = {k: v[0] for k, v in trial_parameters.items()}
        trial_parameters = self.add_parameters(trial_parameters)
        simulated_data = self.simulator(trial_parameters)
        simulated_features = self.feature_extractor(simulated_data)

        for name in self.free_parameters_names:
            # This line logs the tried parameters.
            self.log_of_tried_parameters[name].append(trial_parameters[name])

        for i, name in enumerate(self.logged_feature_names):
            # This line logs the resulting features.
            self.log_of_calculated_features[name].append(simulated_features[i])

        return simulated_features


class Optimizer_State:
    """Handles the heuristics for the EP-BOLFI operation modes."""

    def __init__(
        self,
        input_dim,
        mcmc_chains,
        total_evidence,
        posterior_samples,
        gelman_rubin_threshold,
        ess_ratio_resample=5.0,
        ess_ratio_sampling_from_zero=-1.0,
        ess_ratio_abort=20.0,
        posterior_sampling_increase=1.2,
        model_resampling_increase=1.2,
    ):
        self.input_dim = input_dim
        self.mcmc_chains = mcmc_chains
        self.total_evidence = total_evidence
        self.posterior_samples = posterior_samples
        self.gelman_rubin_threshold = gelman_rubin_threshold
        self.ess_ratio_resample = ess_ratio_resample
        self.ess_ratio_sampling_from_zero = ess_ratio_sampling_from_zero
        self.ess_ratio_abort = ess_ratio_abort
        self.posterior_sampling_increase = posterior_sampling_increase
        self.model_resampling_increase = model_resampling_increase

        self.finished = False
        self.sampling_from_zero = False
        self.resampling = False
        self.initials = None

        self.order_of_actions = []
        for ess_ratio, action in sorted([
            (0.0, "sample"),
            (self.ess_ratio_resample, "resample"),
            (self.ess_ratio_sampling_from_zero, "sample_from_zero"),
            (self.ess_ratio_abort, "abort"),
        ]):
            if ess_ratio >= 0.0:
                self.order_of_actions.append((ess_ratio, action))
        self.current_action = "sample"
        self.verbose_actions = {
            "sample":
                "Sampling the pseudo-posterior...",
            "resample":
                "Resampling the model...",
            "sample_from_zero":
                "Sampling the pseudo-posterior from  0...",
            "abort":
                "Posterior too badly conditioned. Skipping feature.",
        }

    def calculate_next_step(self, ess_ratio, action=None):
        if action is None:
            for ess_criterion, action in self.order_of_actions:
                # By not breaking when the criterion is fulfilled and
                # going through them in ascending order, their ess
                # ratios work as checkpoints; the largest one smaller
                # than the current ess gets triggered. They are
                # exclusive to each other.
                if ess_criterion < ess_ratio:
                    self.current_action = action
        else:
            self.current_action = action
        if self.current_action == "sample":
            self.initials = None
        elif self.current_action == "resample":
            self.initials = None
            self.total_evidence *= self.model_resampling_increase
        elif self.current_action == "sample_from_zero":
            self.initials = np.zeros([self.mcmc_chains, self.input_dim])
        elif self.current_action == "abort":
            self.initials = None
        self.posterior_samples *= self.posterior_sampling_increase


class EP_BOLFI:
    """
    Expectation Propagation and Bayesian Optimization.

    Sets up and runs these two algorithms to infer model parameters.
    Use the variables *Q*, *r*, *Q_features* and *r_features* to copy
    the state of another estimator. Do not use them in any other case.
    Always use either all of them or none of them.
    """

    def __init__(
        self,
        simulators,
        experimental_datasets,
        feature_extractors,
        fixed_parameters,
        free_parameters=None,
        initial_covariance=None,
        free_parameters_boundaries=None,
        boundaries_in_deviations=0,
        Q=None,
        r=None,
        Q_features=None,
        r_features=None,
        transform_parameters={},
        weights=None,
        display_current_feature=None,
        fixed_parameter_order=None,
    ):
        """
        :param simulators:
            A list of functions that take one argument: a dictionary of
            combined *fixed_parameters* and *free_parameters*. They
            return the simulated counterpart to the experimental data.
            Most of the time, one function will be sufficient.
            Additional functions may be used to combine simulators which
            each give a subset of the total experimental method.
        :param experimental_datasets:
            A list of the experimental data. Each entry corresponds to
            the simulator in *simulators* with the same index and has
            the same structure as its output.
        :param feature_extractors:
            A list of functions which each take the corresponding data
            entry and return a list of numbers, which represent its
            reduced features.
        :param fixed_parameters:
            Dictionary of parameters that stay fixed and their values.
        :param free_parameters:
            Dictionary of parameters which shall be inferred and their
            initial guesses or, more accurately, their expected values.
            Please note that these values live in the transformed space.
            Optionally, the values may be a 2-tuple where the second
            entry would be the variance of that parameter. For finer
            tuning with covariances, use *initial_covariance* (will take
            precedence). Alternatively, you may set
            *free_parameters_boundaries* to set the expectation values
            and variances by confidence intervals.
        :param initial_covariance:
            Initial covariance of the parameters. Has to be a symmetric
            matrix (list of list or numpy 2D array). A reasonable simple
            choice is to have a diagonal matrix and set the standard
            deviation σᵢ of each parameter to half of the distance
            between initial guess and biggest/smallest value that shall
            be tried. If the diagonal entries are σᵢ² and the bounds are
            symmetric, the probability distribution of each parameter is
            95% within these bounds. Please note that the same does not
            hold for the whole probability distribution.
        :param free_parameters_boundaries:
            Optional hard boundaries of the space in which optimal
            parameters are searched for. They are given as a dictionary
            with values as 2-tuples with the left and right boundaries.
            Boundaries need to be given for either none or all
            parameters. If None are given, boundaries will be set by
            *boundaries_in_deviations* relative to the covariance.
            The default then is the 95 % confidence ellipsoid.
            If neither *initial_covariance* nor *free_parameters* set
            the covariance, this parameter sets it according to the
            example in the description of *initial_covariance*.
        :param boundaries_in_deviations:
            When <= 0, the boundaries are set as described above.
            When > 0, the boundaries are this multiple of the standard
            deviation. This scales with the shrinking covariance
            as the algorithm progresses. *free_parameters_boundaries*
            takes precedence, i.e., the covariance gets set and then the
            boundaries in the optimization are in standard deviations.
        :param Q:
            If you want to restore a previous EP-BOLFI instance from a
            dump of its data (see "result_to_json" method), put the
            *Q* attribute stored therein into this parameter.
        :param r:
            Same as *Q*, but use the *r* attribute.
        :param Q_features:
            Same as *Q*, but use the *Q_features* attribute.
        :param r_features:
            Same as *Q*, but use the *r_features* attribute.
        :param transform_parameters:
            Optional transformations between the parameter space that is
            used for searching for optimal parameters and the model
            parameters. Any missing free parameter is not transformed.
            The values are 2-tuples. The first entry is a function
            taking the search space parameter and returning the model
            parameter. The second entry is the inverse function.
            For convenience, any value may also be one of the following:
            - 'none' => (identity, identity)
            - 'log' => (exp, log)
            Please note that, for performance reasons, the returned
            inferred values are directly back-transformed from the mean
            of the internal standard distribution. This means that they
            represent the median of the actual distribution.
        :param weights:
            Optional weights to rescale multi-dimensional features.
            Has no effect on scalar features, as BOLFI is invariant with
            respect to constant or linear transformations.
            A list of lists of numpy.array which correspond to the
            *feature_extractors*. The numpy.array have to have the same
            length as their feature, and will be multiplied entry-wise
            onto the feature before taking the distance to the data.
        :param display_current_feature:
            A list of functions. Each corresponds to a feature extractor
            with the same index. Given an index of the array of its
            features, this returns a short description of it.
            If None is given, only the index will be shown in the output.
        :param fixed_parameter_order:
            Establish a numerical order to the parameter names. This
            prevents errors arising from internal reordering of the
            dictionaries. Only necessary when using the same model in
            different contexts.
        """

        self.simulators = simulators
        self.experimental_datasets = experimental_datasets
        self.feature_extractors = feature_extractors
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        self.initial_covariance = initial_covariance
        self.free_parameters_boundaries = free_parameters_boundaries
        self.boundaries_in_deviations = boundaries_in_deviations
        self.transform_parameters = transform_parameters
        self.weights = weights
        self.display_current_feature = display_current_feature

        if self.free_parameters:
            self.fixed_parameter_order = list(
                enumerate(self.free_parameters.keys())
            )
        elif self.free_parameters_boundaries:
            self.fixed_parameter_order = list(
                enumerate(self.free_parameters_boundaries.keys())
            )
        else:
            raise ValueError(
                "Either 'free_parameters' or 'free_parameters_boundaries' "
                "needs to be set!"
            )
        if fixed_parameter_order:
            self.fixed_parameter_order = fixed_parameter_order
            """
            If the parameter order is explicitly given, use that.
            """
        self.log_of_tried_parameters = {
            name: [] for _, name in self.fixed_parameter_order
        }
        """Stores all parameter combinations that have been tried."""

        self.experimental_features = []
        """Experimental features."""
        self.input_dim = len(self.log_of_tried_parameters)
        """Input dimension of the estimation task."""
        self.output_dim = 0
        """
        Output dimension of the estimation task (sum of features).
        """
        self.simulator_index_by_feature = []
        """
        Mapping of index by all features to corresponding simulator.
        """
        self.sub_index_by_feature = []
        """
        Mapping of index by all features to that by one set of them.
        """
        for i, (extractor, dataset) in (enumerate(zip(
                self.feature_extractors, self.experimental_datasets))):
            self.experimental_features.append(extractor(dataset))
            length = len(self.experimental_features[-1])
            for j in range(self.output_dim, self.output_dim + length):
                self.simulator_index_by_feature.append(i)
                self.sub_index_by_feature.append(j - self.output_dim)
            self.output_dim = self.output_dim + length
        if self.weights is None:
            self.weights = []
            """Set the weights to unity if None are given."""
            for features in self.experimental_features:
                self.weights.append([1 for _ in range(len(features))])
        self.initial_guesses = np.zeros(self.input_dim)
        """Container for the initial expectation values."""

        if self.display_current_feature is not None:
            self.log_of_discrepancies = {
                self.display_current_feature[
                    self.simulator_index_by_feature[j]
                ](self.sub_index_by_feature[j]): []
                for j in range(self.output_dim)
            }
            """Stores all discrepancies of the sampled parameters."""
            self.log_of_calculated_features = {
                self.display_current_feature[
                    self.simulator_index_by_feature[j]
                ](self.sub_index_by_feature[j]): []
                for j in range(self.output_dim)
            }
        else:
            self.log_of_discrepancies = {
                str(j): [] for j in range(self.output_dim)
            }
            self.log_of_calculated_features = {
                str(j): [] for j in range(self.output_dim)
            }

        if (boundaries_in_deviations <= 0):
            self.boundaries_in_deviations = np.sqrt(
                chi2(self.input_dim).ppf(0.95)
            )
        else:
            self.boundaries_in_deviations = boundaries_in_deviations

        # Substitute transformations given by name.
        if self.transform_parameters is not {}:
            for name, function in self.transform_parameters.items():
                if type(function) is str:
                    if function == 'none':
                        self.transform_parameters[name] = (
                            lambda s: s, lambda b: b
                        )
                    elif function == 'log':
                        self.transform_parameters[name] = (
                            lambda s: np.exp(s), lambda b: np.log(b)
                        )
        # Fill-in 'none' transformations.
        for _, name in self.fixed_parameter_order:
            if name not in self.transform_parameters.keys():
                self.transform_parameters[name] = (
                    lambda s: s, lambda b: b
                )

        # Set up the covariance matrix if it is not given already.
        variance_info = False
        if self.initial_covariance is None and Q is None:
            # Check if the variance information is present.
            variance_info = True
            try:
                if self.free_parameters:
                    for value in self.free_parameters.values():
                        if len(value) < 2:
                            variance_info = False
                            break
                else:
                    variance_info = False
            except TypeError:
                variance_info = False
            if not variance_info and self.free_parameters_boundaries is None:
                raise ValueError(
                    "'free_parameters' has to contain 2-tuples or "
                    "'free_parameters_boundaries' has to be set if "
                    "'initial_covariance' is not given!"
                )

            self.initial_covariance = np.zeros([
                self.input_dim, self.input_dim
            ])
            if variance_info:
                for i, name in self.fixed_parameter_order:
                    value = self.free_parameters[name]
                    self.initial_guesses[i] = value[0]
                    self.initial_covariance[i][i] = value[1]
            else:
                if self.free_parameters:
                    for i, name in self.fixed_parameter_order:
                        value = self.free_parameters[name]
                        self.initial_guesses[i] = np.atleast_1d(value)[0]
                else:
                    for i, name in self.fixed_parameter_order:
                        value = self.free_parameters_boundaries[name]
                        transform = self.transform_parameters[name]
                        expected_value = 0.5 * (
                            transform[1](value[0]) + transform[1](value[1])
                        )
                        self.initial_guesses[i] = expected_value
                for i, name in self.fixed_parameter_order:
                    value = self.free_parameters_boundaries[name]
                    transform = self.transform_parameters[name]
                    standard_deviation = np.max([
                        np.abs(transform[1](value[1])
                               - self.initial_guesses[i]),
                        np.abs(self.initial_guesses[i]
                               - transform[1](value[0]))
                    ]) / self.boundaries_in_deviations
                    self.initial_covariance[i][i] = standard_deviation**2
        else:
            if self.free_parameters:
                for i, name in self.fixed_parameter_order:
                    value = self.free_parameters[name]
                    self.initial_guesses[i] = np.atleast_1d(value)[0]
            else:
                raise ValueError(
                    "When 'initial_covariance' or 'Q' are set, "
                    "'free_parameters' needs to be set as well!"
                )
        if Q is not None:
            self.initial_covariance = np.linalg.inv(Q)
        if r is not None:
            self.initial_guesses = self.initial_covariance @ r

        self.final_expectation = deepcopy(self.initial_guesses)
        """Stores the inference mean (empty at first)."""
        self.final_covariance = deepcopy(self.initial_covariance)
        """Stores the inference covariance (empty at first)."""
        deviations = np.sqrt(np.diag(self.final_covariance))
        self.final_correlation = (
            self.final_covariance / deviations[:, None]
        ) / deviations[None, :]
        """Stores the inference correlation (empty at first)."""

        if Q is None:
            self.initial_Q = np.linalg.inv(self.initial_covariance)
            """Expectation Propagation covariance matrix (prior)."""
            self.Q = deepcopy(self.initial_Q)
            """Expectation Propagation covariance matrix (posterior)."""
        else:
            self.initial_Q = deepcopy(Q)
            self.Q = deepcopy(Q)
        if r is None:
            self.initial_r = self.initial_Q @ self.initial_guesses
            """Expectation Propagation expectation value (prior)."""
            self.r = deepcopy(self.initial_r)
            """Expectation Propagation expectation value (posterior)."""
        else:
            self.initial_r = deepcopy(r)
            self.r = deepcopy(r)
        self.Q_features = []
        """Expectation Propagation itemized covariance matrices."""
        self.r_features = []
        """Expectation Propagation itemized expectation values."""
        if Q_features is None:
            for i in range(self.output_dim):
                self.Q_features.append(
                    deepcopy(self.Q) * 0
                )
        else:
            self.Q_features = deepcopy(Q_features)
        if r_features is None:
            for i in range(self.output_dim):
                self.r_features.append(
                    deepcopy(self.r) * 0
                )
        else:
            self.r_features = deepcopy(r_features)

        self.inferred_parameters = {
            name: self.transform_parameters[name][0](self.final_expectation[i])
            for i, name in self.fixed_parameter_order
        }
        """The inferred model parameters."""
        error_lower_bound = {}
        error_upper_bound = {}
        for i, name in self.fixed_parameter_order:
            error_lower_bound[name] = (
                self.final_expectation[i]
                - np.sqrt(
                    chi2(self.input_dim).ppf(0.95)
                ) * deviations[i])
            error_upper_bound[name] = (
                self.final_expectation[i]
                + np.sqrt(
                    chi2(self.input_dim).ppf(0.95)
                ) * deviations[i])
        error_lower_bound = {
            name: transform[0](error_lower_bound[name])
            for name, transform in self.transform_parameters.items()
        }
        error_upper_bound = {
            name: transform[0](error_upper_bound[name])
            for name, transform in self.transform_parameters.items()
        }
        self.final_error_bounds = {
            name: (error_lower_bound[name], error_upper_bound[name])
            for name in self.log_of_tried_parameters.keys()
        }
        """
        The 95% confidence bounds (which don't reflect the
        cross-correlations, but are easier to interpret).
        """

    def result_to_json(self, seed=None):
        """
        Formats the relevant optimizer states in JSON.

        :param seed:
            Optionally put the seed you used when running EP-BOLFI.
        """
        return json.dumps({
            "inferred parameters": self.inferred_parameters,
            "covariance":
                [list(line) for line in self.final_covariance],
            "correlation":
                [list(line) for line in self.final_correlation],
            "error bounds": self.final_error_bounds,
            "seed": seed,
            "Q": self.Q,
            "r": self.r,
            "Q_features": self.Q_features,
            "r_features": self.r_features,
            "fixed_parameter_order": self.fixed_parameter_order,
        }, cls=NDArrayEncoder)

    def log_to_json(self):
        """Formats the relevant optimizer logs in JSON."""
        return json.dumps({
            "tried parameters": self.log_of_tried_parameters,
            "features": self.log_of_calculated_features,
            "discrepancies": self.log_of_discrepancies,
        }, cls=NDArrayEncoder)

    def visualize_parameter_distribution(self):
        """
        Plots the features and visualizes the correlation.

        Please note that this function requires that the output of the
        individual simulators and the individual experimental data give
        an x- and y-axis when indexed with [0] and [1], respectively.
        Lists of lists in [0] and [1] with segmented data works as well.
        ``EP_BOLFI.run`` does not have these restrictions.
        Visualizes the comparison that EP_BOLFI was set up to infer
        model parameters with. May be used to check if everything works
        as intended. Additionally, the 95% confidence error bounds for
        the parameters are visualized to check for reasonable bounds.
        If called after ``run``, the expected parameter set, the
        correlation and error bounds are from the finished estimation.
        """

        parameters = deepcopy(self.fixed_parameters)
        parameters.update({
            name: self.transform_parameters[name][0](value) for name, value
            in zip(self.log_of_tried_parameters.keys(), self.final_expectation)
        })
        combinations = combine_parameters_to_try(
            parameters, self.final_error_bounds
        )[0]
        fig_ax = [plt.subplots(figsize=(2**0.5 * 5, 5))
                  for _ in range(len(self.simulators))]
        for i, (sim, exp_data, feature) in enumerate(zip(
                self.simulators, self.experimental_datasets,
                self.feature_extractors)):
            ps = Preprocessed_Simulator(
                sim, self.fixed_parameters,
                list(self.log_of_tried_parameters.keys()),
                self.r, self.Q, exp_data, feature,
                transform_parameters=self.transform_parameters,
                fixed_parameter_order=self.fixed_parameter_order
            )
            sim_data = ps.simulator(parameters)
            sim_data_plot = [[entry for segment in axis
                              for entry in np.atleast_1d(segment)]
                             for axis in sim_data]
            exp_data_plot = [[entry for segment in axis
                              for entry in np.atleast_1d(segment)]
                             for axis in exp_data]
            print("Features of simulator #" + str(i) + ":")
            print(feature(sim_data))
            print("Features of experiment #" + str(i) + ":")
            print(feature(exp_data))
            errorbars = [
                [entry
                 for segment in ps.simulator(combination)[1]
                 for entry in np.atleast_1d(segment)]
                for combination in combinations
            ]
            fig, ax = fig_ax[i]
            ax.plot(sim_data_plot[0], sim_data_plot[1],
                    label="expectation value")
            ax.plot(exp_data_plot[0], exp_data_plot[1],
                    label="experimental data")
            ax.fill_between(sim_data_plot[0],
                            np.min(errorbars, axis=0),
                            np.max(errorbars, axis=0), alpha=0.5,
                            label="95% confidence")
            ax.legend()
        plt.show()

    def run(
        self,
        bolfi_initial_evidence=None,
        bolfi_total_evidence=None,
        bolfi_posterior_samples=None,
        ep_iterations=3,
        ep_dampener=None,
        final_dampening=None,
        ep_dampener_reduction_steps=-1,
        gelman_rubin_threshold=None,
        ess_ratio_resample=5.0,
        ess_ratio_sampling_from_zero=-1.0,
        ess_ratio_abort=20.0,
        max_heuristic_steps=10,
        posterior_sampling_increase=1.2,
        model_resampling_increase=1.1,
        independent_mcmc_chains=4,
        scramble_ep_feature_order=True,
        normalize_features=True,
        show_trials=False,
        verbose=True,
        seed=None,
    ):
        """
        Runs Expectation Propagation together with BOLFI.

        This function can be called multiple times; the estimation will
        take off from where it last stopped.


        Enable parallelization in ELFI (for details, see https://
        # elfi.readthedocs.io/en/latest/usage/parallelization.html):
        - For local multithreading with all cores:
          ``elfi.set_client('multiprocessing')``
        - For scaling to clusters:
          ``elfi.set_client('ipyparallel')``

        :param bolfi_initial_evidence:
            Number of evidence samples BOLFI will take for each feature
            before using Bayesian Optimization sampling. Default:
            ``1 + 2 ** number of estimated parameters``.
        :param bolfi_total_evidence:
            Number of evidence samples BOLFI will take for each feature
            in total (including initial evidence). Default:
            ``2 * bolfi_initial_evidence``.
        :param bolfi_posterior_samples:
            Effective number of samples BOLFI will take from the
            posterior distribution. These are then used to fit a
            Gaussian to the posterior. Fit convergence scales with 1/√n.
            Default: I² + 3 * I with I as the number of
            estimated parameters. This is the number of the
            metaparameters of the underlying probability distribution
            times 2. The "times 2" considers the warmup samples.
        :param ep_iterations:
            The number of iterations of the Expectation Propagation
            algorithm, i.e., the number of passes over each feature.
            Default: 3.
        :param ep_dampener:
            The linear combination factor of the posterior calculated by
            BOLFI and the pseudo-prior. 0 means no dampening, i.e., the
            pseudo-prior gets replaced by the posterior. For values up
            to 1, that fraction of the pseudo-prior remains in each
            site update. Default: with a as the number of features and
            b as *ep_iterations*, ``1 - a * (1 - ᵃᵇ√final_dampening)``.
        :param final_dampening:
            Alternative way to set *ep_dampener*. 0 means no dampening.
            For values up to 1, that fraction of the prior remains after
            the whole estimation. Default: if *ep_dampener* is not set,
            0.5. Else, *ep_dampener* takes precedence.
        :param ep_dampener_reduction_steps:
            Number of iterations over which the *ep_dampener* gets
            reduced to 0. In each iteration, an equal fraction of it
            gets subtracted. Set to a negative number to disable the
            reduction. Default: -1.
        :param gelman_rubin_threshold:
            Optional threshold on top of the effective sample size.
            Values close to one indicate a converged estimate of the
            pseudo-posteriors. Never set to exactly one.
        :param ess_ratio_sampling_from_zero:
            Threshold in the ratio of effective sample size to samples
            in the pseudo-posterior estimation, at which the sampling
            defaults to starting at the center of the pseudo-prior. Set
            higher than *ess_ratio_resample* to disable this behaviour.
        :param ess_ratio_resample:
            Threshold in the ratio of effective sample size to samples
            in the pseudo-posterior estimation, at which before sampling
            the model gets resampled. Set higher than *ess_ratio_abort*
            to disable this behaviour.
        :param ess_ratio_abort:
            Threshold in the ratio of effective sample size to samples
            in the pseudo-posterior estimation, at which the sampling
            aborts and the pseudo-posterior update is skipped.
        :param max_heuristic_steps:
            The heuristics that are set by the *ess_ratio_x* arguments
            could effectively run forever. This parameter limits the
            amount of times these heuristics get employed in on EP
            iteration before it terminates.
        :param posterior_sampling_increase:
            The factor by which the ratio of the effective sample size
            to samples in the pseudo-posterior estimation is multiplied
            each loop (cumulatively). Never set to exactly one or lower,
            as it might result in an infinite loop.
        :param model_resampling_increase:
            The factor by which *bolfi_total_evidence* gets multiplied
            each time the model gets resampled.
        :param independent_mcmc_chains:
            The number of independent Markov-Chain Monte Carlo chains
            that are used for the estimation of the pseudo-posterior.
            Since we did not implement parallelization, more chains will
            not be faster, but more stable.
        :param scramble_ep_feature_order:
            True randomizes the order that the EP features are iterated
            over. Their order is still consistent across EP iterations.
            False uses the order that the *feature_extractors* define.
        :param normalize_features:
            Choose whether simulated features get divided by the
            experimental features before comparison. Has mostly
            numerical benefits, as BOLFI is scale-invariant.
            Set to False if you have features that converge to zero.
        :param show_trials:
            True plots the log of tried parameters live. Please note
            that each plot blocks the execution of the program, so do
            not use this when running the estimation in the background.
        :param verbose:
            True shows verbose error messages and logs of the estimation
            process. With False, you need to get the estimation results
            from *self.final_expectation* and *self.final_covariance*.
            Default: True.
        :param seed:
            Optional seed that is used in the RNG. If None is given, the
            results will be slightly different each time.
        :returns:
            The BOLFI instance of the last EP iteration. As such, it
            contains the Posterior of the overall inference procedure.
        """

        if self.output_dim == 1 and ep_iterations > 1:
            raise RuntimeError(
                "EP with only one feature does not properly update the "
                "posterior after the initial loop. Set ep_iterations to 1. "
                "If you want to re-run, initialize a new EP_BOLFI instance "
                "from this one's Q and r attributes, but leave the attributes "
                "Q_features and r_features as None. "
                "This will make the former posterior the new prior."
            )

        bolfi_initial_evidence = (bolfi_initial_evidence
                                  or 1 + 2**self.input_dim)
        bolfi_total_evidence = (bolfi_total_evidence
                                or 2 * bolfi_initial_evidence)
        bolfi_posterior_samples = (
            bolfi_posterior_samples
            or self.input_dim**2 + 3 * self.input_dim
        )
        final_dampening = final_dampening or 0.5
        final_dampening = np.max([0, np.min([final_dampening, 1])])
        ep_dampener = (
            ep_dampener
            or 1 - self.output_dim * (
                1 - final_dampening**(1 / (self.output_dim * ep_iterations))
            )
        )
        ep_dampener = np.max([0, np.min([ep_dampener, 1])])
        gelman_rubin_threshold = gelman_rubin_threshold or float('inf')
        gelman_rubin_threshold = (
            gelman_rubin_threshold if gelman_rubin_threshold > 1 else 1.1
        )
        posterior_sampling_increase = (
            posterior_sampling_increase if posterior_sampling_increase > 1.0
            else 1.2
        )
        model_resampling_increase = (
            model_resampling_increase if model_resampling_increase > 1.0
            else 1.2
        )

        # Shuffle the order in which the features are processed.
        if scramble_ep_feature_order:
            shuffled_order = (
                np.random.default_rng(seed).permutation(self.output_dim)
            )
        else:
            shuffled_order = np.array(range(self.output_dim))

        skip_sample = False

        for k in range(ep_iterations):

            # Compute the dampening for this iteration.
            if ep_dampener_reduction_steps > 0:
                if (k + 1 > ep_dampener_reduction_steps
                        or ep_dampener_reduction_steps == 0):
                    ep_dampener = 0.0
                else:
                    ep_dampener = ep_dampener * (
                        1.0 - k / ep_dampener_reduction_steps
                    )

            for j in shuffled_order:

                simulator_index = self.simulator_index_by_feature[j]
                sub_index = self.sub_index_by_feature[j]

                if verbose:
                    print()
                    if self.display_current_feature is not None:
                        print(
                            "Current feature: "
                            + self.display_current_feature[simulator_index](
                                sub_index
                            ) + "."
                        )
                    else:
                        print("Current feature: #" + str(j) + ".")

                # Initialize a new model in ELFI to ensure that no old,
                # superfluous nodes get added to this inference task.
                elfi.new_model()

                # Initialize the ELFI priors. For technical reasons, the
                # artificial "stabilising noise" used in BOLFI is equal
                # across all parameters. So in order to not have the
                # parameters with larger variance blow up the noise, all
                # should be scaled to have equal variance. To avoid any
                # accidental mismatch of means, they are normed to 0 and
                # transformed back into the actual parameter ranges when
                # needed.
                priors = []
                for i in range(self.input_dim):
                    priors.append(elfi.Prior("norm", 0.0, 1.0, name=str(i)))

                # Compute the pseudo priors.
                Q_pseudoprior = self.Q - self.Q_features[j]
                r_pseudoprior = self.r - self.r_features[j]

                # Get the simulator for use with ELFI with its
                # normalization and transformation functions.
                list_of_all_simulator_features = []
                for m in shuffled_order:
                    m_simulator_index = self.simulator_index_by_feature[m]
                    if m_simulator_index != simulator_index:
                        continue
                    if self.display_current_feature is not None:
                        list_of_all_simulator_features.append(
                            self.display_current_feature[simulator_index](
                                self.sub_index_by_feature[m]
                            )
                        )
                    else:
                        list_of_all_simulator_features.append(str(m))
                ps = Preprocessed_Simulator(
                    self.simulators[simulator_index], self.fixed_parameters,
                    list(self.log_of_tried_parameters.keys()), r_pseudoprior,
                    Q_pseudoprior, self.experimental_datasets[simulator_index],
                    self.feature_extractors[simulator_index],
                    list_of_all_simulator_features,
                    self.transform_parameters,
                    self.fixed_parameter_order
                )

                # Create the ELFI simulator.
                elfi_simulator = elfi.Simulator(
                    ps.elfi_simulator, *priors,
                    observed=self.experimental_features[simulator_index]
                )

                # Create the summary operation that selects the feature.
                # Note that it normalizes to the experimental features.
                # This gives equal weight to every feature before
                # "weights" are applied.
                def summary(data):
                    try:
                        if normalize_features:
                            return [
                                data[sub_index] / self.experimental_features[
                                    simulator_index
                                ][sub_index]
                                * self.weights[simulator_index][sub_index]
                            ]
                        else:
                            return [
                                data[sub_index]
                                * self.weights[simulator_index][sub_index]
                            ]
                    except IndexError:
                        if verbose:
                            print("Warning: simulator output was too short.")
                        return [
                            0 * self.experimental_features[simulator_index][
                                sub_index
                            ]
                        ]
                processed_data = elfi.Summary(summary, elfi_simulator)

                # Use the Euclidean distance. Either the feature is a
                # number and it is irrelevant, or it is a time series.
                distance = elfi.Distance('euclidean', processed_data)

                # It may be beneficial to take the logarithm of the
                # discrepancy. This is because the Gaussian Process used
                # in BOLFI is highly sensitive to high discrepancies
                # (which shouldn't be accounted for too much,
                # since they just correspond to bad parameter guesses).
                log_distance = elfi.Operation(np.log, distance)

                # Calculate the hard boundaries for BOLFI.
                bolfi_bounds = {
                    str(c): (-np.sqrt(chi2(self.input_dim).ppf(0.95)),
                             +np.sqrt(chi2(self.input_dim).ppf(0.95)))
                    for c in range(self.input_dim)
                }

                # Start BOLFI and show a progressbar if requested.
                bolfi = elfi.BOLFI(
                    log_distance, initial_evidence=bolfi_initial_evidence,
                    batch_size=1, bounds=bolfi_bounds, seed=seed
                )
                bolfi.fit(n_evidence=bolfi_total_evidence, bar=verbose)

                state = Optimizer_State(
                    self.input_dim,
                    independent_mcmc_chains,
                    bolfi_total_evidence,
                    bolfi_posterior_samples,
                    gelman_rubin_threshold,
                    ess_ratio_resample,
                    ess_ratio_sampling_from_zero,
                    ess_ratio_abort,
                    posterior_sampling_increase,
                    model_resampling_increase,
                )

                degenerate_estimate = np.array([0.0] * self.input_dim)
                minimum_ess = 0
                ess = [0 for _ in range(self.input_dim)]
                maximum_gelman_rubin = float('inf')
                gelman_rubin = [float('inf') for _ in range(self.input_dim)]
                sample_ess_ratio = 1.0
                current_action = None
                adapt_sample_size = True
                heuristic_steps = 0

                while (minimum_ess < bolfi_posterior_samples
                        or (maximum_gelman_rubin > gelman_rubin_threshold if
                            gelman_rubin_threshold < float('inf') else False)):
                    state.calculate_next_step(sample_ess_ratio, current_action)
                    if state.current_action == "resample":
                        adapt_sample_size = False
                    current_action = None
                    if (
                        state.current_action == "abort"

                        or heuristic_steps > max_heuristic_steps
                    ):
                        degenerate_estimate = np.zeros(self.input_dim)
                        skip_sample = True
                        print(state.verbose_actions["abort"])
                        break
                    if verbose:
                        # print(bolfi.target_model)
                        print(state.verbose_actions[state.current_action])
                    heuristic_steps += 1
                    # If the number of evidence points stayed the same,
                    # this just returns the current posterior.
                    bolfi.fit(
                        n_evidence=max(
                            [state.total_evidence, bolfi.n_evidence]
                        ),
                        bar=verbose
                    )
                    if verbose:
                        print()  # linebreak after progressbar
                    try:
                        # Make a deepcopy of the BOLFI instance for
                        # sampling.
                        # Reason: when the sampling errors out with
                        # "NUTS: Cannot find acceptable stepsize ...",
                        # something breaks internally so that bolfi.fit
                        # errors out with inconsistent array sizes.
                        # File <lib>/elfi/methods/bo/gpy_regression.py,
                        #   line 132,
                        #   in predict:
                        # r2 = np.sum(x**2., 1)[:, None]
                        #    + self._rbf_x2sum
                        #    - 2. * x.dot(self._gp.X.T)
                        # ValueError: operands could not be broadcast
                        # together with shapes (1,<old evidence>)
                        # (1,<old evidence> + 1)
                        bolfi_for_sampling = deepcopy(bolfi)
                        result = bolfi_for_sampling.sample(
                            int(
                                state.posterior_samples * sample_ess_ratio
                            ) if adapt_sample_size else int(
                                state.posterior_samples
                            ),
                            n_chains=independent_mcmc_chains,
                            initials=state.initials
                        )
                        adapt_sample_size = True
                    except OverflowError as oe:
                        # cannot convert float infinity to integer
                        # Happens when state.posterior_samples gets too
                        # large.
                        if verbose:
                            print("Error message from BOLFI:", oe)
                        skip_sample = True
                        break
                    except ValueError as ve:
                        # NUTS: Cannot find acceptable stepsize
                        # starting from point [best evidence].
                        if verbose:
                            print("Error message from BOLFI:", ve)
                        em = str(ve)
                        degenerate_estimate = np.fromstring(
                            em[em.find('[') + 1:em.find(']')],
                            dtype=float, sep='  '
                        )
                        if current_action == "sample_from_zero":
                            current_action = "resample"
                        else:
                            current_action = "sample_from_zero"
                        continue
                    except ZeroDivisionError:
                        skip_sample = True
                        break
                    chains = result.meta['chains']
                    for i in range(self.input_dim):
                        ess[i] = (
                            elfi.methods.mcmc.eff_sample_size(chains[:, :, i])
                        )
                        gelman_rubin[i] = (
                            elfi.methods.mcmc.gelman_rubin_statistic(
                                chains[:, :, i]
                            )
                        )
                    minimum_ess = np.min(ess)
                    minimum_ess = 1 if np.min(ess) <= 0 else minimum_ess
                    # The samples_array excludes the warmup samples
                    # (one half). sample_ess_ratio is the amount of
                    # samples compared to the minimum ESS of any
                    # parameter. It's >= 1.
                    sample_ess_ratio = (
                        2 * len(result.samples_array)
                        / (minimum_ess * independent_mcmc_chains)
                    )
                    maximum_gelman_rubin = np.max(gelman_rubin)

                if self.display_current_feature is not None:
                    name = self.display_current_feature[
                        self.simulator_index_by_feature[j]
                    ](self.sub_index_by_feature[j])
                else:
                    name = str(j)
                self.log_of_discrepancies[name].extend(
                    bolfi.target_model.Y
                )
                for m in shuffled_order:
                    if m == j:
                        continue
                    if self.display_current_feature is not None:
                        name = self.display_current_feature[
                            self.simulator_index_by_feature[m]
                        ](self.sub_index_by_feature[m])
                    else:
                        name = str(m)
                    self.log_of_discrepancies[name].extend(
                        0 * bolfi.target_model.Y
                    )

                # Append the newly tried parameters.
                for name in self.log_of_tried_parameters.keys():
                    self.log_of_tried_parameters[name].extend(
                        ps.log_of_tried_parameters[name]
                    )

                # Append the corresponding resulting features.
                for m in shuffled_order:
                    if self.display_current_feature is not None:
                        name = self.display_current_feature[
                            self.simulator_index_by_feature[m]
                        ](self.sub_index_by_feature[m])
                    else:
                        name = str(m)
                    m_simulator_index = self.simulator_index_by_feature[m]
                    if m_simulator_index == simulator_index:
                        self.log_of_calculated_features[name].extend(
                            ps.log_of_calculated_features[name]
                        )
                    else:
                        self.log_of_calculated_features[name].extend(
                            [0.0 for i in range(len(
                                list(ps.log_of_calculated_features.items())[0]
                            ))]
                        )

                if show_trials:
                    # Plot the trial log.
                    fig, axes = plt.subplots(
                        self.input_dim, 1, num=0, clear=True,
                        constrained_layout=True
                    )
                    axes = np.atleast_1d(axes)
                    for index, name in zip(
                        range(self.input_dim),
                        self.log_of_tried_parameters.keys()
                    ):
                        axes[index].set_title(name)
                        axes[index].semilogy(range(len(
                            self.log_of_tried_parameters[name]
                        )), self.log_of_tried_parameters[name],
                            marker='.', lw=0, label=name
                        )
                    plt.show()

                if skip_sample:
                    try:
                        inferred_parameters = ps.undo_transformation(
                            ps.search_to_transformed_trial(
                                degenerate_estimate
                            )
                        )
                        if verbose:
                            print("inferred parameters:")
                            print(inferred_parameters)
                    except (ValueError, IndexError):
                        # BOLFI probably didn't take any samples at all.
                        pass
                    skip_sample = False
                    continue

                # if verbose:
                #     print(result)
                #     print(result.samples_array)

                # Get the raw samples. They are given as a list.
                normed_samples_0 = np.array(result.samples_array).T

                # Un-raw the samples.
                samples = ps.search_to_transformed_trial(normed_samples_0)
                samples = [samples[name] for name
                           in self.log_of_tried_parameters.keys()]
                number_of_samples = len(samples[0])

                # Compute mean and covariance directly from the samples.
                µ_h = np.sum(samples, axis=1) / number_of_samples
                # np.outer(vector) computes vector.T * vector.
                covariance = np.sum(
                    np.apply_along_axis(
                        lambda vector: np.outer(vector, vector), 1,
                        np.apply_along_axis(
                            lambda slice: slice - µ_h, 0, samples
                        ).T
                    ),
                    axis=0
                ) / (number_of_samples - 1)

                # Calculate the Expectation Propagation site update.
                Q_h = np.linalg.inv(covariance)
                r_h = Q_h @ µ_h

                self.Q_features[j] += (1.0 - ep_dampener) * (Q_h - self.Q)
                self.r_features[j] += (1.0 - ep_dampener) * (r_h - self.r)

                self.Q = (1.0 - ep_dampener) * Q_h + ep_dampener * self.Q
                self.r = (1.0 - ep_dampener) * r_h + ep_dampener * self.r

                # Update the covariance and expectation values.
                self.final_covariance = np.linalg.inv(self.Q)
                self.final_expectation = self.final_covariance @ self.r

                self.inferred_parameters = ps.undo_transformation(
                    {
                        name: self.final_expectation[i]
                        for i, name in self.fixed_parameter_order
                    }
                )

                # Calculate the correlation matrix for interpretability.
                deviations = np.sqrt(np.diag(self.final_covariance))
                # 'None' in this context is equivalent to 'np.newaxis'.
                self.final_correlation = (
                    self.final_covariance / deviations[:, None]
                ) / deviations[None, :]

                # Calculate the error bounds (which don't reflect the
                # correlations, but are easier to interpret).
                error_lower_bound = {}
                error_upper_bound = {}
                for i, name in self.fixed_parameter_order:
                    error_lower_bound[name] = (
                        self.final_expectation[i]
                        - np.sqrt(
                            chi2(self.input_dim).ppf(0.95)
                        ) * deviations[i])
                    error_upper_bound[name] = (
                        self.final_expectation[i]
                        + np.sqrt(
                            chi2(self.input_dim).ppf(0.95)
                        ) * deviations[i])
                error_lower_bound = ps.undo_transformation(error_lower_bound)
                error_upper_bound = ps.undo_transformation(error_upper_bound)
                self.final_error_bounds = {
                    name: (error_lower_bound[name], error_upper_bound[name])
                    for name in self.log_of_tried_parameters.keys()
                }

                if verbose:
                    print("inferred parameters:", self.inferred_parameters)
                    print("covariance of inferred parameters:")
                    print(repr(self.final_covariance))
                    print("correlation of inferred parameters:")
                    print(repr(self.final_correlation))
                    print("resulting approximate error bounds:")
                    print("{")
                    for name, value in self.final_error_bounds.items():
                        print("'" + name + "':", value, ",")
                    print("}")

        if verbose:
            # Additionally display the posteriors per feature.
            print()
            print("final state of the estimator:")
            print("inferred parameters:", self.inferred_parameters)
            print("covariance of inferred parameters:")
            print(repr(self.final_covariance))
            print("correlation of inferred parameters:")
            print(repr(self.final_correlation))
            print("resulting approximate error bounds:")
            print("{")
            for name, value in self.final_error_bounds.items():
                print("'" + name + "':", value, ",")
            print("}")
            print("precision matrix:")
            print("Q:")
            print(repr(self.Q))
            print("precision-scaled raw parameter guess:")
            print("r:")
            print(repr(self.r))
            print("itemized precisions per feature:")
            print("Q_features:")
            print(repr(self.Q_features))
            print("itemized precision-scaled raw parameter guesses:")
            print("r_features:")
            print(repr(self.r_features))
            for i in range(self.output_dim):
                print()
                if self.display_current_feature is not None:
                    simulator_index = self.simulator_index_by_feature[i]
                    sub_index = self.sub_index_by_feature[i]
                    print(self.display_current_feature[simulator_index](
                        sub_index
                    ) + ":")
                else:
                    print("Feature #" + str(i) + ":")
                try:
                    local_covariance = np.linalg.inv(self.Q_features[i])
                except np.linalg.LinAlgError:
                    local_covariance = np.diag([
                        np.inf for i in range(self.input_dim)
                    ])
                print("covariance:")
                print(repr(local_covariance))
                # Calculate the correlation matrix for interpretability.
                local_deviations = np.sqrt(np.diag(local_covariance))
                # 'None' in this context is equivalent to 'np.newaxis'.
                local_correlation = (
                    local_covariance / local_deviations[:, None]
                ) / local_deviations[None, :]
                print("correlation:")
                print(repr(local_correlation))
                local_guess = local_covariance @ self.r_features[i]
                # Calculate the error bounds (which don't reflect the
                # correlations, but are easier to interpret).
                local_error_lower_bound = {}
                local_error_upper_bound = {}
                for i, name in self.fixed_parameter_order:
                    local_error_lower_bound[name] = (
                        local_guess[i] - np.sqrt(
                            chi2(self.input_dim).ppf(0.95)
                        ) * local_deviations[i]
                    )
                    local_error_upper_bound[name] = (
                        local_guess[i] + np.sqrt(
                            chi2(self.input_dim).ppf(0.95)
                        ) * local_deviations[i]
                    )
                local_error_lower_bound = ps.undo_transformation(
                    local_error_lower_bound)
                local_error_upper_bound = ps.undo_transformation(
                    local_error_upper_bound)
                local_error_bounds = {
                    name: (local_error_lower_bound[name],
                           local_error_upper_bound[name])
                    for name in self.log_of_tried_parameters.keys()
                }
                local_guess = ps.undo_transformation({
                    name: local_guess[i]
                    for i, name in self.fixed_parameter_order
                })
                print("guess:")
                for name, value in local_guess.items():
                    print(name, value)
                print("resulting approximate error bounds:")
                print("{")
                for name, value in local_error_bounds.items():
                    print("'" + name + "':", value, ",")
                print("}")

        # Return the last BOLFI instance, as it contains the Posterior.
        return bolfi
