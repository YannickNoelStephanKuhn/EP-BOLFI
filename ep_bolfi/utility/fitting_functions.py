"""
Various helper and fitting functions for processing measurement curves.
"""

import json
import re

import numpy as np
from os import linesep
from pyimpspec import calculate_drt_tr_nnls, DataSet, TRNNLSResult
import scipy.optimize as so
from scipy.optimize import root_scalar, minimize
from scipy import interpolate as ip
from sklearn.cluster import KMeans
import sympy as sp
import warnings


class NDArrayEncoder(json.JSONEncoder):
    def default(self, item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        return json.JSONEncoder.default(self, item)


class OCV_fit_result:
    """
    Contains OCV fit parameters and related information.

    Reference
    ----------
    C. R. Birkl, E. McTurk, M. R. Roberts, P. G. Bruce and D. A. Howey.
    “A Parametric Open Circuit Voltage Model for Lithium Ion Batteries”.
    Journal of The Electrochemical Society, 162(12):A2271-A2280, 2015
    """

    def __init__(
        self,
        fit,
        SOC,
        OCV,
        SOC_offset=1.0,
        SOC_scale=1.0,
        optimize_result=None,
        spline_interpolation_knots=None,
        spline_interpolation_coefficients=None,
        function_string=None,
    ):
        """
        :param fit:
            The fit parameters of the OCV function from Birkl et al.,
            either with or without the estimated SOC range at the
            beginning, and optionally without the last Δx entry, which
            is then fixed to ensure that the sum over Δx is 1.
            Order of parameters: [E₀_0, a_0, Δx_0, E₀_1, a_1, ...].
        """
        if len(fit) % 3:
            self.SOC_range = np.array(fit[0:2])
            """The SOC range of the data."""
            self.fit = np.array(fit[2:])
            """
            The fit parameters of the OCV function from Birkl et al.,
            excluding the estimated SOC range.
            """
        else:
            self.SOC_range = np.array([0, 1])
            self.fit = np.array(fit)
        self.E_0 = np.array(self.fit[0::3])
        """The E₀ (plateau voltages) parameters."""
        self.a = np.array(self.fit[1::3])
        """The a (inverse plateau widths) parameters."""
        self.Δx = np.array(self.fit[2::3])
        """The Δx (phase proportion) parameters."""
        if len(self.Δx) < len(self.E_0):
            last_Δx = 1 - np.sum(self.Δx)
            self.Δx = np.append(self.Δx, last_Δx)
            self.fit = np.append(self.fit, last_Δx)
        self.E_0, self.a, self.Δx = zip(*sorted(zip(
            self.E_0, self.a, self.Δx
        ), reverse=True))
        self.fit = [
            par
            for plateau in zip(self.E_0, self.a, self.Δx)
            for par in plateau
        ]
        for i in range(len(self.E_0) - 1):
            if (
                np.abs(self.E_0[i + 1] - self.E_0[i])
                + np.abs(self.a[i + 1] - self.a[i]) < 1e-4
            ):
                print("Warning (OCV_fit_result): At least two fitted summands "
                      "coincide.")
        self.SOC = np.array(SOC)
        """The SOC data points."""
        self.OCV = np.array(OCV)
        """
        The OCV data points. May be adjusted from the original data.
        """
        self.SOC_offset = SOC_offset
        """
        If another electrode was factored out in the data, this may
        contain its SOC at SOC 0 of the electrode of interest.
        """
        self.SOC_scale = SOC_scale
        """
        If another electrode was factored out in the data, this may
        contain the the rate of change of its SOC to that of the
        electrode of interest.
        """
        self.optimize_result = optimize_result
        """The scipy.optimize.OptimizeResult that led to the fit."""
        self.spline_interpolation_knots = spline_interpolation_knots
        """
        The knots of the interpolating spline fitted to the inverse.
        """
        self.spline_interpolation_coefficients = (
            spline_interpolation_coefficients
        )
        """
        The coefficients of the interpolating spline fitted to the
        inverse.
        """
        self.function_string = function_string
        """
        The string representation of the interpolating spline fitted to
        the inverse.
        """

    def to_json(self):
        """
        Gives a complete representation in JSON format.

        :returns:
            A JSON-formatted string.
        """
        return json.dumps({
            'fit': {
                'parameters': self.fit,
                'SOC_range': self.SOC_range,
                'E_0': self.E_0,
                'a': self.a,
                'Δx': self.Δx,
            },
            'data': {
                'SOC': self.SOC,
                'OCV': self.OCV,
            },
            'balancing': {
                'SOC_offset': self.SOC_offset,
                'SOC_scale': self.SOC_scale,
            },
            'optimize_result': self.optimize_result,
            'spline': {
                'knots': self.spline_interpolation_knots,
                'coefficients': self.spline_interpolation_coefficients,
            },
            'function': self.function_string,
        }, cls=NDArrayEncoder)

    def SOC_adjusted(self, soc=None):
        """
        Gives the adjusted SOC values.

        :param soc:
            The SOC as assigned in the original data. This usually
            corresponds to the range available during a measurement.
        :returns:
            The SOC as corrected by the OCV model. These values will try
            to correspond to the level of lithiation.
        """

        if soc is None:
            soc = self.SOC

        return self.SOC_range[0] + soc * (
            self.SOC_range[1] - self.SOC_range[0]
        )

    def SOC_other_electrode(self, soc=None):
        """
        Relates the SOCs of the two electrodes to each other.

        If the original data was of a full cell and the other
        electrode was factored out, this may contain the function that
        takes the SOC of the electrode of interest and gives the SOC
        of the other electrode, i.e., the stoichiometric relation.

        :param soc:
            The SOC of the electrode of interest.
        :returns:
            The SOC of the other electrode that was factored out.
        """

        if soc is None:
            soc = self.SOC

        return self.SOC_offset - self.SOC_scale * soc


def find_occurrences(sequence, value):
    """
    Gives indices in sequence where it is closest to value.

    :param sequence:
        A list that represents a differentiable function.
    :param value:
        The value that is searched for in *sequence*. Also, crossings of
        consecutive values in *sequence* with *value* are searched for.
    :returns:
        A list of indices in *sequence* in ascending order where *value*
        or a close match for *value* was found.
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


def smooth_fit(
    x, y, order=3, splits=None, w=None, s=None, display=False, derivatives=0
):
    """
    Calculates a smoothed spline with derivatives.

    Note: the ``roots`` method of a spline only works if it is cubic,
    i.e. of third order. Each ``derivative`` reduces the order by one.

    :param x:
        The independent variable.
    :param y:
        The dependent variable ("plotted over *x*").
    :param order:
        Interpolation order of the spline.
    :param splits:
        Optional tuning parameter. A list of points between which
        splines will be fitted first. The returned spline then is a fit
        of these individual splines.
    :param w:
        Optional list of weights. Works best if ``1 / w`` approximates
        the standard deviation of the noise in *y* at each point.
        Defaults to the SciPy default behaviour of
        ``scipy.interpolate.UnivariateSpline``.
    :param s:
        Optional tuning parameter. Higher values lead to coarser, but
        more smooth interpolations and vice versa. Defaults to SciPy
        default behaviour of ``scipy.interpolate.UnivariateSpline``.
    :param display:
        If set to True, the fit parameters of the spline will be printed
        to console. If possible, a monomial representation is printed.
    :param derivatives:
        The derivatives of the spline to also include in the return.
        Default is 0, which gives the spline. 1 would give the spline,
        followed by its derivative. Can not be higher than spline order.
        Derivatives are only continuous when ``derivatives < order``.
    :returns:
        A smoothing spline in the form of ``scipy.UnivariateSpline``.
    """

    # Clean up multiple-valued ranges.
    x = np.array(x)
    y = np.array(y)

    def recursive_cleanup(x, y):
        increasing = np.append(True, x[1:] > x[:-1])
        # stationary = np.append(True, x[1:] == x[:-1])
        decreasing = np.append(True, x[1:] < x[:-1])
        if np.sum(increasing) == 1 or np.sum(decreasing) == 1:
            return x, y
        # Choose the larger range of monotonic x values.
        if np.sum(increasing) > np.sum(decreasing):
            x = x[increasing]
            y = y[increasing]
        else:
            # Make sure that x is increasing instead of decreasing.
            x = np.flip(x[decreasing])
            y = np.flip(y[decreasing])
        x, y = recursive_cleanup(x, y)
        return x, y
    x, y = recursive_cleanup(x, y)

    y_smoothed = []

    # One could use the smoothing spline for the data directly, but that
    # isn't as stable as subdividing the data first in some cases.
    # One such case is if there are high variations in gradients.
    if splits is None:
        y_smoothed = y
    else:
        split_indices = [np.abs(x-split).argmin() for split in splits]
        x_splits = np.split(x, split_indices)
        number_of_splits = len(x_splits)
        y_splits = np.split(y, split_indices)

        for x_split, y_split in zip(x_splits, y_splits):
            if len(x_split) < order-2:
                print("Warning: not enough points in split for smoothing.")
                y_smoothed.extend(y_split)
            else:
                split_spline = ip.UnivariateSpline(
                    x_split, y_split, w=w, k=order,
                    s=None if s is None else s/number_of_splits
                )
                y_smoothed.extend(split_spline(x_split))

    try:
        spline = ip.UnivariateSpline(x, y_smoothed,
                                     k=order, s=s)
    except ValueError:
        spline = ip.UnivariateSpline(x[::-1], y_smoothed[::-1],
                                     k=order, s=s)

    if display:
        print("Knots of the interpolating spline:")
        print(spline.get_knots())
        print("Coefficients of the interpolating spline:")
        print(spline.get_coeffs())
        try:
            simplification = verbose_spline_parameterization(
                spline.get_coeffs(), spline.get_knots(), order,
                function_name="y", function_args="x", derivatives=derivatives,
                verbose=True
            )
            print("Monomial representation of the interpolating spline:")
            print(simplification)
        except ValueError as e:
            print(e)

    return spline


def fit_exponential_decay_with_warnings(
    timepoints, voltages, recursive_depth=1, threshold=0.95
):
    """
    Extracts a set amount of exponential decay curves.

    :param timepoints:
        The timepoints of the measurements.
    :param voltages:
        The corresponding voltages.
    :param recursive_depth::
        The default 1 fits one exponential curve to the data. For
        higher values that fit is repeated with the data minus the
        preceding fit(s) for this amount of times minus one.
    :param threshold:
        The lower threshold value for the R² coefficient of
        determination. If *threshold* is smaller than 1, the subset of
        the exponential decay data is searched that just fulfills it.
        Defaults to 0.95. Values above 1 are set to 1.
    :returns:
        A list of length *recursive_depth* where each element is a
        3-tuple with the timepoints, the fitted voltage evaluations
        and a 3-tuple of the parameters of the following decay function:
        ``t, (U_0, ΔU, τᵣ⁻¹):
        U_0 + ΔU * np.exp(-τᵣ⁻¹ * (t - timepoints[0]))``.
    """

    t_eval = np.atleast_1d(timepoints)
    u_eval = np.atleast_1d(voltages)
    threshold = threshold if threshold <= 1 else 1

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 3:
        print("Warning: fit_exponential_decay was given insufficient data.")
        return []

    def exp_fit_function(t, b, c, d, t_0=t_eval[0]):
        """Exponential decay function."""
        return b + c * np.exp(-d * (t - t_0))

    def log_inverse_fit_function(y, b, c, d, t_0=t_eval[0]):
        """Corresponding logarithm function."""
        log_arg = (y - b) / c
        log_arg[log_arg <= 0] = 0.1**d
        return t_0 - np.log(log_arg) / d

    end = len(t_eval) - 1
    bracket = [0, end]
    curves = []
    depth_counter = 0
    fit_guess = [
        np.nan_to_num(u_eval[end]),
        np.nan_to_num(u_eval[end // 10] - u_eval[end]),
        np.nan_to_num(1.0 / (t_eval[end] - t_eval[end // 10])),
    ]

    # Evaluate the R² value for a split at the middle of the data.
    split = int(0.5 * (bracket[0] + bracket[1]))
    argmax_R_squared = split

    fit_split = so.minimize(
        lambda x: np.sum((
            exp_fit_function(t_eval[split:end], *x) - u_eval[split:end]
        )**2)**0.5, x0=fit_guess, method='trust-constr'
    ).x
    test_t_split = log_inverse_fit_function(u_eval[split:end], *fit_split)
    R_squared_split = np.nan_to_num(
        1 - np.sum((t_eval[split:end] - test_t_split)**2)
        / np.sum((t_eval[split:end] - np.mean(t_eval[split:end]))**2)
    )

    while True:
        # End prematurely if the threshold is matched.
        if R_squared_split >= threshold:
            fit_eval = exp_fit_function(t_eval, *fit_split)
            R_squared_split = -float('inf')
            end = split
            bracket = [0, end]
            curves.append([t_eval, list(fit_eval), fit_split])
            u_eval = u_eval - fit_eval + fit_eval[0]
            depth_counter = depth_counter + 1
            if depth_counter >= recursive_depth:
                break
            fit_guess = [
                np.nan_to_num(u_eval[end]),
                np.nan_to_num(u_eval[end // 10] - u_eval[end]),
                np.nan_to_num(1.0 / (t_eval[end] - t_eval[end // 10])),
            ]
        # If the threshold wasn't reached, use the highest R² value.
        if bracket[1] - bracket[0] <= 1:
            fit_argmax = so.minimize(lambda x: np.sum(
                (exp_fit_function(t_eval[argmax_R_squared:end], *x)
                    - u_eval[argmax_R_squared:end])**2
            )**0.5, x0=fit_guess, method='trust-constr').x
            fit_eval = exp_fit_function(t_eval, *fit_argmax)
            end = split
            bracket = [0, end]
            curves.append([t_eval, list(fit_eval), fit_argmax])
            u_eval = u_eval - fit_eval + fit_eval[0]
            depth_counter = depth_counter + 1
            if depth_counter >= recursive_depth:
                break
            fit_guess = [
                np.nan_to_num(u_eval[end]),
                np.nan_to_num(u_eval[end // 10] - u_eval[end]),
                np.nan_to_num(1.0 / (t_eval[end] - t_eval[end // 10])),
            ]
        left = int(0.75 * bracket[0] + 0.25 * bracket[1])
        split = int(0.5 * (bracket[0] + bracket[1]))
        right = int(0.25 * bracket[0] + 0.75 * bracket[1])
        # Fit an exponential decay for both "left" and "right" splits.
        fit_left = so.minimize(lambda x: np.sum(
            (exp_fit_function(t_eval[left:end], *x)
                - u_eval[left:end])**2
        )**0.5, x0=fit_guess, method='trust-constr').x
        fit_right = so.minimize(lambda x: np.sum(
            (exp_fit_function(t_eval[right:end], *x)
                - u_eval[right:end])**2
        )**0.5, x0=fit_guess, method='trust-constr').x
        # Take the logarithm corresponding to the fit. If the fit is
        # good, it should give a line, and thus, a high R² value.
        test_t_left = log_inverse_fit_function(
            u_eval[left:end], *fit_left
        )
        R_squared_left = np.nan_to_num(
            1 - np.sum((t_eval[left:end] - test_t_left)**2)
            / np.sum((t_eval[left:end] - np.mean(t_eval[left:end]))**2)
        )
        test_t_right = log_inverse_fit_function(
            u_eval[right:end], *fit_right
        )
        R_squared_right = np.nan_to_num(
            1 - np.sum((t_eval[right:end] - test_t_right)**2)
            / np.sum((t_eval[right:end] - np.mean(t_eval[right:end]))**2)
        )
        # Step in the direction of ascending R² value.
        if R_squared_left > R_squared_right:
            bracket[1] = split
            if R_squared_left > R_squared_split:
                argmax_R_squared = left
            R_squared_split = R_squared_left
            fit_split = fit_left
        else:
            bracket[0] = split
            if R_squared_right > R_squared_split:
                argmax_R_squared = right
            R_squared_split = R_squared_right
            fit_split = fit_right

    return curves


def fit_exponential_decay(
    timepoints, voltages, recursive_depth=1, threshold=0.95
):
    """
    See ``fit_exponential_decay_with_warnings`` for details. This method
    does the same, but suppresses inconsequential NumPy warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'delta_grad == 0.0. Check if the approximated function is '
            'linear. If the function is linear better results can be '
            'obtained by defining the Hessian as zero instead of '
            'using quasi-Newton approximations.'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in scalar multiply'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in scalar divide'
        )
        warnings.filterwarnings(
            'ignore',
            'divide by zero encountered in divide'
        )
        warnings.filterwarnings(
            'ignore',
            'overflow encountered in scalar power'
        )
        warnings.filterwarnings(
            'ignore',
            'The occurrence of roundoff error is detected, which prevents '
            'the requested tolerance from being achieved.  The error may be '
            'underestimated.'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in subtract'
        )
        return fit_exponential_decay_with_warnings(
            timepoints, voltages, recursive_depth, threshold
        )


def fit_sqrt_with_warnings(timepoints, voltages, threshold=0.95):
    """
    Extracts a square root at the beginning of the data.

    :param timepoints:
        The timepoints of the measurements.
    :param voltages:
        The corresponding voltages.
    :param threshold:
        The lower threshold value for the R² coefficient of
        determination. If *threshold* is smaller than 1, the subset of
        the experimental data is searched that just fulfills it.
        Defaults to 0.95. Values above 1 are set to 1.
    :returns:
        A 3-tuple with the timepoints, the fitted voltage evaluations
        and a 2-tuple of the parameters of the following sqrt function:
        ``t, (U_0, dU_d√t): U_0 + dU_d√t * √(t - timepoints[0])``.
    """

    t_eval = np.atleast_1d(timepoints)
    u_eval = np.atleast_1d(voltages)
    threshold = threshold if threshold <= 1 else 1

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 2:
        print("Warning: fit_sqrt was given insufficient data.")
        return []

    def sqrt_fit_function(t, b, c, t_0=t_eval[0]):
        """Square root function."""
        return b + c * np.sqrt(t - t_0)

    def square_inverse_fit_function(y, b, c, t_0=t_eval[0]):
        """Corresponding square function."""
        return t_0 + ((y - b) / c)**2

    end = len(t_eval)
    bracket = [0, end]
    fit_guess = [
        np.nan_to_num(u_eval[0]),
        np.nan_to_num(
            (u_eval[end - 1] - u_eval[0])
            / np.sqrt(t_eval[end - 1] - t_eval[0])
        )
    ]

    # Evaluate the R² value for a split at the middle of the data.
    split = int(0.5 * (bracket[0] + bracket[1]))
    argmax_R_squared = split
    fit_split = so.minimize(lambda x: np.sum(
        (sqrt_fit_function(t_eval[0:split], *x) - u_eval[0:split])**2
    )**0.5, x0=fit_guess, method='trust-constr').x
    test_t_split = square_inverse_fit_function(u_eval[0:split], *fit_split)
    R_squared_split = np.nan_to_num(
        1 - np.sum((t_eval[0:split] - test_t_split)**2)
        / np.sum((t_eval[0:split] - np.mean(t_eval[0:split]))**2)
    )

    while True:
        # Return the result if the threshold is matched.
        if R_squared_split >= threshold:
            fit_eval = sqrt_fit_function(t_eval, *fit_split)
            return [t_eval, list(fit_eval), fit_split]
        # If the threshold wasn't reached, use the highest R² value.
        if bracket[1] - bracket[0] <= 1:
            fit_argmax = so.minimize(
                lambda x: np.sum(
                    (sqrt_fit_function(t_eval[0:argmax_R_squared], *x)
                        - u_eval[0:argmax_R_squared])**2
                )**0.5, x0=fit_guess, method='trust-constr'
            ).x
            fit_eval = sqrt_fit_function(t_eval, *fit_argmax)
            return [t_eval, list(fit_eval), fit_argmax]
        left = int(0.75 * bracket[0] + 0.25 * bracket[1])
        split = int(0.5 * (bracket[0] + bracket[1]))
        right = int(0.25 * bracket[0] + 0.75 * bracket[1])
        # Fit an exponential decay for both "left" and "right" splits.
        fit_left = so.minimize(lambda x: np.sum(
            (sqrt_fit_function(t_eval[0:left], *x) - u_eval[0:left])**2
        )**0.5, x0=fit_guess, method='trust-constr').x
        fit_right = so.minimize(lambda x: np.sum(
            (sqrt_fit_function(t_eval[0:right], *x) - u_eval[0:right])**2
        )**0.5, x0=fit_guess, method='trust-constr').x
        # Take the square corresponding to the fit. If the fit is
        # good, it should give a line, and thus, a high R² value.
        test_t_left = square_inverse_fit_function(
            u_eval[0:left], *fit_left
        )
        R_squared_left = np.nan_to_num(
            1 - np.sum((t_eval[0:left] - test_t_left)**2)
            / np.sum((t_eval[0:left] - np.mean(t_eval[0:left]))**2)
        )
        test_t_right = square_inverse_fit_function(
            u_eval[0:right], *fit_right
        )
        R_squared_right = np.nan_to_num(
            1 - np.sum((t_eval[0:right] - test_t_right)**2)
            / np.sum((t_eval[0:right] - np.mean(t_eval[0:right]))**2)
        )
        # Step in the direction of ascending R² value.
        if R_squared_left > R_squared_right:
            bracket[1] = split
            if R_squared_left > R_squared_split:
                argmax_R_squared = left
            R_squared_split = R_squared_left
            fit_split = fit_left
        else:
            bracket[0] = split
            if R_squared_right > R_squared_split:
                argmax_R_squared = right
            R_squared_split = R_squared_right
            fit_split = fit_right


def fit_sqrt(timepoints, voltages, threshold=0.95):
    """
    See ``fit_sqrt_with_warnings`` for details.  This method
    does the same, but suppresses inconsequential NumPy warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'delta_grad == 0.0. Check if the approximated function is '
            'linear. If the function is linear better results can be '
            'obtained by defining the Hessian as zero instead of '
            'using quasi-Newton approximations.'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in scalar multiply'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in scalar divide'
        )
        warnings.filterwarnings(
            'ignore',
            'divide by zero encountered in divide'
        )
        warnings.filterwarnings(
            'ignore',
            'overflow encountered in scalar power'
        )
        warnings.filterwarnings(
            'ignore',
            'The occurrence of roundoff error is detected, which prevents '
            'the requested tolerance from being achieved.  The error may be '
            'underestimated.'
        )
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in subtract'
        )
        warnings.filterwarnings(
            'ignore',
            'overflow encountered in square'
        )
        return fit_sqrt_with_warnings(timepoints, voltages, threshold)


def fit_pwrlaw(timepoints, quantity, threshold=0.95):
    """
    Extracts a powerlaw from the data..

    :param timepoints:
        The timepoints of the measurements.
    :param quantity:
        The corresponding quantity, to which a powerlaw-behaviour should
        be fitted.
    :param threshold:
        The lower threshold value for the R² coefficient of
        determination. If *threshold* is larger than the R² coefficient,
        a warning is issued and the parameters may be NaN or Inf.
        Defaults to 0.95. Values above 1 are set to 0.98.
    :returns:
        A 3-tuple with the timepoints, the fitted quantity evaluations
        and a 4-tuple of the parameters of the following powerlaw:
        ``t, (q0, dq_dt, q_trans, n): q0 + dq_dt * (t - q_trans)**n``.
    """

    t_eval = np.atleast_1d(timepoints)
    q_eval = np.atleast_1d(quantity)
    threshold = threshold if threshold <= 1 else 0.98

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 2:
        print("Warning: fit_pwrlaw was given insufficient data.")
        return []

    corv = 1e40  # Avoid small differences and numerical issues

    def pwrlaw_fit_function(t, k, n, t_0=t_eval[0], b=q_eval[0]):
        """Power law function."""
        return (b + k * np.maximum(0, (t - t_0))**n)*corv

    def pwrlaw_inverse_fit_function(y, k, n, t_0=t_eval[0], b=q_eval[0]):
        """Corresponding inverse function."""
        return t_0 + ((y - b) / k)**(1/n)

    end = len(t_eval) - 1
    p0 = [
        np.maximum(
            1e-20,
            np.minimum(
                0.9e-11,
                (q_eval[end] - q_eval[0])/(t_eval[end]-t_eval[0])
            )
        ),
        0.9
    ]

    # Check if the NumPy array contains any NaN value
    if (np.isnan(t_eval).any()):
        print("t_eval contains NaN values")
    if (np.isinf(t_eval).any()):
        print("t_eval contains inf values")

    if (np.isnan(q_eval*corv).any()):
        print("q_eval contains NaN values")
    if (np.isinf(q_eval*corv).any()):
        print("q_eval contains inf values")

    if (np.isnan(p0).any()):
        print("p0 contains NaN values")
    if (np.isinf(p0).any()):
        print("p0 contains inf values")

    p0_cor = np.nan_to_num(p0, nan=1e-13, posinf=1e-11, neginf=1e-17)
    params, pcov = so.curve_fit(
        pwrlaw_fit_function,
        t_eval,
        q_eval * corv,
        p0=p0_cor,
        bounds=([0, 0.5], [1e-11, 1.1]),
        maxfev=int(1e6)
    )

    test_t_split = pwrlaw_inverse_fit_function(q_eval, *params)
    R_squared_inv = np.nan_to_num(
        1 - np.sum((t_eval - test_t_split)**2)
        / np.sum((t_eval - np.mean(t_eval))**2)
    )

    # Return the result if the threshold is matched.
    if R_squared_inv >= threshold:
        fit_eval = pwrlaw_fit_function(t_eval, *params)
        params = [np.log(params[0]), params[1]]
        return [t_eval, list(fit_eval), params]
    else:
        print("Power-law fit gone wrong")
        fit_eval = pwrlaw_fit_function(t_eval, *params)
        return [t_eval, list(fit_eval), params]


def fit_pwrlawCL(timepoints, quantity, threshold=0.95):
    """
    Extracts a powerlaw from the data. Does the same as ``fit_pwrlaw``,
    and additionally logs its internal states to stdout.

    :param timepoints:
        The timepoints of the measurements.
    :param quantity:
        The corresponding quantity, to which a powerlaw-behaviour should
        be fitted.
    :param threshold:
        The lower threshold value for the R² coefficient of
        determination. If *threshold* is larger than the R² coefficient,
        a warning is issued and the parameters may be NaN or Inf.
        Defaults to 0.95. Values above 1 are set to 0.98.
    :returns:
        A 3-tuple with the timepoints, the fitted quantity evaluations
        and a 4-tuple of the parameters of the following powerlaw:
        t, (q0, dq_dt, q_trans, n) ↦ q0 + dq_dt * (t - q_trans)**n
    """

    t_eval = np.atleast_1d(timepoints)
    q_eval = np.atleast_1d(quantity)
    threshold = threshold if threshold <= 1 else 0.98

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 3:
        print("Warning: fit_pwrlawCL was given insufficient data.")
        print("t_eval:", t_eval)
        params = [1e-20, -2]
        return [t_eval, list(0), params]
    if len(q_eval) < 3:
        print("Warning: fit_pwrlawCL was given insufficient data.")
        print("q_eval:", q_eval)
        params = [1e-20, -2]
        return [t_eval, [0], params]

    corv = 1e0  # Avoid small differences and resulting numerical issues

    def pwrlaw_fit_function(t, k, n, t_0=t_eval[0], b=q_eval[0]):
        """Power law function."""
        return (b + k * np.maximum(0, (t - t_0))**n)*corv

    def pwrlaw_inverse_fit_function(y, k, n, t_0=t_eval[0], b=q_eval[0]):
        """Corresponding inverse function."""
        return t_0 + ((y - b) / k)**(1/n)

    end = len(t_eval) - 1
    p0 = [
        np.maximum(
            1e-20,
            np.minimum(
                0.9e-11,
                (q_eval[end] - q_eval[0])/(t_eval[end]-t_eval[0])
            )
        ),
        0.9
    ]

    # Check if the NumPy array contains any NaN value
    if (np.isnan(t_eval).any()):
        print("t_eval contains NaN values")
    if (np.isinf(t_eval).any()):
        print("t_eval contains inf values")

    if (np.isnan(q_eval*corv).any()):
        print("q_eval contains NaN values")
    if (np.isinf(q_eval*corv).any()):
        print("q_eval contains inf values")

    if (np.isnan(p0).any()):
        print("p0 contains NaN values")
    if (np.isinf(p0).any()):
        print("p0 contains inf values")

    p0_cor = np.nan_to_num(p0, nan=1e-13, posinf=1e-11, neginf=1e-17)
    params, pcov = so.curve_fit(
        pwrlaw_fit_function,
        t_eval,
        q_eval * corv,
        p0=p0_cor,
        bounds=([0, 0.1], [1e-2, 3]),
        maxfev=int(1e6)
    )
    print(params)
    test_q = pwrlaw_fit_function(t_eval, *params)
    R_squared_split = np.nan_to_num(
        1 - np.sum((q_eval - test_q/corv)**2)
        / np.sum((q_eval - np.mean(q_eval))**2)
    )
    print("R_squared_split:", R_squared_split)

    test_t_split = pwrlaw_inverse_fit_function(q_eval, *params)
    R_squared_inv = np.nan_to_num(
        1 - np.sum((t_eval - test_t_split)**2)
        / np.sum((t_eval - np.mean(t_eval))**2)
    )

    print("R_squared_inv:", R_squared_inv)
    # Return the result if the threshold is matched.
    if R_squared_inv >= threshold:
        fit_eval = pwrlaw_fit_function(t_eval, *params)
        params = [np.log(params[0]), params[1]]
        return [t_eval, list(fit_eval), params]
    else:
        print("Power-law fit gone wrong")
        fit_eval = pwrlaw_fit_function(t_eval, *params)
        return [t_eval, list(fit_eval), params]


def fit_drt(frequencies, impedances, lambda_value=-2.0):
    """
    A Distribution of Relaxation Times approximation via ``pyimpspec``.

    :param frequencies:
        Array of the measured frequencies.
    :param impedances:
        Array of the measured complex impedances.
    :param lambda_value:
        Takes the place of the R² value as a tuning parameter.
        Consult the ``pyimpspec`` documentation for the precise usage.
        Default is -2, which "uses the L-curve approach to estimate λ".
        -1 uses a different heuristic, and values > 0 set λ directly.
    :returns:
        A 3-tuple of characteristic DRT time constants, their
        corresponding resistances, and the whole ``TRNNLSResult`` object
        returned by ``pyimpspec``. Note that the ``.pseudo_chisqr``
        attribute is not useful for the same reasons that went into the
        R² calculations in the other fit functions: a "pseudo-R²"
        does not use the inverse function to obtain a sensible R².
    """

    # Clean up any "inductive" impedances, i.e., those with imag < 0.
    positive_indices = np.imag(impedances) < 0
    frequencies = np.array(frequencies)[positive_indices]
    impedances = np.array(impedances)[positive_indices]
    pyimpspec_dataset = DataSet(frequencies, impedances)
    drt: TRNNLSResult = calculate_drt_tr_nnls(
        pyimpspec_dataset, lambda_value=lambda_value
    )
    # Assume that the peaks are sharp. So the adjacent discrete time
    # constants have all the weight of every peak (rest of the gammas
    # should be zero).
    # Since Z = R_0 + ∫₀∞ gamma / (1 + s w) dw, the actual resistances
    # are the integrals around the peaks.
    # pyimpspec uses a logarithmic transform for the integral, i.e.,
    # the "dw" is actually a "d(log(w))".
    peaks = drt.get_peaks(threshold=0.0)
    peak_indices = [
        find_occurrences(drt.time_constants, time_constant)[0]
        for time_constant in peaks[0]
    ]
    # peak_values = peaks[1]
    resistances = []
    for pi in peak_indices:
        peak_surrounding = np.array(
            [0] + list(drt.gammas[pi - 1: pi + 2]) + [0]
        )
        extended_time_constants = (
            [np.exp(
                2 * np.log(drt.time_constants[0])
                - np.log(drt.time_constants[1])
            )]
            + list(drt.time_constants)
            + [np.exp(
                2 * np.log(drt.time_constants[-1])
                - np.log(drt.time_constants[-2])
            )]
        )
        log_time_constants = np.log(extended_time_constants)
        # Count from one index further, since the front got appended to.
        integration_weights = np.diff(log_time_constants[pi - 1: pi + 4])
        peak_integral = np.sum(
            0.5 * (peak_surrounding[1:] + peak_surrounding[:-1])
            * integration_weights
        )
        resistances.append(peak_integral)
    return (drt.time_constants[peak_indices], resistances, drt)


def laplace_transform(x, y, s):
    """
    Performs a basic Laplace transformation.

    :param x:
        The independent variable.
    :param y:
        The dependent variable.
    :param s:
        The (possibly complex) frequencies for which to perform the
        transform.
    :returns:
        The evaluation of the laplace transform at *s*.
    """

    x = np.array(x)
    y = np.array(y)
    Δx = x[1:] - x[:-1]
    x_int = 0.5 * (x[1:] + x[:-1])
    y_int = 0.5 * (y[1:] + y[:-1])
    return np.sum(y_int * np.exp(-(s[:, None] + 0.0j) * x_int) * Δx, axis=1)


def a_fit(γUeminus1):
    """
    Calculates the conversion from "A parametric OCV model" (Birkl).

    :param γUeminus1:
        γ * Uᵢ / e (see "A parametric OCV model").
    :returns:
        The approximation factor aᵢ from "A parametric OCV model".
    """

    return 0.0789207 / (γUeminus1 - 0.08048093)


def OCV_fit_function(
    E_OCV,
    *args,
    z=1.0,
    T=298.15,
    individual=False,
    fit_SOC_range=False,
    rescale=False,
):
    """
    The OCV model from "A parametric OCV model".

    Reference
    ----------
    C. R. Birkl, E. McTurk, M. R. Roberts, P. G. Bruce and D. A. Howey.
    “A Parametric Open Circuit Voltage Model for Lithium Ion Batteries”.
    Journal of The Electrochemical Society, 162(12):A2271-A2280, 2015

    :param E_OCV:
        The voltages for which the SOCs shall be evaluated.
    :param *args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from the paper referenced above in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...). If *fit_SOC_range*
        is True, two additional arguments are at the front (see there).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param individual:
        If True, the model function summands are not summed up.
    :param fit_SOC_range:
        If True, this function takes two additional arguments at the
        start of *args* that may be used to adjust the data SOC range.
    :param rescale:
        If True, the expected *args* now contain the slopes of the
        summands at their respective origins instead of a. Formula:
        ``a / slope = -4 * (k_B / e) * T / (Δx * z)``.
    :returns:
        The evaluation of the model function. This is the referenced fit
        function on a linearly transformed SOC range if *fit_SOC_range*
        is True. If *individual* is True, the individual summands in the
        model function get returned as a list.
    """

    if fit_SOC_range:
        offset = 2
        SOC_start = args[0]
        SOC_end = args[1]
    else:
        offset = 0
        SOC_start = 0
        SOC_end = 1

    if (len(args) - offset) % 3 and (len(args) - offset + 1) % 3:
        raise ValueError("OCV_fit_function: incorrect length of args.")

    E_0 = np.array(args[offset + 0::3])
    temp_args = np.array(args[offset + 1::3])
    Δx = np.array(args[offset + 2::3])

    E_OCV = np.array(E_OCV)

    # Ensure that the sum of Δx is 1.
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

    # Rescale to obtain a.
    if rescale:
        a = -temp_args * 4 * (k_B / e) * T / (Δx * z)
    else:
        a = temp_args

    try:
        exponential = np.exp((E_OCV[:, None] - E_0) * a * z * e / (k_B * T))
        axis = 1
    except (IndexError, TypeError):
        exponential = np.exp((E_OCV - E_0) * a * z * e / (k_B * T))
        axis = 0
    summands = Δx / (1 + exponential)

    if individual:
        return [np.nan_to_num(SOC_start / len(Δx) + s * (SOC_end - SOC_start))
                for s in summands]
    else:
        return np.nan_to_num(
            SOC_start + np.sum(summands, axis=axis) * (SOC_end - SOC_start)
        )


def d_dE_OCV_fit_function(
    E_OCV,
    *args,
    z=1.0,
    T=298.15,
    individual=False,
    fit_SOC_range=False,
    rescale=False,
):
    """
    The derivative of ``fitting_functions.OCV_fit_function``.

    :param E_OCV:
        The voltages for which the derivative shall be evaluated.
    :param *args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param individual:
        If True, the model function summands are not summed up.
    :param fit_SOC_range:
        If True, this function takes two additional arguments at the
        start of "args" that may be used to adjust the data SOC range.
    :param rescale:
        If True, the expected "args" now contain the slopes of the
        summands at their respective origins instead of a. Formula:
        ``a / slope = -4 * (k_B / e) * T / (Δx * z)``.
    :return:
        The evaluation of ∂OCV_fit_function(OCV) / ∂OCV.
    """

    if fit_SOC_range:
        offset = 2
        SOC_start = args[0]
        SOC_end = args[1]
    else:
        offset = 0
        SOC_start = 0
        SOC_end = 1

    if (len(args) - offset) % 3 and (len(args) - offset + 1) % 3:
        raise ValueError("d_dE_OCV_fit_function: incorrect length of args.")

    E_0 = np.array(args[offset + 0::3])
    temp_args = np.array(args[offset + 1::3])
    Δx = np.array(args[offset + 2::3])

    E_OCV = np.array(E_OCV)

    # Ensure that the sum of Δx is 1
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

    # Rescale to obtain a.
    if rescale:
        a = -temp_args * 4 * (k_B / e) * T / (Δx * z)
    else:
        a = temp_args

    try:
        exponential = np.exp((E_OCV[:, None] - E_0) * a * z * e / (k_B * T))
        axis = 1
    except (IndexError, TypeError):
        exponential = np.exp((E_OCV - E_0) * a * z * e / (k_B * T))
        axis = 0
    summands = -Δx * a * z * e / (k_B * T) * exponential / (1 + exponential)**2

    if individual:
        return [np.nan_to_num(s * (SOC_end - SOC_start)) for s in summands]
    else:
        return np.nan_to_num(np.sum(summands, axis=axis)) * (
            SOC_end - SOC_start
        )


def d2_dE2_OCV_fit_function(
    E_OCV,
    *args,
    z=1.0,
    T=298.15,
    individual=False,
    fit_SOC_range=False,
    rescale=False,
):
    """
    The 2ⁿᵈ derivative of ``fitting_functions.OCV_fit_function``.

    :param E_OCV:
        The voltages for which the 2ⁿᵈ derivative shall be evaluated.
    :param *args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param individual:
        If True, the model function summands are not summed up.
    :param fit_SOC_range:
        If True, this function takes two additional arguments at the
        start of *args* that may be used to adjust the data SOC range.
    :param rescale:
        If True, the expected *args* now contain the slopes of the
        summands at their respective origins instead of a. Formula:
        ``a / slope = -4 * (k_B / e) * T / (Δx * z)``.
    :returns:
        The evaluation of ∂²OCV_fit_function(OCV) / ∂OCV².
    """

    if fit_SOC_range:
        offset = 2
        SOC_start = args[0]
        SOC_end = args[1]
    else:
        offset = 0
        SOC_start = 0
        SOC_end = 1

    if (len(args) - offset) % 3 and (len(args) - offset + 1) % 3:
        raise ValueError("d2_dE2_OCV_fit_function: incorrect length of args.")

    E_0 = np.array(args[offset + 0::3])
    temp_args = np.array(args[offset + 1::3])
    Δx = np.array(args[offset + 2::3])

    E_OCV = np.array(E_OCV)

    # Ensure that the sum of Δx is 1
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

    # Rescale to obtain a.
    if rescale:
        a = -temp_args * 4 * (k_B / e) * T / (Δx * z)
    else:
        a = temp_args

    try:
        exponential = np.exp((E_OCV[:, None] - E_0) * a * z * e / (k_B * T))
        axis = 1
    except (IndexError, TypeError):
        exponential = np.exp((E_OCV - E_0) * a * z * e / (k_B * T))
        axis = 0
    summands = (
        Δx * a * z * e / (k_B * T) * exponential / (1 + exponential)**2
        * (2 / (1 + exponential) - a * z * e / (k_B * T))
    )

    if individual:
        return [np.nan_to_num(s * (SOC_end - SOC_start)) for s in summands]
    else:
        return np.nan_to_num(np.sum(summands, axis=axis)) * (
            SOC_end - SOC_start
        )


def inverse_OCV_fit_function(SOC, *args, z=1.0, T=298.15, inverted=True):
    """
    The inverse of ``fitting_functions.OCV_fit_function``.

    Approximately OCV(SOC). Requires that Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    :param E_OCV:
        The SOCs for which the voltages shall be evaluated.
    :param *args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param inverted:
        False uses the formulation from "A parametric OCV model".
        The default True flips the SOC argument internally to correspond
        to the more widely adopted convention for the SOC direction.
    :returns:
        The evaluation of the inverse model function.
    """

    if inverted:
        try:
            SOC = [1 - entry for entry in SOC]
        except TypeError:
            SOC = 1 - SOC

    E_0 = args[0::3]
    a = args[1::3]
    Δx = np.array(args[2::3])

    e = 1.602176634e-19
    k_B = 1.380649e-23

    # Define sensible bounds for the "root_scalar" optimizers.
    bracket = [np.min(E_0) - 7 * k_B * T / (np.min(np.abs(a)) * z * e),
               np.max(E_0) + 7 * k_B * T / (np.min(np.abs(a)) * z * e)]

    try:
        len(SOC)
    except TypeError:
        initial = E_0[(np.cumsum(Δx[::-1]) - SOC).argmin()]
        try:
            return root_scalar(
                lambda E: OCV_fit_function(E, *args, z=z, T=T)
                - SOC, method="toms748", bracket=bracket, x0=initial
            ).root
        #    fprime=lambda E: d_dE_OCV_fit_function(E, *args, z=z, T=T),
        #    fprime2=lambda E: d2_dE2_OCV_fit_function(E, *args, z=z, T=T)
        except ValueError:
            return 0.0

    roots = []
    try:
        initial = E_0[(np.cumsum(Δx[::-1]) - SOC[0]).argmin()]
    except ValueError:
        initial = E_0[0]
    for SOC_i in SOC:
        try:
            root = root_scalar(
                lambda E: OCV_fit_function(E, *args, z=z, T=T) - SOC_i,
                method="toms748", bracket=bracket, x0=initial
            ).root
            initial = root
        except ValueError:
            root = float("NaN")
        #    fprime=lambda E: d_dE_OCV_fit_function(E, *args, z=z, T=T),
        #    fprime2=lambda E: d2_dE2_OCV_fit_function(E, *args, z=z, T=T)
        roots.append(root)

    return np.array(roots)


def inverse_d_dSOC_OCV_fit_function(
    SOC, *args, z=1.0, T=298.15, inverted=True
):
    """!
    The derivative of the inverse of ``OCV_fit_function``.

    Approximately OCV'(SOC). Requires that Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    :param E_OCV:
        The SOCs for which the voltages shall be evaluated.
    :param args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param inverted:
        False uses the formulation from "A parametric OCV model".
        The default True flips the SOC argument internally to correspond
        to the more widely adopted convention for the SOC direction.
    :returns:
        The evaluation of the derivative of the inverse model function.
    """

    roots = np.array(inverse_OCV_fit_function(
        SOC, *args, z=z, T=T, inverted=inverted
    ))

    return 1 / d_dE_OCV_fit_function(roots, *args, z=z, T=T)


def inverse_d2_dSOC2_OCV_fit_function(
    SOC, *args, z=1.0, T=298.15, inverted=True
):
    """
    The 2ⁿᵈ derivative of the inverse of ``OCV_fit_function``.

    Approximately OCV''(SOC). Requires that Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    :param E_OCV:
        The SOCs for which the voltages shall be evaluated.
    :param args:
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param inverted:
        False uses the formulation from "A parametric OCV model".
        The default True flips the SOC argument internally to correspond
        to the more widely adopted convention for the SOC direction.
    :returns:
        The second derivative of the inverse model function.
    """

    roots = np.array(inverse_OCV_fit_function(
        SOC, *args, z=z, T=T, inverted=inverted
    ))

    return (
        -d2_dE2_OCV_fit_function(roots, *args, z=z, T=T)
        / d_dE_OCV_fit_function(roots, *args, z=z, T=T)**3
    )


def fit_OCV(
    SOC,
    OCV,
    N=4,
    SOC_range_bounds=(0.2, 0.8),
    SOC_range_limits=(0.0, 1.0),
    z=1.0,
    T=298.15,
    inverted=True,
    fit_SOC_range=True,
    distance_order=2,
    weights=None,
    initial_parameters=None,
    minimize_options=None,
):
    """
    Fits data to ``fitting_functions.OCV_fit_function``.

    In addition to the fit itself, a model-based correction
    to the provided SOC-OCV-data is made. If *SOC* lives in a
    (0,1)-range, the correction is given as its transformation to the
    (SOC_start, SOC_end)-range given as the first two returned numbers.

    :param SOC:
        The SOCs at which measurements were made.
    :param OCV:
        The corresponding open-circuit voltages.
    :param N:
        The number of phases of the OCV model.
    :param SOC_range_bounds:
        Optional hard upper and lower bounds for the SOC correction from
        the left and the right side, respectively, as a 2-tuple. Use it
        as a limiting guess for the actual SOC range represented in the
        measurement. Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to
        effectively disable SOC range estimation.
    :param SOC_range_limits:
        Optional hard lower and upper bounds for the SOC correction from
        the left and the right side, respectively, as a 2-tuple. Use it
        if you know that your OCV data is incomplete and by how much.
        Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to allow the
        SOC range estimation to assign datapoints to the asymptotes.
    :param z:
        The charge number of the electrode interface reaction.
    :param T:
        The temperature of the electrode.
    :param inverted:
        If True (default), the widely adopted SOC convention is assumed.
        If False, the formulation of "A parametric OCV model" is used.
    :param fit_SOC_range:
        If True (default), a model-based correction to the provided
        SOC-OCV-data is made.
    :param distance_order:
        The order of the norm of the vector of the distances between OCV
        data and OCV model. Default is 2, i.e., the Euclidean norm.
        1 sets it to absolute distance, and ``float('inf')`` sets it to
        maximum distance. Note that 1 will lead to worse performance.
    :param weights:
        Optional weights to apply to the vector of the distances between
        OCV data and OCV model. Defaults to equal weights.
    :param initial_parameters:
        Optional initial guess for the model parameters. If left as-is,
        this will be automatically gleaned from the data. Use only if
        you have another fit to data of the same electrode material.
    :param minimize_options:
        Dictionary that gets passed to scipy.optimize.minimize with the
        method ``trust-constr``. See scipy.optimize.show_options with
        the arguments 'minimize' and 'trust-constr' for details.
    :returns:
        The fitted parameters of ``fitting_functions.OCV_fit_function``
        plus the fitted SOC range prepended.
    """

    e = 1.602176634e-19
    k_B = 1.380649e-23

    # The OCV fit function interprets SOC inversely to the popular way.
    if inverted:
        SOC = [1 - entry for entry in SOC]
    if weights is None:
        SOC, OCV = zip(*sorted(zip(SOC, OCV)))
        weights = np.array([1.0] * len(SOC))
    else:
        SOC, OCV, weights = zip(*sorted(zip(SOC, OCV, weights)))
        weights = np.array(weights)
    SOC = np.array(SOC)
    OCV = np.array(OCV)

    # Get the SOC correction bounds for the estimation procedure.
    SOC_start = SOC_range_bounds[0]
    SOC_end = SOC_range_bounds[1]
    SOC_limit_start = SOC_range_limits[0]
    SOC_limit_end = SOC_range_limits[1]
    if inverted:
        # [0, 1] ⊂ [-(y - 1), 1 - x]; x = x_start, y = x_end
        # [-(y - 1), 1 - x] → [0, 1], z ↦ (z + y - 1) / (y - x)
        # [0, 1] ↦ [(y - 1) / (y - x), y / (y - x)]
        x_start = (SOC_end - 1) / (SOC_end - SOC_start)
        x_end = SOC_end / (SOC_end - SOC_start)
        x_limit_start = (SOC_limit_end - 1) / (SOC_limit_end - SOC_limit_start)
        x_limit_end = SOC_limit_end / (SOC_limit_end - SOC_limit_start)
    else:
        # [0, 1] ⊂ [x, y]; x = x_start, y = x_end
        # [x, y] → [0, 1], z ↦ (z - x) / (y - x)
        # [0, 1] ↦ [- x / (y - x), (1 - x) / (y - x)]
        x_start = SOC_start / (SOC_start - SOC_end)
        x_end = (SOC_start - 1) / (SOC_start - SOC_end)
        x_limit_start = SOC_limit_start / (SOC_limit_start - SOC_limit_end)
        x_limit_end = (SOC_limit_start - 1) / (SOC_limit_start - SOC_limit_end)

    OCV_for_clustering = np.log(OCV.reshape(-1, 1))
    labels = KMeans(n_clusters=N, n_init='auto').fit_predict(
        OCV_for_clustering
    )
    E_0 = [np.median(OCV[labels == i]) for i in range(N)]
    Δx = []
    for i in range(N):
        Δx.append(np.abs(
            (SOC[labels == i][-1] - SOC[labels == i][0])
            / (SOC[-1] - SOC[0])
        ))
    # Δx = np.array([np.max([dx, 0.01]) for dx in Δx])
    # Δx = np.array(Δx) / np.sum(Δx)
    # Motivation for the weights: E(|X-µ|) = σ * √(2 / π)
    SOC_std_estimate = list(np.abs(
        SOC[1:-1] - 0.5 * (SOC[2:] + SOC[:-2])
    ) * np.sqrt(0.5 * np.pi))
    SOC_std_estimate = np.array(
        [SOC_std_estimate[0]]
        + SOC_std_estimate
        + [SOC_std_estimate[-1]]
    )
    ICA = smooth_fit(
        OCV, SOC, w=1 / SOC_std_estimate
    ).derivative()(OCV)
    slopes = [np.median(ICA[labels == i]) for i in range(N)]
    slopes = [s if s > 0.0 else 1.0 for s in slopes]

    if initial_parameters is None:
        initial = (
            [0.0, 1.0] * fit_SOC_range
            + [par for i in range(N)
               for par in (E_0[i], slopes[i], Δx[i])][:-1]
        )
    else:
        # Replace "a" with "slope" in the fit result.
        initial_a = initial_parameters[1::3]
        initial_Δx = initial_parameters[2::3]
        if len(initial_Δx) < len(initial_a):
            initial_Δx = np.append(initial_Δx, 1 - np.sum(initial_Δx))
        for i, (a, dx) in enumerate(zip(initial_a, initial_Δx)):
            initial_parameters[1 + 3 * i] = (
                -a / 4 * e / (k_B * T) * (dx * z)
            )
        initial = (
            [0.0, 1.0] * fit_SOC_range + list(initial_parameters)
        )

    E_0_ranges = [
        (np.min(OCV[labels == i]), np.max(OCV[labels == i])) for i in range(N)
    ]
    slope_ranges = [(0.0, 100.0) for _ in range(N)]
    Δx_ranges = [(0.0, 1.0) for _ in range(N)]

    bounds = list(zip(
        [x_start, x_limit_end] * fit_SOC_range
        + [par[0] for i in range(N)
           for par in (E_0_ranges[i], slope_ranges[i], Δx_ranges[i])][:-1],
        [x_limit_start, x_end] * fit_SOC_range
        + [par[1] for i in range(N)
           for par in (E_0_ranges[i], slope_ranges[i], Δx_ranges[i])][:-1]
    ))

    # Make sure that the sum of Δx without the implicit one of them is
    # not larger than 1.
    Δx_summation = np.array(
        [0, 0] * fit_SOC_range + (N - 1) * [0, 0, 1] + [0, 0]
    )
    Δx_constraint = so.LinearConstraint(Δx_summation, -np.inf, 1, True)

    if distance_order < float('inf'):
        if distance_order % 2:
            optimize_result = minimize(
                lambda x: np.sum(np.abs(
                    ((
                        OCV_fit_function(
                            OCV, *x, z=z, T=T, fit_SOC_range=True, rescale=True
                        ) - SOC
                    ) * weights)**distance_order / (
                        1 + d_dE_OCV_fit_function(
                            OCV, *x, z=z, T=T, fit_SOC_range=True, rescale=True
                        )**distance_order
                    )
                ))**(1 / distance_order),
                x0=initial,
                bounds=bounds,
                method='trust-constr',
                constraints=Δx_constraint,
                options=minimize_options,
            )
        else:
            optimize_result = minimize(
                lambda x: np.sum(
                    ((
                        OCV_fit_function(
                            OCV, *x, z=z, T=T, fit_SOC_range=True, rescale=True
                        ) - SOC
                    ) * weights)**distance_order / (
                        1 + d_dE_OCV_fit_function(
                            OCV, *x, z=z, T=T, fit_SOC_range=True, rescale=True
                        )**distance_order
                    )
                )**(1 / distance_order),
                jac='cs',
                x0=initial,
                bounds=bounds,
                method='trust-constr',
                constraints=Δx_constraint,
                options=minimize_options,
            )
    else:
        optimize_result = minimize(
            lambda x: np.amax(
                (OCV_fit_function(
                    OCV, *x, z=z, T=T, fit_SOC_range=True, rescale=True
                ) - SOC) * weights
            ),
            jac='cs',
            x0=initial,
            bounds=bounds,
            method='trust-constr',
            constraints=Δx_constraint,
            options=minimize_options,
        )
    fit = optimize_result.x
    slope_fit = fit[fit_SOC_range * 2 + 1::3]
    Δx_fit = fit[fit_SOC_range * 2 + 2::3]
    Δx_fit = np.append(Δx_fit, 1 - np.sum(Δx_fit))
    # Replace "slope" with "a" in the fit result.
    for i, (s, dx) in enumerate(zip(slope_fit, Δx_fit)):
        fit[fit_SOC_range * 2 + 1 + 3 * i] = (
            -s * 4 * (k_B / e) * T / (dx * z)
        )

    if fit_SOC_range:
        x_start = fit[0]
        x_end = fit[1]
        if inverted:
            # [0, 1] ⊂ [-(y - 1), 1 - x]; x = x_start, y = x_end
            # [-(y - 1), 1 - x] → [0, 1], z ↦ (z + y - 1) / (y - x)
            # [0, 1] ↦ [(y - 1) / (y - x), y / (y - x)]
            fit[0] = (x_end - 1) / (x_end - x_start)
            fit[1] = x_end / (x_end - x_start)
        else:
            # [0, 1] ⊂ [x, y]; x = x_start, y = x_end
            # [x, y] → [0, 1], z ↦ (z - x) / (y - x)
            # [0, 1] ↦ [- x / (y - x), (1 - x) / (y - x)]
            fit[0] = - x_start / (x_end - x_start)
            fit[1] = (1 - x_start) / (x_end - x_start)
    else:
        fit = [0, 1] + fit

    if inverted:
        SOC = SOC[::-1]

    return OCV_fit_result(fit, SOC, OCV, optimize_result=optimize_result)


def verbose_spline_parameterization(
    coeffs,
    knots,
    order,
    format='python',
    function_name="OCV",
    function_args="SOC",
    derivatives=0,
    spline_transformation='',
    verbose=False,
):
    """
    Gives the monomic representation of a B-spline.

    :param coeffs:
        The B-spline coefficients as used in ``scipy.interpolate``.
    :param knots:
        The B-spline knots as used in ``scipy.interpolate``.
    :param order:
        The order of the B-spline.
    :param format:
        Gives the file/language format for the function representation.
        Default is 'python'. The other choice is 'matlab'.
    :param function_name:
        An optional name for the printed Python function.
    :param function_args:
        An optional string for the arguments of the function.
    :param derivatives:
        The derivatives of the spline to also include in the return.
        Default is 0, which gives the spline. 1 would give the spline,
        followed by its derivative. Can not be higher than spline order.
        Derivatives are only continuous when ``derivatives < order``.
    :param spline_transformation:
        Give a string if you want to include a function that gets
        applied to the whole spline, e.g. 'exp'. Note that only this
        one extra case gets handled for the first derivative.
    :param verbose:
        Print information about the progress of the conversion.
    :returns:
        A string that gives a Python function when "exec"-uted.
    """

    if derivatives > 1:
        raise NotImplementedError(
            "Spline derivatives greater than 1 are not implemented."
        )
    if derivatives > order:
        raise ValueError(
            "A " + str(order) + "-order spline can not be differentiated "
            + str(derivatives) + " times."
        )

    # Ensure that 'function_name' and 'function_args' are valid Python
    # variable names. This convention should work for MatLab as well.
    function_name = re.sub(r'\W|^(?=\d)', '_', function_name)
    function_args = re.sub(r'\W|^(?=\d)', '_', function_args)

    tab = " " * 4
    nl = " ..." + linesep if format == 'matlab' else linesep
    if format == 'python':
        func = "def " + function_name + "(" + function_args + "):" + nl
        func += tab + "return " + spline_transformation + "("
    elif format == 'matlab':
        func = ("function [U] = " + function_name + "(" + function_args
                + ")" + nl + "U = (")
    else:
        func = ""

    t = [knots[0]] * order + list(knots) + [knots[-1]] * order

    if order == 1:

        # "p_c": shorthand for piecewise coefficients.
        p_c = np.zeros([len(knots) - 1, order + 1])

        for i in range(len(knots) - 1 + order):
            if t[i] != t[i + 1]:
                p_c[i - order][1] = p_c[i - order][1] + coeffs[i] * (
                    1 / (t[i + 1] - t[i])
                )
                p_c[i - order][0] = p_c[i - order][0] + coeffs[i] * (
                    -t[i] / (t[i + 1] - t[i])
                )
            if t[i + 1] != t[i + 2]:
                p_c[i - order + 1][1] = p_c[i + 1 - order][1] + coeffs[i] * (
                    -1 / (t[i + 2] - t[i + 1])
                )
                p_c[i - order + 1][0] = p_c[i + 1 - order][0] + coeffs[i] * (
                    t[i + 2] / (t[i + 2] - t[i + 1])
                )

        for i, poly in enumerate(p_c):
            func += nl
            if i == 0:
                if len(p_c) > 1:
                    func += (tab * 2 + "  (" + function_args + " < "
                             + str(knots[i + 1]) + ") * ")
                else:
                    func += tab * 2 + "  "
            elif i == len(p_c) - 1:
                func += (tab * 2 + "+ (" + function_args + " >= "
                         + str(knots[i]) + ") * ")
            else:
                func += (tab * 2 + "+ (" + function_args + " >= "
                         + str(knots[i]) + ") * (" + function_args + " < "
                         + str(knots[i + 1]) + ") * ")
            func += ("(" * (len(poly) - 1) + nl + tab * 3 + "  "
                     + str(poly[order]))
            for j in range(len(poly) - 2):
                func += (" * " + function_args + "" + nl + tab * 3 + "+ "
                         + str(poly[order - j - 1]) + ")")
            func += (" * " + function_args + nl + tab * 3 + "+ " + str(poly[0])
                     + ")")

        func += (nl + tab + ")" + nl + "end" if format == 'matlab' else
                 nl + tab + ")" + nl)

        if derivatives == 1:

            func += nl
            if spline_transformation == '':
                if format == 'python':
                    func += ("def derivative_" + function_name + "("
                             + function_args + "):" + nl)
                    func += tab + "return ("
                elif format == 'matlab':
                    func += ("function [U] = derivative_" + function_name + "("
                             + function_args + ")" + nl + "U = (")
            elif spline_transformation == 'exp':
                if format == 'python':
                    func += ("def derivative_" + function_name + "("
                             + function_args + "):" + nl)
                    func += (tab + "return " + function_name + "("
                             + function_args + ") * (")
                elif format == 'matlab':
                    func += ("function [U] = derivative_" + function_name + "("
                             + function_args + ")" + nl + "U = (")
                    func += (tab + "return " + function_name + "("
                             + function_args + ") * (")
            else:
                raise ValueError(
                    "verbose_spline_parameterization got a "
                    "spline_transformation that is not supported: "
                    + spline_transformation + ". Use '' or 'exp'."
                )

            for i, poly in enumerate(p_c):
                func += nl
                if i == 0:
                    if len(p_c) > 1:
                        func += (tab * 2 + "  (" + function_args + " < "
                                 + str(knots[i + 1]) + ") * ")
                    else:
                        func += tab * 2 + "  "
                elif i == len(p_c) - 1:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(knots[i]) + ") * ")
                else:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(knots[i]) + ") * (" + function_args + " < "
                             + str(knots[i + 1]) + ") * ")
                # func += ("(" * (len(poly) - 2) + nl + tab * 3 + "  "
                #          + str(order * poly[order]))
                # for j in range(len(poly) - 3):
                #     func += (" * " + function_args + "" + nl + tab * 3 + "+ "
                #              + str((order - j - 1) * poly[order - j - 1])
                #              + ")")
                func += "(" + nl + tab * 3 + str(poly[1]) + ")"

            func += (nl + tab + ")" + nl + "end" if format == 'matlab' else
                     nl + tab + ")" + nl)

    elif order == 2:

        # "p_c": shorthand for piecewise coefficients.
        p_c = np.zeros([len(knots) - 1, order + 1])

        for i in range(len(knots) - 1 + order):
            if t[i] != t[i + 1] and t[i] != t[i + 2]:
                p_c[i - order][2] = p_c[i - order][2] + coeffs[i] * (
                    1 / ((t[i + 2] - t[i]) * (t[i + 1] - t[i]))
                )
                p_c[i - order][1] = p_c[i - order][1] + coeffs[i] * (
                    -2 * t[i] / ((t[i + 2] - t[i]) * (t[i + 1]-t[i]))
                )
                p_c[i - order][0] = p_c[i - order][0] + coeffs[i] * (
                    t[i]**2 / ((t[i + 2] - t[i]) * (t[i + 1]-t[i]))
                )
            if t[i + 1] != t[i + 2]:
                if t[i] != t[i + 2]:
                    p_c[i - order + 1][2] = (
                        p_c[i - order + 1][2] + coeffs[i] * (
                            -1 / ((t[i + 2] - t[i + 1]) * (t[i + 2] - t[i]))
                        )
                    )
                    p_c[i - order + 1][1] = (
                        p_c[i - order + 1][1] + coeffs[i] * (
                            (t[i + 2] + t[i])
                            / ((t[i + 2] - t[i + 1]) * (t[i + 2] - t[i]))
                        )
                    )
                    p_c[i - order + 1][0] = (
                        p_c[i - order + 1][0] + coeffs[i] * (
                            -t[i]
                            * t[i + 2] / ((t[i + 2] - t[i + 1])
                                          * (t[i + 2] - t[i]))
                        )
                    )
                if t[i + 1] != t[i + 3]:
                    p_c[i - order + 1][2] = p_c[i - order + 1][2] + coeffs[i]*(
                        -1 / ((t[i + 2] - t[i + 1]) * (t[i + 3] - t[i + 1])))
                    p_c[i - order + 1][1] = p_c[i - order + 1][1] + coeffs[i]*(
                        (t[i + 3] + t[i + 1]) / ((t[i + 2] - t[i + 1])
                                                 * (t[i + 3] - t[i + 1])))
                    p_c[i - order + 1][0] = p_c[i - order + 1][0] + coeffs[i]*(
                        -t[i + 3] * t[i + 1] / ((t[i + 2] - t[i + 1])
                                                * (t[i + 3] - t[i + 1])))
            if t[i + 1] != t[i + 3] and t[i + 2] != t[i + 3]:
                p_c[i - order + 2][2] = p_c[i - order + 2][2] + coeffs[i] * (
                    1 / ((t[i + 3] - t[i + 1]) * (t[i + 3] - t[i + 2]))
                )
                p_c[i - order + 2][1] = p_c[i - order + 2][1] + coeffs[i] * (
                    -2 * t[i + 3]
                    / ((t[i + 3] - t[i + 1]) * (t[i + 3] - t[i + 2]))
                )
                p_c[i - order + 2][0] = p_c[i - order + 2][0] + coeffs[i] * (
                    t[i + 3]**2
                    / ((t[i + 3] - t[i + 1]) * (t[i + 3] - t[i + 2]))
                )

        for i, poly in enumerate(p_c):
            func += nl
            if i == 0:
                if len(p_c) > 1:
                    func += (tab * 2 + "  (" + function_args + " < "
                             + str(knots[i + 1]) + ") * ")
                else:
                    func += tab * 2 + "  "
            elif i == len(p_c) - 1:
                func += (tab * 2 + "+ (" + function_args + " >= "
                         + str(knots[i]) + ") * ")
            else:
                func += (tab * 2 + "+ (" + function_args + " >= "
                         + str(knots[i]) + ") * (" + function_args + " < "
                         + str(knots[i + 1]) + ") * ")
            func += ("(" * (len(poly) - 1) + nl + tab * 3 + "  "
                     + str(poly[order]))
            for j in range(len(poly) - 2):
                func += (" * " + function_args + nl + tab * 3 + "+ "
                         + str(poly[order - j - 1]) + ")")
            func += (" * " + function_args + nl + tab * 3 + "+ " + str(poly[0])
                     + ")")

        func += (nl + tab + ")" + nl + "end" if format == 'matlab' else
                 nl + tab + ")" + nl)

        if derivatives == 1:

            func += nl
            if spline_transformation == '':
                if format == 'python':
                    func += (
                        "def derivative_" + function_name + "("
                        + function_args + "):" + nl
                    )
                    func += tab + "return ("
                elif format == 'matlab':
                    func += (
                        "function [U] = derivative_" + function_name
                        + "(" + function_args + ")" + nl + "U = ("
                    )
            elif spline_transformation == 'exp':
                if format == 'python':
                    func += (
                        "def derivative_" + function_name + "("
                        + function_args + "):" + nl
                    )
                    func += (
                        tab + "return " + function_name + "("
                        + function_args + ") * ("
                    )
                elif format == 'matlab':
                    func += (
                        "function [U] = derivative_" + function_name
                        + "(" + function_args + ")" + nl + "U = ("
                    )
                    func += (
                        tab + "return " + function_name + "("
                        + function_args + ") * ("
                    )
            else:
                raise ValueError(
                    "verbose_spline_parameterization got a "
                    "spline_transformation that is not supported: "
                    + spline_transformation + ". Use '' or 'exp'."
                )

            for i, poly in enumerate(p_c):
                func += nl
                if i == 0:
                    if len(p_c) > 1:
                        func += (tab * 2 + "  (" + function_args + " < "
                                 + str(knots[i + 1]) + ") * ")
                    else:
                        func += tab * 2 + "  "
                elif i == len(p_c) - 1:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(knots[i]) + ") * ")
                else:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(knots[i]) + ") * (" + function_args + " < "
                             + str(knots[i + 1]) + ") * ")
                func += ("(" * (len(poly) - 2) + nl + tab * 3 + "  "
                         + str(order * poly[order]))
                for j in range(len(poly) - 3):
                    func += (" * " + function_args + nl + tab * 3 + "+ "
                             + str((order - j - 1) * poly[order - j - 1])
                             + ")")
                func += (" * " + function_args + nl + tab * 3 + "+ "
                         + str(poly[1]) + ")")

            func += (nl + tab + ")" + nl + "end" if format == 'matlab' else
                     nl + tab + ")" + nl)

    else:

        if len(knots) > 13 - 2 * order:
            raise ValueError(
                "Symbolical computation of a higher-order spline "
                "with this many knots would take too long. Reduce the amount "
                "of knots or use an order 2 spline for dense interpolation."
            )

        x = sp.Symbol("x")

        # Definition of B-splines by recursion.
        def B(i, k=order):

            if k > 0:
                if t[i + k] == t[i] and t[i + k + 1] == t[i + 1]:
                    return 0
                elif t[i + k] == t[i]:
                    return ((t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1])
                            * B(i + 1, k - 1))
                elif t[i + k + 1] == t[i + 1]:
                    return (x - t[i]) / (t[i + k] - t[i]) * B(i, k - 1)
                else:
                    return ((x - t[i]) / (t[i + k] - t[i]) * B(i, k - 1)
                            + (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1])
                            * B(i + 1, k - 1))

            else:
                # Note: "&" is the bit-wise operator for integers and the
                # "sp.And(⋅,⋅)" function for everything else for convenience.
                # Another note: later in the code, it is asssumed that "x"
                # is on the lhs and the interval boundary on the rhs.
                return sp.Piecewise(
                    (1, sp.And(x >= t[i], x < t[i+1])),
                    (0, True)
                )

        def B_derivative(i, k=order):

            if k > 0:
                if t[i + k] == t[i] and t[i + k + 1] == t[i + 1]:
                    return 0
                elif t[i + k] == t[i]:
                    return -k / (t[i + k + 1] - t[i + 1]) * B(i + 1, k - 1)
                elif [i + k + 1] == t[i + 1]:
                    return k / (t[i + k] - t[i]) * B(i, k - 1)
                else:
                    return (
                        k / (t[i + k] - t[i]) * B(i, k - 1)
                        - k / (t[i + k + 1] - t[i + 1])
                        * B(i + 1, k - 1)
                    )

            else:
                return 0

        if verbose:
            sp.init_printing()
            print("Simplifying the B-splines. This may take a while...")

        B_splines = []
        for i in range(len(knots) - 1 + order):
            print("Simplifying B-spline #" + str(i+1) + " of "
                  + str(len(knots) - 1 + order))
            B_splines.append(sp.simplify(B(i, order)))

        if derivatives == 1:
            B_spline_derivatives = []
            print("Simplifying derivated B-spline #" + str(i+1) + " of "
                  + str(len(knots) - 1 + order))
            B_spline_derivatives.append(sp.simplify(B_derivative(i, order)))

        if verbose:
            print("Simplifying the spline. This may take another while...")
        ip_spline = 0
        for i, B_spline in enumerate(B_splines):
            ip_spline = ip_spline + coeffs[i] * B_spline
            ip_spline = sp.simplify(ip_spline)
        if derivatives == 1:
            ip_spline_der = 0
            for i, B_spline_derivative in enumerate(B_spline_derivatives):
                ip_spline_der = ip_spline_der + coeffs[i] * B_spline_derivative
                ip_spline_der = sp.simplify(ip_spline_der)
        if verbose:
            print("Interpolating spline:")
            print()
            sp.pprint(ip_spline)
            print()
            if derivatives == 1:
                print()
                sp.pprint(ip_spline_der)
                print()

        if derivatives == 0:
            polys = [ip_spline]
        elif derivatives == 1:
            polys = [ip_spline, ip_spline_der]

        for i, poly in enumerate(polys):

            if i == 1:
                func += nl
                if spline_transformation == '':
                    if format == 'python':
                        func += ("def derivative_" + function_name + "("
                                 + function_args + "):" + nl)
                        func += tab + "return ("
                    elif format == 'matlab':
                        func += ("function [U] = derivative_" + function_name
                                 + "(" + function_args + ")" + nl + "U = (")
                elif spline_transformation == 'exp':
                    if format == 'python':
                        func += ("def derivative_" + function_name + "("
                                 + function_args + "):" + nl)
                        func += (tab + "return " + function_name + "("
                                 + function_args + ") * (")
                    elif format == 'matlab':
                        func += ("function [U] = derivative_" + function_name
                                 + "(" + function_args + ")" + nl + "U = (")
                        func += (tab + "return " + function_name + "("
                                 + function_args + ") * (")
                else:
                    raise ValueError(
                        "verbose_spline_parameterization got a "
                        "spline_transformation that is not supported: "
                        + spline_transformation + ". Use '' or 'exp'."
                    )

            polys_sorted_by_definition = [[], [], []]
            for part in ip_spline.args:
                try:
                    polynomial = sp.Poly(part[0])
                except sp.polys.polyerrors.GeneratorsNeeded:
                    # "part[0]" was 0 probably.
                    continue
                polys_sorted_by_definition[0].append(part[1].args[0].rhs)
                polys_sorted_by_definition[1].append(part[1].args[1].rhs)
                polys_sorted_by_definition[2].append(polynomial.all_coeffs())
            polys_sorted_by_definition = list(sorted(zip(
                *polys_sorted_by_definition)))

            for i, (left, right, poly) in enumerate(
                polys_sorted_by_definition
            ):
                func += nl
                if i == 0:
                    if len(polys_sorted_by_definition) > 1:
                        func += (tab * 2 + "  (" + function_args + " < "
                                 + str(right) + ") * ")
                    else:
                        func += tab * 2 + "  "
                elif i == len(polys_sorted_by_definition) - 1:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(left) + ") * ")
                else:
                    func += (tab * 2 + "+ (" + function_args + " >= "
                             + str(left) + ") * (" + function_args + " < "
                             + str(right) + ") * ")
                func += ("(" * (len(poly) - 1) + nl + tab * 3 + "  "
                         + str(poly[0]))
                for j in range(len(poly) - 2):
                    func += (" * " + function_args + nl + tab * 3 + "+ "
                             + str(poly[j + 1]) + ")")
                func += (" * " + function_args + nl + tab * 3 + "+ "
                         + str(poly[-1]) + ")")
            func += (nl + tab + ")" + nl + "end" if format == 'matlab' else
                     nl + tab + ")" + nl + nl)

    return func
