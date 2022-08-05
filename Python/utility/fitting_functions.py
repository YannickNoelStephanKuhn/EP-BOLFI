"""!@package utility.fitting_functions
Various helper and fitting functions for processing measurement curves.
"""

import numpy as np
import scipy.optimize as so
from scipy.optimize import root_scalar, minimize
from scipy import interpolate as ip
import sympy as sp


class OCV_fit_result(object):
    """!@brief Contains OCV fit parameters and related information.

    Reference
    ----------
    C. R. Birkl, E. McTurk, M. R. Roberts, P. G. Bruce and D. A. Howey.
    “A Parametric Open Circuit Voltage Model for Lithium Ion Batteries”.
    Journal of The Electrochemical Society, 162(12):A2271-A2280, 2015
    """

    def __init__(self, fit, SOC, OCV, SOC_offset=1.0, SOC_scale=1.0):
        """!
        The fit parameters of the OCV function from Birkl. et al.,
        excluding the estimated SOC range.
        """
        self.fit = np.array(fit[2:])
        """! The SOC range of the data. """
        self.SOC_range = np.array(fit[0:2])
        """! The E₀ (plateau voltages) parameters. """
        self.E_0 = np.array(fit[2::3])
        """! The a (inverse plateau widths) parameters. """
        self.a = np.array(fit[3::3])
        """! The Δx (phase proportion) parameters. """
        self.Δx = np.array(fit[4::3])
        if len(self.Δx) < len(self.E_0):
            self.Δx = np.append(self.Δx, 1 - np.sum(self.Δx))
        self.E_0, self.a, self.Δx = zip(*sorted(zip(
            self.E_0, self.a, self.Δx
        ), reverse=True))
        for i in range(len(self.E_0) - 1):
            if (
                np.abs(self.E_0[i + 1] - self.E_0[i])
                + np.abs(self.a[i + 1] - self.a[i]) < 1e-4
            ):
                print("Warning (OCV_fit_result): At least two fitted summands "
                      "coincide.")
        """! The SOC data points. """
        self.SOC = np.array(SOC)
        """! The OCV data points. May be adjusted from the original data. """
        self.OCV = np.array(OCV)
        """!
        If another electrode was factored out in the data, this may
        contain its SOC at SOC 0 of the electrode of interest.
        """
        self.SOC_offset = SOC_offset
        """!
        If another electrode was factored out in the data, this may
        contain the the rate of change of its SOC to that of the
        electrode of interest.
        """
        self.SOC_scale = SOC_scale

    def SOC_adjusted(self, soc=None):
        """!@brief Gives the adjusted SOC values.

        @par soc
            The SOC as assigned in the original data. This usually
            corresponds to the range available during a measurement.
        @return
            The SOC as corrected by the OCV model. These values will try
            to correspond to the level of lithiation.
        """

        if soc is None:
            soc = self.SOC

        return self.SOC_range[0] + soc * (
            self.SOC_range[1] - self.SOC_range[0]
        )

    def SOC_other_electrode(self, soc=None):
        """!@brief Relates the SOCs of the two electrodes to each other.

        If the original data was of a full cell and the other
        electrode was factored out, this may contain the function that
        takes the SOC of the electrode of interest and gives the SOC
        of the other electrode, i.e., the stoichiometric relation.

        @par soc
            The SOC of the electrode of interest.
        @return
            The SOC of the other electrode that was factored out.
        """

        if soc is None:
            soc = self.SOC

        return self.SOC_offset - self.SOC_scale * soc


def smooth_fit(
    x, y, order=5, splits=None, smoothing_factor=2e-3, display=False
):
    """!@brief Calculates a smoothed spline with derivatives.

    Note: the "roots" function of a spline only works if it is cubic,
    i.e. of third order. Each "derivative" reduces the order by one.

    @par x
        The independent variable.
    @par y
        The dependent variable ("plotted over x").
    @par order
        Interpolation order of the spline.
    @par splits
        Optional tuning parameter. A list of points between which
        splines will be fitted first. The returned spline then is a fit
        of these individual splines.
    @par smoothing_factor
        Optional tuning parameter. Higher values lead to coarser, but
        more smooth interpolations and vice versa.
    @par display
        If set to True, the fit parameters of the spline will be printed
        to console. If possible, a monomial representation is printed.
    @return
        A smoothing spline in the form of scipy.UnivariateSpline.
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
                    x_split, y_split, k=order,
                    s=smoothing_factor/number_of_splits
                )
                y_smoothed.extend(split_spline(x_split))

    try:
        spline = ip.UnivariateSpline(x, y_smoothed,
                                     k=order, s=smoothing_factor)
    except ValueError:
        spline = ip.UnivariateSpline(x[::-1], y_smoothed[::-1],
                                     k=order, s=smoothing_factor)

    if display:
        print("Knots of the interpolating spline:")
        print(spline.get_knots())
        print("Coefficients of the interpolating spline:")
        print(spline.get_coeffs())
        try:
            simplification = verbose_spline_parameterization(
                spline.get_coeffs(), spline.get_knots(), order,
                function_name="y", function_args="x", verbose=True
            )
            print("Monomial representation of the interpolating spline:")
            print(simplification)
        except ValueError as e:
            print(e)

    return spline


def fit_exponential_decay(
    timepoints, voltages, recursive_depth=1, threshold=0.95
):
    """!@brief Extracts a set amount of exponential decay curves.

    @par timepoints
        The timepoints of the measurements.
    @par voltages
        The corresponding voltages.
    @par recursive_depth
        The default (1) fits one exponential curve to the data. For
        higher values that fit is repeated with the data minus the
        preceding fit(s) for this amount of times minus one.
    @par threshold
        The lower threshold value for the R² coefficient of
        determination. If "threshold" is smaller than 1, the subset of
        the exponential decay data is searched that just fulfills it.
        Defaults to 0.95. Values above 1 are set to 1.
    @return
        A list of length "recursive_depth" where each element is a
        3-tuple with the timepoints, the fitted voltage evaluations
        and a 3-tuple of the parameters of the following decay function:
        t, (U_0, ΔU, τᵣ⁻¹) ↦ U_0 + ΔU * np.exp(-τᵣ⁻¹ * (t - timepoints[0])).
    """

    t_eval = np.atleast_1d(timepoints)
    u_eval = np.atleast_1d(voltages)
    threshold = threshold if threshold <= 1 else 1

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 3:
        print("Warning: fit_exponential_decay was given insufficient data.")
        return []

    def exp_fit_function(t, b, c, d, t_0=t_eval[0]):
        """! Exponential decay function. """
        return b + c * np.exp(-d * (t - t_0))

    def log_inverse_fit_function(y, b, c, d, t_0=t_eval[0]):
        """! Corresponding logarithm function. """
        log_arg = (y - b) / c
        log_arg[log_arg <= 0] = 0.1**d
        return t_0 - np.log(log_arg) / d

    end = len(t_eval) - 1
    bracket = [0, end]
    curves = []
    depth_counter = 0
    fit_guess = [u_eval[end], u_eval[end // 10] - u_eval[end],
                 1.0 / (t_eval[end] - t_eval[end // 10])]

    # Evaluate the R² value for a split at the middle of the data.
    split = int(0.5 * (bracket[0] + bracket[1]))
    argmax_R_squared = split
    fit_split = so.minimize(
        lambda x: np.sum(
            (exp_fit_function(t_eval[split:end], *x) - u_eval[split:end])**2
        )**0.5, x0=fit_guess, method='trust-constr'
    ).x
    test_t_split = log_inverse_fit_function(u_eval[split:end], *fit_split)
    R_squared_split = (
        1 - np.sum(np.nan_to_num(t_eval[split:end] - test_t_split)**2)
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
                u_eval[end], u_eval[end // 10] - u_eval[end],
                1.0 / (t_eval[end] - t_eval[end // 10])
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
                u_eval[end], u_eval[end // 10] - u_eval[end],
                1.0 / (t_eval[end] - t_eval[end // 10])
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
        test_t_left = log_inverse_fit_function(u_eval[left:end], *fit_left)
        R_squared_left = (
            1 - np.sum(np.nan_to_num(t_eval[left:end] - test_t_left)**2)
            / np.sum((t_eval[left:end] - np.mean(t_eval[left:end]))**2)
        )
        test_t_right = log_inverse_fit_function(u_eval[right:end], *fit_right)
        R_squared_right = (
            1 - np.sum(np.nan_to_num(t_eval[right:end] - test_t_right)**2)
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


def fit_sqrt(timepoints, voltages, threshold=0.95):
    """!@brief Extracts a square root at the beginning of the data.

    @par timepoints
        The timepoints of the measurements.
    @par voltages
        The corresponding voltages.
    @par threshold
        The lower threshold value for the R² coefficient of
        determination. If "threshold" is smaller than 1, the subset of
        the experimental data is searched that just fulfills it.
        Defaults to 0.95. Values above 1 are set to 1.
    @return
        A 3-tuple with the timepoints, the fitted voltage evaluations
        and a 2-tuple of the parameters of the following sqrt function:
        t, (U_0, dU_d√t) ↦ U_0 + dU_d√t * √(t - timepoints[0]).
    """

    t_eval = np.atleast_1d(timepoints)
    u_eval = np.atleast_1d(voltages)
    threshold = threshold if threshold <= 1 else 1

    # Make sure that invalid inputs don't crash anything.
    if len(t_eval) < 2:
        print("Warning: fit_sqrt was given insufficient data.")
        return []

    def sqrt_fit_function(t, b, c, t_0=t_eval[0]):
        """! Square root function. """
        return b + c * np.sqrt(t - t_0)

    def square_inverse_fit_function(y, b, c, t_0=t_eval[0]):
        """! Corresponding square function. """
        return t_0 + ((y - b) / c)**2

    end = len(t_eval) - 1
    bracket = [0, end]
    fit_guess = [u_eval[0],
                 (u_eval[end] - u_eval[0]) / np.sqrt(t_eval[end] - t_eval[0])]

    # Evaluate the R² value for a split at the middle of the data.
    split = int(0.5 * (bracket[0] + bracket[1]))
    argmax_R_squared = split
    fit_split = so.minimize(lambda x: np.sum(
            (sqrt_fit_function(t_eval[0:split], *x) - u_eval[0:split])**2
        )**0.5, x0=fit_guess, method='trust-constr').x
    test_t_split = square_inverse_fit_function(u_eval[0:split], *fit_split)
    R_squared_split = (
        1 - np.sum(np.nan_to_num(t_eval[0:split] - test_t_split)**2)
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
        test_t_left = square_inverse_fit_function(u_eval[0:left], *fit_left)
        R_squared_left = (
            1 - np.sum(np.nan_to_num(t_eval[0:left] - test_t_left)**2)
            / np.sum((t_eval[0:left] - np.mean(t_eval[0:left]))**2)
        )
        test_t_right = square_inverse_fit_function(u_eval[0:right], *fit_right)
        R_squared_right = (
            1 - np.sum(np.nan_to_num(t_eval[0:right] - test_t_right)**2)
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


def a_fit(γUeminus1):
    """!@brief Calculates the conversion from "A parametric OCV model".

    @par γUeminus1
        γ * Uᵢ / e from "A parametric OCV model".
    @return
        The approximation factor aᵢ from "A parametric OCV model".
    """

    return 0.0789207 / (γUeminus1 - 0.08048093)


def OCV_fit_function(
    E_OCV, *args, z=1.0, T=298.15, individual=False, fit_SOC_range=False
):
    """!@brief The OCV model from "A parametric OCV model".

    Reference
    ----------
    C. R. Birkl, E. McTurk, M. R. Roberts, P. G. Bruce and D. A. Howey.
    “A Parametric Open Circuit Voltage Model for Lithium Ion Batteries”.
    Journal of The Electrochemical Society, 162(12):A2271-A2280, 2015

    @par E_OCV
        The voltages for which the SOCs shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from the paper referenced above in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...). If "fit_SOC_range"
        is True, two additional arguments are at the front (see there).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par individual
        If True, the model function summands are not summed up.
    @par fit_SOC_range
        If True, this function takes two additional arguments at the
        start of "args" that may be used to adjust the data SOC range.
    @return
        The evaluation of the model function. This is the referenced fit
        function on a linearly transformed SOC range if "fit_SOC_range"
        is True. If "individual" is True, the individual summands in the
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
    a = np.array(args[offset + 1::3])
    Δx = np.array(args[offset + 2::3])

    # Ensure that the sum of Δx is 1.
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

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


def d_dE_OCV_fit_function(E_OCV, *args, z=1.0, T=298.15):
    """!@brief The derivative of fitting_functions.OCV_fit_function.

    @par E_OCV
        The voltages for which the derivative shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @return
        The evaluation of ∂OCV_fit_function(OCV) / ∂OCV.
    """

    if (len(args) + 1) % 3 and len(args) % 3:
        raise ValueError("d_dE_OCV_fit_function: incorrect length of args.")

    E_0 = np.array(args[0::3])
    a = np.array(args[1::3])
    Δx = np.array(args[2::3])

    E_OCV = np.array(E_OCV)

    # Ensure that the sum of Δx is 1
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

    try:
        exponential = np.exp((E_OCV[:, None] - E_0) * a * z * e / (k_B * T))
        axis = 1
    except (IndexError, TypeError):
        exponential = np.exp((E_OCV - E_0) * a * z * e / (k_B * T))
        axis = 0
    summands = -Δx * a * z * e / (k_B * T) * exponential / (1 + exponential)**2

    return np.nan_to_num(np.sum(summands, axis=axis))


def d2_dE2_OCV_fit_function(E_OCV, *args, z=1.0, T=298.15):
    """!@brief The 2ⁿᵈ derivative of fitting_functions.OCV_fit_function.

    @par E_OCV
        The voltages for which the 2ⁿᵈ derivative shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
        The last Δx_j may be omitted to force ∑ⱼ Δx_j = 1.
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @return
        The evaluation of ∂²OCV_fit_function(OCV) / ∂²OCV.
    """

    if (len(args) + 1) % 3 and len(args) % 3:
        raise ValueError("d2_dE2_OCV_fit_function: incorrect length of args.")

    E_0 = np.array(args[0::3])
    a = np.array(args[1::3])
    Δx = np.array(args[2::3])

    E_OCV = np.array(E_OCV)

    # Ensure that the sum of Δx is 1
    if len(Δx) < len(E_0):
        Δx = np.append(Δx, 1 - np.sum(Δx))

    e = 1.602176634e-19
    k_B = 1.380649e-23

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

    return np.nan_to_num(np.sum(summands, axis=axis))


def inverse_OCV_fit_function(SOC, *args, z=1.0, T=298.15, inverted=True):
    """!@brief The inverse of fitting_functions.OCV_fit_function.

    Basically OCV(SOC). Requires that the Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    @par E_OCV
        The SOCs for which the voltages shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par inverted
        The default (False) uses the formulation from "A parametric
        OCV model". If True, the SOC argument gets flipped internally to
        give the more widely adopted convention for the SOC direction.
    @return
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
    for SOC_i in SOC:

        initial = E_0[(np.cumsum(Δx[::-1]) - SOC_i).argmin()]
        try:
            root = root_scalar(
                lambda E: OCV_fit_function(E, *args, z=z, T=T) - SOC_i,
                method="toms748", bracket=bracket, x0=initial
            ).root
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
    @brief The derivative of the inverse of fitting_functions.OCV_fit_function.

    Basically OCV'(SOC). Requires that the Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    @par E_OCV
        The SOCs for which the voltages shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par inverted
        The default (False) uses the formulation from "A parametric
        OCV model". If True, the SOC argument gets flipped internally to
        give the more widely adopted convention for the SOC direction.
    @return
        The evaluation of the derivative of the inverse model function.
    """

    roots = np.array(inverse_OCV_fit_function(
        SOC, *args, z=z, T=T, inverted=inverted
    ))

    return 1 / d_dE_OCV_fit_function(roots, *args, z=z, T=T)


def inverse_d2_dSOC2_OCV_fit_function(
    SOC, *args, z=1.0, T=298.15, inverted=True
):
    """! The 2nd derivative of the inverse of .OCV_fit_function.

    Basically OCV'(SOC). Requires that the Δx entries are sorted by x.
    This corresponds to the parameters being sorted by decreasing E_0.

    @par E_OCV
        The SOCs for which the voltages shall be evaluated.
    @par args
        A list which length is dividable by three. These are the
        parameters E₀, a and Δx from "A parametric OCV model" in the
        order (E₀_0, a_0, Δx_0, E₀_1, a_1, Δx_1...).
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par inverted
        The default (False) uses the formulation from "A parametric
        OCV model". If True, the SOC argument gets flipped internally to
        give the more widely adopted convention for the SOC direction.
    @return
        The evaluation of the derivative of the inverse model function.
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
    initial=None,
    bounds=None,
    z=1.0,
    T=298.15,
    inverted=True,
    fit_SOC_range=True
):
    """!@brief Fits data to fitting_functions.OCV_fit_function.

    In addition to fitting_functions.fit_OCV, a model-based correction
    to the provided SOC-OCV-data is made. If "SOC" lives in a
    (0,1)-range, the correction is given as its transformation to the
    (SOC_start, SOC_end)-range given as the first two returned numbers.
    If "other_electrode" is not None, the stoichiometric offset and the
    scaling that gives the (adjusted) SOC of this electrode given the
    SOC of the "other_electrode" are put before SOC_start and SOC_end.

    @par SOC
        The SOCs at which measurements were made.
    @par OCV
        The corresponding open-circuit voltages.
    @par initial
        An optional initial guess for the parameters of the model.
    @par bounds
        Optional hard bounds for the parameters of the model. Has the
        form of a list of 2-tuples for each parameter.
    @par z
        The charge number of the electrode interface reaction.
    @par T
        The temperature of the electrode.
    @par inverted
        If True (default), the widely adopted SOC convention is assumed.
        If False, the formulation of "A parametric OCV model" is used.
    @par fit_SOC_range
        If True (default), a model-based correction to the provided
        SOC-OCV-data is made.
    @return
        The fitted parameters of fitting_functions.OCV_fit_function plus
        the fitted SOC range prepended.
    """

    # The OCV fit function interprets SOC inversely to the popular way.
    if inverted:
        SOC = [1 - entry for entry in SOC]
    SOC, OCV = zip(*sorted(zip(SOC, OCV)))
    SOC = np.array(SOC)
    OCV = np.array(OCV)

    V_min, V_max, V_mean = (np.min(OCV), np.max(OCV), np.mean(OCV))
    if initial is None:
        E_0 = [V_mean] * N
        a = [-5.0] * N
        Δx = [0.0] * (N - 1)
        initial = (
            [0.0, 1.0] * fit_SOC_range
            + [par for i in range(N - 1) for par in (E_0[i], a[i], Δx[i])]
            + [E_0[N - 1], a[N - 1]]
        )
    if bounds is None:
        bounds = list(zip(
            [-0.4, 1.0] * fit_SOC_range + [V_min, -100.0, 0.0] * (N - 1)
            + [V_min, -100.0],
            [0.0, 1.4] * fit_SOC_range + [V_max, 0.0, 1.0] * (N - 1)
            + [V_max, 0.0]
        ))

    # Make sure that the sum of Δx without the implicit one of them is
    # not larger than 1.
    Δx_summation = np.array(
        [0, 0] * fit_SOC_range + (N - 1) * [0, 0, 1] + [0, 0]
    )
    Δx_constraint = so.LinearConstraint(Δx_summation, -np.inf, 1, True)

    fit = minimize(lambda x: np.sum(
            (OCV_fit_function(OCV, *x, z=z, T=T, fit_SOC_range=fit_SOC_range)
             - SOC)**2
        )**0.5, jac='cs', x0=initial, bounds=bounds, method='trust-constr',
        constraints=Δx_constraint).x

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

    return OCV_fit_result(fit, SOC, OCV)


def verbose_spline_parameterization(
    coeffs,
    knots,
    order,
    format='python',
    function_name="OCV",
    function_args="SOC",
    verbose=False
):
    """!@brief Gives the monomic representation of a B-spline.

    @par coeffs
        The B-spline coefficients as structured by scipy.interpolate.
    @par knots
        The B-spline knots as structured by scipy.interpolate.
    @par order
        The order of the B-spline.
    @par format
        Gives the file/language format for the function representation.
        Default is 'python'. The other choice is 'matlab'.
    @par function_name
        An optional name for the printed Python function.
    @par function_args
        An optional string for the arguments of the function.
    @par verbose
        Print information about the progress of the conversion.
    @return
        A string that gives a Python function when "exec"-uted.
    """

    tab = " " * 4
    nl = " ...\r\n" if format == 'matlab' else "\r\n"
    if format == 'python':
        func = "def " + function_name + "(" + function_args + "):" + nl
        func += tab + "return ("
    elif format == 'matlab':
        func = ("function [U] = " + function_name + "(" + function_args
                + ")\r\nU = (")
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
                    (0, True))

        if verbose:
            sp.init_printing()
            print("Simplifying the B-splines. This may take a while...")

        B_splines = []
        for i in range(len(knots) - 1 + order):
            print("Simplifying B-spline #" + str(i+1) + " of "
                  + str(len(knots) - 1 + order))
            B_splines.append(sp.simplify(B(i, order)))

        if verbose:
            print("Simplifying the spline. This may take another while...")
        ip_spline = 0
        for i, B_spline in enumerate(B_splines):
            ip_spline = ip_spline + coeffs[i] * B_spline
            ip_spline = sp.simplify(ip_spline)
        if verbose:
            print("Interpolating spline:")
            print()
            sp.pprint(ip_spline)
            print()

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

        for i, (left, right, poly) in enumerate(polys_sorted_by_definition):
            func += nl
            if i == 0:
                if len(polys_sorted_by_definition) > 1:
                    func += (tab * 2 + "  (" + function_args + " < "
                             + str(right) + ") * ")
                else:
                    func += tab * 2 + "  "
            elif i == len(polys_sorted_by_definition) - 1:
                func += (tab * 2 + "+ (" + function_args + " >= " + str(left)
                         + ") * ")
            else:
                func += (tab * 2 + "+ (" + function_args + " >= " + str(left) +
                         ") * (" + function_args + " < " + str(right) + ") * ")
            func += "(" * (len(poly) - 1) + nl + tab * 3 + "  " + str(poly[0])
            for j in range(len(poly) - 2):
                func += (" * " + function_args + nl + tab * 3 + "+ "
                         + str(poly[j + 1]) + ")")
            func += (" * " + function_args + nl + tab * 3 + "+ "
                     + str(poly[-1]) + ")")

    func += (nl + tab + ")\r\nend" if format == 'matlab' else
             nl + tab + ")" + nl)
    return func
