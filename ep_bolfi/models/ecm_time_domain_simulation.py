# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.integrate import solve_ivp

from models.ecm.equivalent_circuits import ECM


def laplace_transform(y, x, s):
    """!@brief Performs a basic laplace transformation.

    @par y
        The dependent variable.
    @par x
        The independent variable.
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


def apply_ocv(
    cycles_solution,
    coulomb_to_cell_soc,
    positive_ocv,
    negative_ocv=None,
    cell_soc_to_positive_soc=lambda cell_soc: cell_soc,
    cell_soc_to_negative_soc=lambda cell_soc: 1.0 - cell_soc,
):
    """Adds the OCV curve to the result of 'simulate_cycling_protocol'.

    @par cycles_solution
        A return value of 'simulate_cycling_protocol', see there.
    @par coulomb_to_cell_soc
        A function that takes the coulomb counted capacity of the cell
        and returns the SOC of the cell.
    @par positive_ocv
        The OCV curve of the positive electrode.
    @par negative_ocv
        The OCV curve of the negative electrode.
    @par cell_soc_to_positive_soc
        Takes the cell SOC and returns the value that 'positive_ocv' expects.
    @par cell_soc_to_negative_soc
        Takes the cell SOC and returns the value that 'negative_ocv' expects.
    """
    for cycle in cycles_solution:
        cell_soc = coulomb_to_cell_soc(cycle['Q'])
        if negative_ocv:
            cell_ocv = (
                positive_ocv(cell_soc_to_positive_soc(cell_soc))
                - negative_ocv(cell_soc_to_negative_soc(cell_soc))
            )
        else:
            cell_ocv = positive_ocv(cell_soc_to_positive_soc(cell_soc))
        cycle['U'] = cycle['U'] + cell_ocv


def simulate_cycling_protocol(
    ecm: ECM,
    cycles: list,
    initial_state: np.array,
    t_eval: np.array = None,
    **kwargs
):
    """Implements cycles with variable voltage or current control.

    @par ecm
        Equivalent Circuit Model that implements the ODEs.
    @par cycles
        List of cycles. The entries are dictionaries with this structure:
        - "control": either "U" or "I" to indicate whether voltage or
            current control is desired, respectively.
        - "duration": duration of the cycle in seconds.
        - "input": callable that takes the time since the start of the
            cycle and returns the operating voltage ("U") or current
            ("I").
        - "d_dt_input": first derivative by time of the callable in
            "input".
        - "d2_dt2_input": second derivative by time of the callable in
            "input".
    @par initial_state
        Initial state of the simulation. A 3-array of the initial SOC as
        well as state and first derivative of the state that is variable
        at the start, e.g., I if voltage control is selected.
        ToDo: implement OCV and events for termination at extreme SOC.
    @par t_eval
        Time points at which the solution shall be evaluated. These do
        not influence the time points the internal solver chooses.
    @par kwargs
        Get passed on to the ODE solver. Please refer to the
        documentation of scipy.solve_ivp(..., method='RK45').
    @return
        A list of solutions for each cycle. Each list entry is a
        dictionary with "t", "I" and "U" for time points and current as
        well as voltage evaluated at those points, respectively.
    """

    whole_solution = []
    preceding_control = cycles[0]['control']
    previous_input = None
    elapsed_time = 0.0
    first_cycle = True
    dense_output = True  # False if t_eval is None else True
    cycle_solution = None
    """
    # Differential evaluation independent on interpolant
    # implementation. Induces some error as it does not perform an
    # exact polynomial regression.
    interpolant_order = 4
    if dense_output:
        last_interpolant_duration = cycle_solution.sol.interpolants[-1].h
        t_poly = np.array([
            elapsed_time + last_interpolant_duration
            * (i / interpolation_order - 1)
            for i in range(interpolation_order + 1)
        ])
        poly_fit = odr.ODR(
            odr.Data(t_poly, cycle_solution.sol(t_poly)[0]),
            odr.polynomial(interpolation_order)
        ).run()
    """
    for cycle in cycles:
        whole_input = [
            cycle['input'], cycle['d_dt_input'], cycle['d2_dt2_input']
        ]
        if cycle['control'] == 'U':
            rhs = ecm.I_rhs
            calculate_for_new_control = (
                ecm.calculate_I_ode_state_for_new_control
            )
            update_to_new_input = ecm.update_I_ode_state_to_new_input
        elif cycle['control'] == 'I':
            rhs = ecm.U_rhs
            calculate_for_new_control = (
                ecm.calculate_U_ode_state_for_new_control
            )
            update_to_new_input = ecm.update_U_ode_state_to_new_input
        else:
            raise ValueError(
                "Provided control states have to be either 'U' or 'I'."
            )
        if not first_cycle:
            if dense_output:
                old_state = cycle_solution.sol(elapsed_time)
            else:
                old_state = cycle_solution.y[:, -1]
        if preceding_control != cycle['control']:
            # Differential evaluation specific to RkDenseOutput.
            if dense_output:
                if preceding_control == 'I':
                    global_index = ecm.U_global_index
                    integrated_output = False
                elif preceding_control == 'U':
                    # may be < 0 to indicate that only Q was calculated
                    if ecm.global_I_index < 0:
                        global_index = ecm.Q_global_index
                        integrated_output = True
                    else:
                        global_index = ecm.I_global_index
                        integrated_output = False
                interpolant = cycle_solution.sol.interpolants[-1]
                # Contains the prefactors for x^i at index i.
                monomial_base_representation = (
                    [interpolant.y_old[global_index]] +
                    list(interpolant.h * interpolant.Q[global_index])
                )
                # The interpolation is defined on the interval [0, 1].
                monomial_base_first_derivative = [0] + [
                    i * monomial_base_representation[i] / interpolant.h
                    for i in range(1, len(monomial_base_representation))
                ]
                monomial_base_second_derivative = [0, 0] + [
                    i * (i - 1)
                    * monomial_base_representation[i] / interpolant.h**2
                    for i in range(2, len(monomial_base_representation))
                ]
                if integrated_output:
                    monomial_base_third_derivative = [0, 0, 0] + [
                        i * (i - 1) * (i - 2)
                        * monomial_base_representation[i] / interpolant.h**3
                        for i in range(3, len(monomial_base_representation))
                    ]
                    old_global_derivatives = [
                        np.sum(monomial_base_first_derivative),
                        np.sum(monomial_base_second_derivative),
                        np.sum(monomial_base_third_derivative)
                    ]
                else:
                    old_global_derivatives = [
                        np.sum(monomial_base_representation),
                        np.sum(monomial_base_first_derivative),
                        np.sum(monomial_base_second_derivative)
                    ]
            else:
                raise NotImplementedError(
                    "Dense output of IVP solver required for control switch."
                )
            initial_state = calculate_for_new_control(
                old_state,
                previous_input,
                old_global_derivatives,
                [i(0) for i in whole_input]
            )
        elif not first_cycle:
            initial_state = update_to_new_input(
                old_state, previous_input, [i(0) for i in whole_input]
            )
        cycle_solution = solve_ivp(
            lambda t, y: rhs(t, y, whole_input),
            (elapsed_time, elapsed_time + cycle['duration']),
            initial_state,
            dense_output=dense_output,
            **kwargs
        )
        if t_eval is None:
            t_sol = cycle_solution.t
            y_sol = cycle_solution.y
        else:
            t_sol = t_eval[
                (elapsed_time <= t_eval)
                &
                (t_eval < elapsed_time + cycle['duration'])
            ]
            y_sol = cycle_solution.sol(t_sol)
        if cycle['control'] == 'U':
            whole_solution.append({
                't': t_sol,
                'Q': y_sol[ecm.Q_global_index],
                'I': y_sol[ecm.I_global_index],
                'U': cycle['input'](t_sol - t_sol[0]),
            })
        elif cycle['control'] == 'I':
            whole_solution.append({
                't': t_sol,
                'Q': y_sol[ecm.Q_global_index],
                'I': cycle['input'](t_sol - t_sol[0]),
                'U': y_sol[ecm.U_global_index],
            })
        preceding_control = cycle['control']
        previous_input = [i(cycle['duration']) for i in whole_input]
        elapsed_time = elapsed_time + cycle['duration']
        first_cycle = False
    return whole_solution
