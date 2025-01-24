# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sympy import (
    Array, Symbol,
    lambdify, simplify, summation,
    coth, tanh,
)
from sympy.printing.pretty.pretty import pretty_print
from sympy.utilities.iterables import numbered_symbols
import torch
from torch.distributions.normal import Normal


def series(*ecms):
    i = Symbol('i')
    impedances = [ecm.EC for ecm in ecms]
    return ECM(
        EC=summation(Array(impedances)[i], (i, 0, len(ecms) - 1)),
    )


def parallel(*ecms):
    i = Symbol('i')
    impedances = [ecm.EC for ecm in ecms]
    return ECM(
        EC=1 / summation(1 / Array(impedances)[i], (i, 0, len(ecms) - 1)),
    )


class ECM:

    def __init__(
        self,
        EC=None,
        ECM_parameters=None,
        Q_global_index=None,
        I_global_index=None,
        U_global_index=None,
        ECM=None
    ):
        if ECM is not None:
            self.EC = ECM.EC
            self.ECM_parameters = ECM.ECM_parameters
            self.Q_global_index = ECM.Q_global_index
            self.I_global_index = ECM.I_global_index
            self.U_global_index = ECM.U_global_index
        else:
            self.EC = EC
            self.ECM_parameters = ECM_parameters
            self.Q_global_index = Q_global_index
            self.I_global_index = I_global_index
            self.U_global_index = U_global_index

    def check_parameters_validity(self, reference_list):
        for name in self.ECM_parameters.keys():
            if name not in reference_list:
                raise ValueError(
                    "Unknown EC parameter " + name + ". "
                    + "List of appropriate EC parameters: ["
                    + ", ".join(reference_list) + "]."
                )
        for name in reference_list:
            if name not in self.ECM_parameters.keys():
                raise ValueError(
                    "Missing EC parameter " + name + ". "
                    + "List of passed EC parameters: ["
                    + ", ".join(self.ECM_parameters.keys()) + "]."
                )

    def lambdify_ECM(self):
        symbolic_parameters = {
            Symbol(key): value for key, value in self.ECM_parameters.items()
        }
        impedance_frequency = lambdify(
            [Symbol('s')] + list(symbolic_parameters.keys()),
            self.EC,
            'numpy'
        )

        def impedance_frequency_eval(s):
            return impedance_frequency(s, *symbolic_parameters.values())

        return impedance_frequency_eval

    def U_rhs(self, t, state, I_derivatives):
        raise NotImplementedError

    def I_rhs(self, t, state, U_derivatives):
        raise NotImplementedError

    def update_U_ode_state_to_new_input(
        self, old_U_state, old_I_derivatives, new_I_derivatives
    ):
        raise NotImplementedError

    def calculate_U_ode_state_for_new_control(
        self,
        old_I_state,
        U_derivatives_from_old_input,
        I_derivatives_from_old_ode,
        I_derivatives_from_new_input
    ):
        raise NotImplementedError

    def update_I_ode_state_to_new_input(
        self, old_I_state, old_U_derivatives, new_U_derivatives
    ):
        raise NotImplementedError

    def calculate_I_ode_state_for_new_control(
        self,
        old_U_state,
        I_derivatives_from_old_input,
        U_derivatives_from_old_ode,
        U_derivatives_from_new_input
    ):
        raise NotImplementedError


class R(ECM):

    def __init__(self, resistance):
        super().__init__(
            EC=Symbol(resistance),
        )


class C(ECM):

    def __init__(self, capacitance):
        super().__init__(
            EC=1 / (Symbol(capacitance) * Symbol('s')),
        )


class L(ECM):

    def __init__(self, inductance):
        super().__init__(
            EC=Symbol(inductance) * Symbol('s'),
        )


class Q(ECM):
    """Constant Phase Element (CPE). exponent is in (0, 1].

    Impedance is phase-shifted by a constant factor n.
    Nyquist plot looks like C, but rotated by (1 - n) * 90°.

    Models imperfect capacitors. Hence, in general use values >> 0.5.
    Origins:
     - surface roughness (fractal dimension),
     - distribution of reaction rates (varying activation energies),
     - varying thickness or composition of (surface) coatings,
     - non-uniform current distribution (n decreases towards electrode edge).

    In RQ elements, the center of the semicircle rotates below the real
    axis. A "true" capacitance C may be obtained via RQ = τⁿ = (RC)ⁿ.
    """

    def __init__(self, cpe, exponent):
        super().__init__(
            EC=1 / (Symbol(cpe) * Symbol('s'))**exponent,
        )


class W(ECM):
    """Warburg with semi-infinite diffusion condition.

    Represents an infinite transmission line of RC elements.
    "coefficient" refers to the Warburg coefficient σ.
    "characteristic" does nothing, as there is no characteristic time-scale.

    σ = RT / (n² F² A sqrt(2)) * 1 / (
        sqrt(D_O) c_O^(b) sqrt(D_R) c_R^(b)
    )
    with D_x being the diffusion coefficients of reduced and oxidized species
    and c_x bulk concentrations (^(b)) of reduced and oxidized species.
    """

    def __init__(self, coefficient):
        # EC = Symbol(name) * (
        #     1 / (Symbol('s') * -1j)**0.5 - 1j / (Symbol('s') * -1j)**0.5
        # )
        super().__init__(
            ECM=Q(coefficient, 0.5),
        )


class warburg_open(ECM):
    """Also known as finite-space Warburg. Closes with a capacitor.

    One plate of the closing capacitor may be thought of as the
    counter electrode. Nyquist plot diverges to imaginary infinity.

    "coefficient" refers to the Warburg coefficient.
    "characteristic" refers to the ratio between thickness of the
    diffusion layer and the square-root of the diffusivity.
    """

    def __init__(self, coefficient, characteristic):
        super().__init__(
            EC=Symbol(coefficient) * coth(
                Symbol(characteristic) * Symbol('s')**0.5
            ) / Symbol('s')**0.5,
        )


class warburg_short(ECM):
    """Also known as finite-length Warburg. Closes with a resistor.

    Represents diffusion layers with controlled thickness. Nyquist plot
    goes from the 45° line into a quarter circle back to the real axis.

    "coefficient" refers to the Warburg coefficient.
    "characteristic" refers to the ratio between thickness of the
    diffusion layer and the square-root of the diffusivity.
    """

    def __init__(self, coefficient, characteristic):
        super().__init__(
            EC=Symbol(coefficient) * tanh(
                Symbol(characteristic) * Symbol('s')**0.5
            ) / Symbol('s')**0.5,
        )


class RC_chain(ECM):
    """RC pairs in series with a resistor.

    Keyword arguments for *_rhs: R_0, R_1, C_1, R_2, C_2, ...
    """

    def __init__(self, ECM_parameters, chain_length=1):
        super().__init__(
            EC=series(
                R('R_0'),
                *[
                    parallel(
                        R('R_' + str(i + 1)),
                        C('C_' + str(i + 1))
                    )
                    for i in range(chain_length)
                ]
            ).EC,
            ECM_parameters=ECM_parameters,
            Q_global_index=0,
            I_global_index=1,
            U_global_index=1,
        )
        self.chain_length = chain_length
        self.check_parameters_validity([
            'R_0',
            *['R_' + str(i + 1) for i in range(self.chain_length)],
            *['C_' + str(i + 1) for i in range(self.chain_length)],
        ])

    def U_rhs(self, t, state, I_derivatives):
        # state[0]: Q
        # state[1]: U
        # state[i], i > 1: voltage of RC pair #(i - 1)
        return np.array([
            I_derivatives[0](t),
            (
                self.ECM_parameters['R_0'] * I_derivatives[1](t)
                + np.sum([
                    (
                        I_derivatives[0](t)
                        - state[i + 1] / self.ECM_parameters['R_' + str(i)]
                    ) / self.ECM_parameters['C_' + str(i)]
                    for i in range(1, self.chain_length + 1)
                ])
            ),
            *[
                (
                    I_derivatives[0](t)
                    - state[i + 1] / self.ECM_parameters['R_' + str(i)]
                ) / self.ECM_parameters['C_' + str(i)]
                for i in range(1, self.chain_length + 1)
            ]
        ])

    def I_rhs(self, t, state, U_derivatives):
        # state[0]: Q
        # state[1]: I
        # state[i], i > 1: voltage of RC pair #(i - 1)
        return np.array([
            state[1],
            (
                U_derivatives[1](t)
                - np.sum([
                    (
                        state[1]
                        - state[i + 1] / self.ECM_parameters['R_' + str(i)]
                    ) / self.ECM_parameters['C_' + str(i)]
                    for i in range(1, self.chain_length + 1)])
            ) / self.ECM_parameters['R_0'],
            *[
                (
                    state[1]
                    - state[i + 1] / self.ECM_parameters['R_' + str(i)]
                ) / self.ECM_parameters['C_' + str(i)]
                for i in range(1, self.chain_length + 1)
            ]
        ])

    def update_U_ode_state_to_new_input(
        self, old_U_state, old_I_derivatives, new_I_derivatives
    ):
        return np.array([
            old_U_state[0],
            old_U_state[1] + self.ECM_parameters['R_0'] * (
                new_I_derivatives[0] - old_I_derivatives[0]
            ),
            *old_U_state[2:]
        ])

    def calculate_U_ode_state_for_new_control(
        self,
        old_I_state,
        U_derivatives_from_old_input,
        I_derivatives_from_old_ode,
        I_derivatives_from_new_input
    ):
        return self.update_U_ode_state_to_new_input(
            [old_I_state[0]]
            + list(U_derivatives_from_old_input[0:1])
            + list(old_I_state[2:]),
            I_derivatives_from_old_ode,
            I_derivatives_from_new_input
        )

    def update_I_ode_state_to_new_input(
        self, old_I_state, old_U_derivatives, new_U_derivatives
    ):
        return np.array([
            old_I_state[0],
            old_I_state[1] + (
                new_U_derivatives[0] - old_U_derivatives[0]
            ) / self.ECM_parameters['R_0'],
            *old_I_state[2:]
        ])

    def calculate_I_ode_state_for_new_control(
        self,
        old_U_state,
        I_derivatives_from_old_input,
        U_derivatives_from_old_ode,
        U_derivatives_from_new_input
    ):
        return self.update_I_ode_state_to_new_input(
            [old_U_state[0]]
            + list(I_derivatives_from_old_input[0:1])
            + list(old_U_state[2:]),
            U_derivatives_from_old_ode,
            U_derivatives_from_new_input
        )


class debye(ECM):
    """Debye circuit for ideally blocking electrodes.

    B: bulk electrolyte
    D: dielectric capacitance
    DL: double-layer

       __R_B__
      |       |
    --|       |--C_DL--
      |       |
      |__C_D__|
    """

    def __init__(self, ECM_parameters):
        super().__init__(
            EC=series(
                parallel(
                    R('R_B'),
                    C('C_D'),
                ),
                C('C_DL')
            ).EC,
            ECM_parameters=ECM_parameters,
            Q_global_index=0,
            I_global_index=-1,
            U_global_index=1,
        )
        self.check_parameters_validity(['R_B', 'C_DL', 'C_D'])

    def U_rhs(self, t, state, I_derivatives):
        # state[0]: Q
        # state[1]: U
        R_B = self.ECM_parameters['R_B']
        C_DL = self.ECM_parameters['C_DL']
        C_D = self.ECM_parameters['C_D']
        return np.array([
            I_derivatives[0](t),
            (C_DL + C_D) / (C_DL * C_D) * I_derivatives[0](t)
            + (state[0] / C_DL - state[1]) / (R_B * C_D)
        ])

    def I_rhs(self, t, state, U_derivatives):
        # state[0]: Q
        R_B = self.ECM_parameters['R_B']
        C_DL = self.ECM_parameters['C_DL']
        C_D = self.ECM_parameters['C_D']
        return np.array([
            (C_DL * C_D) / (C_DL + C_D) * U_derivatives[1](t)
            + (U_derivatives[0](t) * C_DL - state[0]) / (R_B * (C_DL + C_D))
        ])

    def update_U_ode_state_to_new_input(
        self, old_U_state, old_I_derivatives, new_I_derivatives
    ):
        # integral of U_rhs[1] is continuous
        return old_U_state

    def calculate_U_ode_state_for_new_control(
        self,
        old_I_state,
        U_derivatives_from_old_input,
        I_derivatives_from_old_ode,
        I_derivatives_from_new_input
    ):
        return self.update_U_ode_state_to_new_input(
            [old_I_state[0]] + list(U_derivatives_from_old_input[0:1]),
            I_derivatives_from_old_ode,
            I_derivatives_from_new_input,
        )

    def update_I_ode_state_to_new_input(
        self, old_I_state, old_U_derivatives, new_U_derivatives
    ):
        # I_rhs is only [Q], which has to be continuous
        return old_I_state

    def calculate_I_ode_state_for_new_control(
        self,
        old_U_state,
        I_derivatives_from_old_input,
        U_derivatives_from_old_ode,
        U_derivatives_from_new_input
    ):
        return self.update_I_ode_state_to_new_input(
            [old_U_state[0]],
            U_derivatives_from_old_ode,
            U_derivatives_from_new_input
        )


class debye_variant(ECM):
    """Often used variant of the Debye circuit. Valid as well.

       __R_B__C_DL__
      |             |
    --|             |--
      |             |
      |_____C_D_____|
    """

    def __init__(self, ECM_parameters):
        super().__init__(
            EC=parallel(
                series(
                    R('R_B'),
                    C('C_DL'),
                ),
                C('C_D')
            ).EC,
            ECM_parameters=ECM_parameters,
            Q_global_index=0,
            I_global_index=1,
            U_global_index=1,
        )
        self.check_parameters_validity(['R_B', 'C_DL', 'C_D'])

    def U_rhs(self, t, state, I_derivatives):
        # state[0]: Q
        # state[1]: U
        # state[2]: ∂ₜU
        R_B = self.ECM_parameters['R_B']
        C_DL = self.ECM_parameters['C_DL']
        C_D = self.ECM_parameters['C_D']
        return np.array([
            I_derivatives[0](t),
            state[2],
            - (1 + C_D / C_DL) / (R_B * C_D) * state[2]
            + I_derivatives[1](t) / C_D
            + I_derivatives[0](t) / (R_B * C_D * C_DL)
        ])

    def I_rhs(self, t, state, U_derivatives):
        # state[0]: Q
        # state[1]: I
        R_B = self.ECM_parameters['R_B']
        C_DL = self.ECM_parameters['C_DL']
        C_D = self.ECM_parameters['C_D']
        return np.array([
            state[1],
            - state[1] / (R_B * C_DL)
            + C_D * U_derivatives[2](t)
            + (1 + C_D / C_DL) / R_B * U_derivatives[1](t)
        ])

    def update_U_ode_state_to_new_input(
        self, old_U_state, old_I_derivatives, new_I_derivatives
    ):
        # derivation:
        # 1. ∂ₜ²U_new - ∂ₜ²U_old = U_rhs_new[2] - U_rhs_old[2].
        # 2. Integrate to obtain ∂ₜU_new - ∂ₜU_old.
        #    Naturally, this will depend on U_new - U_old; this term is
        #    a direct consequence of the reason the second derivative
        #    needed to be calculated in the first place. The integration
        #    constant results from the continuity of integration over I.
        # 3. Integrate again to obtain U_new - U_old. The integration
        #    constant results from the continuity of integration over U.
        # 4. Substitute U_new - U_old into ∂ₜU_new - ∂ₜU_old.
        return np.array([
            old_U_state[0],
            old_U_state[1],
            old_U_state[2] + (
                new_I_derivatives[0] - old_I_derivatives[0]
            ) / self.ECM_parameters['C_D']
        ])

    def calculate_U_ode_state_for_new_control(
        self,
        old_I_state,
        U_derivatives_from_old_input,
        I_derivatives_from_old_ode,
        I_derivatives_from_new_input
    ):
        return self.update_U_ode_state_to_new_input(
            [old_I_state[0]] + list(U_derivatives_from_old_input[0:2]),
            I_derivatives_from_old_ode,
            I_derivatives_from_new_input
        )

    def update_I_ode_state_to_new_input(
        self, old_I_state, old_U_derivatives, new_U_derivatives
    ):
        # derivation:
        # 1. ∂ₜI_new - ∂ₜI_old = I_rhs_new[1] - I_rhs_old[1].
        # 2. Integrate to obtain I_new - I_old.
        #    The integral of state[0] becomes the accumulated charge,
        #    which has to be continuous. This condition gives the
        #    integration constant.
        R_B = self.ECM_parameters['R_B']
        C_DL = self.ECM_parameters['C_DL']
        C_D = self.ECM_parameters['C_D']
        return np.array([
            old_I_state[0],
            old_I_state[1]
            + C_D * (
                new_U_derivatives[1] - old_U_derivatives[1]
            )
            + (1 + C_D / C_DL) / R_B * (
                new_U_derivatives[0] - old_U_derivatives[0]
            )
        ])

    def calculate_I_ode_state_for_new_control(
        self,
        old_U_state,
        I_derivatives_from_old_input,
        U_derivatives_from_old_ode,
        U_derivatives_from_new_input
    ):
        # Special case where the state of the ODE for I is just [I].
        return self.update_I_ode_state_to_new_input(
            [old_U_state[0]] + list(I_derivatives_from_old_input[0:1]),
            U_derivatives_from_old_ode,
            U_derivatives_from_new_input
        )


class randles(ECM):
    """Randles circuit. Very similar to SPM impedance with open Warburg.

    Pass one of the Warburg classes to choose the right Warburg.

    U: uncompensated resistance between reference electrode and metal surface
    DL: double-layer
    CT: charge-transfer resistance
    W: Warburg for mass-transport

            ___C_DL____
           |           |
    --R_U--|           |--
           |           |
           |__R_CT--W__|
    """

    def __init__(self, warburg):
        super().__init__(
            ECM=series(
                R('R_U'),
                parallel(
                    C('C_DL'),
                    series(
                        R('R_CT'),
                        warburg('σ', 'τ')
                    )
                )
            )
        )


class randles_variant(ECM):
    """Randles circuit variant.

    Pass one of the Warburg classes to choose the right Warburg.

    U: uncompensated resistance between reference electrode and metal surface
    DL: double-layer
    CT: charge-transfer resistance
    W: Warburg for mass-transport

            ___C_DL____
           |           |
    --R_U--|           |--C_X--
           |           |
           |__R_CT--W__|
    """

    def __init__(self, warburg):
        super().__init__(
            ECM=series(
                R('R_U'),
                parallel(
                    C('C_DL'),
                    series(
                        R('R_CT'),
                        warburg('σ', 'τ')
                    )
                ),
                C('C_X')
            )
        )


class wrong_randles(ECM):
    """Wrong Randles circuit. Often used even if it is not applicable.

    Would imply that CT resistance and mass transport are not coupled (wrong),
    while CT resistance and double-layer capacitance are (also wrong).

    Pass one of the Warburg classes to choose the right Warburg.

    U: uncompensated resistance between reference electrode and metal surface
    DL: double-layer
    CT: charge-transfer resistance
    W: Warburg for mass-transport

            ___C_DL____
           |           |
    --R_U--|           |--W--
           |           |
           |___R_CT____|
    """

    def __init__(self, warburg):
        super().__init__(
            ECM=series(
                R('R_U'),
                parallel(
                    C('C_DL'),
                    R('R_CT'),
                ),
                warburg('σ', 'τ')
            )
        )


class wrong_randles_variant(ECM):
    """Variant of the wrong Randles circuit.

    Contains a redundant CPE.

    Pass one of the Warburg classes to choose the right Warburg.

    U: uncompensated resistance between reference electrode and metal surface
    DL: double-layer
    CT: charge-transfer resistance
    W: Warburg for mass-transport

            ___C_DL____
           |           |
    --R_U--|           |--W--Q_X--
           |           |
           |___R_CT____|
    """

    def __init__(self, warburg, exponent):
        super().__init__(
            ECM=series(
                R('R_U'),
                parallel(
                    C('C_DL'),
                    R('R_CT'),
                ),
                warburg('σ', 'τ'),
                Q('Q_X', exponent)
            )
        )


class SCR(ECM):
    """SCR circuit for cathodes. Stems from a simplification of a TLM.

    R_B: bulk electrolyte
    C_D: dielectric capacitance
    R_C: resistance between current collector and electrode
    C_DL: double-layer
    R_CT: charge-transfer resistance
    W: Warburg for mass-transport


            _________C_D_________
           |                     |
    --R_B--|                     |
           |        ___C_DL___   |
           |       |          |  |
           |--R_C--|          |--|--
                   |          |
                   |__R_CT--W_|
    """

    def __init__(self, warburg):
        super().__init__(
            ECM=series(
                R('R_B'),
                parallel(
                    C('C_D'),
                    series(
                        R('R_C'),
                        parallel(
                            C('C_DL'),
                            series(
                                R('R_CT'),
                                warburg('σ', 'τ')
                            )
                        )
                    )
                )
            )
        )


class SCRF(ECM):
    """SCRF circuit for anodes with SEI. Stems from a simplification of a TLM.

    R_B: bulk electrolyte
    C_D: dielectric capacitance
    R_C: resistance between current collector and electrode
    C_F: capacitance of the passivating film (i.e. SEI)
    C_DL: double-layer
    R_CT: charge-transfer resistance
    W: Warburg for mass-transport
    R_F: resistance of the passivating film (i.e. SEI)


            _________C_D____________________
           |                                |
           |        _____C_F_____________   |
    --R_B--|       |                     |  |
           |       |   ___C_DL___        |  |
           |       |  |          |       |  |
           |--R_C--|--|          |--R_F--|--|--
                      |          |
                      |__R_CT--W_|
    """

    def __init__(self, warburg):
        super().__init__(
            ECM=series(
                R('R_B'),
                parallel(
                    C('C_D'),
                    series(
                        R('R_C'),
                        parallel(
                            C('C_F'),
                            series(
                                parallel(
                                    C('C_DL'),
                                    series(
                                        R('R_CT'),
                                        warburg('σ', 'τ')
                                    )
                                ),
                                R('R_F')
                            )
                        )
                    )
                )
            )
        )


class aluminium_electrode(ECM):
    """Models an aluminium electrode incl. adsorption process.

    R_s: resistance of the electrolyte (measurements are typically
         in frequencies too low to see its capacitance)
    C_1: double-layer capacitance.
    R_1: unclear.
    L_1: models the loop attributed to the adsorption process.
    R_2: resistance of the adsorption process.
    C: charge-transfer capacitance.
    R_3: charge-transfer resistance.
    """

    def __init__(self):
        super().__init__(
            ECM=series(
                R('R_s'),
                parallel(
                    C('C_1'),
                    R('R_1'),
                    series(L('L_1'), R('R_2')),
                ),
                parallel(
                    C('C'),
                    R('R_3'),
                )
            )
        )


class aluminium_electrode_variant(ECM):
    """Can fit the same spectrum, but with less meaningful parameters.

    R_s: resistance of the electrolyte (measurements are typically
         in frequencies too low to see its capacitance)
    C_1: double-layer capacitance.
    R_1: unclear.
    L_1: models the loop attributed to the adsorption process.
    R_2: resistance of the adsorption process.
    C: charge-transfer capacitance.
    R_3: charge-transfer resistance.
    """

    def __init__(self):
        super().__init__(
            series(
                R('R_s'),
                parallel(
                    C('C_1'),
                    series(
                        R('R_1'),
                        parallel(
                            L('L_1'), R('R_2')
                        ),
                    )
                ),
                parallel(
                    C('C'),
                    R('R_3'),
                )
            )
        )


def condense(eq, *x):
    """collapse additive/multiplicative constants into single
    variables, returning condensed expression and replacement
    values.

    https://stackoverflow.com/questions/71315789/optimize-sympy-expression-
    evaluation-by-combining-as-many-free-symbols-as-possib
    by smichr under CC-BY-SA 4.0:
    https://creativecommons.org/licenses/by-sa/4.0/

    Examples
    ========

    Simple constants are left unchanged

    >>> condense(2*x + 2, x)
    (2*x + 2, {})

    More complex constants are replaced by a single variable

    >>> first = condense(eq, x); first
    (c6*(c5 - 2*sqrt(d*(c4 + x))), {c4: a*b - c - e, c6: 1/(b - 1),
     c5: a*b*c**2})

    If a condensed expression is expanded, there may be more simplification
    possible:

    >>> second = condense(first[0].expand(), x); second
    (c0 + c2*sqrt(c1 + d*x), {c1: c4*d, c2: -2*c6, c0: c5*c6})
    >>> full_reps = {k: v.xreplace(first[1]) for k, v in second[1].items()};
        full_reps
    {c1: d*(a*b - c - e), c2: -2/(b - 1), c0: a*b*c**2/(b - 1)}

    More than 1 variable can be designated:

    >>> condense(eq, c, e)
    (c4*(c**2*c1 - 2*sqrt(d*(-c + c2 - e))), {c4: 1/(b - 1), c1: a*b,
     c2: a*b + x})
    """
    reps = {}
    con = numbered_symbols('c')
    free = eq.free_symbols

    def c():
        while True:
            rv = next(con)
            if rv not in free:
                return rv

    def do(e):
        i, d = e.as_independent(*x)
        if not i.args:
            return e
        return e.func(reps.get(i, reps.setdefault(i, c())), d)

    rv = eq.replace(lambda x: x.is_Add or x.is_Mul, lambda x: do(x))
    reps = {v: k for k, v in reps.items()}
    keep = rv.free_symbols & set(reps)
    reps = {k: reps[k].xreplace(reps) for k in keep}
    return rv, reps


if __name__ == '__main__':
    partially_polarizable = series(
        parallel(
            R('R_B'),
            C('C_B'),
        ),
        parallel(
            R('R_F'),
            C('C_D'),
        ),
    )

    ideally_conducting = parallel(
        R('R_B'),
        C('C_B'),
    )

    ideally_blocking = series(
        parallel(
            R('R_B'),
            C('C_B'),
        ),
        C('C_D'),
    )

    aluminium_corrosion = series(
        R('R_s'),
        parallel(
            C('C_1'),
            R('R_1'),
            series(
                L('L_1'),
                R('R_2')
            ),
        ),
        parallel(
            C('C'),
            R('R_3'),
        )
    )

    for name, model in [
        ("Partially polarizable", partially_polarizable),
        ("Ideally conducting", ideally_conducting),
        ("Ideally blocking", ideally_blocking),
        ("Aluminium corrosion", aluminium_corrosion),
    ]:
        print(name + ":")
        pretty_print(simplify(model.EC))
        pretty_print(condense(simplify(model.EC), Symbol('s')))
        print()


class Two_RC_Optimized_for_Torch:
    def __init__(
        self,
        omega,
        true_parameters=None,
        measured_real_part=None,
        measured_imaginary_part=None,
    ):
        """
        @param omega
            A torch.Tensor of the angular frequencies in rad / s.
        @param true_parameters
            If given, synthetic data will be generated for measured Z.
        @param measured_real_part
            If 'true_parameters' is not set, you may give experimental
            data directly instead.
        @param measured_imaginary_part
            If 'true_parameters' is not set, you may give experimental
            data directly instead.
        """
        self.omega = omega
        self.mu = torch.mean(torch.log(self.omega))
        self.sigma = torch.std(torch.log(self.omega))
        if true_parameters is None:
            self.measured_real_part = measured_real_part
            self.measured_imaginary_part = measured_imaginary_part
        else:
            # true_parameters = torch.atleast_2d(true_parameters)
            rt = true_parameters[0]
            r1 = (-true_parameters[1].exp()).exp()
            t1 = true_parameters[2]
            r2 = (-true_parameters[3].exp()).exp()
            t2 = true_parameters[4]
            noise_sigma = true_parameters[5]
            Rt = rt.exp()
            r0 = 1 - r1 - r2

            self.measured_real_part = (
                self.real_part(Rt, r0, r1, t1, r2, t2) + Normal(0, 1).sample(
                    torch.Size([len(self.omega)])
                ) * torch.sqrt(noise_sigma)
            )
            self.measured_imaginary_part = (
                self.imaginary_part(Rt, r1, t1, r2, t2) + Normal(0, 1).sample(
                    torch.Size([len(self.omega)])
                ) * torch.sqrt(noise_sigma)
            )

    def unnormalise_tau(self, tau):
        """
        @param tau: torch.tensor, time constant tau in log space;
                tau = ln(omega * t_i)
        Returns:
            - tau: torch.tensor, time constant tau in raw space;
                tau = omega * t_i
        """
        return torch.exp(-(self.sigma * tau + self.mu))

    def normalised_input(self, tau):
        """
        Args:
            - tau: torch.tensor, time constant tau in raw space;
                tau = omega * t_i
        Returns:
            - tau: torch.tensor, time constant tau in log space;
                tau = ln(omega * t_i)
        """
        return (
            torch.log(self.omega) - (self.sigma * tau + self.mu)
        )

    def real_part(self, Rt, r0, r1, t1, r2, t2):
        """
        Returns:
            - Z.real: torch.tensor, real part of impedance spectrum
        """
        return Rt * (
            r0 + r1 / 2 * (
                1 - torch.tanh(self.normalised_input(t1))
            )
            + r2 / 2 * (1 - torch.tanh(self.normalised_input(t2)))
        )

    def imaginary_part(self, Rt, r1, t1, r2, t2):
        """
        Returns:
            - Z.imaginary: torch.tensor, imaginary part of impedance spectrum
        """
        return Rt * (
            (r1 / 2) / torch.cosh(self.normalised_input(t1))
            + (r2 / 2) / torch.cosh(self.normalised_input(t2))
        )

    def circuit_elements(self, theta):
        rt = theta[0]
        r1 = (-theta[1].exp()).exp()
        t1 = theta[2]
        r2 = (-theta[3].exp()).exp()
        t2 = theta[4]
        Rt = rt.exp()
        r0 = 1 - r1 - r2

        R0 = Rt * r0
        R1 = Rt * r1
        R2 = Rt * r2
        C1 = (-(self.sigma * t1 + self.mu)).exp() / R1
        C2 = (-(self.sigma * t2 + self.mu)).exp() / R2

        return R0, R1, C1, R2, C2

    def __call__(self, theta):
        """
        @param theta
            A torch.Tensor of the normalized circuit parameters.
            Θ = [rt, r1_norm, t1_norm, r2_norm, t2_norm].
        Returns:
            - LL: torch.tensor, log-likelihood
        """
        rt = theta[:, [0]]
        r1 = (-theta[:, [1]].exp()).exp()
        t1 = theta[:, [2]]
        r2 = (-theta[:, [3]].exp()).exp()
        t2 = theta[:, [4]]
        Rt = rt.exp()
        r0 = 1 - r1 - r2

        err_reZ = self.measured_real_part - self.real_part(
            Rt, r0, r1, t1, r2, t2
        )
        err_imZ = self.measured_imaginary_part - self.imaginary_part(
            Rt, r1, t1, r2, t2
        )
        err = (
            torch.linalg.vecdot(err_reZ, err_reZ)
            + torch.linalg.vecdot(err_imZ, err_imZ)
        )
        discrepancy = -(err / (2 * len(self.omega))).log()
        LL = -len(self.omega) * (
            1 + torch.log(torch.pi / len(self.omega) * err)
        )
        return discrepancy, LL
