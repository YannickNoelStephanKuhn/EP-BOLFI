"""!@package ep_bolfi.models.assess_effective_parameters
Evaluates a parameter set for, e.g., overpotential and capacity.
"""

import numpy as np
from scipy.optimize import root_scalar
from tabulate import tabulate
import pybamm
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


class Effective_Parameters:
    """!@brief Calculates, stores and prints effective parameters. """

    def __init__(self, parameters, working_electrode='both'):
        """!@brief Preprocesses the model parameters.
        @param parameters
            A dictionary of parameter values with the parameter names as
            keys. For these names, please refer to
            ep_bolfi.models.standard_parameters.
        @param working_electrode
            When set to either 'negative' or 'positive', the parameters
            will be treated as a half-cell setup with said electrode.
        """

        from ep_bolfi.models.standard_parameters import (
            cₙ, cₚ, SOCₙ_init, SOCₚ_init,
            iₛₑₙ_0, iₛₑₚ_0,
            zₙ, zₚ,
            αₙₙ, αₚₙ, αₙₚ, αₚₚ,
            σₙ, σₚ,
            cₑ_init,
            εₛ_scalar, βₑₛ_scalar,
            Dₑ, κₑ, κₑ_hat, t_plus, one_plus_dlnf_dlnc,
            Lₙ_dim, Lₚ_dim, Lₛ,
            Cₑ, Cᵣₙ, Cᵣₚ,
            γₑ, γₙ, γₚ,
            A, F,
            T_init, thermal_voltage,
        )

        ## The parameter dictionary converted to PyBaMM form.
        # Convert SubstitutionDict to dict by iterating over it.
        self.parameters = pybamm.ParameterValues({
            k: v for k, v in parameters.items()
        })
        ## Sets whether full- or half-cell parameters are calculated.
        self.working_electrode = working_electrode
        if self.working_electrode == 'positive':
            Lₙ = pybamm.Scalar(0)
            εₙ_scalar = pybamm.Scalar(0)
            βₑₙ_scalar = pybamm.Scalar(0)
            βₛₙ_scalar = pybamm.Scalar(1)
            from ep_bolfi.models.standard_parameters import (
                Lₚ, εₚ_scalar, βₑₚ_scalar, βₛₚ_scalar
            )
        elif self.working_electrode == 'negative':
            Lₚ = pybamm.Scalar(0)
            εₚ_scalar = pybamm.Scalar(0)
            βₑₚ_scalar = pybamm.Scalar(0)
            βₛₚ_scalar = pybamm.Scalar(1)
            from ep_bolfi.models.standard_parameters import (
                Lₙ, εₙ_scalar, βₑₙ_scalar, βₛₙ_scalar
            )
        else:
            from ep_bolfi.models.standard_parameters import (
                Lₙ, Lₚ, εₙ_scalar, εₚ_scalar,
                βₑₙ_scalar, βₛₙ_scalar, βₑₚ_scalar, βₛₚ_scalar
            )

        ## Negative electrode theoretical capacity.
        self.Qₙ = (1 - εₙ_scalar) * Lₙ_dim * cₙ * zₙ * F * A
        ## Positive electrode theoretical capacity.
        self.Qₚ = (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A

        ## Integration constant for the electrolyte concentration / I.
        self.R_cₑ_0_1 = (
            (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init, T_init) * γₑ) * (
                2 * Lₚ**2 / εₚ_scalar**βₑₚ_scalar
                - 2 * Lₙ**2 / εₙ_scalar**βₑₙ_scalar
                + 3 * (Lₙ**2 - Lₚ**2 + 1) / εₛ_scalar**βₑₛ_scalar
            )
        )
        ## Electrolyte concentration / I at the negative electrode.
        self.R_bar_cₑₙ_1 = (
            (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init, T_init) * γₑ) * (
                2 * Lₙ / εₙ_scalar**βₑₙ_scalar
                - 6 * Lₙ / εₛ_scalar**βₑₛ_scalar
            ) + self.R_cₑ_0_1
        )
        ## Electrolyte concentration / I at the positive electrode.
        self.R_bar_cₑₚ_1 = (
            (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init, T_init) * γₑ) * (
                -2 * Lₚ / εₚ_scalar**βₑₚ_scalar
                - 6 * (1 - Lₚ) / εₛ_scalar**βₑₛ_scalar
            ) + self.R_cₑ_0_1
        )
        if self.working_electrode == 'positive':
            ## Effective resistance of the negative electrode.
            self.R_bar_φₛₙ_1 = -Lₙ / (σₙ * Cₑ)
        else:
            self.R_bar_φₛₙ_1 = -Lₙ / (
                3 * σₙ * (1 - εₙ_scalar)**βₛₙ_scalar * Cₑ
            )
        if self.working_electrode == 'negative':
            ## Effective resistance of the positive electrode.
            self.R_bar_φₛₚ_1 = Lₚ / (
                3 * σₚ * (1 - εₚ_scalar)**βₛₚ_scalar * Cₑ
            )
        else:
            self.R_bar_φₛₚ_1 = Lₚ / (3 * σₚ * (1 - εₚ_scalar)**βₛₚ_scalar * Cₑ)

        ## Negative electrode resistance (as calculated by the SPMe(S)).
        self.Rₙₛ = self.eval(thermal_voltage) * self.eval(
            -Lₙ / (3 * (1 - εₙ_scalar)**βₛₙ_scalar * σₙ)
        )
        ## Positive electrode resistance (as calculated by the SPMe(S)).
        self.Rₚₛ = self.eval(thermal_voltage) * self.eval(
            -Lₚ / (3 * (1 - εₚ_scalar)**βₛₚ_scalar * σₚ)
        )
        ## Electrolyte resistance (as calculated by the SPMe(S)).
        self.Rₑ = self.eval(thermal_voltage * Cₑ) * self.eval(
            (2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init))
            * (self.R_bar_cₑₚ_1 - self.R_bar_cₑₙ_1)
            + self.R_bar_cₑₚ_1 / zₚ - self.R_bar_cₑₙ_1 / zₙ - (
                Lₚ / (3 * εₚ_scalar**βₑₚ_scalar)
                + Lₙ / (3 * εₙ_scalar**βₑₙ_scalar)
                + Lₛ / εₛ_scalar**βₑₛ_scalar
            ) / (κₑ(cₑ_init, T_init) * Cₑ * κₑ_hat)
        )

        ## SPM(e) negative electrode exchange-current.
        self.iₛₑₙ = lambda ηₙ: (
            self.eval(
                (γₙ / Cᵣₙ) * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), cₙ, T_init)
            ) * (
                np.exp(self.eval(αₙₙ * zₙ) * ηₙ)
                - np.exp(self.eval(-αₚₙ * zₙ) * ηₙ)
            )
        )
        ## SPM(e) positive electrode exchange-current.
        self.iₛₑₚ = lambda ηₚ: (
            self.eval(
                (γₚ / Cᵣₚ) * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T_init)
            ) * (
                np.exp(self.eval(αₙₚ * zₚ) * ηₚ)
                - np.exp(self.eval(-αₚₚ * zₚ) * ηₚ)
            )
        )

    def eval(self, expression):
        """!@brief Short-hand for PyBaMM symbol evaluation.
        @param expression
            A pybamm.Symbol.
        @return
            The numeric value of "expression".
        """
        return self.parameters.process_symbol(expression).evaluate()

    def print(self, c_rates=[0.1, 0.2, 0.5, 1.0]):
        """!@brief Prints the voltage losses for the given C-rates.
        @param c_rates
            The C-rates (as fraction of "Typical current [A]").
        """

        from ep_bolfi.models.standard_parameters import (
            SOCₙ_init, SOCₚ_init,
            iₛₑₙ_0, d_cₑₙ_iₛₑₙ_0, iₛₑₚ_0, d_cₑₚ_iₛₑₚ_0,
            zₙ, zₚ,
            αₙₙ, αₚₙ, αₙₚ, αₚₚ,
            cₑ_init, cₙ, cₚ,
            Lₙ, Lₚ,
            Cₑ,
            T_init, thermal_voltage,
        )

        c_rates = np.array(c_rates)

        if self.working_electrode == 'positive':
            Lₙ_inverse = 1
            Lₚ_inverse = self.eval(1 / Lₚ)
        elif self.working_electrode == 'negative':
            Lₙ_inverse = self.eval(1 / Lₙ)
            Lₚ_inverse = 1
        else:
            Lₙ_inverse = self.eval(1 / Lₙ)
            Lₚ_inverse = self.eval(1 / Lₚ)
        # SPM negative electrode overpotential.
        ηₙ_0 = np.array([root_scalar(
            lambda ηₙ: self.iₛₑₙ(ηₙ) - c * Lₙ_inverse,
            method='toms748', bracket=[-40, 40], x0=0
        ).root for c in c_rates])
        # SPM positive electrode overpotential.
        ηₚ_0 = np.array([root_scalar(
            lambda ηₚ: self.iₛₑₚ(ηₚ) + c * Lₚ_inverse,
            method='toms748', bracket=[-40, 40], x0=0
        ).root for c in c_rates])
        # SPMe negative electrolyte concentration.
        bar_cₑₙ_1 = self.eval(self.R_bar_cₑₙ_1) * c_rates
        # SPMe positive electrolyte concentration.
        bar_cₑₚ_1 = self.eval(self.R_bar_cₑₚ_1) * c_rates
        # SPMe negative electrode overpotential correction.
        bar_ηₙ_1 = (
            -bar_cₑₙ_1 / self.eval(zₙ)
            * self.eval(d_cₑₙ_iₛₑₙ_0(cₑ_init, SOCₙ_init(0), cₙ, T_init)
                        / iₛₑₙ_0(cₑ_init, SOCₙ_init(0), cₙ, T_init))
            * (np.exp(self.eval(αₙₙ * zₙ) * ηₙ_0)
               - np.exp(self.eval(-αₚₙ * zₙ) * ηₙ_0))
            / (self.eval(αₙₙ) * np.exp(self.eval(αₙₙ * zₙ) * ηₙ_0)
               + self.eval(αₚₙ) * np.exp(self.eval(-αₚₙ * zₙ) * ηₙ_0))
        )
        # SPMe positive electrode overpotential correction.
        bar_ηₚ_1 = (
            -bar_cₑₚ_1 / self.eval(zₚ)
            * self.eval(d_cₑₚ_iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T_init)
                        / iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T_init))
            * (np.exp(self.eval(αₙₚ * zₚ) * ηₚ_0)
               - np.exp(self.eval(-αₚₚ * zₚ) * ηₚ_0))
            / (self.eval(αₙₚ) * np.exp(self.eval(αₙₚ * zₚ) * ηₚ_0)
               + self.eval(αₚₚ) * np.exp(self.eval(-αₚₚ * zₚ) * ηₚ_0))
        )
        # SPM electrolyte potential.
        # if self.halfcell:
        #     φₑ_0 = -ηₙ_0
        # else:
        #     φₑ_0 = -ηₙ_0 - self.eval(OCVₙ(SOCₙ_init(0), T_init))
        # SPMe electrolyte potential integration constant
        # from ep_bolfi.models.standard_parameters import (
        #     σₙ, εₙ_scalar, εₛ_scalar, βₑₙ_scalar, βₑₛ_scalar, βₛₙ_scalar,
        #     κₑ, κₑ_hat, t_plus, one_plus_dlnf_dlnc,
        # )
        # if self.working_electrode == 'positive':
        #     tilde_φₑ = (
        #         -bar_ηₙ_1 - bar_cₑₙ_1 / self.eval(zₙ)
        #         - self.eval(2 * (1 - t_plus(cₑ_init))
        #                     * one_plus_dlnf_dlnc(cₑ_init)) * bar_cₑₙ_1
        #     )
        # else:
        #     tilde_φₑ = (
        #         -c_rates * self.eval(
        #             Lₙ / (3 * σₙ * (1 - εₙ_scalar)**βₛₙ_scalar * Cₑ)
        #         ) - bar_ηₙ_1 - bar_cₑₙ_1 / self.eval(zₙ)
        #         - self.eval(2 * (1 - t_plus(cₑ_init))
        #                     * one_plus_dlnf_dlnc(cₑ_init)) * bar_cₑₙ_1
        #         - c_rates / self.eval(κₑ_hat * Cₑ * κₑ(cₑ_init, T_init))
        #         * self.eval((-Lₙ / (3 * εₙ_scalar**βₑₙ_scalar)
        #                     + Lₙ / εₛ_scalar**βₑₛ_scalar))
        #     )

        # SPMe electrolyte potential correction
        # bar_φₑₙ_1 = tilde_φₑ + self.eval(
        #     2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init)
        # ) * bar_cₑₙ_1 - c_rates / self.eval(
        #     κₑ_hat * Cₑ * κₑ(cₑ_init, T_init)
        # ) * (
        #     self.eval(-Lₙ / (3 * εₙ_scalar**βₑₙ_scalar)
        #               + Lₙ / εₛ_scalar**βₑₛ_scalar)
        # )
        # bar_φₑₚ_1 = tilde_φₑ + self.eval(
        #     2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init)
        # ) * bar_cₑₚ_1 - c_rates / self.eval(
        #     κₑ_hat * Cₑ * κₑ(cₑ_init, T_init)
        # ) * (
        #     self.eval(Lₚ / (3 * εₚ_scalar**βₑₚ_scalar)
        #               + (1 - Lₚ) / εₛ_scalar**βₑₛ_scalar)
        # )

        # Tabulate the voltage losses.
        table = [
            ["Current / C-rate"] + list(c_rates),
            ["SPM anode OP [mV]"] + list(self.eval(thermal_voltage) * ηₙ_0),
            ["SPM cathode OP [mV]"] + list(-self.eval(thermal_voltage) * ηₚ_0),
            ["SPMe anode OP [mV]"] + list(self.eval(thermal_voltage) * (
                ηₙ_0 - self.eval(Cₑ) * bar_ηₙ_1
            )),
            ["SPMe cathode OP [mV]"] + list(-self.eval(thermal_voltage) * (
                ηₚ_0 + self.eval(Cₑ) * bar_ηₚ_1
            )),
            ["SPMe electrolyte drop [mV]"] + list(-self.Rₑ * c_rates),
            ["SPMe anode drop [mV]"] + list(-self.Rₙₛ * c_rates),
            ["SPMe cathode drop [mV]"] + list(-self.Rₚₛ * c_rates),
        ]
        for row in range(1, len(table)):
            for column in range(1, len(table[row])):
                table[row][column] = '{:.3f}'.format(table[row][column] * 1e3)

        print("Negative electrode theoretical capacity: "
              + str(self.eval(self.Qₙ)) + " C")
        print("Positive electrode theoretical capacity: "
              + str(self.eval(self.Qₚ)) + " C")
        print(tabulate(table, headers='firstrow', tablefmt='pretty'))
