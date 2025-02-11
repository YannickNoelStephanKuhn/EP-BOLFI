"""
Contains equations to evaluate the analytic impedance of the SPMe model.
"""

import pybamm
import warnings

from mpmath import mp, mpmathify, pi, sqrt, exp, tan, sinh, cosh, tanh, coth
from numpy import array, atleast_1d, seterr

# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


class AnalyticImpedance:
    """
    Analytic impedance of the SPMe.
    """

    def __init__(
        self, parameters, catch_warnings=True, precision=106, verbose=False
    ):
        """
        Preprocesses the model parameters.

        :param parameters:
            A dictionary of parameter values with the parameter names as
            keys. For these names, please refer to
            models.standard_parameters.
        :param catch_warnings:
            Set to False if you want to disable the over-/underflow
            protection.
        :param precision:
            Precision (in bits) in which the computations are carried
            out. Note that the library used for this (mpmath) sets this
            globally. The default is double of what a 64-bit float
            usually assigns to the decimal number (53 bits). Exponents
            are stored exactly.
        :param verbose:
            If set to True, information about the so-called resonance
            frequencies and resistances will be printed during model
            calculations. The resonance frequencies are the frequencies
            with maximum imaginary part of the impedance for the various
            model components. The resistances are the widths of the
            semi-circles or Warburg two-thirds-circles with 45° line.
        """

        if catch_warnings:
            # Make RuntimeWarnings catchable.
            seterr(all='warn')
            warnings.filterwarnings('error')

        mp.dps = precision
        self.verbose = verbose

        from ep_bolfi.models.standard_parameters import (
            dOCVₙ_dSOCₙ, dOCVₚ_dSOCₚ,
            SOCₙ_init, SOCₚ_init,
            cₙ, cₚ,
            iₛₑₙ_0, iₛₑₚ_0,
            aₙ, aₚ,
            zₙ, zₚ,
            σₙ, σₚ,
            cₑ_init,
            εₙ_scalar, εₛ_scalar, εₚ_scalar,
            βₑₙ_scalar, βₑₛ_scalar, βₑₚ_scalar, βₛₙ_scalar, βₛₚ_scalar,
            Dₑ, Dₑ_dim, κₑ, κₑ_hat, t_plus, one_plus_dlnf_dlnc,
            Lₙ, Lₛ, Lₚ,
            Cₑ, Cₙ, Cₚ, Cᵣₙ, Cᵣₚ,
            γₑ, γₙ, γₚ,
            C, τᵈ,
            T_init, thermal_voltage,
            C_DLₙ, C_DLₚ, L_SEI, R_SEI, C_SEI,
            nₙ, nₚ, zₙ_salt, zₚ_salt,
            ρₑ, ρₑ_plus, M_N, c_N, ρ_N, v_N, tilde_ρ_N,
            ε_SEI, β_SEI, L_electrolyte_for_SEI_model, t_SEI_minus, F
        )

        self.parameters = pybamm.ParameterValues(parameters)
        """The parameter dictionary converted to PyBaMM form."""

        T = T_init
        self.symbolic_constants = {
            "Positive electrode intercalation resistance":
                Cₚ / (Lₚ * aₚ * γₚ) * dOCVₚ_dSOCₚ(SOCₚ_init(1), T),
            "Positive electrode intercalation impedance scaling":
                Cₚ**0.5,
            "Negative electrode intercalation resistance":
                Cₙ / (Lₙ * aₙ * γₙ) * dOCVₙ_dSOCₙ(SOCₙ_init(0), T),
            "Negative electrode intercalation impedance scaling":
                Cₙ ** 0.5,
            "Positive electrolyte concentration impedance scaling":
                (Cₑ * εₚ_scalar / (εₚ_scalar**βₑₚ_scalar
                                   * Dₑ(cₑ_init, T)))**0.5,
            "Separator electrolyte concentration impedance scaling":
                (Cₑ * εₛ_scalar / (εₛ_scalar**βₑₛ_scalar
                                   * Dₑ(cₑ_init, T)))**0.5,
            "Negative electrolyte concentration offset":
                (1 - t_plus(cₑ_init)) / (Cₑ * γₑ * εₙ_scalar * Lₙ),
            "Negative electrolyte concentration impedance scaling":
                (Cₑ * εₙ_scalar / (εₙ_scalar**βₑₙ_scalar
                                   * Dₑ(cₑ_init, T)))**0.5,
            "Negative electrode charge transfer resistance":
                (zₙ * γₙ / Cᵣₙ
                 * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), cₙ, T) * Lₙ)**(-1),
            "Positive electrode charge transfer resistance":
                (zₚ * γₚ / Cᵣₚ
                 * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T) * Lₚ)**(-1),
            "Charge transfer resistance":
                (zₚ * γₚ / Cᵣₚ
                 * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T) * Lₚ)**(-1)
                + (zₙ * γₙ / Cᵣₙ
                   * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), cₙ, T) * Lₙ)**(-1),
            "Charge transfer resistance for half-cell":
                (zₚ * γₚ / Cᵣₚ
                 * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), cₚ, T) * Lₚ)**(-1)
                + (zₙ * γₙ / Cᵣₙ
                   * iₛₑₙ_0(cₑ_init, pybamm.Scalar(0.5), cₙ, T))**(-1),
            "Electrolyte conductivity resistance":
                (
                    Lₚ / (3 * εₚ_scalar**βₑₚ_scalar)
                    + Lₙ / (3 * εₙ_scalar**βₑₙ_scalar)
                    + Lₛ / εₛ_scalar**βₑₛ_scalar
                ) / (κₑ(cₑ_init, T) * κₑ_hat),
            "Electrolyte conductivity resistance for half-cell":
                (
                    Lₚ / (3 * εₚ_scalar**βₑₚ_scalar)
                    + Lₛ / εₛ_scalar**βₑₛ_scalar
                ) / (κₑ(cₑ_init, T) * κₑ_hat),
            "Electrode conductivity resistance":
                (
                    Lₚ / ((1 - εₚ_scalar)**βₛₚ_scalar * σₚ)
                    + Lₙ / ((1 - εₙ_scalar)**βₛₙ_scalar * σₙ)
                ) / 3,
            "Negative electrode conductivity resistance":
                (Lₙ / ((1 - εₙ_scalar)**βₛₙ_scalar * σₙ)) / 3,
            "Positive electrode conductivity resistance":
                (Lₚ / ((1 - εₚ_scalar)**βₛₚ_scalar * σₚ)) / 3,
            "Electrode conductivity resistance for half-cell":
                (Lₚ / ((1 - εₚ_scalar)**βₛₚ_scalar * σₚ)) / 3 + Lₙ / σₙ,
            "Ionic conductivity": κₑ(cₑ_init, T) * κₑ_hat,
        }
        """
        All relevant constants in a dictionary.
        Note: any term here that corresponds to U¹ - ηₚ¹ + ηₙ¹ is
        already scaled by Cₑ / I. All constants herein are duplicated
        as members after this. Their descriptions match their names
        (the keys of this dictionary).
        """

        self.constants = {
            key: self.parameters.process_symbol(value).evaluate()
            for key, value in self.symbolic_constants.items()
        }

        # For the stationary solution of cₑ¹ for "long" timescales.
        """
        self.constants["cₑ₀¹ / I"] = (
            (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init) * γₑ) * (
            2 * Lₚ**2 / εₚ_scalar**βₑₚ - 2 * Lₙ**2 / εₙ_scalar**βₑₙ
            + 3 / εₛ_scalar**βₑₛ * (Lₙ**2 - Lₚ**2 + 1)))

        self.constants.update({
            "̄cₑₚ¹ / I": self.constants["cₑ₀¹ / I"]
                + (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init) * γₑ)
                * (2 * Lₙ / εₙ_scalar**βₑₙ - 6 * Lₙ / εₛ_scalar**βₑₛ),
            "̄cₑₚ¹ / I": self.constants["cₑ₀¹ / I"]
                + (1 - t_plus(cₑ_init)) / (6 * Dₑ(cₑ_init) * γₑ)
                * (-2 * Lₚ / εₚ_scalar**βₑₚ - 6 * (1 - Lₚ)/εₛ_scalar**βₑₛ),
        })
        """

        self.Rₚ_int = self.constants[
            "Positive electrode intercalation resistance"
        ]
        """Positive electrode intercalation resistance."""
        self.Zₚ_int = self.constants[
            "Positive electrode intercalation impedance scaling"
        ]
        """Positive electrode intercalation impedance scaling."""
        self.Rₙ_int = self.constants[
            "Negative electrode intercalation resistance"
        ]
        """Negative electrode intercalation resistance."""
        self.Zₙ_int = self.constants[
            "Negative electrode intercalation impedance scaling"
        ]
        """Negative electrode intercalation impedance scaling."""
        self.Z_cₑₚ = self.constants[
            "Positive electrolyte concentration impedance scaling"
        ]
        """Positive electrolyte concentration impedance scaling"""
        self.Z_cₑₛ = self.constants[
            "Separator electrolyte concentration impedance scaling"
        ]
        """Separator electrolyte concentration impedance scaling."""
        self.Z_cₑₙ = self.constants[
            "Negative electrolyte concentration impedance scaling"
        ]
        """Negative electrolyte concentration impedance scaling."""
        self.Rₛₑₙ = self.constants[
            "Negative electrode charge transfer resistance"
        ]
        """Negative electrode charge transfer resistance."""
        self.Rₛₑₚ = self.constants[
            "Positive electrode charge transfer resistance"
        ]
        """Positive electrode charge transfer resistance."""
        self.Rₛₑ = self.constants[
            "Charge transfer resistance"
        ]
        """Charge transfer resistance."""
        self.Rₑ = self.constants[
            "Electrolyte conductivity resistance"
        ]
        """Electrolyte conductivity resistance."""
        self.Rₑ_metal_counter = self.constants[
            "Electrolyte conductivity resistance for half-cell"
        ]
        """
        Electrolyte conductivity resistance (metal counter electrode).
        """
        self.Rₛ = self.constants[
            "Electrode conductivity resistance"
        ]
        """Electrode conductivity resistance."""
        self.Rₛₙ = self.constants[
            "Negative electrode conductivity resistance"
        ]
        """Negative electrode conductivity resistance."""
        self.Rₛₚ = self.constants[
            "Positive electrode conductivity resistance"
        ]
        """Positive electrode conductivity resistance."""
        self.Rₛ_metal_counter = self.constants[
            "Electrode conductivity resistance for half-cell"
        ]
        """
        Electrode conductivity resistance (metal counter electrode).
        """
        self.κₑ = self.constants["Ionic conductivity"]
        """Ionic conductivity of the electrolyte."""

        self.timescale = self.parameters.process_symbol(τᵈ).evaluate()
        """Timescale used for non-dimensionalization in s."""
        # self.OCVₚ_ref = self.parameters.process_symbol(OCVₚ_ref)
        # self.OCVₙ_ref = self.parameters.process_symbol(OCVₙ_ref)
        self.thermal_voltage = self.parameters.process_symbol(
            thermal_voltage
        ).evaluate()
        """Voltage used for non-dimensionalization in V."""
        self.C = self.parameters.process_symbol(C).evaluate()
        """C-rate of the battery in A."""
        self.Cₑ = self.parameters.process_symbol(Cₑ).evaluate()
        """Non-dimensionalized timescale of the electrolyte."""
        self.Dₑ = self.parameters.process_symbol(
            Dₑ(cₑ_init, T_init)
        ).evaluate()
        """Non-dimensionalized diffusivity of the electrolyte."""
        self.Dₑ_dim = self.parameters.process_symbol(
            Dₑ_dim(cₑ_init, T_init)
        ).evaluate()
        """Diffusivity of the electrolyte."""
        self.βₚ = self.parameters.process_symbol(βₑₚ_scalar).evaluate()
        """Bruggeman coefficient of the cathode."""
        self.βₛ = self.parameters.process_symbol(βₑₛ_scalar).evaluate()
        """Bruggeman coefficient of the separator."""
        self.βₙ = self.parameters.process_symbol(βₑₙ_scalar).evaluate()
        """Bruggeman coefficient of the anode."""
        self.εₚ = self.parameters.process_symbol(εₚ_scalar).evaluate()
        """Porosity of the cathode."""
        self.εₛ = self.parameters.process_symbol(εₛ_scalar).evaluate()
        """Porosity of the separator."""
        self.εₙ = self.parameters.process_symbol(εₙ_scalar).evaluate()
        """Porosity of the anode."""
        self.Lₚ = self.parameters.process_symbol(Lₚ).evaluate()
        """Fraction of the cathode length of the battery length."""
        self.Lₛ = self.parameters.process_symbol(Lₛ).evaluate()
        """Fraction of the separator length of the battery length."""
        self.Lₙ = self.parameters.process_symbol(Lₙ).evaluate()
        """Fraction of the anode length of the battery length."""
        self.t_plus = self.parameters.process_symbol(
            t_plus(cₑ_init)
        ).evaluate()
        """Transference number."""
        self.γₑ = self.parameters.process_symbol(γₑ).evaluate()
        """Non-dimensionalized electrolyte concentration."""
        self.one_plus_dlnf_dlnc = self.parameters.process_symbol(
            one_plus_dlnf_dlnc(cₑ_init)
        ).evaluate()
        """Thermodynamic factor."""
        self.zₚ = self.parameters.process_symbol(zₚ).evaluate()
        """Charge number for the cathode."""
        self.zₙ = self.parameters.process_symbol(zₙ).evaluate()
        """Charge number for the anode."""
        self.C_DLₙ = self.parameters.process_symbol(C_DLₙ).evaluate()
        """Negative electrode double-layer capacity."""
        self.C_DLₚ = self.parameters.process_symbol(C_DLₚ).evaluate()
        """Positive electrode double-layer capacity."""
        self.L_SEI = self.parameters.process_symbol(L_SEI).evaluate()
        """Thickness of the SEI."""
        self.R_SEI = self.parameters.process_symbol(R_SEI).evaluate()
        """Resistance of the SEI."""
        self.C_SEI = self.parameters.process_symbol(C_SEI).evaluate()
        """Capacitance of the SEI."""
        self.nₚ = self.parameters.process_symbol(nₚ).evaluate()
        """Stoichiometry of salt dissociation (cation)."""
        self.nₙ = self.parameters.process_symbol(nₙ).evaluate()
        """Stoichiometry of salt dissociation (anion)."""
        self.zₚ_salt = self.parameters.process_symbol(zₚ_salt).evaluate()
        """Charge number of salt dissociation (cation)."""
        self.zₙ_salt = self.parameters.process_symbol(zₙ_salt).evaluate()
        """Charge number of salt dissociation (anion)."""
        self.ρₑ = self.parameters.process_symbol(ρₑ).evaluate()
        """Electrolyte mass density."""
        self.ρₑ_plus = self.parameters.process_symbol(ρₑ_plus).evaluate()
        """Cation mass density in electrolyte."""
        self.M_N = self.parameters.process_symbol(M_N).evaluate()
        """Molar mass of electrolyte solvent."""
        self.c_N = self.parameters.process_symbol(c_N).evaluate()
        """Solvent concentration."""
        self.ρ_N = self.parameters.process_symbol(ρ_N).evaluate()
        """Solvent mass density."""
        self.v_N = self.parameters.process_symbol(v_N).evaluate()
        """Solvent partial molar volume."""
        self.tilde_ρ_N = self.parameters.process_symbol(tilde_ρ_N).evaluate()
        """Solvent partial mass density."""
        self.ε_SEI = self.parameters.process_symbol(ε_SEI).evaluate()
        """SEI porosity."""
        self.β_SEI = self.parameters.process_symbol(β_SEI).evaluate()
        """SEI Bruggeman coefficient (for tortuosity)."""
        self.L_electrolyte_for_SEI_model = (
            self.parameters.process_symbol(
                L_electrolyte_for_SEI_model
            ).evaluate()
        )
        """
        Length-scale for bulk electrolyte contribution in SEI model.
        """
        self.t_SEI_minus = (
            self.parameters.process_symbol(t_SEI_minus).evaluate()
        )
        """Anion transference number in SEI."""
        self.F = self.parameters.process_symbol(F).evaluate()
        """Faraday constant."""

    def Z_SPM(self, s_dim):
        """
        Transfer function U(s) / I(s) of the SPM.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """

        s_array = [
            2 * pi * mpmathify(s) for s in atleast_1d(self.timescale * s_dim)
        ]
        sqrt_s_array = [s**0.5 for s in s_array]
        tanh_p_array = [tanh(sqrt_s * self.Zₚ_int) for sqrt_s in sqrt_s_array]
        tanh_n_array = [tanh(sqrt_s * self.Zₙ_int) for sqrt_s in sqrt_s_array]
        if self.verbose:
            print("Positive electrode resistance:", self.Rₚ_int)
            print("Negative electrode resistance:", -self.Rₙ_int)
            print(
                "Positive electrode diffusion resonance frequency:",
                1 / self.Zₚ_int**2
            )
            print(
                "Negative electrode diffusion resonance frequency:",
                1 / self.Zₙ_int**2
            )
        return array([
            complex(
                (
                    self.Rₚ_int * tanh_p
                    / (tanh_p - sqrt_s * self.Zₚ_int)
                    + self.Rₙ_int * tanh_n
                    / (tanh_n - sqrt_s * self.Zₙ_int)
                    + self.Rₛₑ
                ) * self.thermal_voltage / self.C
            )
            for sqrt_s, tanh_p, tanh_n in zip(
                sqrt_s_array, tanh_p_array, tanh_n_array
            )
        ])

    def Z_SPM_offset(self):
        """
        Static part of the transfer function of the SPM.

        :returns:
            The part of the SPM's impedance that doesn't depend on
            frequency.
        """
        return self.Rₛₑ * self.thermal_voltage / self.C

    def Aₛ(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the SPMe.
        Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        cosh_n_array = [
            cosh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_n_array = [
            sinh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        return [
            exp(-self.Lₙ * self.Z_cₑₛ * sqrt_s) * (
                (1 - self.t_plus)
                / (2 * self.εₙ * self.Lₙ * self.Cₑ * self.γₑ * s)
                + A_n * (
                    cosh_n
                    + (self.εₙ ** self.βₙ) / (self.εₛ ** self.βₛ)
                    * self.Z_cₑₙ / self.Z_cₑₛ
                    * sinh_n
                )
            )
            for s, sqrt_s, cosh_n, sinh_n, A_n in zip(
                s_array,
                sqrt_s_array,
                cosh_n_array,
                sinh_n_array,
                self.Aₙ(s_nondim)
            )
        ]

    def Bₛ(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the SPMe.
        Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        cosh_n_array = [
            cosh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_n_array = [
            sinh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        return [
            exp(self.Lₙ * self.Z_cₑₛ * sqrt_s) * (
                (1 - self.t_plus)
                / (2 * self.εₙ * self.Lₙ * self.Cₑ * self.γₑ * s)
                + A_n * (
                    cosh_n
                    - (self.εₙ ** self.βₙ) / (self.εₛ ** self.βₛ)
                    * self.Z_cₑₙ / self.Z_cₑₛ
                    * sinh_n
                )
            )
            for s, sqrt_s, cosh_n, sinh_n, A_n in zip(
                s_array,
                sqrt_s_array,
                cosh_n_array,
                sinh_n_array,
                self.Aₙ(s_nondim)
            )
        ]

    def Aₙ(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the SPMe.
        Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        sinh_n_array = [
            sinh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_s_array = [
            sinh(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_n_array = [
            coth(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_s_array = [
            coth(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_p_array = [
            coth(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]

        return [
            1 / self.Cₑ * (
                (1 - self.t_plus)
                / (2 * self.γₑ * s)
                * (
                    1 / (self.εₚ * self.Lₚ)
                    + sinh_s * (
                        self.εₛ ** self.βₛ / (self.εₚ ** self.βₚ)
                        * self.Z_cₑₛ / self.Z_cₑₚ
                        * coth_p
                        + coth_s
                    ) / (self.εₙ * self.Lₙ)
                )
                * self.εₛ ** self.βₛ / (self.εₙ ** self.βₙ)
                * self.Z_cₑₛ / self.Z_cₑₙ
            ) / (
                1 / sinh_s
                - sinh_s
                * (
                    self.εₛ ** self.βₛ / (self.εₙ ** self.βₙ)
                    * self.Z_cₑₛ / self.Z_cₑₙ
                    * coth_n
                    + coth_s
                ) * (
                    self.εₛ ** self.βₛ / (self.εₚ ** self.βₚ)
                    * self.Z_cₑₛ / self.Z_cₑₚ
                    * coth_p
                    + coth_s
                )
            ) / sinh_n  # this sinh cancels out in the concentration expression
            for s, sinh_n, sinh_s, coth_n, coth_s, coth_p in zip(
                s_array,
                sinh_n_array,
                sinh_s_array,
                coth_n_array,
                coth_s_array,
                coth_p_array
            )
        ]

    def Cₚ(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the SPMe.
        Note: compared to its formulation in the appended PDF,
        Cₚ(s) here is Aₚ(s) exp(Z_cₑₚ sqrt(s)) in the PDF.

        :param s:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        sinh_s_array = [
            sinh(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_p_array = [
            sinh(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_n_array = [
            coth(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_s_array = [
            coth(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_p_array = [
            coth(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]

        return [
            1 / self.Cₑ * (
                -(1 - self.t_plus)
                / (2 * self.γₑ * s)
                * (
                    1 / (self.εₙ * self.Lₙ)
                    + sinh_s * (
                        self.εₛ ** self.βₛ / (self.εₙ ** self.βₙ)
                        * self.Z_cₑₛ / self.Z_cₑₙ
                        * coth_n
                        + coth_s
                    ) / (self.εₚ * self.Lₚ)
                )
                * self.εₛ ** self.βₛ / (self.εₚ ** self.βₚ)
                * self.Z_cₑₛ / self.Z_cₑₚ
            ) / (
                1 / sinh_s
                - sinh_s
                * (
                    self.εₛ ** self.βₛ / (self.εₙ ** self.βₙ)
                    * self.Z_cₑₛ / self.Z_cₑₙ
                    * coth_n
                    + coth_s
                ) * (
                    self.εₛ ** self.βₛ / (self.εₚ ** self.βₚ)
                    * self.Z_cₑₛ / self.Z_cₑₚ
                    * coth_p
                    + coth_s
                )
            ) / sinh_p  # this sinh cancels out in the concentration expression
            for s, sinh_s, sinh_p, coth_n, coth_s, coth_p in zip(
                s_array,
                sinh_s_array,
                sinh_p_array,
                coth_n_array,
                coth_s_array,
                coth_p_array
            )
        ]

    def cₑₛ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration within the separator.

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the separator.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        if self.verbose:
            print(
                "Separator electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₛ**2
            )

        # Handle high frequencies separately, since no amount of
        # 'just more precision' fixes cancelling exponentials.
        return [
            A_s * exp(x_nondim * self.Z_cₑₛ * sqrt_s)
            + B_s * exp(-x_nondim * self.Z_cₑₛ * sqrt_s)
            if abs(self.Z_cₑₛ * sqrt_s) < 80 else
            0
            for sqrt_s, A_s, B_s in zip(
                sqrt_s_array, self.Aₛ(s_nondim), self.Bₛ(s_nondim)
            )
        ]

    def d_dx_cₑₛ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration within the separator.

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the separator.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]

        # Since this is not actually used for parameterization,
        # it remains in this numerically unstable form.
        return [
            self.Z_cₑₛ * sqrt_s * (
                A_s * exp(x_nondim * self.Z_cₑₛ * sqrt_s)
                - B_s * exp(-x_nondim * self.Z_cₑₛ * sqrt_s)
            )
            for sqrt_s, A_s, B_s in zip(
                sqrt_s_array, self.Aₛ(s_nondim), self.Bₛ(s_nondim)
            )
        ]

    def cₑₙ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration within the neg. electrode.

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the neg. electrode.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        if self.verbose:
            print(
                "Negative electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₙ**2
            )

        return [
            + (1 - self.t_plus)
            / (self.Lₙ * self.Cₑ * self.γₑ * self.εₙ * s)
            + A_n * 2 * cosh(x_nondim * self.Z_cₑₙ * sqrt_s)
            for s, sqrt_s, A_n in zip(s_array, sqrt_s_array, self.Aₙ(s_nondim))
        ]

    def d_dx_cₑₙ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration gradient (neg. electrode).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the neg. electrode.
        :returns:
            The electrolyte concentration gradient.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]

        return [
            A_n * 2 * self.Z_cₑₙ * sqrt_s
            * sinh(x_nondim * self.Z_cₑₙ * sqrt_s)
            for sqrt_s, A_n in zip(sqrt_s_array, self.Aₙ(s_nondim))
        ]

    def cₑₚ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration within the pos. electrode.

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the pos. electrode.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        if self.verbose:
            print(
                "Positive electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₚ**2
            )

        return [
            - (1 - self.t_plus)
            / (self.Lₚ * self.Cₑ * self.γₑ * self.εₚ * s)
            + C_p * 2 * cosh((1 - x_nondim) * self.Z_cₑₚ * sqrt_s)
            for s, sqrt_s, C_p in zip(s_array, sqrt_s_array, self.Cₚ(s_nondim))
        ]

    def d_dx_cₑₚ_1(self, s_nondim, x_nondim):
        """
        Electrolyte concentration gradient (pos. electrode).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the pos. electrode.
        :returns:
            The electrolyte concentration gradient.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]

        return [
            - C_p * 2 * self.Z_cₑₚ * sqrt_s
            * sinh((1 - x_nondim) * self.Z_cₑₚ * sqrt_s)
            for sqrt_s, C_p in zip(sqrt_s_array, self.Cₚ(s_nondim))
        ]

    def bar_cₑₙ_1(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the full-cell
        SPMe. Note: compared to its formulation in the appended PDF,
        it has already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        sinh_n_array = [
            sinh(self.Lₙ * self.Z_cₑₙ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        if self.verbose:
            print(
                "Negative electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₙ**2
            )

        return [
            +(1 - self.t_plus) / (self.Cₑ * self.γₑ * self.εₙ * s)
            + 2 * sinh_n / (self.Z_cₑₙ * sqrt_s) * A_n
            for s, sqrt_s, sinh_n, A_n in zip(
                s_array, sqrt_s_array, sinh_n_array, self.Aₙ(s_nondim)
            )
        ]

    def bar_cₑₚ_1(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the full-cell
        SPMe. Note: compared to its formulation in the appended PDF,
        it has already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        sinh_p_array = [
            sinh(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        if self.verbose:
            print(
                "Positive electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₚ**2
            )

        return [
            -(1 - self.t_plus) / (self.Cₑ * self.γₑ * self.εₚ * s)
            + 2 * sinh_p / (self.Z_cₑₚ * sqrt_s) * C_p
            for s, sqrt_s, sinh_p, C_p in zip(
                s_array, sqrt_s_array, sinh_p_array, self.Cₚ(s_nondim)
            )
        ]

    def Z_SPMe_1(self, s_dim):
        """
        Additional correction to the SPM's transfer function.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """
        s = atleast_1d(self.timescale * s_dim)

        # To match the convention of positive real part of impedance,
        # the concentration expressions have to be negated.
        return array([
            complex(
                (self.Rₛ + self.Rₑ) * self.thermal_voltage / self.C
                - self.Cₑ * (  # <- this - negates the concentration expression
                    2 * (1 - self.t_plus) * self.one_plus_dlnf_dlnc * (
                        right - left
                    )
                    + right / self.zₚ - left / self.zₙ
                ) * self.thermal_voltage / self.C
            )
            for left, right in zip(self.bar_cₑₙ_1(s), self.bar_cₑₚ_1(s))
        ])

    def Z_SPMe_1_offset(self):
        """
        Static part of the correction to the SPM's impedance.

        :returns:
            The part of the difference between the SPM's and SPMe's
            impedance that doesn't depend on frequency.
        """
        return (self.Rₛ + self.Rₑ) * self.thermal_voltage / self.C

    def Z_SPMe(self, s_dim):
        """
        Transfer function U(s) / I(s) of the SPMe.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """
        return self.Z_SPM(s_dim) + self.Z_SPMe_1(s_dim)

    def Z_SPMe_offset(self):
        """
        Static part of the transfer function of the SPMe.

        :returns:
            The part of the SPMe's impedance that doesn't depend on
            frequency.
        """
        return self.Z_SPM_offset() + self.Z_SPMe_1_offset()

    def Z_SPM_metal_counter(self, s_dim):
        """
        Transfer function U(s) / I(s) of the SPM (metal CE).

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """
        s_array = [
            2 * pi * mpmathify(s) for s in atleast_1d(self.timescale * s_dim)
        ]
        sqrt_s_array = [s**0.5 for s in s_array]
        tanh_p_array = [tanh(sqrt_s * self.Zₚ_int) for sqrt_s in sqrt_s_array]
        if self.verbose:
            print("Positive electrode resistance:", self.Rₚ_int)
            print(
                "Positive electrode diffusion resonance frequency:",
                1 / self.Zₚ_int**2
            )

        return array([
            complex(
                (
                    self.Rₚ_int * tanh_p / (tanh_p - sqrt_s * self.Zₚ_int)
                    + self.constants[
                        "Charge transfer resistance for half-cell"
                    ]
                ) * self.thermal_voltage / self.C
            )
            for sqrt_s, tanh_p in zip(sqrt_s_array, tanh_p_array)
        ])

    def Z_SPM_offset_metal_counter(self):
        """
        Static part of the transfer function of the SPM (metal CE).

        :returns:
            The part of the half-cell SPM's impedance that doesn't
            depend on frequency.
        """
        return self.constants[
            "Charge transfer resistance for half-cell"
        ] * self.thermal_voltage / self.C

    def Cₚ_metal_counter(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the metal CE SPMe

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """
        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        coth_p_array = [
            coth(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        coth_s_array = [
            coth(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_p_array = [
            sinh(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_s_array = [
            sinh(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]

        return [
            0.5 * (1 - self.t_plus) / (self.γₑ * sqrt_s) * (
                sinh_s / (self.Cₑ * self.εₚ * self.Lₚ * sqrt_s)
                + 1.0 / (self.εₛ ** self.βₛ * self.Z_cₑₛ * self.Dₑ)
            ) / sinh_p / (
                self.εₚ ** self.βₚ * self.Z_cₑₚ / (
                    self.εₛ ** self.βₛ * self.Z_cₑₛ
                ) * sinh_s * (
                    self.εₛ ** self.βₛ * self.Z_cₑₛ / (
                        self.εₚ**self.βₚ * self.Z_cₑₚ
                    ) * coth_p + coth_s
                )
            )
            for sqrt_s, coth_p, coth_s, sinh_p, sinh_s in zip(
                sqrt_s_array,
                coth_p_array,
                coth_s_array,
                sinh_p_array,
                sinh_s_array
            )
        ]

    def Aₛ_metal_counter(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the metal CE
        SPMe. Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        exp_s_array = [
            exp(-self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_p_array = [
            sinh(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]
        sinh_s_array = [
            sinh(self.Lₛ * self.Z_cₑₛ * sqrt_s) for sqrt_s in sqrt_s_array
        ]

        return [
            - (self.εₚ ** self.βₚ * self.Z_cₑₚ) / (
                self.εₛ ** self.βₛ * self.Z_cₑₛ
            ) * sinh_p / sinh_s * C_p_mc
            + (1 - self.t_plus) / (
                2 * self.εₛ ** self.βₛ * self.Z_cₑₛ
                * self.Dₑ * self.γₑ * sqrt_s
            ) * exp_s / sinh_s
            for sqrt_s, exp_s, sinh_p, sinh_s, C_p_mc in zip(
                sqrt_s_array,
                exp_s_array,
                sinh_p_array,
                sinh_s_array,
                self.Cₚ_metal_counter(s_nondim)
            )
        ]

    def Bₛ_metal_counter(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the metal CE
        SPMe. Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]

        return [
            A_s_mc + (1 - self.t_plus) / (
                self.εₛ ** self.βₛ * self.Z_cₑₛ
                * self.Dₑ * self.γₑ * s**0.5
            )
            for s, A_s_mc in zip(s_array, self.Aₛ_metal_counter(s_nondim))
        ]

    def cₑₚ_1_metal_counter(self, s_nondim, x_nondim):
        """
        Electrolyte concentration against Li metal (pos. electrode).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the pos. electrode.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        if self.verbose:
            print(
                "Positive electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₚ**2
            )

        return [
            - (1 - self.t_plus) / (
                self.Cₑ * self.γₑ * self.εₚ * self.Lₚ * s
            ) + 2 * C_p_mc * (
                cosh((1 - x_nondim) * self.Z_cₑₚ * s**0.5)
            )
            for s, C_p_mc in zip(s_array, self.Cₚ_metal_counter(s_nondim))
        ]

    def cₑₛ_1_metal_counter(self, s_nondim, x_nondim):
        """
        Electrolyte concentration against Li metal (separator).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the separator.
        :returns:
            The electrolyte concentration.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        if self.verbose:
            print(
                "Separator electrolyte diffusion resonance frequency:",
                1 / self.Z_cₑₛ**2
            )

        return [
            A_s_mc * exp(x_nondim * self.Z_cₑₛ * sqrt_s)
            + B_s_mc * exp(-x_nondim * self.Z_cₑₛ * sqrt_s)
            for sqrt_s, A_s_mc, B_s_mc in zip(
                sqrt_s_array,
                self.Aₛ_metal_counter(s_nondim),
                self.Bₛ_metal_counter(s_nondim)
            )
        ]

    def d_dx_cₑₚ_1_metal_counter(self, s_nondim, x_nondim):
        """
        Electrolyte gradient against Li metal (pos. electrode).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the pos. electrode.
        :returns:
            The electrolyte concentration gradient.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]

        return [
            - 2 * C_p_mc * self.Z_cₑₚ * sqrt_s * (
                sinh((1 - x_nondim) * self.Z_cₑₚ * sqrt_s)
            )
            for sqrt_s, C_p_mc in zip(
                sqrt_s_array, self.Cₚ_metal_counter(s_nondim)
            )
        ]

    def d_dx_cₑₛ_1_metal_counter(self, s_nondim, x_nondim):
        """
        Electrolyte gradient against Li metal (separator).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :param x_nondim:
            The non-dimensionalized location within the cell.
            0 is the negative current collector and 1 the positive one.
            Return values are only valid within the separator.
        :returns:
            The electrolyte concentration gradient.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]

        return [
            self.Z_cₑₛ * sqrt_s * (
                A_s_mc * exp(x_nondim * self.Z_cₑₛ * sqrt_s)
                - B_s_mc * exp(-x_nondim * self.Z_cₑₛ * sqrt_s)
            )
            for sqrt_s, A_s_mc, B_s_mc in zip(
                sqrt_s_array,
                self.Aₛ_metal_counter(s_nondim),
                self.Bₛ_metal_counter(s_nondim)
            )
        ]

    def bar_cₑₚ_1_metal_counter(self, s_nondim):
        """
        Integration constant; please refer to the PDF.

        Integration constant of the solution of cₑ¹ in the metal CE
        SPMe. Note: compared to its formulation in the appended PDF,
        they have already been scaled with I(s).

        :param s_nondim:
            The non-dimensionalized frequencies as an array.
        :returns:
            Integration constant of the solution of cₑ¹ in the half-cell
            SPMe.
        """

        s_array = [2 * pi * mpmathify(s) for s in atleast_1d(s_nondim)]
        sqrt_s_array = [s**0.5 for s in s_array]
        sinh_p_array = [
            sinh(self.Lₚ * self.Z_cₑₚ * sqrt_s) for sqrt_s in sqrt_s_array
        ]

        return [
            -(1 - self.t_plus) / (self.Cₑ * self.γₑ * self.εₚ * s)
            + 2 * sinh_p / (self.Z_cₑₚ * sqrt_s) * C_p_mc
            for s, sqrt_s, sinh_p, C_p_mc in zip(
                s_array,
                sqrt_s_array,
                sinh_p_array,
                self.Cₚ_metal_counter(s_nondim)
            )
        ]

    def Z_SPMe_1_metal_counter(self, s_dim):
        """
        Correction to the transfer function of the SPM (metal CE).

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """
        s = atleast_1d(self.timescale * s_dim)

        # To match the convention of positive real part of impedance,
        # the concentration expressions have to be negated.
        return (
            self.Rₛ_metal_counter + self.Rₑ_metal_counter
            - self.Cₑ * (  # <- this - negates the concentration expressions
                2 * (1 - self.t_plus) * self.one_plus_dlnf_dlnc * (
                    self.bar_cₑₚ_1_metal_counter(s)
                    - self.cₑₛ_1_metal_counter(s, 0)
                )
                + self.bar_cₑₚ_1_metal_counter(s) / self.zₚ
                - self.cₑₛ_1_metal_counter(s, 0) / self.zₙ
            )
        ) * (self.thermal_voltage / self.C)

    def Z_SPMe_1_offset_metal_counter(self):
        """#
        Static part of the impedance of the SPMe(1) (metal CE).

        :returns:
            The part of the difference between the SPM's and SPMe's
            impedance for half-cells that doesn't depend on frequency.
        """
        return (
            (self.Rₛ_metal_counter + self.Rₑ_metal_counter)
            * self.thermal_voltage / self.C
        )

    def Z_SPMe_metal_counter(self, s_dim):
        """
        Transfer function U(s) / I(s) of the SPMe (metal CE).

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """
        return (
            self.Z_SPM_metal_counter(s_dim)
            + self.Z_SPMe_1_metal_counter(s_dim)
        )

    def Z_SPMe_offset_metal_counter(self):
        """
        Static part of the half-cell SPMe's transfer function.

        :returns:
            The part of the half-cell SPMe's impedance that doesn't
            depend on frequency.
        """
        return (
            self.Z_SPM_offset_metal_counter()
            + self.Z_SPMe_1_offset_metal_counter()
        )

    def Z_SPM_reference_electrode(
            self, s_dim, working_electrode="positive"
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :returns:
            The evaluated impedances as an array.
        """

        s_array = [
            2 * pi * mpmathify(s) for s in atleast_1d(self.timescale * s_dim)
        ]
        sqrt_s_array = [s**0.5 for s in s_array]
        if working_electrode == "positive":
            tanh_p_array = [
                tanh(sqrt_s * self.Zₚ_int) for sqrt_s in sqrt_s_array
            ]
            if self.verbose:
                print("Positive electrode resistance:", self.Rₚ_int)
                print(
                    "Positive electrode diffusion resonance frequency:",
                    1 / self.Zₚ_int**2
                )
            return array([
                complex(
                    (
                        self.Rₚ_int * tanh_p / (tanh_p - sqrt_s * self.Zₚ_int)
                        + self.Rₛₑₚ
                    ) * self.thermal_voltage / self.C
                )
                for sqrt_s, tanh_p in zip(sqrt_s_array, tanh_p_array)
            ])
        elif working_electrode == "negative":
            tanh_n_array = [
                tanh(sqrt_s * self.Zₙ_int) for sqrt_s in sqrt_s_array
            ]
            if self.verbose:
                print("Negative electrode resistance:", -self.Rₙ_int)
                print(
                    "Negative electrode diffusion resonance frequency:",
                    1 / self.Zₙ_int**2
                )
            return array([
                complex(
                    (
                        self.Rₙ_int * tanh_n / (tanh_n - sqrt_s * self.Zₙ_int)
                        + self.Rₛₑₙ
                    ) * self.thermal_voltage / self.C
                )
                for sqrt_s, tanh_n in zip(sqrt_s_array, tanh_n_array)
            ])
        else:
            raise ValueError(
                "Working electrode has to be either 'positive' or 'negative'."
            )

    def Z_SPM_offset_reference_electrode(
        self, working_electrode="positive"
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :returns:
            The evaluated impedances as an array.
        """
        if working_electrode == "positive":
            return self.Rₛₑₚ * self.thermal_voltage / self.C
        elif working_electrode == "negative":
            return self.Rₛₑₙ * self.thermal_voltage / self.C
        else:
            raise ValueError(
                "Working electrode has to be either 'positive' or 'negative'."
            )

    def Z_SPMe_1_reference_electrode(
        self, s_dim, working_electrode="positive", dimensionless_location=0.5
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :param dimensionless_location:
            The location of the reference electrode. 0 refers to the
            point where negative electrode and separator touch, and 1
            refers to the point where separator and positive electrode
            touch.
        :returns:
            The evaluated impedances as an array.
        """
        s = atleast_1d(self.timescale * s_dim)

        if working_electrode == "positive":
            Rₑ = (
                self.Lₚ / (3 * self.εₚ**self.βₚ)
                + self.Lₛ / self.εₛ**self.βₛ * (1 - dimensionless_location)
            ) / self.κₑ
            Rₛ = self.Rₛₚ
            left = self.cₑₛ_1(
                s, self.Lₙ + self.Lₛ * dimensionless_location
            )
            right = self.bar_cₑₚ_1(s)
        elif working_electrode == "negative":
            Rₑ = (
                self.Lₙ / (3 * self.εₙ**self.βₙ)
                + self.Lₛ / self.εₛ**self.βₛ * dimensionless_location
            ) / self.κₑ
            Rₛ = self.Rₛₙ
            left = self.bar_cₑₙ_1(s)
            right = self.cₑₛ_1(
                s, self.Lₙ + self.Lₛ * dimensionless_location
            )
        else:
            raise ValueError(
                "Working electrode has to be either 'positive' or 'negative'."
            )

        # To match the convention of positive real part of impedance,
        # the concentration expressions have to be negated.
        return array([
            complex(
                (Rₛ + Rₑ) * self.thermal_voltage / self.C
                - self.Cₑ * (  # <- this - negates the concentration terms
                    2 * (1 - self.t_plus)
                    * self.one_plus_dlnf_dlnc * (r - l)
                    + r / self.zₚ - l / self.zₙ
                ) * self.thermal_voltage / self.C
            )
            for r, l in zip(right, left)
        ])

    def Z_SPMe_1_offset_reference_electrode(
        self, working_electrode="positive", dimensionless_location=0.5
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :param dimensionless_location:
            The location of the reference electrode. 0 refers to the
            point where negative electrode and separator touch, and 1
            refers to the point where separator and positive electrode
            touch.
        :returns:
            The evaluated impedances as an array.
        """

        if working_electrode == "positive":
            Rₑ = (
                self.Lₚ / (3 * self.εₚ**self.βₚ)
                + self.Lₛ / self.εₛ**self.βₛ * (1 - dimensionless_location)
            ) / self.κₑ
            Rₛ = self.Rₛₚ
        elif working_electrode == "negative":
            Rₑ = (
                self.Lₙ / (3 * self.εₙ**self.βₙ)
                + self.Lₛ / self.εₛ**self.βₛ * dimensionless_location
            ) / self.κₑ
            Rₛ = self.Rₛₙ
        else:
            raise ValueError(
                "Working electrode has to be either 'positive' or 'negative'."
            )
        return (
            (Rₛ + Rₑ) * self.thermal_voltage / self.C
        )

    def Z_SPMe_reference_electrode(
        self, s_dim, working_electrode="positive", dimensionless_location=0.5
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :param dimensionless_location:
            The location of the reference electrode. 0 refers to the
            point where negative electrode and separator touch, and 1
            refers to the point where separator and positive electrode
            touch.
        :returns:
            The evaluated impedances as an array.
        """
        return (
            self.Z_SPM_reference_electrode(
                s_dim, working_electrode
            )
            + self.Z_SPMe_1_reference_electrode(
                s_dim, working_electrode, dimensionless_location
            )
        )

    def Z_SPMe_offset_reference_electrode(
        self, working_electrode="positive"
    ):
        """
        Impedance against a reference electrode.

        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :returns:
            The evaluated impedances as an array.
        """
        return (
            self.Z_SPM_offset_reference_electrode(
                working_electrode
            )
            + self.Z_SPMe_1_offset_reference_electrode(
                working_electrode
            )
        )

    def Z_DL(self, s_dim, working_electrode=None):
        """
        Impedance of the Double-Layers.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
            Defaults to both electrodes contributing to the impedance.
        :returns:
            The evaluated impedances as an array.
        """

        if working_electrode == "negative":
            if self.verbose:
                print(
                    "Negative double-layer resonance frequency:",
                    -1 / (self.Rₙ_int * self.C_DLₙ)
                )
            return 1 / (2 * pi * s_dim * self.C_DLₙ)
        elif working_electrode == "positive":
            if self.verbose:
                print(
                    "Positive double-layer resonance frequency:",
                    1 / (self.Rₚ_int * self.C_DLₚ)
                )
            return 1 / (2 * pi * s_dim * self.C_DLₚ)
        else:
            if self.verbose:
                print(
                    "Negative double-layer resonance frequency:",
                    -1 / (self.Rₙ_int * self.C_DLₙ)
                )
                print(
                    "Positive double-layer resonance frequency:",
                    1 / (self.Rₚ_int * self.C_DLₚ)
                )
            return (
                1 / (2 * pi * s_dim * self.C_DLₙ)
                + 1 / (2 * pi * s_dim * self.C_DLₚ)
            )

    def Z_SEI(self, s_dim):
        """
        Impedance of the Double-Layer.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """

        # Notations refer to Single2019.
        k_electrolyte = [
            (1 - 1j) * sqrt(
                self.εₙ ** (-self.βₙ) * 2 * pi * s / 1j / (2 * self.Dₑ_dim)
            )
            for s in s_dim
        ]
        k_SEI = [
            (1 - 1j) * sqrt(
                self.ε_SEI ** (1 - self.β_SEI)
                * 2 * pi * s / 1j / (2 * self.Dₑ_dim)
            )
            for s in s_dim
        ]
        Theta = (
            - (self.nₚ + self.nₙ) / (self.nₚ * self.nₙ)
            / (self.zₚ_salt * self.zₙ_salt * self.F**2)
            * self.ρₑ**2 / (self.ρ_N * self.tilde_ρ_N)
            * self.one_plus_dlnf_dlnc
        )
        Psi = [
            1
            - self.ε_SEI ** ((1 + self.β_SEI) / 2)
            * tan(k_e * self.L_electrolyte_for_SEI_model)
            * tan(k_sei * self.L_SEI)
            for k_e, k_sei in zip(k_electrolyte, k_SEI)
        ]
        D_electrolyte_eff = self.Dₑ_dim * self.ε_SEI ** self.β_SEI
        Z_D_SEI = [
            self.L_SEI * Theta / D_electrolyte_eff
            * (self.t_SEI_minus - self.ρₑ_plus / self.ρₑ)**2
            * tan(k_sei * self.L_SEI)
            / (psi * k_sei * self.L_SEI)
            for k_sei, psi in zip(k_SEI, Psi)
        ]

        if self.verbose:
            print("SEI ohmic resistance:", self.R_SEI)
            print(
                "SEI capacitance resonance frequency:",
                1 / (self.R_SEI * self.C_SEI)
            )
            print(
                "SEI diffusion resonance frequency:",
                float(1.2703 * D_electrolyte_eff / (pi * self.L_SEI**2))
            )
            print(
                "SEI diffusion resistance:",
                float(
                    self.L_SEI * Theta / D_electrolyte_eff
                    * (self.t_SEI_minus - self.ρₑ_plus / self.ρₑ)**2
                )
            )

        return array([
            complex(1 / (2 * pi * s * self.C_SEI + 1 / (self.R_SEI + zdsei)))
            for s, zdsei in zip(s_dim, Z_D_SEI)
        ])

    def Z_SPM_with_double_layer_and_SEI(self, s_dim):
        """
        Impedance of the SPM with Double-Layer and SEI.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """

        Z_SEI_contribution = self.Z_SEI(s_dim)
        Z_SPM_contribution = self.Z_SPM(s_dim)
        Z_DL_contribution = self.Z_DL(s_dim)

        return array([
            complex(zsei + 1 / (1 / zspm + 1 / zdl))
            for zsei, zspm, zdl in zip(
                Z_SEI_contribution,
                Z_SPM_contribution,
                Z_DL_contribution
            )
        ])

    def Z_SPM_with_double_layer_and_SEI_offset(self):
        """
        Impedance of the SPM with Double-Layer and SEI.

        :returns:
            The evaluated impedances as an array.
        """
        Theta = (
            - (self.nₚ + self.nₙ) / (self.nₚ * self.nₙ)
            / (self.zₚ_salt * self.zₙ_salt * self.F**2)
            * self.ρₑ**2 / (self.ρ_N * self.tilde_ρ_N)
            * self.one_plus_dlnf_dlnc
        )
        D_electrolyte_eff = self.Dₑ_dim * self.ε_SEI ** self.β_SEI
        return (
            self.Z_SPM_offset()
            + self.R_SEI
            + self.L_SEI * Theta / D_electrolyte_eff
            * (self.t_SEI_minus - self.ρₑ_plus / self.ρₑ)**2
        )

    def Z_SPM_with_double_layer_and_SEI_reference_electrode(
        self, s_dim, working_electrode="positive", dimensionless_location=0.5
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :param dimensionless_location:
            The location of the reference electrode. 0 refers to the
            point where negative electrode and separator touch, and 1
            refers to the point where separator and positive electrode
            touch.
        :returns:
            The evaluated impedances as an array.
        """

        if working_electrode == "negative":
            Z_SEI_contribution = self.Z_SEI(s_dim)
        else:
            Z_SEI_contribution = [0] * len(s_dim)
        Z_SPM_contribution = self.Z_SPM_reference_electrode(
            s_dim, working_electrode
        )
        Z_DL_contribution = self.Z_DL(s_dim, working_electrode)

        return array([
            complex(zsei + 1 / (1 / zspm + 1 / zdl))
            for zsei, zspm, zdl in zip(
                Z_SEI_contribution,
                Z_SPM_contribution,
                Z_DL_contribution
            )
        ])

    def Z_SPMe_with_double_layer_and_SEI(self, s_dim):
        """
        Impedance of the SPMe/DFN with Double-Layer and SEI.

        :param s_dim:
            An array of the frequencies to evaluate.
        :returns:
            The evaluated impedances as an array.
        """

        Z_SPMe_1_contribution = self.Z_SPMe_1(s_dim)
        Z_SEI_contribution = self.Z_SEI(s_dim)
        Z_SPM_contribution = self.Z_SPM(s_dim)
        Z_DL_contribution = self.Z_DL(s_dim)

        return array([
            complex(zspme1 + zsei + 1 / (1 / zspm + 1 / zdl))
            for zspme1, zsei, zspm, zdl in zip(
                Z_SPMe_1_contribution,
                Z_SEI_contribution,
                Z_SPM_contribution,
                Z_DL_contribution
            )
        ])

    def Z_SPMe_with_double_layer_and_SEI_offset(self):
        """
        Impedance of the SPMe/DFN with Double-Layer and SEI.

        :returns:
            The evaluated impedances as an array.
        """
        Theta = (
            - (self.nₚ + self.nₙ) / (self.nₚ * self.nₙ)
            / (self.zₚ_salt * self.zₙ_salt * self.F**2)
            * self.ρₑ**2 / (self.ρ_N * self.tilde_ρ_N)
            * self.one_plus_dlnf_dlnc
        )
        D_electrolyte_eff = self.Dₑ_dim * self.ε_SEI ** self.β_SEI
        return (
            self.Z_SPMe_offset()
            + self.R_SEI
            + self.L_SEI * Theta / D_electrolyte_eff
            * (self.t_SEI_minus - self.ρₑ_plus / self.ρₑ)**2
        )

    def Z_SPMe_with_double_layer_and_SEI_reference_electrode(
        self, s_dim, working_electrode="positive", dimensionless_location=0.5
    ):
        """
        Impedance against a reference electrode.

        :param s_dim:
            An array of the frequencies to evaluate.
        :param working_electrode:
            The electrode against which the voltage was measured with
            respect to the reference electrode.
            'positive' or 'negative'.
        :param dimensionless_location:
            The location of the reference electrode. 0 refers to the
            point where negative electrode and separator touch, and 1
            refers to the point where separator and positive electrode
            touch.
        :returns:
            The evaluated impedances as an array.
        """

        Z_SPMe_1_contribution = self.Z_SPMe_1_reference_electrode(
            s_dim, working_electrode, dimensionless_location
        )
        if working_electrode == "negative":
            Z_SEI_contribution = self.Z_SEI(s_dim)
        else:
            Z_SEI_contribution = [0] * len(s_dim)
        Z_SPM_contribution = self.Z_SPM_reference_electrode(
            s_dim, working_electrode
        )
        Z_DL_contribution = self.Z_DL(s_dim, working_electrode)

        return array([
            complex(zspme1 + zsei + 1 / (1 / zspm + 1 / zdl))
            for zspme1, zsei, zspm, zdl in zip(
                Z_SPMe_1_contribution,
                Z_SEI_contribution,
                Z_SPM_contribution,
                Z_DL_contribution
            )
        ])
