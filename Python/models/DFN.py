# Copyright (c): German Aerospace Center (DLR)
"""!@package models.DFN
Contains a PyBaMM-compatible Doyle-Fuller-Newman (DFN) model.
"""

import pybamm
import models.standard_parameters as standard_parameters
from models.standard_parameters import (
    OCVₙ_ref, OCVₚ_ref, OCVₙ, OCVₚ, OCVₙ_dim, OCVₚ_dim,
    Dₙ, Dₚ,
    cₙ, cₚ, SOCₙ_init, SOCₚ_init,
    iₛₑₙ_0, iₛₑₚ_0,
    aₙ, aₙ_dim, aₚ, aₚ_dim,
    zₙ, zₚ,
    αₙₙ, αₚₙ, αₙₚ, αₚₚ,
    σₙ, σₚ,
    Rₙ, Rₚ,
    cₑ_typ, cₑ_init,
    εₙ, εₛ, εₚ, εₙ_scalar, εₛ_scalar, εₚ_scalar,
    βₑₙ_scalar, βₑₛ_scalar, βₑₚ_scalar, βₛₙ_scalar, βₛₚ_scalar,
    Dₑ, κₑ, κₑ_hat, t_plus, one_plus_dlnf_dlnc,
    Lₙ_dim, Lₚ_dim, L_dim, Lₙ, Lₚ,
    Cₑ, Cₙ, Cₚ, Cᵣₙ, Cᵣₚ,
    γₑ, γₙ, γₚ,
    C, A, F, τᵈ, capacity,
    T_init, thermal_voltage,
    I_extern,
    Uₗ, Uᵤ,
)
from numpy import pi as π
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


class DFN_internal(pybamm.BaseSubModel):
    """!@brief Defining equations for a Doyle-Fuller-Newman-Model.

    Reference
    ----------
    SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman.
    “An asymptotic derivation of a single particle model with
    electrolyte”. Journal of The Electrochemical Society,
    166(15):A3693–A3706, 2019
    """

    def __init__(
        self,
        param,
        halfcell=False,
        pybamm_control=False,
        options={},
        build=True
    ):
        """!@brief Sets the model properties.

        @par param
            A class containing all the relevant parameters for this
            model. For example, models.standard_parameters represents a
            valid choice for this parameter.
        @par halfcell
            Per default False, which indicates a full-cell setup. If set
            to True, the equations for a half-cell will be used instead.
        @par pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @par options
            Not used; only here for compatibility with the base class.
        @par build
            Not used; only here for compatibility with the base class.
        """

        super().__init__(param)
        """! Equations build a full-cell if False and a half-cell if True. """
        self.halfcell = halfcell
        """! Current is fixed if False and a variable if True. """
        self.pybamm_control = pybamm_control

    def get_fundamental_variables(self):
        """!@brief Builds all relevant model variables' symbols.

        @return
            A dictionary with the variables' names as keys and their
            symbols (of type pybamm.Symbol) as values.
        """

        if not self.halfcell:
            xₙ = pybamm.standard_spatial_vars.x_n
            rₙ = pybamm.standard_spatial_vars.r_n
        xₚ = pybamm.standard_spatial_vars.x_p
        rₚ = pybamm.standard_spatial_vars.r_p

        # Define the variables.
        t = pybamm.Variable("Time [h]")
        if not self.pybamm_control:
            Q = pybamm.Variable("Discharge capacity [A.h]")

        # Variables that vary spatially are created with a domain
        if not self.halfcell:
            cₑₙ = pybamm.Variable(
                "Negative electrolyte concentration",
                domain="negative electrode"
            )
        cₑₛ = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        cₑₚ = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        # Concatenations combine several variables into a single
        # variable, to simplify implementing equations that hold over
        # several domains
        if self.halfcell:
            cₑ = pybamm.concatenation(cₑₛ, cₑₚ)
        else:
            cₑ = pybamm.concatenation(cₑₙ, cₑₛ, cₑₚ)

        # Electrolyte potential
        if not self.halfcell:
            φₑₙ = pybamm.Variable(
                "Negative electrolyte potential", domain="negative electrode"
            )
        φₑₛ = pybamm.Variable(
            "Separator electrolyte potential", domain="separator"
        )
        if self.halfcell:
            φₑₙ = pybamm.boundary_value(φₑₛ, "left")
        φₑₚ = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        if self.halfcell:
            φₑ = pybamm.concatenation(φₑₛ, φₑₚ)
        else:
            φₑ = pybamm.concatenation(φₑₙ, φₑₛ, φₑₚ)

        # Electrode potential
        if self.halfcell:
            # Note: the "porosity" of the electrode is 1.
            # φₛₙ = -I * (Lₙ / σₙ) * xₙ / Lₙ
            # φₛₙ = -I * (Lₙ / σₙ)
            φₛₙ = pybamm.Variable("Negative electrode potential")
        else:
            φₛₙ = pybamm.Variable(
                "Negative electrode potential", domain="negative electrode"
            )
        φₛₚ = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode"
        )
        # Particle concentrations are variables on the particle domain,
        # but also vary in the x-direction (electrode domain) and so
        # must be provided with auxiliary domains
        if not self.halfcell:
            SOCₙ = pybamm.Variable(
                "Negative particle concentration",
                domain="negative particle",
                auxiliary_domains={"secondary": "negative electrode"},
            )
            # total_SOCₙ = pybamm.Variable("Negative electrode SOC")
        else:
            SOCₙ = pybamm.Scalar(0)
            # total_SOCₙ = pybamm.Scalar(0)
        SOCₚ = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        # total_SOCₚ = pybamm.Variable("Positive electrode SOC")
        if self.halfcell:
            SOCₙ_surf = pybamm.Scalar(0)
        else:
            SOCₙ_surf = pybamm.surf(SOCₙ)
        SOCₚ_surf = pybamm.surf(SOCₚ)

        # Constant temperature
        T = T_init
        # Porosity
        # if self.halfcell:
        #     ε = pybamm.concatenation(εₛ, εₚ)
        # else:
        #     ε = pybamm.concatenation(εₙ, εₛ, εₚ)
        # Tortuosity
        if self.halfcell:
            εᵝ = pybamm.concatenation(εₛ**βₑₛ_scalar, εₚ**βₑₚ_scalar)
        else:
            εᵝ = pybamm.concatenation(εₙ**βₑₙ_scalar, εₛ**βₑₛ_scalar,
                                      εₚ**βₑₚ_scalar)

        # Interfacial reactions

        if self.halfcell:
            ηₙ = pybamm.Variable("Negative electrode overpotential")
        else:
            ηₙ = φₛₙ - φₑₙ - OCVₙ(SOCₙ_surf, T) - pybamm.log(cₑₙ) / zₙ
        ηₚ = φₛₚ - φₑₚ - OCVₚ(SOCₚ_surf, T) - pybamm.log(cₑₚ) / zₚ

        # Butler-Volmer reaction fluxes.
        # Note: not using the simplified version with sinh, which is
        # fixed to symmetric charge-transfer, is significantly slower.
        if self.halfcell:
            # Note: γₙ / Cᵣₙ does not depend on cₙ; it cancels out.
            iₛₑₙ = (
                (γₙ / Cᵣₙ)
                * iₛₑₙ_0(pybamm.boundary_value(cₑₛ, "left"), 0.5, T)
                * (pybamm.exp(αₙₙ * zₙ * ηₙ) - pybamm.exp(-αₚₙ * zₙ * ηₙ))
            )
        else:
            iₛₑₙ = (
                (γₙ / Cᵣₙ)
                * iₛₑₙ_0(cₑₙ, SOCₙ_surf, T)
                * (pybamm.exp(αₙₙ * zₙ * ηₙ) - pybamm.exp(-αₚₙ * zₙ * ηₙ))
            )
        iₛₑₛ = pybamm.PrimaryBroadcast(0, "separator")
        iₛₑₚ = (
            (γₚ / Cᵣₚ)
            * iₛₑₚ_0(cₑₚ, SOCₚ_surf, T)
            * (pybamm.exp(αₙₚ * zₚ * ηₚ) - pybamm.exp(-αₚₚ * zₚ * ηₚ))
        )
        if self.halfcell:
            iₛₑ = pybamm.concatenation(iₛₑₛ, iₛₑₚ)
        else:
            iₛₑ = pybamm.concatenation(iₛₑₙ, iₛₑₛ, iₛₑₚ)

        # Current in the solid
        if not self.halfcell:
            iₛₙ = -σₙ * (1 - εₙ)**βₛₙ_scalar * pybamm.grad(φₛₙ)
        else:
            iₛₙ = pybamm.Variable("Negative electrode current")
        iₛₚ = -σₚ * (1 - εₚ)**βₛₚ_scalar * pybamm.grad(φₛₚ)

        # Current in the electrolyte
        iₑ = εᵝ * κₑ_hat * κₑ(cₑ, T) * (
            2 * one_plus_dlnf_dlnc(cₑ)
            * (1 - t_plus(cₑ))
            * pybamm.grad(cₑ) / cₑ
            - pybamm.grad(φₑ)
        )

        # Cell voltage
        voltage = pybamm.boundary_value(φₛₚ, "right")
        if self.halfcell:
            voltage_dim = OCVₚ_ref + voltage * thermal_voltage
        else:
            voltage_dim = OCVₚ_ref - OCVₙ_ref + voltage * thermal_voltage
        # Calculations for overpotential
        if not self.halfcell:
            total_SOCₙ = pybamm.Integral(pybamm.Integral(SOCₙ / rₙ**2, xₙ),
                                         rₙ) / (4 * π * Lₙ)
        else:
            total_SOCₙ = pybamm.Scalar(0)
        total_SOCₚ = pybamm.Integral(pybamm.Integral(SOCₚ / rₚ**2, xₚ),
                                     rₚ) / (4 * π * Lₚ)
        if self.halfcell:
            overpotential_dim = voltage_dim - OCVₚ_dim(total_SOCₚ, T)
        else:
            overpotential_dim = (
                voltage_dim
                - OCVₚ_dim(total_SOCₚ, T)
                + OCVₙ_dim(total_SOCₙ, T)
            )
        overpotential = overpotential_dim / thermal_voltage
        # The `variables` dictionary contains all variables that might
        # be useful for visualising the solution of the model
        variables = {
            "Negative electrode SOC": total_SOCₙ,
            "Negative particle concentration": SOCₙ,
            "Negative particle concentration [mol.m-3]": SOCₙ * cₙ,
            "Negative particle surface concentration": SOCₙ_surf,
            "Negative particle surface concentration [mol.m-3]":
                SOCₙ_surf * cₙ,
            "Negative electrolyte concentration":
                pybamm.Scalar(0) if self.halfcell else cₑₙ,
            "Negative electrolyte concentration [mol.m-3]":
                pybamm.Scalar(0) if self.halfcell else cₑₙ * cₑ_typ,
            "Separator electrolyte concentration": cₑₛ,
            "Separator electrolyte concentration [mol.m-3]": cₑₛ * cₑ_typ,
            "Positive electrolyte concentration": cₑₚ,
            "Positive electrolyte concentration [mol.m-3]": cₑₚ * cₑ_typ,
            "Electrolyte concentration": cₑ,
            "Electrolyte concentration [mol.m-3]": cₑ * cₑ_typ,
            "Positive electrode SOC": total_SOCₚ,
            "Positive particle concentration": SOCₚ,
            "Positive particle concentration [mol.m-3]": SOCₚ * cₚ,
            "Positive particle surface concentration": SOCₚ_surf,
            "Positive particle surface concentration [mol.m-3]":
                SOCₚ_surf * cₚ,
            "Negative electrode interface current density": iₛₑₙ,
            "Positive electrode interface current density": iₛₑₚ,
            "Interface current density": iₛₑ,
            "Interface current density [A.m-2]":
                pybamm.concatenation(
                    C / (A * aₙ_dim * L_dim) * iₛₑₙ,
                    iₛₑₛ,
                    C / (A * aₚ_dim * L_dim) * iₛₑₚ
                )
                if not self.halfcell else
                pybamm.concatenation(
                    iₛₑₛ, C / (A * aₚ_dim * L_dim) * iₛₑₚ
                ),
            "Overpotential":
                pybamm.concatenation(
                    ηₙ, pybamm.PrimaryBroadcast(0, "separator"), ηₚ
                )
                if not self.halfcell else
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(0, "separator"), ηₚ
                ),
            "Overpotential [V]":
                pybamm.concatenation(
                    ηₙ, pybamm.PrimaryBroadcast(0, "separator"), ηₚ
                ) * thermal_voltage
                if not self.halfcell else
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(0, "separator"), ηₚ
                ) * thermal_voltage,
            "Negative electrode current": iₛₙ,
            "Positive electrode current": iₛₚ,
            "Negative electrode potential": φₛₙ,
            "Negative electrode potential [V]": φₛₙ * thermal_voltage,
            "Negative electrode overpotential": ηₙ,
            "Negative electrode overpotential [V]": ηₙ * thermal_voltage,
            "Electrolyte current": iₑ,
            # For computation, use the version without NaN values.
            "Negative electrolyte potential": φₑₙ,
            "Negative electrolyte potential [V]": φₑₙ * thermal_voltage,
            "Separator electrolyte potential": φₑₛ,
            "Separator electrolyte potential [V]": φₑₛ * thermal_voltage,
            "Positive electrolyte potential": φₑₚ,
            "Positive electrolyte potential [V]": φₑₚ * thermal_voltage,
            "Electrolyte potential": φₑ,
            "Electrolyte potential [V]": φₑ * thermal_voltage,
            "Positive electrode potential": φₛₚ,
            "Positive electrode potential [V]":
                OCVₚ_ref - OCVₙ_ref + φₛₚ * thermal_voltage,
            "Positive electrode overpotential": ηₚ,
            "Positive electrode overpotential [V]": ηₚ * thermal_voltage,
            # The following variable names are for PyBaMM compatibility.
            # In particular, they are required if pybamm_control==True.
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Total overpotential": overpotential,
            "Total overpotential [V]": overpotential_dim,
            "Time [h]": t,
            "C-rate": C,
            "Negative electrode capacity [A.h]":
                (1 - εₙ_scalar) * Lₙ_dim * cₙ * zₙ * F * A,
            "Positive electrode capacity [A.h]":
                (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A,
            "Negative electrode open circuit potential": OCVₙ(SOCₙ_surf, T),
            "Negative electrode open circuit potential [V]":
                OCVₙ_ref + thermal_voltage * OCVₙ(SOCₙ_surf, T),
            "Negative electrode open circuit potential over time [V]":
                OCVₙ_ref + thermal_voltage * OCVₙ(total_SOCₙ, T),
            "Positive electrode open circuit potential": OCVₚ(SOCₚ_surf, T),
            "Positive electrode open circuit potential [V]":
                OCVₚ_ref + thermal_voltage * OCVₚ(SOCₚ_surf, T),
            "Positive electrode open circuit potential over time [V]":
                OCVₚ_ref + thermal_voltage * OCVₚ(total_SOCₚ, T),
            "X-averaged negative electrode open circuit potential":
                OCVₙ(SOCₙ_surf, T),
            "X-averaged negative electrode open circuit potential [V]":
                OCVₙ_ref + thermal_voltage * OCVₙ(SOCₙ_surf, T),
            "X-averaged positive electrode open circuit potential":
                OCVₚ(SOCₚ_surf, T),
            "X-averaged positive electrode open circuit potential [V]":
                OCVₚ_ref + thermal_voltage * OCVₚ(SOCₚ_surf, T),
            "X-averaged negative electrode reaction overpotential":
                ηₙ if self.halfcell else pybamm.Integral(ηₙ, xₙ) / Lₙ,
            "X-averaged negative electrode reaction overpotential [V]":
                thermal_voltage
                * (ηₙ if self.halfcell else pybamm.Integral(ηₙ, xₙ) / Lₙ),
            "X-averaged positive electrode reaction overpotential":
                pybamm.Integral(ηₚ, xₚ) / Lₚ,
            "X-averaged positive electrode reaction overpotential [V]":
                thermal_voltage * pybamm.Integral(ηₚ, xₚ) / Lₚ,
            "X-averaged negative electrode ohmic losses":
                φₛₙ if self.halfcell else pybamm.Integral(φₛₙ, xₙ) / Lₙ,
            "X-averaged negative electrode ohmic losses [V]": thermal_voltage
                * (φₛₙ if self.halfcell else pybamm.Integral(φₛₙ, xₙ) / Lₙ),
            "X-averaged positive electrode ohmic losses":
                pybamm.Integral(φₛₚ, xₚ) / Lₚ,
            "X-averaged positive electrode ohmic losses [V]":
                thermal_voltage * (pybamm.Integral(φₛₚ, xₚ) / Lₚ),
            "X-averaged negative electrode sei film overpotential":
                pybamm.Scalar(0),
            "X-averaged negative electrode sei film overpotential [V]":
                pybamm.Scalar(0),
            "X-averaged positive electrode sei film overpotential":
                pybamm.Scalar(0),
            "X-averaged positive electrode sei film overpotential [V]":
                pybamm.Scalar(0),
            "Nominal cell capacity [A.h]": capacity,
            "Loss of active material in negative electrode [%]":
                pybamm.Scalar(0),
            "Loss of active material in positive electrode [%]":
                pybamm.Scalar(0),
            "Loss of lithium inventory [%]": pybamm.Scalar(0),
            "Loss of lithium inventory, including electrolyte [%]":
                pybamm.Scalar(0),
            "Total lithium lost [mol]": pybamm.Scalar(0),
            "Total lithium lost from particles [mol]": pybamm.Scalar(0),
            "Total lithium lost from electrolyte [mol]": pybamm.Scalar(0),
            "Loss of lithium to negative electrode SEI [mol]":
                pybamm.Scalar(0),
            "Loss of lithium to positive electrode SEI [mol]":
                pybamm.Scalar(0),
            "Loss of lithium to negative electrode lithium plating [mol]":
                pybamm.Scalar(0),
            "Loss of lithium to positive electrode lithium plating [mol]":
                pybamm.Scalar(0),
            "Loss of capacity to negative electrode SEI [A.h]":
                pybamm.Scalar(0),
            "Loss of capacity to positive electrode SEI [A.h]":
                pybamm.Scalar(0),
            "Loss of capacity to negative electrode lithium plating [A.h]":
                pybamm.Scalar(0),
            "Loss of capacity to positive electrode lithium plating [A.h]":
                pybamm.Scalar(0),
            "Total lithium lost to side reactions [mol]": pybamm.Scalar(0),
            "Total capacity lost to side reactions [A.h]": pybamm.Scalar(0),
            "Loss of lithium to SEI [mol]": pybamm.Scalar(0),
            "Loss of lithium to lithium plating [mol]": pybamm.Scalar(0),
            "Loss of capacity to SEI [A.h]": pybamm.Scalar(0),
            "Loss of capacity to lithium plating [A.h]": pybamm.Scalar(0),
            # These parameters are only a placeholder.
            "Total lithium [mol]": pybamm.Scalar(0),
            "Total lithium in electrolyte [mol]": pybamm.Scalar(0),
            "Total lithium in negative electrode [mol]": pybamm.Scalar(0),
            "Total lithium in positive electrode [mol]": pybamm.Scalar(0),
            "Total lithium in particles [mol]": pybamm.Scalar(0),
            "Local ECM resistance [Ohm]": pybamm.Scalar(0),
        }
        if not self.pybamm_control:
            variables["Discharge capacity [A.h]"] = Q
            variables["Current [A]"] = C * I_extern
        return variables

    def get_coupled_variables(self, variables):
        """!@brief Builds all model symbols that rely on other models.

        @par variables
            A dictionary containing at least all variable symbols that
            are required for the variable symbols built here.
        @return
            A dictionary with the new variables' names as keys and their
            symbols (of type pybamm.Symbol) as values.
        """

        return {}

    def set_rhs(self, variables):
        """!@brief Sets up the right-hand-side equations in self.rhs.
        """

        # if not self.halfcell:
        #     xₙ = pybamm.standard_spatial_vars.x_n
        # xₚ = pybamm.standard_spatial_vars.x_p

        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
            # total_SOCₙ = variables["Negative electrode SOC"]
        SOCₚ = variables["Positive particle concentration"]
        # total_SOCₚ = variables["Positive electrode SOC"]
        iₑ = variables["Electrolyte current"]
        # iₛₑₙ = variables["Negative electrode interface current density"]
        # iₛₑₚ = variables["Positive electrode interface current density"]
        iₛₑ = variables["Interface current density"]
        cₑ = variables["Electrolyte concentration"]
        # Constant temperature
        T = T_init
        # Porosity
        if self.halfcell:
            ε = pybamm.concatenation(εₛ, εₚ)
        else:
            ε = pybamm.concatenation(εₙ, εₛ, εₚ)
        # Tortuosity
        if self.halfcell:
            εᵝ = pybamm.concatenation(εₛ**βₑₛ_scalar, εₚ**βₑₚ_scalar)
        else:
            εᵝ = pybamm.concatenation(εₙ**βₑₙ_scalar, εₛ**βₑₛ_scalar,
                                      εₚ**βₑₚ_scalar)

        self.rhs[t] = τᵈ / 3600
        if not self.pybamm_control:
            self.rhs[Q] = C * I_cell * τᵈ / 3600
        if not self.halfcell:
            Nₛₙ = -Dₙ(SOCₙ, T) * pybamm.grad(SOCₙ)
            self.rhs[SOCₙ] = -(1 / Cₙ) * pybamm.div(Nₛₙ)
            # self.rhs[total_SOCₙ] = pybamm.Integral(
            #     -iₛₑₙ * 3 / (aₙ * γₙ), xₙ
            # ) / Lₙ
        Nₛₚ = -Dₚ(SOCₚ, T) * pybamm.grad(SOCₚ)
        self.rhs[SOCₚ] = -(1 / Cₚ) * pybamm.div(Nₛₚ)
        # self.rhs[total_SOCₚ] = pybamm.Integral(
        #     -iₛₑₚ * 3 / (aₚ * γₚ), xₚ
        # ) / Lₚ
        # Electrolyte concentration
        Nₑ = -εᵝ * Dₑ(cₑ, T) * pybamm.grad(cₑ) + (Cₑ / γₑ) * t_plus(cₑ) * iₑ
        self.rhs[cₑ] = (1 / ε) * (-pybamm.div(Nₑ) / Cₑ + iₛₑ / γₑ)

    def set_algebraic(self, variables):
        """!@brief Sets up the algebraic equations in self.algebraic.
        """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        ηₙ = variables["Negative electrode overpotential"]
        iₛₑₙ = variables["Negative electrode interface current density"]
        iₛₑₚ = variables["Positive electrode interface current density"]
        iₛₑ = variables["Interface current density"]
        iₛₙ = variables["Negative electrode current"]
        iₛₚ = variables["Positive electrode current"]
        φₛₙ = variables["Negative electrode potential"]
        φₛₚ = variables["Positive electrode potential"]
        iₑ = variables["Electrolyte current"]
        φₑ = variables["Electrolyte potential"]

        if self.halfcell:
            self.algebraic[ηₙ] = iₛₑₙ - I_cell
            self.algebraic[φₛₙ] = φₛₙ + I_cell * (Lₙ / σₙ)
            self.algebraic[iₛₙ] = iₛₙ - I_cell
        if not self.halfcell:
            self.algebraic[φₛₙ] = pybamm.div(iₛₙ) + iₛₑₙ
        self.algebraic[φₛₚ] = pybamm.div(iₛₚ) + iₛₑₚ
        self.algebraic[φₑ] = pybamm.div(iₑ) - iₛₑ

    def set_boundary_conditions(self, variables):
        """!@brief Sets the (self.)boundary(_)conditions.
        """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
        SOCₚ = variables["Positive particle concentration"]
        if not self.halfcell:
            SOCₙ_surf = variables[
                "Negative particle surface concentration"
            ]
        SOCₚ_surf = variables[
            "Positive particle surface concentration"
        ]
        ηₙ = variables["Negative electrode overpotential"]
        iₛₑₙ = variables["Negative electrode interface current density"]
        iₛₑₚ = variables["Positive electrode interface current density"]
        φₛₙ = variables["Negative electrode potential"]
        φₛₚ = variables["Positive electrode potential"]
        φₑ = variables["Electrolyte potential"]
        cₑ = variables["Electrolyte concentration"]
        # Constant temperature
        T = T_init

        if not self.halfcell:
            self.boundary_conditions[SOCₙ] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (
                    -Cₙ * iₛₑₙ / (aₙ * γₙ * Dₙ(SOCₙ_surf, T)), "Neumann"
                ),
            }
        self.boundary_conditions[SOCₚ] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-Cₚ * iₛₑₚ / (aₚ * γₚ * Dₚ(SOCₚ_surf, T)), "Neumann"),
        }
        if not self.halfcell:
            self.boundary_conditions[φₛₙ] = {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        self.boundary_conditions[φₛₚ] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (I_cell / pybamm.boundary_value(
                -σₚ * (1 - εₚ)**βₛₚ_scalar, "right"
            ), "Neumann"),
        }
        if self.halfcell:
            # At the anode surface, a Dirichlet boundary condition is
            # now required; a Neumann one won't compute (literally).
            # Reason: formerly, the offset of φₑ would be determined by
            # the anode overpotential, which itself is a variable now.
            cₑₛ_boundary = pybamm.boundary_value(cₑ, "left")
            self.boundary_conditions[φₑ] = {
                "left": (
                    -I_cell * (Lₙ / σₙ) - ηₙ - pybamm.log(cₑₛ_boundary) / zₙ,
                    "Dirichlet"
                ),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        else:
            self.boundary_conditions[φₑ] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        if self.halfcell:
            cₑₛ_boundary = pybamm.boundary_value(cₑ, "left")
            self.boundary_conditions[cₑ] = {
                "left": (
                    -Cₑ * (1 - t_plus(cₑₛ_boundary)) * I_cell
                    / (γₑ * εₛ_scalar**βₑₛ_scalar * Dₑ(cₑₛ_boundary, T)),
                    "Neumann"
                ),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        else:
            self.boundary_conditions[cₑ] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

    def set_initial_conditions(self, variables):
        """!@brief Sets the (self.)initial(_)conditions.
        """

        if not self.halfcell:
            xₙ = pybamm.standard_spatial_vars.x_n
        xₚ = pybamm.standard_spatial_vars.x_p

        # Initial conditions for rhs: actual start values.
        # Initial conditions for algebraic: help for root-finding.
        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
            # total_SOCₙ = variables["Negative electrode SOC"]
        SOCₚ = variables["Positive particle concentration"]
        # total_SOCₚ = variables["Positive electrode SOC"]
        ηₙ = variables["Negative electrode overpotential"]
        φₛₙ = variables["Negative electrode potential"]
        φₛₚ = variables["Positive electrode potential"]
        iₛₙ = variables["Negative electrode current"]
        φₑ = variables["Electrolyte potential"]
        cₑ = variables["Electrolyte concentration"]

        self.initial_conditions[t] = pybamm.Scalar(0)
        if not self.pybamm_control:
            self.initial_conditions[Q] = pybamm.Scalar(0)
        if self.halfcell:
            self.initial_conditions[ηₙ] = (2 / zₙ) * pybamm.arcsinh(
                I_cell / (
                    2 * (γₙ / Cᵣₙ) * iₛₑₙ_0(cₑ_init, 0.5, T_init)
                )
            )
            self.initial_conditions[φₛₙ] = -I_cell * (Lₙ / σₙ)
            self.initial_conditions[iₛₙ] = I_cell
        # c_n_init and c_p_init can in general be functions of x
        # Note the broadcasting, for domains
        if not self.halfcell:
            self.initial_conditions[SOCₙ] = SOCₙ_init(
                pybamm.PrimaryBroadcast(xₙ, "negative particle")
            )
            # self.initial_conditions[total_SOCₙ] = (
            #     pybamm.Integral(SOCₙ_init(xₙ), xₙ) / Lₙ
            # )
        self.initial_conditions[SOCₚ] = SOCₚ_init(
            pybamm.PrimaryBroadcast(xₚ, "positive particle")
        )
        # self.initial_conditions[total_SOCₚ] = (
        #     pybamm.Integral(SOCₚ_init(xₚ), xₚ) / Lₚ
        # )
        # We evaluate c_n_init at x=0 and c_p_init at x=1 (this is just
        # an initial guess so actual value is not too important)
        if not self.halfcell:
            self.initial_conditions[φₛₙ] = pybamm.Scalar(0)
        self.initial_conditions[φₛₚ] = (
            OCVₚ(SOCₚ_init(1), T_init) - OCVₙ(SOCₙ_init(0), T_init)
        )
        if self.halfcell:
            self.initial_conditions[φₑ] = -(2 / zₙ) * pybamm.arcsinh(
                I_cell / (2 * (γₙ / Cᵣₙ) * iₛₑₙ_0(cₑ_init, 0.5, T_init))
            )
        else:
            self.initial_conditions[φₑ] = (
                -OCVₙ(SOCₙ_init(0), T_init)
                - (2 / zₙ) * pybamm.arcsinh(
                    I_cell / (
                        2 * (γₙ / Cᵣₙ)
                        * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), T_init)
                        * Lₙ
                    )
                )
            )
        self.initial_conditions[cₑ] = cₑ_init

    def set_events(self, variables):
        """!@brief Sets up the termination switches in self.events.
        """

        if self.halfcell:
            SOCₙ_surf = pybamm.Scalar(0)
        else:
            SOCₙ_surf = variables[
                "Negative particle surface concentration"
            ]
        SOCₚ_surf = variables[
            "Positive particle surface concentration"
        ]
        cₑ = variables["Electrolyte concentration"]
        voltage_dim = variables["Terminal voltage [V]"]
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface concentration",
                pybamm.min(SOCₙ_surf) - 0.001,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration",
                (1 - 0.001) - pybamm.max(SOCₙ_surf),
            ),
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(SOCₚ_surf) - 0.001,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.001) - pybamm.max(SOCₚ_surf),
            ),
            pybamm.Event(
                "Zero electrolyte concentration cut-off",
                pybamm.min(cₑ) - 0.002
            ),
            pybamm.Event("Minimum voltage", voltage_dim - Uₗ),
            pybamm.Event("Maximum voltage", voltage_dim - Uᵤ),
        ]


class DFN(pybamm.BaseBatteryModel):
    """!@brief Doyle-Fuller-Newman (DFN) model.
    """

    def __init__(
        self,
        halfcell=False,
        pybamm_control=False,
        name="DFN",
        options={},
        build=True
    ):
        """!@brief Sets up a DFN model usable by PyBaMM.

        @par halfcell
            Per default False, which indicates a full-cell setup. If set
            to True, the equations for a half-cell will be used instead.
        @par pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @par name
            The optional name of the model. Default is "DFN".
        @par options
            Used for external circuit if pybamm_control is True.
        @par build
            Per default True, which builds the model equations right
            away. If set to False, they remain as symbols for later.
        """

        super().__init__(name=name)
        """! Equations build a full-cell if False and a half-cell if True. """
        self.halfcell = halfcell
        """! Current is fixed if False and a variable if True. """
        self.pybamm_control = pybamm_control
        pybamm.citations.register("Marquis2019")
        self.options = options

        """! Contains all the relevant parameters for this model. """
        self.param = standard_parameters
        """! Non-dimensionalization timescale. """
        self.timescale = τᵈ
        """! Non-dimensionalization length scales. """
        self.length_scales = {
            "negative electrode": L_dim,
            "separator": L_dim,
            "positive electrode": L_dim,
            "negative particle": Rₙ,
            "positive particle": Rₚ,
            # "current collector y": self.param.L_y,
            # "current collector z": self.param.L_z,
        }
        self.set_standard_output_variables()

        if self.pybamm_control:
            self.set_external_circuit_submodel()
        self.submodels["internal"] = DFN_internal(self.param, self.halfcell,
                                                  self.pybamm_control)

        if build:
            self.build_model()

    def set_standard_output_variables(self):
        """!@brief Adds "the horizontal axis" to self.variables.

        Don't use the super() version of this function, as it
        introduces keys with integer values in self.variables.
        """

        # super().set_standard_output_variables()
        var = pybamm.standard_spatial_vars
        param = self.param
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * self.timescale,
                "Time [min]": pybamm.t * self.timescale / 60,
                "Time [h]": pybamm.t * self.timescale / 3600,
                # "y": var.y,
                # "y [m]": var.y * param.L_y,
                # "z": var.z,
                # "z [m]": var.z * param.L_z,
                "x_s": var.x_s,
                "x_s [m]": var.x_s * param.L_dim,
                "x_p": var.x_p,
                "x_p [m]": var.x_p * param.L_dim,
                "r_p": var.r_p,
                "r_p [m]": var.r_p * param.Rₚ,
            }
        )
        if not self.halfcell:
            self.variables.update(
                {
                    "x": var.x,
                    "x [m]": var.x * param.L_dim,
                    "x_n": var.x_n,
                    "x_n [m]": var.x_n * param.L_dim,
                    "r_n": var.r_n,
                    "r_n [m]": var.r_n * param.Rₙ,
                }
            )
        else:
            x = pybamm.concatenation(var.x_s, var.x_p)
            self.variables.update(
                {
                    "x": x,
                    "x [m]": x * param.L_dim,
                }
            )

    def new_copy(self, build=True):
        """!@brief Create an empty copy with identical options.

        @par build
            If True, the new model gets built right away. This is the
            default behavior. If set to False, it remains as symbols.
        @return
            The copy of this model.
        """

        new_model = self.__class__(self.halfcell, self.pybamm_control,
                                   self.name, build=False)
        # update submodels
        new_model.submodels = self.submodels
        # clear submodel equations to avoid weird conflicts
        for submodel in self.submodels.values():
            submodel._rhs = {}
            submodel._algebraic = {}
            submodel._initial_conditions = {}
            submodel._boundary_conditions = {}
            submodel._variables = {}
            submodel._events = []

        # now build
        if build:
            new_model.build_model()
        new_model.use_jacobian = self.use_jacobian
        # new_model.use_simplify = self.use_simplify
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model
        """
        replacer = pybamm.SymbolReplacer({})
        return replacer.process_model(self, inplace=False)
        """

    def set_voltage_variables(self):
        """!@brief Adds voltage-specific variables to self.variables.

        Override this inherited function, since it adds superfluous
        variables.
        """

        pass

    @property
    def default_geometry(self):
        """!@brief Override: corrects the geometry for half-cells.
        """

        geometry = super().default_geometry
        if self.halfcell:
            del geometry["negative electrode"]
            del geometry["negative particle"]
        return geometry
