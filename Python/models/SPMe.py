# Copyright (c): German Aerospace Center (DLR)
"""!@package models.SPMe
Contains a PyBaMM-compatible Single-Particle Model with electrolyte
(SPMe).
"""

import pybamm
import models.standard_parameters as standard_parameters
from models.standard_parameters import (
    OCVₙ_ref, OCVₚ_ref, OCVₙ, OCVₚ, OCVₙ_dim, OCVₚ_dim,
    Dₙ, Dₚ,
    cₙ, cₚ, SOCₙ_init, SOCₚ_init,
    iₛₑₙ_0, d_cₑₙ_iₛₑₙ_0, iₛₑₚ_0, d_cₑₚ_iₛₑₚ_0,
    aₙ, aₙ_dim, aₚ, aₚ_dim,
    zₙ, zₚ,
    αₙₙ, αₚₙ, αₙₚ, αₚₚ,
    σₙ, σₚ,
    Rₙ, Rₚ,
    cₑ_typ, cₑ_init,
    εₙ_scalar, εₛ_scalar, εₚ_scalar,
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


class SPMe_internal(pybamm.BaseSubModel):
    """!@brief Defining equations for the SPMe.

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
        # xₛ = pybamm.standard_spatial_vars.x_s
        xₚ = pybamm.standard_spatial_vars.x_p
        rₚ = pybamm.standard_spatial_vars.r_p

        # Define the variables.
        t = pybamm.Variable("Time [h]")
        ηₙ_0 = pybamm.Variable("Negative electrode overpotential (0th order)")
        ηₚ_0 = pybamm.Variable("Positive electrode overpotential (0th order)")

        # Variables that vary spatially are created with a domain.
        # Note: the secondary current collector domain is only needed
        # for compatibility with pybamm.standard_spatial_vars.x_k
        # (since the terms for ϕₑₖ_1 include both cₑₖ_1 and xₖ).
        if not self.halfcell:
            cₑₙ_1 = pybamm.Variable(
                "Negative electrolyte concentration correction",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"}
            )
        cₑₛ_1 = pybamm.Variable(
            "Separator electrolyte concentration correction",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"}
        )
        if self.halfcell:
            cₑₙ_1 = pybamm.boundary_value(cₑₛ_1, "left")
        cₑₚ_1 = pybamm.Variable(
            "Positive electrolyte concentration correction",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"}
        )
        # Integrate to obtain the mean values.
        if self.halfcell:
            # Replace the mean value with the surface value.
            bar_cₑₙ_1 = pybamm.boundary_value(cₑₛ_1, "left")
        else:
            bar_cₑₙ_1 = (1 / Lₙ) * pybamm.Integral(cₑₙ_1, xₙ)
        bar_cₑₚ_1 = (1 / Lₚ) * pybamm.Integral(cₑₚ_1, xₚ)
        # Concatenations combine several variables into a single
        # variable, to simplify implementing equations that hold over
        # several domains
        if self.halfcell:
            cₑ_1 = pybamm.concatenation(cₑₛ_1, cₑₚ_1)
        else:
            cₑ_1 = pybamm.concatenation(cₑₙ_1, cₑₛ_1, cₑₚ_1)

        if not self.halfcell:
            SOCₙ = pybamm.Variable(
                "Negative particle concentration",
                domain="negative particle",
            )
            # total_SOCₙ = pybamm.Variable("Negative electrode SOC")
        else:
            SOCₙ = pybamm.Scalar(0)
            # total_SOCₙ = pybamm.Scalar(0)
        SOCₚ = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
        )
        # total_SOCₚ = pybamm.Variable("Positive electrode SOC")
        if self.halfcell:
            SOCₙ_surf = pybamm.Scalar(0)
        else:
            SOCₙ_surf = pybamm.surf(SOCₙ)
        SOCₚ_surf = pybamm.surf(SOCₚ)

        # Constant temperature
        T = T_init

        # Interfacial reactions

        if self.halfcell:
            # Note: γₙ / Cᵣₙ does not depend on cₙ; it cancels out.
            iₛₑₙ = (
                (γₙ / Cᵣₙ)
                * iₛₑₙ_0(cₑ_init, 0.5, T)
                * (pybamm.exp(αₙₙ * zₙ * ηₙ_0) - pybamm.exp(-αₚₙ * zₙ * ηₙ_0))
            )
        else:
            iₛₑₙ = (
                (γₙ / Cᵣₙ)
                * iₛₑₙ_0(cₑ_init, SOCₙ_surf, T)
                * (pybamm.exp(αₙₙ * zₙ * ηₙ_0) - pybamm.exp(-αₚₙ * zₙ * ηₙ_0))
            )
        iₛₑₛ = pybamm.PrimaryBroadcast(0, "separator")
        iₛₑₚ = (
            (γₚ / Cᵣₚ)
            * iₛₑₚ_0(cₑ_init, SOCₚ_surf, T)
            * (pybamm.exp(αₙₚ * zₚ * ηₚ_0) - pybamm.exp(-αₚₚ * zₚ * ηₚ_0))
        )
        if self.halfcell:
            iₛₑ = pybamm.concatenation(
                iₛₑₛ,
                pybamm.PrimaryBroadcast(iₛₑₚ, "positive electrode")
            )
        else:
            iₛₑ = pybamm.concatenation(
                pybamm.PrimaryBroadcast(iₛₑₙ, "negative electrode"),
                iₛₑₛ,
                pybamm.PrimaryBroadcast(iₛₑₚ, "positive electrode")
            )

        # The first order correction is a function of the zeroth one.
        if self.halfcell:
            bar_ηₙ_1 = (
                (-bar_cₑₙ_1 / zₙ)
                * (d_cₑₙ_iₛₑₙ_0(cₑ_init, 0.5, T) / iₛₑₙ_0(cₑ_init, 0.5, T))
                * (pybamm.exp(αₙₙ * zₙ * ηₙ_0) - pybamm.exp(-αₚₙ * zₙ * ηₙ_0))
                / (
                    αₙₙ * pybamm.exp(αₙₙ * zₙ * ηₙ_0)
                    + αₚₙ * pybamm.exp(-αₚₙ * zₙ * ηₙ_0)
                )
            )
        else:
            bar_ηₙ_1 = (
                (-bar_cₑₙ_1 / zₙ)
                * (
                    d_cₑₙ_iₛₑₙ_0(cₑ_init, SOCₙ_surf, T)
                    / iₛₑₙ_0(cₑ_init, SOCₙ_surf, T)
                )
                * (pybamm.exp(αₙₙ * zₙ * ηₙ_0) - pybamm.exp(-αₚₙ * zₙ * ηₙ_0))
                / (
                    αₙₙ * pybamm.exp(αₙₙ * zₙ * ηₙ_0)
                    + αₚₙ * pybamm.exp(-αₚₙ * zₙ * ηₙ_0)
                )
            )
        bar_ηₚ_1 = (
            (-bar_cₑₚ_1 / zₚ)
            * (
                d_cₑₚ_iₛₑₚ_0(cₑ_init, SOCₚ_surf, T)
                / iₛₑₚ_0(cₑ_init, SOCₚ_surf, T)
            )
            * (pybamm.exp(αₙₚ * zₚ * ηₚ_0) - pybamm.exp(-αₚₚ * zₚ * ηₚ_0))
            / (
                αₙₚ * pybamm.exp(αₙₚ * zₚ * ηₚ_0)
                + αₚₚ * pybamm.exp(-αₚₚ * zₚ * ηₚ_0)
            )
        )

        cₑ = cₑ_init + Cₑ * cₑ_1

        # Calculations for overpotential
        if not self.halfcell:
            total_SOCₙ = pybamm.Integral(SOCₙ / pybamm.Integral(
                rₙ / Lₙ, xₙ)**2, rₙ
            ) / (4 * π)
        else:
            total_SOCₙ = pybamm.Scalar(0)
        total_SOCₚ = pybamm.Integral(SOCₚ / pybamm.Integral(
            rₚ / Lₚ, xₚ)**2, rₚ
        ) / (4 * π)

        variables = {
            "Negative electrode SOC": total_SOCₙ,
            "Negative particle concentration": SOCₙ,
            "Negative particle concentration [mol.m-3]": SOCₙ * cₙ,
            "Negative particle surface concentration over time": SOCₙ_surf,
            "Negative particle surface concentration over time [mol.m-3]":
                SOCₙ_surf * cₙ,
            "Negative particle surface concentration": SOCₙ_surf
                if self.halfcell else
                pybamm.PrimaryBroadcast(SOCₙ_surf, "negative electrode"),
            "Negative particle surface concentration [mol.m-3]":
                SOCₙ_surf * cₙ if self.halfcell else
                pybamm.PrimaryBroadcast(SOCₙ_surf * cₙ, "negative electrode"),
            "Electrolyte concentration correction": cₑ_1,
            "Negative electrolyte concentration correction": cₑₙ_1,
            "Negative electrolyte concentration correction average": bar_cₑₙ_1,
            "Negative electrolyte concentration":
                pybamm.Scalar(0) if self.halfcell else cₑ_init + Cₑ * cₑₙ_1,
            "Negative electrolyte concentration [mol.m-3]":
                pybamm.Scalar(0) if self.halfcell else
                (cₑ_init + Cₑ * cₑₙ_1) * cₑ_typ,
            "Separator electrolyte concentration correction": cₑₛ_1,
            "Separator electrolyte concentration": cₑ_init + Cₑ * cₑₛ_1,
            "Separator electrolyte concentration [mol.m-3]":
                (cₑ_init + Cₑ * cₑₛ_1) * cₑ_typ,
            "Positive electrolyte concentration correction": cₑₚ_1,
            "Positive electrolyte concentration correction average": bar_cₑₚ_1,
            "Positive electrolyte concentration": cₑ_init + Cₑ * cₑₚ_1,
            "Positive electrolyte concentration [mol.m-3]":
                (cₑ_init + Cₑ * cₑₚ_1) * cₑ_typ,
            "Electrolyte concentration": cₑ,
            "Electrolyte concentration [mol.m-3]": cₑ * cₑ_typ,
            "Positive electrode SOC": total_SOCₚ,
            "Positive particle concentration": SOCₚ,
            "Positive particle concentration [mol.m-3]": SOCₚ * cₚ,
            "Positive particle surface concentration over time": SOCₚ_surf,
            "Positive particle surface concentration over time [mol.m-3]":
                SOCₚ_surf * cₚ,
            "Positive particle surface concentration":
                pybamm.PrimaryBroadcast(
                    SOCₚ_surf, "positive electrode"
                ),
            "Positive particle surface concentration [mol.m-3]":
                pybamm.PrimaryBroadcast(SOCₚ_surf * cₚ, "positive electrode"),
            "Negative electrode interface current density": iₛₑₙ,
            "Positive electrode interface current density": iₛₑₚ,
            "Interface current density": iₛₑ,
            "Interface current density [A.m-2]":
                pybamm.concatenation(
                    C / (A * aₙ_dim * L_dim) * pybamm.PrimaryBroadcast(
                        iₛₑₙ, "negative electrode"
                    ),
                    iₛₑₛ,
                    C / (A * aₚ_dim * L_dim) * pybamm.PrimaryBroadcast(
                        iₛₑₚ, "positive electrode"
                    )
                )
                if not self.halfcell else
                pybamm.concatenation(
                    iₛₑₛ, C / (A * aₚ_dim * L_dim) * pybamm.PrimaryBroadcast(
                        iₛₑₚ, "positive electrode"
                    )
                ),
            "Negative electrode overpotential (0th order)": ηₙ_0,
            "Negative electrode overpotential correction": bar_ηₙ_1,
            "Negative electrode overpotential": ηₙ_0 + Cₑ * bar_ηₙ_1,
            "Negative electrode overpotential [V]": (ηₙ_0 + Cₑ * bar_ηₙ_1)
                * thermal_voltage,
            "Positive electrode overpotential (0th order)": ηₚ_0,
            "Positive electrode overpotential correction": bar_ηₚ_1,
            "Positive electrode overpotential": ηₚ_0 + Cₑ * bar_ηₚ_1,
            "Positive electrode overpotential [V]": (ηₚ_0 + Cₑ * bar_ηₚ_1)
                * thermal_voltage,
            "Overpotential":
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(0, "separator"),
                    pybamm.PrimaryBroadcast(ηₚ_0 + Cₑ * bar_ηₚ_1,
                                            "positive electrode")
                )
                if self.halfcell else
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(ηₙ_0 + Cₑ * bar_ηₙ_1,
                                            "negative electrode"),
                    pybamm.PrimaryBroadcast(0, "separator"),
                    pybamm.PrimaryBroadcast(ηₚ_0 + Cₑ * bar_ηₚ_1,
                                            "positive electrode")
                ),
            "Overpotential [V]":
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(0, "separator"),
                    pybamm.PrimaryBroadcast(ηₚ_0 + Cₑ * bar_ηₚ_1,
                                            "positive electrode")
                ) * thermal_voltage
                if self.halfcell else
                pybamm.concatenation(
                    pybamm.PrimaryBroadcast(ηₙ_0 + Cₑ * bar_ηₙ_1,
                                            "negative electrode"),
                    pybamm.PrimaryBroadcast(0, "separator"),
                    pybamm.PrimaryBroadcast(ηₚ_0 + Cₑ * bar_ηₚ_1,
                                            "positive electrode")
                ) * thermal_voltage,
            "Time [h]": t,
            "Negative electrode capacity [A.h]":
                (1 - εₙ_scalar) * Lₙ_dim * cₙ * zₙ * F * A,
            "Positive electrode capacity [A.h]":
                (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A,
            "Negative electrode open circuit potential": OCVₙ(SOCₙ_surf, T),
            "Negative electrode open circuit potential [V]":
                OCVₙ_ref + thermal_voltage * OCVₙ(SOCₙ_surf, T),
            "Positive electrode open circuit potential": OCVₚ(SOCₚ_surf, T),
            "Positive electrode open circuit potential [V]":
                OCVₚ_ref + thermal_voltage * OCVₚ(SOCₚ_surf, T),
            "X-averaged negative electrode open circuit potential":
                OCVₙ(SOCₙ_surf, T),
            "X-averaged negative electrode open circuit potential [V]":
                OCVₙ_ref + thermal_voltage * OCVₙ(SOCₙ_surf, T),
            "X-averaged positive electrode open circuit potential":
                OCVₚ(SOCₚ_surf, T),
            "X-averaged positive electrode open circuit potential [V]":
                OCVₚ_ref + thermal_voltage * OCVₚ(SOCₚ_surf, T),
            "X-averaged negative electrode reaction overpotential":
                ηₙ_0 + Cₑ * bar_ηₙ_1,
            "X-averaged negative electrode reaction overpotential [V]":
                thermal_voltage * (ηₙ_0 + Cₑ * bar_ηₙ_1),
            "X-averaged positive electrode reaction overpotential":
                ηₚ_0 + Cₑ * bar_ηₚ_1,
            "X-averaged positive electrode reaction overpotential [V]":
                thermal_voltage * (ηₚ_0 + Cₑ * bar_ηₚ_1),
            "X-averaged negative electrode sei film overpotential":
                pybamm.Scalar(0),
            "X-averaged negative electrode sei film overpotential [V]":
                pybamm.Scalar(0),
            "X-averaged positive electrode sei film overpotential":
                pybamm.Scalar(0),
            "X-averaged positive electrode sei film overpotential [V]":
                pybamm.Scalar(0),
            "Cell capacity [A.h]": capacity,
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

        if not self.halfcell:
            xₙ = pybamm.standard_spatial_vars.x_n
            # rₙ = pybamm.standard_spatial_vars.r_n
        xₛ = pybamm.standard_spatial_vars.x_s
        xₚ = pybamm.standard_spatial_vars.x_p
        # rₚ = pybamm.standard_spatial_vars.r_p

        # Constant temperature
        T = T_init

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = pybamm.Variable("Discharge capacity [A.h]")

        cₑₙ_1 = variables["Negative electrolyte concentration correction"]
        cₑₛ_1 = variables["Separator electrolyte concentration correction"]
        cₑₚ_1 = variables["Positive electrolyte concentration correction"]
        bar_cₑₙ_1 = (
            variables["Negative electrolyte concentration correction average"]
        )
        bar_cₑₚ_1 = (
            variables["Positive electrolyte concentration correction average"]
        )
        ηₙ_0 = variables["Negative electrode overpotential (0th order)"]
        ηₚ_0 = variables["Positive electrode overpotential (0th order)"]
        bar_ηₙ_1 = variables["Negative electrode overpotential correction"]
        bar_ηₚ_1 = variables["Positive electrode overpotential correction"]
        total_SOCₙ = variables["Negative electrode SOC"]
        total_SOCₚ = variables["Positive electrode SOC"]
        SOCₙ_surf = variables[
            "Negative particle surface concentration over time"
        ]
        SOCₚ_surf = variables[
            "Positive particle surface concentration over time"
        ]

        # Electrode potentials
        if self.halfcell:
            # Note: the "porosity" of the electrode is 1.
            # φₛₙ_1 = -I_cell * Lₙ / (σₙ * Cₑ) * xₙ / Lₙ
            φₛₙ_1 = -I_cell * Lₙ / (σₙ * Cₑ)
        else:
            φₛₙ_1 = (
                -I_cell / (2 * Lₙ * σₙ * (1 - εₙ_scalar)**βₛₙ_scalar * Cₑ)
                * (2 * Lₙ * xₙ - xₙ**2)
            )
        φₛₚ_1 = (
            I_cell / (2 * Lₚ * σₚ * (1 - εₚ_scalar)**βₛₚ_scalar * Cₑ)
            * (2 * Lₚ * (1 - xₚ) - (1 - xₚ)**2)
        )
        if self.halfcell:
            bar_φₛₙ_1 = -I_cell * Lₙ / (σₙ * Cₑ)
        else:
            bar_φₛₙ_1 = (
                -I_cell * Lₙ / (3 * σₙ * (1 - εₙ_scalar)**βₛₙ_scalar * Cₑ)
            )
        bar_φₛₚ_1 = I_cell * Lₚ / (3 * σₚ * (1 - εₚ_scalar)**βₛₚ_scalar * Cₑ)

        # Electrolyte potential
        if self.halfcell:
            tilde_φₑ = 0.5 * (
                -bar_ηₙ_1 - bar_cₑₙ_1 / zₙ - 2 * (1 - t_plus(cₑ_init))
                * one_plus_dlnf_dlnc(cₑ_init) * bar_cₑₙ_1
            )
        else:
            tilde_φₑ = 0.5 * (
                -I_cell * Lₙ / (3 * σₙ * (1 - εₙ_scalar)**βₛₙ_scalar * Cₑ)
                - bar_ηₙ_1 - bar_cₑₙ_1 / zₙ - 2 * (1 - t_plus(cₑ_init))
                * one_plus_dlnf_dlnc(cₑ_init) * bar_cₑₙ_1
                - I_cell / (κₑ_hat * Cₑ * κₑ(cₑ_init, T))
                * (-Lₙ / (3 * εₙ_scalar**βₑₙ_scalar)
                   + Lₙ / εₛ_scalar**βₑₛ_scalar)
            )
        if self.halfcell:
            φₑ_0 = -ηₙ_0
        else:
            φₑ_0 = -ηₙ_0 - OCVₙ(SOCₙ_surf, T)
        # iₑₙ_0 = xₙ * I / Lₙ
        # iₑₛ_0 = pybamm.PrimaryBroadcast(I, "separator")
        # iₑₚ_0 = (1 - xₚ) * I / Lₚ
        # iₑ_0 = pybamm.concatenation(iₑₙ_0, iₑₛ_0, iₑₚ_0)
        if not self.halfcell:
            φₑₙ_1 = (
                tilde_φₑ
                + 2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init)
                * cₑₙ_1
                - I_cell / (κₑ_hat * Cₑ * κₑ(cₑ_init, T))
                * (
                    (xₙ**2 - Lₙ**2) / (2 * εₙ_scalar**βₑₙ_scalar * Lₙ)
                    + Lₙ / εₛ_scalar**βₑₛ_scalar
                )
            )
        φₑₛ_1 = (
            tilde_φₑ
            + 2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init) * cₑₛ_1
            - I_cell / (κₑ_hat * Cₑ * κₑ(cₑ_init, T))
            * xₛ / εₛ_scalar**βₑₛ_scalar
        )
        φₑₚ_1 = (
            tilde_φₑ
            + 2 * (1 - t_plus(cₑ_init)) * one_plus_dlnf_dlnc(cₑ_init) * cₑₚ_1
            - I_cell / (κₑ_hat * Cₑ * κₑ(cₑ_init, T))
            * (
                (xₚ * (2 - xₚ) + Lₚ**2 - 1) / (2 * εₚ_scalar**βₑₚ_scalar * Lₚ)
                + (1 - Lₚ) / εₛ_scalar**βₑₛ_scalar
            )
        )
        if self.halfcell:
            φₑₙ_1 = pybamm.boundary_value(φₑₛ_1, "left")
            # Replace the mean value with the surface value.
            bar_φₑₙ_1 = pybamm.boundary_value(φₑₛ_1, "left")
        else:
            bar_φₑₙ_1 = (1 / Lₙ) * pybamm.Integral(φₑₙ_1, xₙ)
        bar_φₑₚ_1 = (1 / Lₚ) * pybamm.Integral(φₑₚ_1, xₚ)
        if self.halfcell:
            φₑ_1 = pybamm.concatenation(φₑₛ_1, φₑₚ_1)
        else:
            φₑ_1 = pybamm.concatenation(φₑₙ_1, φₑₛ_1, φₑₚ_1)

        # Cell voltage
        if self.halfcell:
            voltage_0 = OCVₚ(SOCₚ_surf, T) + ηₚ_0 - ηₙ_0
        else:
            voltage_0 = OCVₚ(SOCₚ_surf, T) - OCVₙ(SOCₙ_surf, T) + ηₚ_0 - ηₙ_0
        overpotential_0 = ηₚ_0 - ηₙ_0
        voltage_1 = (
            -bar_φₛₚ_1 + bar_φₛₙ_1
            + bar_ηₚ_1 - bar_ηₙ_1
            + bar_φₑₚ_1 - bar_φₑₙ_1
            + bar_cₑₚ_1 / zₚ - bar_cₑₙ_1 / zₙ
        )
        overpotential_1 = voltage_1
        voltage = voltage_0 + Cₑ * voltage_1
        overpotential = overpotential_0 + Cₑ * overpotential_1
        if not self.halfcell:
            global OCVₙ_ref
        else:
            OCVₙ_ref = pybamm.Scalar(0)
        voltage_dim = OCVₚ_ref - OCVₙ_ref + thermal_voltage * voltage
        # Calculations for overpotential
        # if not self.halfcell:
        #     total_SOCₙ = pybamm.Integral(SOCₙ / pybamm.Integral(
        #         rₙ / Lₙ, xₙ)**2, rₙ
        #     ) / (4 * π * Lₙ)
        # else:
        #     total_SOCₙ = pybamm.Scalar(0)
        # total_SOCₚ = pybamm.Integral(SOCₚ / pybamm.Integral(
        #     rₚ / Lₚ, xₚ)**2, rₚ
        # )  / (4 * π * Lₚ)
        φₑ = φₑ_0 + Cₑ * φₑ_1
        if self.halfcell:
            overpotential_dim = voltage_dim - OCVₚ_dim(total_SOCₚ, T)
        else:
            overpotential_dim = (
                voltage_dim
                - OCVₚ_dim(total_SOCₚ, T)
                + OCVₙ_dim(total_SOCₙ, T)
            )
        overpotential = overpotential_dim / thermal_voltage

        new_variables = {
            "Negative electrode potential over time": Cₑ * bar_φₛₙ_1,
            "Negative electrode potential over time [V]":
                (Cₑ * bar_φₛₙ_1) * thermal_voltage,
            "Negative electrode potential": Cₑ * φₛₙ_1,
            "Negative electrode potential [V]": (Cₑ * φₛₙ_1) * thermal_voltage,
            "Negative electrolyte potential":
                pybamm.Scalar(0) if self.halfcell else φₑ_0 + Cₑ * φₑₙ_1,
            "Negative electrolyte potential [V]":
                pybamm.Scalar(0) if self.halfcell else
                (φₑ_0 + Cₑ * φₑₙ_1) * thermal_voltage,
            "Separator electrolyte potential": φₑ_0 + Cₑ * φₑₛ_1,
            "Separator electrolyte potential [V]":
                (φₑ_0 + Cₑ * φₑₛ_1) * thermal_voltage,
            "Positive electrolyte potential": φₑ_0 + Cₑ * φₑₚ_1,
            "Positive electrolyte potential [V]":
                (φₑ_0 + Cₑ * φₑₚ_1) * thermal_voltage,
            "Electrolyte potential": φₑ,
            "Electrolyte potential [V]": φₑ * thermal_voltage,
            "Positive electrode potential over time": voltage + Cₑ * bar_φₛₚ_1,
            "Positive electrode potential over time [V]":
                voltage_dim + (Cₑ * bar_φₛₚ_1) * thermal_voltage,
            "Positive electrode potential": voltage_dim + Cₑ * φₛₚ_1,
            "Positive electrode potential [V]":
                voltage_dim + (Cₑ * φₛₚ_1) * thermal_voltage,
            # The following variable names are for PyBaMM compatibility.
            # In particular, they are required if pybamm_control==True.
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Total overpotential": overpotential,
            "Total overpotential [V]": overpotential_dim,
            "X-averaged negative electrode ohmic losses": Cₑ * bar_φₛₙ_1,
            "X-averaged negative electrode ohmic losses [V]":
                thermal_voltage * Cₑ * bar_φₛₙ_1,
            "X-averaged positive electrode ohmic losses": Cₑ * bar_φₛₚ_1,
            "X-averaged positive electrode ohmic losses [V]":
                thermal_voltage * Cₑ * bar_φₛₚ_1,
        }
        if not self.pybamm_control:
            new_variables["Discharge capacity [A.h]"] = Q
            new_variables["Current [A]"] = C * I_extern
        return new_variables

    def set_rhs(self, variables):
        """!@brief Sets up the right-hand-side equations in self.rhs.
        """

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
        cₑ_1 = variables["Electrolyte concentration correction"]
        # Constant temperature
        T = T_init
        # Porosity
        if self.halfcell:
            ε = pybamm.concatenation(
                pybamm.FullBroadcast(εₛ_scalar, "separator",
                                     "current collector"),
                pybamm.FullBroadcast(εₚ_scalar, "positive electrode",
                                     "current collector")
            )
        else:
            ε = pybamm.concatenation(
                pybamm.FullBroadcast(εₙ_scalar, "negative electrode",
                                     "current collector"),
                pybamm.FullBroadcast(εₛ_scalar, "separator",
                                     "current collector"),
                pybamm.FullBroadcast(εₚ_scalar, "positive electrode",
                                     "current collector"))
        # Tortuosity
        if self.halfcell:
            εᵝ = pybamm.concatenation(
                pybamm.FullBroadcast(εₛ_scalar**βₑₛ_scalar, "separator",
                                     "current collector"),
                pybamm.FullBroadcast(εₚ_scalar**βₑₚ_scalar,
                                     "positive electrode",
                                     "current collector"))
        else:
            εᵝ = pybamm.concatenation(
                pybamm.FullBroadcast(εₙ_scalar**βₑₙ_scalar,
                                     "negative electrode",
                                     "current collector"),
                pybamm.FullBroadcast(εₛ_scalar**βₑₛ_scalar, "separator",
                                     "current collector"),
                pybamm.FullBroadcast(εₚ_scalar**βₑₚ_scalar,
                                     "positive electrode",
                                     "current collector"))

        self.rhs[t] = τᵈ / 3600
        if not self.pybamm_control:
            self.rhs[Q] = C * I_cell * τᵈ / 3600
        if not self.halfcell:
            Nₛₙ = -Dₙ(SOCₙ, T) * pybamm.grad(SOCₙ)
            self.rhs[SOCₙ] = -(1 / Cₙ) * pybamm.div(Nₛₙ)
            # self.rhs[total_SOCₙ] = -I * 3 / (Lₙ * aₙ * γₙ)
        Nₛₚ = -Dₚ(SOCₚ, T) * pybamm.grad(SOCₚ)
        self.rhs[SOCₚ] = -(1 / Cₚ) * pybamm.div(Nₛₚ)
        # self.rhs[total_SOCₚ] = I * 3 / (Lₚ * aₚ * γₚ)

        # Electrolyte concentration
        if not self.halfcell:
            cₑₙ_1_I_term = pybamm.FullBroadcast(
                (1 - t_plus(cₑ_init)) * I_cell / Lₙ,
                "negative electrode", "current collector")
        cₑₛ_1_I_term = pybamm.FullBroadcast(
            0, "separator", "current collector"
        )
        cₑₚ_1_I_term = pybamm.FullBroadcast(
            -(1 - t_plus(cₑ_init)) * I_cell / Lₚ,
            "positive electrode",
            "current collector"
        )
        if self.halfcell:
            cₑ_1_I_term = pybamm.concatenation(
                cₑₛ_1_I_term, cₑₚ_1_I_term
            )
        else:
            cₑ_1_I_term = pybamm.concatenation(
                cₑₙ_1_I_term, cₑₛ_1_I_term, cₑₚ_1_I_term
            )
        self.rhs[cₑ_1] = (
            -γₑ * pybamm.div(-εᵝ * Dₑ(cₑ_init, T) * pybamm.grad(cₑ_1))
            + cₑ_1_I_term
        ) / (Cₑ * ε * γₑ)

    def set_algebraic(self, variables):
        """!@brief Sets up the algebraic equations in self.algebraic.
        """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        ηₙ_0 = variables["Negative electrode overpotential (0th order)"]
        ηₚ_0 = variables["Positive electrode overpotential (0th order)"]
        iₛₑₙ = variables["Negative electrode interface current density"]
        iₛₑₚ = variables["Positive electrode interface current density"]

        if self.halfcell:
            self.algebraic[ηₙ_0] = iₛₑₙ - I_cell
        else:
            self.algebraic[ηₙ_0] = iₛₑₙ - I_cell / Lₙ
        self.algebraic[ηₚ_0] = iₛₑₚ + I_cell / Lₚ

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
                "Negative particle surface concentration over time"
            ]
        SOCₚ_surf = variables[
            "Positive particle surface concentration over time"
        ]
        cₑ_1 = variables["Electrolyte concentration correction"]
        # Constant temperature
        T = T_init

        if not self.halfcell:
            self.boundary_conditions[SOCₙ] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (
                    -Cₙ * (I_cell / Lₙ) / (aₙ * γₙ * Dₙ(SOCₙ_surf, T)),
                    "Neumann"
                ),
            }
        self.boundary_conditions[SOCₚ] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -Cₚ * (-I_cell / Lₚ) / (aₚ * γₚ * Dₚ(SOCₚ_surf, T)),
                "Neumann"
            ),
        }
        if self.halfcell:
            self.boundary_conditions[cₑ_1] = {
                "left": (
                    -(1 - t_plus(cₑ_init)) * I_cell
                    / (γₑ * εₛ_scalar**βₑₛ_scalar * Dₑ(cₑ_init, T)),
                    "Neumann"
                ),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        else:
            self.boundary_conditions[cₑ_1] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

    def set_initial_conditions(self, variables):
        """!@brief Sets the (self.)initial(_)conditions.
        """

        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        ηₙ_0 = variables["Negative electrode overpotential (0th order)"]
        ηₚ_0 = variables["Positive electrode overpotential (0th order)"]
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
            # total_SOCₙ = variables["Negative electrode SOC"]
        SOCₚ = variables["Positive particle concentration"]
        # total_SOCₚ = variables["Positive electrode SOC"]
        cₑ_1 = variables["Electrolyte concentration correction"]

        self.initial_conditions[t] = pybamm.Scalar(0)
        if not self.pybamm_control:
            self.initial_conditions[Q] = pybamm.Scalar(0)
        if self.halfcell:
            self.initial_conditions[ηₙ_0] = (2 / zₙ) * pybamm.arcsinh(
                I_cell / (2 * (γₙ / Cᵣₙ) * iₛₑₙ_0(cₑ_init, 0.5, T_init))
            )
        else:
            self.initial_conditions[ηₙ_0] = (2 / zₙ) * pybamm.arcsinh(
                I_cell / (
                    2 * (γₙ / Cᵣₙ)
                    * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), T_init)
                    * Lₙ
                )
            )
        self.initial_conditions[ηₚ_0] = (2 / zₚ) * pybamm.arcsinh(
            -I_cell / (
                2 * (γₚ / Cᵣₚ)
                * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), T_init)
                * Lₚ
            )
        )
        # c_n_init and c_p_init can in general be functions of x
        if not self.halfcell:
            # xₙ = pybamm.standard_spatial_vars.x_n
            # self.initial_conditions[SOCₙ] = (
            #     pybamm.Integral(SOCₙ_init(xₙ), xₙ) / Lₙ
            # )
            # self.initial_conditions[total_SOCₙ] = (
            #     pybamm.Integral(SOCₙ_init(xₙ), xₙ) / Lₙ
            # )
            self.initial_conditions[SOCₙ] = SOCₙ_init(0)
            # self.initial_conditions[total_SOCₙ] = SOCₙ_init(0)
        # xₚ = pybamm.standard_spatial_vars.x_p
        # self.initial_conditions[SOCₚ] = (
        #     pybamm.Integral(SOCₚ_init(xₚ), xₚ) / Lₚ
        # )
        # self.initial_conditions[total_SOCₚ] = (
        #     pybamm.Integral(SOCₚ_init(xₚ), xₚ) / Lₚ
        # )
        self.initial_conditions[SOCₚ] = SOCₚ_init(1)
        # self.initial_conditions[total_SOCₚ] = SOCₚ_init(1)
        self.initial_conditions[cₑ_1] = pybamm.Scalar(0)

    def set_events(self, variables):
        """!@brief Sets up the termination switches in self.events.
        """

        if self.halfcell:
            SOCₙ_surf = pybamm.Scalar(0)
        else:
            SOCₙ_surf = variables[
                "Negative particle surface concentration over time"
            ]
        SOCₚ_surf = variables[
            "Positive particle surface concentration over time"
        ]
        cₑ_1 = variables["Electrolyte concentration correction"]
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
                (1 + Cₑ * pybamm.min(cₑ_1)) - 0.002
            ),
            pybamm.Event("Minimum voltage", voltage_dim - Uₗ),
            pybamm.Event("Maximum voltage", voltage_dim - Uᵤ),
        ]


class SPMe(pybamm.BaseBatteryModel):
    """!@brief Single-Particle Model with electrolyte (SPMe).
    """

    def __init__(
        self,
        halfcell=False,
        pybamm_control=False,
        name="SPMe",
        options={},
        build=True
    ):
        """!@brief Sets up a SPMe model usable by PyBaMM.

        @par halfcell
            Per default False, which indicates a full-cell setup. If set
            to True, the equations for a half-cell will be used instead.
        @par pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @par name
            The optional name of the model. Default is "SPMe".
        @par options
            Used for external circuit if pybamm_control is True.
        @par build
            Per default True, which builds the model equations right
            away. If set to False, they remain as symbols for later.
        """

        super().__init__(name=name, options=options)
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
        self.submodels["internal"] = SPMe_internal(self.param, self.halfcell,
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
