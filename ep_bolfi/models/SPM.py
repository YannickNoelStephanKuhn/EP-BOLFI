# Copyright (c): German Aerospace Center (DLR)
"""!@package models.SPM
Contains a PyBaMM-compatible Single-Particle Model (SPM).
"""

import pybamm
import models.standard_parameters as standard_parameters
from models.standard_parameters import (
    OCVₙ_ref, OCVₚ_ref, OCVₙ_dim, OCVₚ_dim, OCVₙ, OCVₚ,
    Dₙ, Dₚ,
    cₙ, cₚ, SOCₙ_init, SOCₚ_init,
    iₛₑₙ_0, iₛₑₚ_0,
    aₙ, aₚ,
    zₙ, zₚ,
    αₙₙ, αₙₚ, αₚₙ, αₚₚ,
    Rₙ, Rₚ,
    cₑ_typ, cₑ_init,
    εₙ_scalar, εₚ_scalar,
    Lₙ_dim, Lₚ_dim, L_dim, Lₙ, Lₚ,
    Cₙ, Cₚ, Cᵣₙ, Cᵣₚ,
    γₙ, γₚ,
    C, A, F, τᵈ, capacity,
    T_init, thermal_voltage,
    I_extern,
    Uₗ, Uᵤ,
)
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


class SPM_internal(pybamm.BaseSubModel):
    """!@brief Defining equations for a Single-Particle Model (SPM).

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

        # if not self.halfcell:
        #     xₙ = pybamm.standard_spatial_vars.x_n
        #     rₙ = pybamm.standard_spatial_vars.r_n
        # xₚ = pybamm.standard_spatial_vars.x_p
        # rₚ = pybamm.standard_spatial_vars.r_p

        if not self.pybamm_control:
            # I_cell = I_extern
            Q = pybamm.Variable("Discharge capacity [A.h]")

        # Define the variables.
        t = pybamm.Variable("Time [h]")
        ηₙ = pybamm.Variable("Negative electrode overpotential")
        ηₚ = pybamm.Variable("Positive electrode overpotential")
        if not self.halfcell:
            SOCₙ = pybamm.Variable(
                "Negative particle concentration",
                domain="negative particle",
            )
            total_SOCₙ = pybamm.Variable("Negative electrode SOC")
        else:
            SOCₙ = pybamm.Scalar(0)
            total_SOCₙ = pybamm.Scalar(0)
        SOCₚ = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
        )
        total_SOCₚ = pybamm.Variable("Positive electrode SOC")
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
                * (pybamm.exp(αₙₙ * zₙ * ηₙ) - pybamm.exp(-αₚₙ * zₙ * ηₙ))
            )
        else:
            iₛₑₙ = (
                (γₙ / Cᵣₙ)
                * iₛₑₙ_0(cₑ_init, SOCₙ_surf, T)
                * (pybamm.exp(αₙₙ * zₙ * ηₙ) - pybamm.exp(-αₚₙ * zₙ * ηₙ))
            )
        iₛₑₚ = (
            (γₚ / Cᵣₚ)
            * iₛₑₚ_0(cₑ_init, SOCₚ_surf, T)
            * (pybamm.exp(αₙₚ * zₚ * ηₚ) - pybamm.exp(-αₚₚ * zₚ * ηₚ))
        )
        if self.halfcell:
            φₑ = -ηₙ
        else:
            φₑ = -ηₙ - OCVₙ(SOCₙ_surf, T)
        φₛₙ = pybamm.Scalar(0)
        if self.halfcell:
            voltage = OCVₚ(SOCₚ_surf, T) + ηₚ - ηₙ
            voltage_dim = OCVₚ_ref + voltage * thermal_voltage
        else:
            voltage = OCVₚ(SOCₚ_surf, T) - OCVₙ(SOCₙ_surf, T) + ηₚ - ηₙ
            voltage_dim = OCVₚ_ref - OCVₙ_ref + voltage * thermal_voltage
        φₛₚ = voltage
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
        if self.halfcell:
            overpotential_dim = voltage_dim - OCVₚ_dim(total_SOCₚ, T)
        else:
            overpotential_dim = (
                voltage_dim
                - OCVₚ_dim(total_SOCₚ, T)
                + OCVₙ_dim(total_SOCₙ, T)
            )
        overpotential = overpotential_dim / thermal_voltage
        if self.halfcell:
            whole_cell = ["separator", "positive electrode"]
        else:
            whole_cell = ["negative electrode", "separator",
                          "positive electrode"]

        variables = {
            "Negative electrode SOC": total_SOCₙ,
            "Negative particle concentration": SOCₙ,
            "Negative particle concentration [mol.m-3]": SOCₙ * cₙ,
            "Negative particle surface concentration over time": SOCₙ_surf,
            "Negative particle surface concentration over time [mol.m-3]":
                SOCₙ_surf * cₙ,
            "Negative particle surface concentration":
                SOCₙ_surf if self.halfcell else
                pybamm.PrimaryBroadcast(SOCₙ_surf, "negative electrode"),
            "Negative particle surface concentration [mol.m-3]":
                SOCₙ_surf * cₙ if self.halfcell else
                pybamm.PrimaryBroadcast(SOCₙ_surf * cₙ, "negative electrode"),
            "Electrolyte concentration": pybamm.PrimaryBroadcast(
                cₑ_init, whole_cell),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                cₑ_init * cₑ_typ, whole_cell),
            "Positive electrode SOC": total_SOCₚ,
            "Positive particle concentration": SOCₚ,
            "Positive particle concentration [mol.m-3]": SOCₚ * cₚ,
            "Positive particle surface concentration over time": SOCₚ_surf,
            "Positive particle surface concentration over time [mol.m-3]":
                SOCₚ_surf * cₚ,
            "Positive particle surface concentration": pybamm.PrimaryBroadcast(
                SOCₚ_surf, "positive electrode"),
            "Positive particle surface concentration [mol.m-3]":
                pybamm.PrimaryBroadcast(SOCₚ_surf * cₚ, "positive electrode"),
            "Negative electrode interface current density": iₛₑₙ,
            "Positive electrode interface current density": iₛₑₚ,
            "Negative electrode potential over time": φₛₙ,
            "Negative electrode potential over time [V]":
                φₛₙ * thermal_voltage,
            "Negative electrode potential":
                φₛₙ
                if self.halfcell else
                pybamm.PrimaryBroadcast(φₛₙ, "negative electrode"),
            "Negative electrode potential [V]":
                φₛₙ * thermal_voltage
                if self.halfcell else
                pybamm.PrimaryBroadcast(
                    φₛₙ * thermal_voltage, "negative electrode"
                ),
            "Negative electrode overpotential": ηₙ,
            "Negative electrode overpotential [V]": ηₙ * thermal_voltage,
            "Electrolyte potential over time": φₑ,
            "Electrolyte potential over time [V]": φₑ * thermal_voltage,
            "Electrolyte potential": pybamm.PrimaryBroadcast(φₑ, whole_cell),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(
                φₑ * thermal_voltage, whole_cell),
            "Positive electrode potential over time": voltage,
            "Positive electrode potential over time [V]": voltage_dim,
            "Positive electrode potential": pybamm.PrimaryBroadcast(
                voltage, "positive electrode"),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                voltage_dim, "positive electrode"),
            "Positive electrode overpotential": ηₚ,
            "Positive electrode overpotential [V]": ηₚ * thermal_voltage,
            # The following variable names are for PyBaMM compatibility.
            # In particular, they are required if pybamm_control==True.
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Total overpotential": overpotential,
            "Total overpotential [V]": overpotential_dim,
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
            "X-averaged negative electrode reaction overpotential": ηₙ,
            "X-averaged negative electrode reaction overpotential [V]":
                thermal_voltage * ηₙ,
            "X-averaged positive electrode reaction overpotential": ηₚ,
            "X-averaged positive electrode reaction overpotential [V]":
                thermal_voltage * ηₚ,
            "X-averaged negative electrode ohmic losses": φₛₙ,
            "X-averaged negative electrode ohmic losses [V]":
                thermal_voltage * φₛₙ,
            "X-averaged positive electrode ohmic losses": φₛₚ,
            "X-averaged positive electrode ohmic losses [V]":
                thermal_voltage * φₛₚ,
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

        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
            total_SOCₙ = variables["Negative electrode SOC"]
        SOCₚ = variables["Positive particle concentration"]
        total_SOCₚ = variables["Positive electrode SOC"]
        # Constant temperature
        T = T_init

        self.rhs[t] = τᵈ / 3600
        if not self.pybamm_control:
            self.rhs[Q] = C * I_cell * τᵈ / 3600
        if not self.halfcell:
            Nₛₙ = -Dₙ(SOCₙ, T) * pybamm.grad(SOCₙ)
            self.rhs[SOCₙ] = -(1 / Cₙ) * pybamm.div(Nₛₙ)
            self.rhs[total_SOCₙ] = -I_cell * 3 / (Lₙ * aₙ * γₙ)
        Nₛₚ = -Dₚ(SOCₚ, T) * pybamm.grad(SOCₚ)
        self.rhs[SOCₚ] = -(1 / Cₚ) * pybamm.div(Nₛₚ)
        self.rhs[total_SOCₚ] = I_cell * 3 / (Lₚ * aₚ * γₚ)

    def set_algebraic(self, variables):
        """!@brief Sets up the algebraic equations in self.algebraic.
        """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        ηₙ = variables["Negative electrode overpotential"]
        ηₚ = variables["Positive electrode overpotential"]
        iₛₑₙ = variables["Negative electrode interface current density"]
        iₛₑₚ = variables["Positive electrode interface current density"]

        if self.halfcell:
            self.algebraic[ηₙ] = iₛₑₙ - I_cell
        else:
            self.algebraic[ηₙ] = iₛₑₙ - I_cell / Lₙ
        self.algebraic[ηₚ] = iₛₑₚ + I_cell / Lₚ

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

    def set_initial_conditions(self, variables):
        """!@brief Sets the (self.)initial(_)conditions.
        """

        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        ηₙ = variables["Negative electrode overpotential"]
        ηₚ = variables["Positive electrode overpotential"]
        if not self.halfcell:
            SOCₙ = variables["Negative particle concentration"]
            total_SOCₙ = variables["Negative electrode SOC"]
        SOCₚ = variables["Positive particle concentration"]
        total_SOCₚ = variables["Positive electrode SOC"]

        self.initial_conditions[t] = pybamm.Scalar(0)
        if not self.pybamm_control:
            self.initial_conditions[Q] = pybamm.Scalar(0)
        if self.halfcell:
            self.initial_conditions[ηₙ] = (2 / zₙ) * pybamm.arcsinh(
                I_cell / (2 * (γₙ / Cᵣₙ) * iₛₑₙ_0(cₑ_init, 0.5, T_init))
            )
        else:
            self.initial_conditions[ηₙ] = (2 / zₙ) * pybamm.arcsinh(
                I_cell / (
                    2 * (γₙ / Cᵣₙ)
                    * iₛₑₙ_0(cₑ_init, SOCₙ_init(0), T_init)
                    * Lₙ
                )
            )
        self.initial_conditions[ηₚ] = (2 / zₚ) * pybamm.arcsinh(
            -I_cell / (
                2 * (γₚ / Cᵣₚ)
                * iₛₑₚ_0(cₑ_init, SOCₚ_init(1), T_init)
                * Lₚ
            )
        )
        # c_n_init and c_p_init can in general be functions of x
        if not self.halfcell:
            self.initial_conditions[SOCₙ] = SOCₙ_init(0)
            self.initial_conditions[total_SOCₙ] = SOCₙ_init(0)
        self.initial_conditions[SOCₚ] = SOCₚ_init(1)
        self.initial_conditions[total_SOCₚ] = SOCₚ_init(1)

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
            pybamm.Event("Minimum voltage", voltage_dim - Uₗ),
            pybamm.Event("Maximum voltage", voltage_dim - Uᵤ),
        ]


class SPM(pybamm.BaseBatteryModel):
    """!@brief Single-Particle Model (SPM).
    """

    def __init__(
        self,
        halfcell=False,
        pybamm_control=False,
        name="SPM",
        options={},
        build=True
    ):
        """!@brief Sets up a SPM model usable by PyBaMM.

        @par halfcell
            Per default False, which indicates a full-cell setup. If set
            to True, the equations for a half-cell will be used instead.
        @par pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @par name
            The optional name of the model. Default is "SPM".
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
        # Incorporate the SPM equations as a "submodel". This way, it
        # can be combined with PyBaMM's "external circuit".
        self.submodels["internal"] = SPM_internal(self.param, self.halfcell,
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

        new_model = self.__class__(name=self.name, halfcell=self.halfcell,
                                   pybamm_control=self.pybamm_control,
                                   build=False)
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
