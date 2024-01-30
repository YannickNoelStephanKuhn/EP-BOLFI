"""!@package ep_bolfi.models.electrolyte
Contains a PyBaMM-compatible electrolyte model.
"""

import pybamm
import ep_bolfi.models.standard_parameters as standard_parameters
from ep_bolfi.models.standard_parameters import (
    OCVₙ_ref, OCVₚ_ref, OCVₙ, OCVₚ,
    cₙ, cₚ,
    iₛₑₙ_0, iₛₑₚ_0,
    zₙ, zₚ,
    αₙₙ, αₚₙ, αₙₚ, αₚₚ,
    σₙ, σₚ,
    Rₙ, Rₚ,
    cₑ_typ, cₑ_init,
    εₛ, εₙ_scalar, εₛ_scalar, εₚ_scalar,
    βₑₛ_scalar,
    Dₑ, κₑ, κₑ_hat, t_plus, one_plus_dlnf_dlnc,
    Lₙ_dim, Lₚ_dim, L_dim, Lₙ, Lₚ,
    Cₑ, Cᵣₙ, Cᵣₚ,
    γₑ, γₙ, γₚ,
    C, A, F, τᵈ, capacity,
    T_init, thermal_voltage,
    I_extern,
    Uₗ, Uᵤ,
)
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


class Electrolyte_internal(pybamm.BaseSubModel):
    """!@brief Defining equations for a symmetric Li cell with electrolyte.
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
        pybamm_control=False,
        options={},
        build=True
    ):
        """!@brief Sets the model properties.
        @param param
            A class containing all the relevant parameters for this
            model. For example, models.standard_parameters represents a
            valid choice for this parameter.
        @param pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @param options
            Not used; only here for compatibility with the base class.
        @param build
            Not used; only here for compatibility with the base class.
        """

        super().__init__(param)
        ## Current is fixed if False and a variable if True.
        self.pybamm_control = pybamm_control

    def get_fundamental_variables(self):
        """!@brief Builds all relevant model variables' symbols.
        @return
            A dictionary with the variables' names as keys and their
            symbols (of type pybamm.Symbol) as values.
        """

        # Define the variables.
        t = pybamm.Variable("Time [h]")
        if not self.pybamm_control:
            # I_cell = I_extern
            Q = pybamm.Variable("Discharge capacity [A.h]")

        # Variables that vary spatially are created with a domain
        cₑₛ = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        cₑ = cₑₛ

        # Electrolyte potential
        φₑₛ = pybamm.Variable(
            "Separator electrolyte potential", domain="separator"
        )
        φₑ = φₑₛ

        SOCₙ = pybamm.Scalar(0)
        SOCₚ = pybamm.Scalar(0)
        # total_SOCₙ = pybamm.Scalar(0)
        # total_SOCₚ = pybamm.Scalar(0)
        SOCₙ_surf = pybamm.Scalar(0)
        SOCₚ_surf = pybamm.Scalar(0)

        # Constant temperature
        T = T_init
        # Porosity
        # ε = εₛ
        # Tortuosity
        εᵝ = εₛ**βₑₛ_scalar

        # Interfacial reactions

        ηₙ = pybamm.Variable("Negative electrode overpotential")
        ηₚ = pybamm.Variable("Positive electrode overpotential")

        # Butler-Volmer reaction fluxes.
        # Note: not using the simplified version with sinh, which is
        # fixed to symmetric charge-transfer, is significantly slower.
        # Note: γₙ / Cᵣₙ does not depend on cₙ; it cancels out.
        iₛₑₙ = (
            (γₙ / Cᵣₙ)
            * iₛₑₙ_0(pybamm.boundary_value(cₑₛ, "left"), 0.5, T)
            * (pybamm.exp(αₙₙ * zₙ * ηₙ) - pybamm.exp(-αₚₙ * zₙ * ηₙ))
        )
        iₛₑₛ = pybamm.PrimaryBroadcast(0, "separator")
        iₛₑₚ = (
            (γₚ / Cᵣₚ) * iₛₑₚ_0(pybamm.boundary_value(cₑₛ, "right"), 0.5, T)
            * (pybamm.exp(αₙₚ * zₚ * ηₚ) - pybamm.exp(-αₚₚ * zₚ * ηₚ))
        )
        iₛₑ = iₛₑₛ

        # Current in the electrolyte
        iₑ = εᵝ * κₑ_hat * κₑ(cₑ, T) * (
            2 * one_plus_dlnf_dlnc(cₑ)
            * (1 - t_plus(cₑ)) * pybamm.grad(cₑ) / cₑ
            - pybamm.grad(φₑ)
        )

        # Calculations for overpotential
        total_SOCₙ = pybamm.Scalar(0)
        total_SOCₚ = pybamm.Scalar(0)
        # The `variables` dictionary contains all variables that might
        # be useful for visualising the solution of the model
        variables = {
            "Negative electrode SOC": total_SOCₙ,
            "Negative particle concentration": SOCₙ,
            "Negative particle concentration [mol.m-3]": SOCₙ * cₙ,
            "Negative particle surface concentration": SOCₙ_surf,
            "Negative particle surface concentration [mol.m-3]":
                SOCₙ_surf * cₙ,
            "Negative electrolyte concentration": pybamm.Scalar(0),
            "Negative electrolyte concentration [mol.m-3]": pybamm.Scalar(0),
            "Separator electrolyte concentration": cₑₛ,
            "Separator electrolyte concentration [mol.m-3]": cₑₛ * cₑ_typ,
            "Positive electrolyte concentration": pybamm.Scalar(0),
            "Positive electrolyte concentration [mol.m-3]": pybamm.Scalar(0),
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
            "Interface current density [A.m-2]": iₛₑₛ,
            "Negative electrode overpotential": ηₙ,
            "Negative electrode overpotential [V]": ηₙ * thermal_voltage,
            "Electrolyte current": iₑ,
            # For computation, use the version without NaN values.
            # "Negative electrolyte potential": φₑₙ,
            # "Negative electrolyte potential [V]": φₑₙ * thermal_voltage,
            "Separator electrolyte potential": φₑₛ,
            "Separator electrolyte potential [V]": φₑₛ * thermal_voltage,
            # "Positive electrolyte potential": φₑₚ,
            # "Positive electrolyte potential [V]": φₑₚ * thermal_voltage,
            "Electrolyte potential": φₑ,
            "Electrolyte potential [V]": φₑ * thermal_voltage,
            "Positive electrode overpotential": ηₚ,
            "Positive electrode overpotential [V]": ηₚ * thermal_voltage,
            # The following variable names are for PyBaMM compatibility.
            # In particular, they are required if pybamm_control==True.
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
                ηₙ,
            "X-averaged negative electrode reaction overpotential [V]":
                thermal_voltage * ηₙ,
            "X-averaged positive electrode reaction overpotential":
                ηₚ,
            "X-averaged positive electrode reaction overpotential [V]":
                thermal_voltage * ηₚ,
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
        return variables

    def get_coupled_variables(self, variables):
        """!@brief Builds all model symbols that rely on other models.
        @param variables
            A dictionary containing at least all variable symbols that
            are required for the variable symbols built here.
        @return
            A dictionary with the new variables' names as keys and their
            symbols (of type pybamm.Symbol) as values.
        """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            # Q = pybamm.Variable("Discharge capacity [A.h]")

        # Electrode potential
        # Note: the "porosity" of the electrode is 1.
        # φₛₙ = -I_cell * (Lₙ / σₙ) * xₙ / Lₙ
        φₛₙ = -I_cell * (Lₙ / σₙ)
        φₛₚ_contribution = I_cell * (Lₚ / σₚ)

        φₑₛ = variables["Separator electrolyte potential"]
        cₑₛ = variables["Separator electrolyte concentration"]

        # Current in the solid
        iₛₙ = I_cell
        iₛₚ = I_cell

        ηₙ = variables["Negative electrode overpotential"]
        ηₚ = variables["Positive electrode overpotential"]

        φₛₚ = (
            φₛₙ
            - ηₙ
            - pybamm.boundary_value(φₑₛ, "left")
            - pybamm.log(pybamm.boundary_value(cₑₛ, "left")) / zₙ
            + pybamm.boundary_value(φₑₛ, "right")
            + pybamm.log(pybamm.boundary_value(cₑₛ, "right")) / zₚ
            + ηₚ
        )
        # Cell voltage
        voltage = φₛₚ - φₛₚ_contribution
        voltage_dim = voltage * thermal_voltage
        overpotential_dim = voltage_dim
        overpotential = overpotential_dim / thermal_voltage

        return {
            "Negative electrode potential": φₛₙ,
            "Negative electrode potential [V]": φₛₙ * thermal_voltage,
            "Positive electrode potential": φₛₚ,
            "Positive electrode potential [V]":
                OCVₚ_ref - OCVₙ_ref + φₛₚ * thermal_voltage,
            "Negative electrode current": iₛₙ,
            "Positive electrode current": iₛₚ,
            # The following variable names are for PyBaMM compatibility.
            # In particular, they are required if pybamm_control==True.
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Battery voltage [V]": voltage_dim,
            "Total overpotential": overpotential,
            "Total overpotential [V]": overpotential_dim,
            "X-averaged negative electrode ohmic losses":
                φₛₙ,
            "X-averaged negative electrode ohmic losses [V]":
                thermal_voltage * φₛₙ,
            "X-averaged positive electrode ohmic losses":
                φₛₚ,
            "X-averaged positive electrode ohmic losses [V]":
                thermal_voltage * φₛₚ,
        }

    def set_rhs(self, variables):
        """!@brief Sets up the right-hand-side equations in self.rhs. """

        t = variables["Time [h]"]
        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
            Q = variables["Discharge capacity [A.h]"]
        iₑ = variables["Electrolyte current"]
        iₛₑ = variables["Interface current density"]
        cₑ = variables["Electrolyte concentration"]
        # Constant temperature
        T = T_init
        # Porosity
        ε = εₛ
        # Tortuosity
        εᵝ = εₛ**βₑₛ_scalar

        self.rhs[t] = τᵈ / 3600
        if not self.pybamm_control:
            self.rhs[Q] = C * I_cell * τᵈ / 3600
        # Electrolyte concentration
        Nₑ = -εᵝ * Dₑ(cₑ, T) * pybamm.grad(cₑ) + (Cₑ / γₑ) * t_plus(cₑ) * iₑ
        self.rhs[cₑ] = (1 / ε) * (-pybamm.div(Nₑ) / Cₑ + iₛₑ / γₑ)

    def set_algebraic(self, variables):
        """!@brief Sets up the algebraic equations in self.algebraic. """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        ηₙ = variables["Negative electrode overpotential"]
        ηₚ = variables["Positive electrode overpotential"]
        iₛₑₙ = variables["Negative electrode interface current density"]
        iₛₑₚ = variables["Positive electrode interface current density"]
        iₛₑ = variables["Interface current density"]
        iₑ = variables["Electrolyte current"]
        φₑ = variables["Electrolyte potential"]

        self.algebraic[ηₙ] = iₛₑₙ - I_cell
        self.algebraic[ηₚ] = iₛₑₚ + I_cell
        self.algebraic[φₑ] = pybamm.div(iₑ) - iₛₑ

    def set_boundary_conditions(self, variables):
        """!@brief Sets the (self.)boundary(_)conditions. """

        if self.pybamm_control:
            I_cell = variables["Total current density"]
        else:
            I_cell = I_extern
        ηₙ = variables["Negative electrode overpotential"]
        φₛₙ = variables["Negative electrode potential"]
        φₑ = variables["Electrolyte potential"]
        cₑ = variables["Electrolyte concentration"]
        # Constant temperature
        T = T_init

        cₑₙ = pybamm.boundary_value(cₑ, "left")
        cₑₚ = pybamm.boundary_value(cₑ, "right")

        # At one electrode surface, a Dirichlet boundary condition is
        # required. At the other electrode surface, a Neumann boundary
        # condition is required to determine the slope.
        self.boundary_conditions[φₑ] = {
            "left": (φₛₙ - ηₙ - pybamm.log(cₑₙ) / zₙ, "Dirichlet"),
            "right": (
                (
                    2 * one_plus_dlnf_dlnc(cₑₚ) * (1 - t_plus(cₑₚ)) * (
                        -Cₑ * (1 - t_plus(cₑₚ)) * I_cell / (
                            γₑ * εₛ_scalar**βₑₛ_scalar * Dₑ(cₑₚ, T)
                        )
                    ) / cₑₚ
                ) - I_cell / (εₛ_scalar**βₑₛ_scalar * κₑ_hat * κₑ(cₑₚ, T)),
                "Neumann"
            )
        }
        self.boundary_conditions[cₑ] = {
            "left": (
                -Cₑ * (1 - t_plus(cₑₙ)) * I_cell
                / (γₑ * εₛ_scalar**βₑₛ_scalar * Dₑ(cₑₙ, T)),
                "Neumann"
            ),
            "right": (
                -Cₑ * (1 - t_plus(cₑₚ)) * I_cell
                / (γₑ * εₛ_scalar**βₑₛ_scalar * Dₑ(cₑₚ, T)),
                "Neumann"
            ),
        }

    def set_initial_conditions(self, variables):
        """!@brief Sets the (self.)initial(_)conditions. """

        # Initial conditions for rhs: actual start values.
        # Initial conditions for algebraic: help for root-finding.
        t = variables["Time [h]"]
        if not self.pybamm_control:
            Q = variables["Discharge capacity [A.h]"]
        ηₙ = variables["Negative electrode overpotential"]
        ηₚ = variables["Positive electrode overpotential"]
        φₑ = variables["Electrolyte potential"]
        cₑ = variables["Electrolyte concentration"]

        self.initial_conditions[t] = pybamm.Scalar(0)
        if not self.pybamm_control:
            self.initial_conditions[Q] = pybamm.Scalar(0)
        self.initial_conditions[ηₙ] = pybamm.Scalar(0)
        self.initial_conditions[ηₚ] = pybamm.Scalar(0)
        self.initial_conditions[φₑ] = pybamm.Scalar(0)
        self.initial_conditions[cₑ] = cₑ_init

    def set_events(self, variables):
        """!@brief Sets up the termination switches in self.events. """

        SOCₙ_surf = pybamm.Scalar(0)
        SOCₚ_surf = pybamm.Scalar(0)
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


class Electrolyte(pybamm.BaseBatteryModel):
    """!@brief Electrolyte model assuming a symmetric Li-metal cell. """

    def __init__(
        self,
        pybamm_control=False,
        name="electrolyte",
        options={},
        build=True
    ):
        """!@brief Sets up an electrolyte model usable by PyBaMM.
        @param pybamm_control
            Per default False, which indicates that the current is given
            as a function. If set to True, this model is compatible with
            PyBaMM experiments, e.g. CC-CV simulations. The current is
            then a variable and it or voltage can be fixed functions.
        @param name
            The optional name of the model. Default is "electrolyte".
        @param options
            Used for external circuit if pybamm_control is True.
        @param build
            Per default True, which builds the model equations right
            away. If set to False, they remain as symbols for later.
        """

        super().__init__(name=name)
        ## Current is fixed if False and a variable if True.
        self.pybamm_control = pybamm_control
        pybamm.citations.register("Marquis2019")
        self.options = options

        ## Contains all the relevant parameters for this model.
        self.param = standard_parameters
        ## Non-dimensionalization timescale.
        self.timescale = τᵈ
        ## Non-dimensionalization length scales.
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
        self.submodels["internal"] = Electrolyte_internal(
            self.param, self.pybamm_control
        )

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
            }
        )
        x = var.x_s
        self.variables.update(
            {
                "x": x,
                "x [m]": x * param.L_dim,
            }
        )

    def new_copy(self, build=True):
        """!@brief Create an empty copy with identical options.
        @param build
            If True, the new model gets built right away. This is the
            default behavior. If set to False, it remains as symbols.
        @return
            The copy of this model.
        """

        new_model = self.__class__(self.pybamm_control,
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
        """
        replacer = pybamm.SymbolReplacer({})
        return replacer.process_model(self, inplace=False)
        """
        return new_model

    def set_voltage_variables(self):
        """!@brief Adds voltage-specific variables to self.variables.
        Override this inherited function, since it adds superfluous
        variables.
        """

        pass

    @property
    def default_geometry(self):
        """!@brief Override: corrects the geometry for half-cells. """

        geometry = super().default_geometry
        del geometry["negative electrode"]
        del geometry["negative particle"]
        del geometry["positive electrode"]
        del geometry["positive particle"]
        return geometry
