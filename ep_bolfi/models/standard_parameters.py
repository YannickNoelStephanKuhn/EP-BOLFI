# Copyright (c): German Aerospace Center (DLR)
"""!@file
@brief Comprehensive list of parameters of every model.
SI units are assumed unless stated otherwise. When imported, this
package turns into the list of variables contained in here.
The "syntactic sugar" with the lower and upper indexed letters is
treated by Python to be the same as their normal counterparts.
E.g., "aₙ" and "an" refer to the exact same variable.
Greek letters aren't converted, unless they are also indexed, in which
case the same as before applies: "εᵝ" is the same as "εβ".
"""

import pybamm
from pybamm import Scalar, Parameter, FunctionParameter
from scipy import constants
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


# ----------------------------------------------------------------------
# "File Layout:"
# 1. Scalar Parameters
# 2. Function Parameters
# 3. Scalings
# 4. Non-Dimensionalised Parameters
# 5. Input / External driving force

# ----------------------------------------------------------------------
# "1. Scalar Parameters"

# Physical constants
"""! Gas constant. """
R = Scalar(constants.R)
"""! Faraday constant. """
F = Scalar(constants.physical_constants["Faraday constant"][0])
"""! Boltzmann constant. """
k_B = constants.physical_constants["Boltzmann constant"][0]
"""! Electron volt. """
qₑ = constants.physical_constants["electron volt"][0]

"""! Reference temperature for non-dimensionalization. """
T_ref = Parameter("Reference temperature [K]")
"""! Initial temperature. """
T_init = Parameter("Initial temperature [K]")
thermal_voltage = R * T_ref / F
"""! Typical temperature rise. """
ΔT = Scalar(1)

"""! Anode thickness. """
Lₙ_dim = Parameter("Negative electrode thickness [m]")
"""! Separator thickness. """
Lₛ_dim = Parameter("Separator thickness [m]")
"""! Cathode thickness. """
Lₚ_dim = Parameter("Positive electrode thickness [m]")
"""! Cell thickness. """
L_dim = Lₙ_dim + Lₛ_dim + Lₚ_dim
"""! Cross-section area of the cell. """
A = Parameter("Current collector perpendicular area [m2]")
"""! Volume of the cell. Left as a separate Parameter for compatibility. """
V = Parameter("Cell volume [m3]")
"""! PyBaMM-compatible name for the cell thickness. """
L_x = L_dim
"""! Width of the electrode for PyBaMM compatibility. """
L_y = Parameter("Electrode width [m]")
"""! Height of the electrode for PyBaMM compatibility. """
L_z = Parameter("Electrode height [m]")

"""! C-rate of the battery (in A). """
C = Parameter("Typical current [A]")
"""! Lower threshold for the cell voltage. """
Uₗ = pybamm.Parameter("Lower voltage cut-off [V]")
"""! Upper threshold for the cell voltage. """
Uᵤ = pybamm.Parameter("Upper voltage cut-off [V]")

"""! Reference electrolyte concentration for non-dimensionalization. """
cₑ_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")

"""! Maximum lithium concentration in the anode active material. """
cₙ = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
"""! Maximum lithium concentration in the cathode active material. """
cₚ = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
"""! Electronic conductivity of the anode. """
σₙ_dim = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
"""! Electronic conductivity of the cathode. """
σₚ_dim = pybamm.Parameter("Positive electrode conductivity [S.m-1]")

"""! Specific surface area of the anode. """
aₙ_dim = Parameter("Negative electrode surface area to volume ratio [m-1]")
"""! Specific surface area of the cathode. """
aₚ_dim = Parameter("Positive electrode surface area to volume ratio [m-1]")
"""! Mean/Median radius of the anode particles. """
Rₙ = Parameter("Negative particle radius [m]")
"""! Mean/Median radius of the cathode particles. """
Rₚ = Parameter("Positive particle radius [m]")
"""! Anodic symmetry factor for the anode interface reaction. """
αₙₙ = Parameter("Negative electrode anodic charge-transfer coefficient")
"""! Cathodic symmetry factor for the anode interface reaction. """
αₚₙ = Parameter("Negative electrode cathodic charge-transfer coefficient")
"""! Anodic symmetry factor for the cathode interface reaction. """
αₙₚ = Parameter("Positive electrode anodic charge-transfer coefficient")
"""! Cathodic symmetry factor for the cathode interface reaction. """
αₚₚ = Parameter("Positive electrode cathodic charge-transfer coefficient")
"""! Bruggeman coefficient for the electrolyte in the anode (scalar). """
βₑₙ_scalar = Parameter(
    "Negative electrode Bruggeman coefficient (electrolyte)"
)
"""! Bruggeman coefficient for the electrolyte in the separator (scalar). """
βₑₛ_scalar = Parameter("Separator Bruggeman coefficient (electrolyte)")
"""! Bruggeman coefficient for the electrolyte in the cathode (scalar). """
βₑₚ_scalar = Parameter(
    "Positive electrode Bruggeman coefficient (electrolyte)"
)
"""! Bruggeman coefficient for the electrolyte in the anode (vector). """
βₑₙ = pybamm.PrimaryBroadcast(βₑₙ_scalar, "negative electrode")
"""! Bruggeman coefficient for the electrolyte in the separator (vector). """
βₑₛ = pybamm.PrimaryBroadcast(βₑₛ_scalar, "separator")
"""! Bruggeman coefficient for the electrolyte in the cathode (vector). """
βₑₚ = pybamm.PrimaryBroadcast(βₑₚ_scalar, "positive electrode")
"""! Bruggeman coefficient for the solid part in the anode. """
βₛₙ_scalar = Parameter("Negative electrode Bruggeman coefficient (electrode)")
"""! Bruggeman coefficient for the solid part in the seperator. """
βₛₛ_scalar = Parameter("Separator Bruggeman coefficient (electrode)")
"""! Bruggeman coefficient for the solid part in the cathode. """
βₛₚ_scalar = Parameter("Positive electrode Bruggeman coefficient (electrode)")
"""! Bruggeman coefficient for the electrolyte in the whole cell. """
βₑ = pybamm.Concatenation(βₑₙ, βₑₛ, βₑₚ)
"""! Porosity of the anode (scalar). """
εₙ_scalar = Parameter("Negative electrode porosity")
"""! Porosity of the separator (scalar). """
εₛ_scalar = Parameter("Separator porosity")
"""! Porosity of the cathode (scalar). """
εₚ_scalar = Parameter("Positive electrode porosity")
"""! Porosity of the anode (vector). """
εₙ = pybamm.PrimaryBroadcast(εₙ_scalar, "negative electrode")
"""! Porosity of the separator (vector). """
εₛ = pybamm.PrimaryBroadcast(εₛ_scalar, "separator")
"""! Porosity of the cathode (vector). """
εₚ = pybamm.PrimaryBroadcast(εₚ_scalar, "positive electrode")
"""! Porosity of the whole cell. """
ε = pybamm.Concatenation(εₙ, εₛ, εₚ)
"""! Tortuosity of the whole cell. """
εᵝ = ε**βₑ

# Example on how to make εₑₙ a Function Parameter:
# εₑₙ = pybamm.FunctionParameter(
#     "Porosity of anode",
#     {"Through-cell distance (x_n) [m]":
#      pybamm.standard_spatial_vars.x_n}
# )

"""! Charge number of the anode interface reaction. """
zₙ = Parameter("Negative electrode electrons in reaction")
"""! Charge number of the cathode interface reaction. """
zₚ = Parameter("Positive electrode electrons in reaction")

"""! Initial electrolyte concentration. """
cₑ_dim_init = Parameter("Initial concentration in electrolyte [mol.m-3]")
"""! Non-dimensionalized initial electrolyte concentration. """
cₑ_init = cₑ_dim_init / cₑ_typ


def SOCₙ_dim_init(x):
    """! Initial SOC of the anode. """
    return FunctionParameter(
        "Initial concentration in negative electrode [mol.m-3]",
        {"Dimensionless through-cell position (x_n)": x}
    )


def SOCₙ_init(x):
    """! Non-dimensionalized initial SOC of the anode. """
    return SOCₙ_dim_init(x) / cₙ


def SOCₚ_dim_init(x):
    """! Initial SOC of the cathode. """
    return FunctionParameter(
        "Initial concentration in positive electrode [mol.m-3]",
        {"Dimensionless through-cell position (x_p)": x}
    )


def SOCₚ_init(x):
    """! Non-dimensionalized initial SOC of the cathode. """
    return SOCₚ_dim_init(x) / cₚ

# ----------------------------------------------------------------------
# "2. Function Parameters"


def Dₑ_dim(cₑ_dim, T_dim):
    """! Electrolyte diffusivity. """
    return FunctionParameter(
        "Electrolyte diffusivity [m2.s-1]",
        {
            "Electrolyte concentration [mol.m-3]": cₑ_dim,
            "Temperature [K]": T_dim,
        }
    )


"""! Reference electrolyte diffusivity for non-dimensionalization. """
Dₑ_typ = Dₑ_dim(cₑ_typ, T_ref)


def Dₑ(cₑ, T):
    """! Non-dimensionalized electrolyte diffusivity. """
    return Dₑ_dim(cₑ_typ * cₑ, ΔT * T + T_ref) / Dₑ_typ


def κₑ_dim(cₑ_dim, T_dim):
    """! Electrolyte conductivity. """
    return FunctionParameter(
        "Electrolyte conductivity [S.m-1]",
        {
            "Electrolyte concentration [mol.m-3]": cₑ_dim,
            "Temperature [K]": T_dim,
        }
    )


"""! Reference electrolyte conductivity for non-dimensionalization. """
κₑ_typ = κₑ_dim(cₑ_typ, T_ref)


def κₑ(cₑ, T):
    """! Non-dimensionalized electrolyte conductivity. """
    return κₑ_dim(cₑ_typ * cₑ, ΔT * T + T_ref) / κₑ_typ


"""! Thermal voltage divided by ionic resistance. """
κₑ_hat = (R * T_ref / F) / (C / A * L_dim / κₑ_typ)


def t_plus_dim(cₑ_dim):
    """! Transference number. """
    return FunctionParameter(
        "Cation transference number",
        {"Electrolyte concentration [mol.m3]": cₑ_dim}
    )


def t_plus(cₑ):
    """! Non-dimensionalized (referring to the input) transference number. """
    return t_plus_dim(cₑ_typ * cₑ)


def one_plus_dlnf_dlnc_dim(cₑ_dim):
    """! Thermodynamic factor. """
    return FunctionParameter(
        "1 + dlnf/dlnc",
        {"Electrolyte concentration [mol.m3]": cₑ_dim}
    )


def one_plus_dlnf_dlnc(cₑ):
    """! Non-dimensionalized (referring to the input) thermodynamic factor. """
    return one_plus_dlnf_dlnc_dim(cₑ_typ * cₑ)


def Dₙ_dim(SOCₙ, T_dim):
    """! Anode diffusivity. """
    return FunctionParameter(
        "Negative electrode diffusivity [m2.s-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Temperature [K]": T_dim,
        }
    )


"""! Reference anode diffusivity for non-dimensionalization. """
Dₙ_typ = Dₙ_dim(Scalar(0.5), T_ref)


def Dₙ(SOCₙ, T):
    """! Non-dimensionalized anode diffusivity. """
    return Dₙ_dim(SOCₙ, ΔT * T + T_ref) / Dₙ_typ


def Dₚ_dim(SOCₚ, T_dim):
    """! Cathode diffusivity. """
    return FunctionParameter(
        "Positive electrode diffusivity [m2.s-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Temperature [K]": T_dim,
        }
    )


"""! Reference cathode diffusivity for non-dimensionalization. """
Dₚ_typ = Dₚ_dim(Scalar(0.5), T_ref)


def Dₚ(SOCₚ, T):
    """! Non-dimensionalized cathode diffusivity. """
    return Dₚ_dim(SOCₚ, ΔT * T + T_ref) / Dₚ_typ


def iₛₑₙ_0_dim(cₑₙ_dim, SOCₙ_surf_dim, T_dim):
    """! Anode exchange current density. """
    return FunctionParameter(
        "Negative electrode exchange-current density [A.m-2]",
        {
            "Electrolyte concentration [mol.m-3]": cₑₙ_dim,
            "Negative particle surface concentration [mol.m-3]": SOCₙ_surf_dim,
            "Temperature [K]": T_dim,
         }
    )


"""! Reference anode exchange current density for non-dimensionalization. """
iₛₑₙ_0_ref = iₛₑₙ_0_dim(cₑ_typ, 0.5 * cₙ, T_ref)


def iₛₑₙ_0(cₑₙ, SOCₙ_surf, T):
    """! Non-dimensionalized anode exchange current density. """
    return (
        iₛₑₙ_0_dim(cₑ_typ * cₑₙ, SOCₙ_surf * cₙ, ΔT * T + T_ref)
        / iₛₑₙ_0_ref
    )


def d_cₑₙ_iₛₑₙ_0_dim(cₑₙ_dim, SOCₙ_surf_dim, T_dim):
    """! ∂ anode exchange current density / ∂ electrolyte concentration. """
    return FunctionParameter(
        "Negative electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]",
        {
            "Electrolyte concentration": cₑₙ_dim,
            "Negative particle surface concentration": SOCₙ_surf_dim,
            "Temperature": T_dim,
        }
    )


def d_cₑₙ_iₛₑₙ_0(cₑₙ, SOCₙ_surf, T):
    """! The non-dimensionalized version of the prior variable. """
    return (
        d_cₑₙ_iₛₑₙ_0_dim(cₑ_typ * cₑₙ, SOCₙ_surf * cₙ, ΔT * T + T_ref)
        * cₑ_typ
        / iₛₑₙ_0_ref
    )


def iₛₑₚ_0_dim(cₑₚ_dim, SOCₚ_surf_dim, T_dim):
    """! Cathode exchange current density. """
    return FunctionParameter(
        "Positive electrode exchange-current density [A.m-2]",
        {
            "Electrolyte concentration [mol.m-3]": cₑₚ_dim,
            "Positive particle surface concentration [mol.m-3]": SOCₚ_surf_dim,
            "Temperature [K]": T_dim,
        }
    )


"""! Reference cathode exchange current density for non-dimensionalization. """
iₛₑₚ_0_ref = iₛₑₚ_0_dim(cₑ_typ, 0.5 * cₚ, T_ref)


def iₛₑₚ_0(cₑₚ, SOCₚ_surf, T):
    """! Non-dimensionalized cathode exchange current density. """
    return (
        iₛₑₚ_0_dim(cₑ_typ * cₑₚ, SOCₚ_surf * cₚ, ΔT * T + T_ref)
        / iₛₑₚ_0_ref
    )


def d_cₑₚ_iₛₑₚ_0_dim(cₑₚ_dim, SOCₚ_surf_dim, T_dim):
    """! ∂ cathode exchange current density / ∂ electrolyte concentration. """
    return FunctionParameter(
        "Positive electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]",
        {
            "Electrolyte concentration": cₑₚ_dim,
            "Positive particle surface concentration": SOCₚ_surf_dim,
            "Temperature": T_dim,
        }
    )


def d_cₑₚ_iₛₑₚ_0(cₑₚ, SOCₚ_surf, T):
    """! The non-dimensionalized version of the prior variable. """
    return (
        d_cₑₚ_iₛₑₚ_0_dim(cₑ_typ * cₑₚ, SOCₚ_surf * cₚ, ΔT * T + T_ref)
        * cₑ_typ
        / iₛₑₚ_0_ref
    )


def dOCVₙ_dT_dim(SOCₙ):
    """! ∂ anode OCV / ∂ temperature. """
    return FunctionParameter(
        "Negative electrode OCP entropic change [V.K-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Max negative particle concentration [mol.m-3]": cₙ,
        }
    )


def dOCVₙ_dT(SOCₙ):
    """! Non-dimensionalized ∂ anode OCV / ∂ temperature. """
    return dOCVₙ_dT_dim(SOCₙ) * ΔT / thermal_voltage


def dOCVₙ_dT_dSOCₙ_dim(SOCₙ):
    """! (∂ anode OCV / ∂ temperature) / ∂ anode SOC. """
    return FunctionParameter(
        "Negative electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Max negative particle concentration [mol.m-3]": cₙ,
        }
    )


def dOCVₙ_dT_dSOCₙ(SOCₙ):
    """! Non-dimensionalized (∂ anode OCV / ∂ temperature) / ∂ anode SOC. """
    return dOCVₙ_dT_dSOCₙ_dim(SOCₙ) * ΔT / thermal_voltage


def dOCVₚ_dT_dim(SOCₚ):
    """! ∂ cathode OCV / ∂ temperature. """
    return FunctionParameter(
        "Positive electrode OCP entropic change [V.K-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Max positive particle concentration [mol.m-3]": cₚ,
        }
    )


def dOCVₚ_dT(SOCₚ):
    """! Non-dimensionalized ∂ cathode OCV / ∂ temperature. """
    return dOCVₚ_dT_dim(SOCₚ) * ΔT / thermal_voltage


def dOCVₚ_dT_dSOCₚ_dim(SOCₚ):
    """! (∂ cathode OCV / ∂ temperature) / ∂ cathode SOC. """
    return FunctionParameter(
        "Positive electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Max positive particle concentration [mol.m-3]": cₚ,
        }
    )


def dOCVₚ_dT_dSOCₚ(SOCₚ):
    """! Non-dimensionalized (∂ cathode OCV / ∂ temperature) / ∂ cathode SOC.
    """
    return dOCVₚ_dT_dSOCₚ_dim(SOCₚ) * ΔT / thermal_voltage


def OCVₙ_dim(SOCₙ, T_dim):
    """! Anode OCV. """
    OCVₙ_at_T_ref = FunctionParameter(
        "Negative electrode OCP [V]",
        {"Negative particle stoichiometry": SOCₙ}
    )
    return OCVₙ_at_T_ref + (T_dim - T_ref) * dOCVₙ_dT_dim(SOCₙ)


"""! Reference anode OCV for non-dimensionalization. """
OCVₙ_ref = OCVₙ_dim(SOCₙ_init(0), T_ref)


def OCVₙ(SOCₙ, T):
    """! Non-dimensionalized anode OCV. """
    return (OCVₙ_dim(SOCₙ, ΔT * T + T_ref) - OCVₙ_ref) / thermal_voltage


def dOCVₙ_dim_dSOCₙ(SOCₙ, T_dim):
    """! ∂ anode OCV / ∂ anode SOC. """
    dOCVₙ_dSOCₙ_at_T_ref = FunctionParameter(
        "Negative electrode OCP derivative by SOC [V]",
        {"Negative particle stoichiometry": SOCₙ}
    )
    return dOCVₙ_dSOCₙ_at_T_ref + (T_dim - T_ref) * dOCVₙ_dT_dSOCₙ_dim(SOCₙ)


def dOCVₙ_dSOCₙ(SOCₙ, T):
    """! Non-dimensionalized ∂ anode OCV / ∂ anode SOC. """
    return dOCVₙ_dim_dSOCₙ(SOCₙ, ΔT * T + T_ref) / thermal_voltage


def OCVₚ_dim(SOCₚ, T_dim):
    """! Cathode OCV. """
    OCVₚ_at_T_ref = FunctionParameter(
        "Positive electrode OCP [V]",
        {"Positive particle stoichiometry": SOCₚ}
    )
    return OCVₚ_at_T_ref + (T_dim - T_ref) * dOCVₚ_dT_dim(SOCₚ)


"""! Reference cathode OCV for non-dimensionalization. """
OCVₚ_ref = OCVₚ_dim(SOCₚ_init(1), T_ref)


def OCVₚ(SOCₚ, T):
    """! Non-dimensionalized cathode OCV. """
    return (OCVₚ_dim(SOCₚ, ΔT * T + T_ref) - OCVₚ_ref) / thermal_voltage


def dOCVₚ_dim_dSOCₚ(SOCₚ, T_dim):
    """! ∂ cathode OCV / ∂ cathode SOC. """
    dOCVₚ_dSOCₚ_at_T_ref = FunctionParameter(
        "Positive electrode OCP derivative by SOC [V]",
        {"Positive particle stoichiometry": SOCₚ}
    )
    return dOCVₚ_dSOCₚ_at_T_ref + (T_dim - T_ref) * dOCVₚ_dT_dSOCₚ_dim(SOCₚ)


def dOCVₚ_dSOCₚ(SOCₚ, T):
    """! Non-dimensionalized ∂ cathode OCV / ∂ cathode SOC. """
    return dOCVₚ_dim_dSOCₚ(SOCₚ, ΔT * T + T_ref) / thermal_voltage


# ----------------------------------------------------------------------
# "3. Scales"

"""! Non-dimensionalized specific surface area of the anode. """
aₙ = aₙ_dim * Rₙ
"""! Non-dimensionalized specific surface area of the cathode. """
aₚ = aₚ_dim * Rₚ
# For the capacity Q, apply the function np.min
# Q = pybamm.Function(
#     np.min,
#         (1 - εₙ_scalar) * Lₙ_dim * cₙ * zₙ * F * A,
#         (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A
# )
# The above doesn't work; the cathode is usually smaller anyways.
Q = (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A

"""! Discharge timescale. """
τᵈ = F * cₚ * L_dim / (C / A)
"""! Electrolyte diffusion timescale. """
τₑ = L_dim**2 / Dₑ_typ
"""! Anode diffusion timescale. """
τₙ = Rₙ**2 / Dₙ_typ
"""! Cathode diffusion timescale. """
τₚ = Rₚ**2 / Dₚ_typ
"""! Anode interface reaction timescale. """
τᵣₙ = F * cₙ / (iₛₑₙ_0_ref * aₙ_dim)
"""! Cathode interface reaction timescale. """
τᵣₚ = F * cₚ / (iₛₑₚ_0_ref * aₚ_dim)

"""! Choose the discharge timescale for non-dimensionalization. """
timescale = τᵈ
# timescale = Parameter("Typical timescale [s]")

# ----------------------------------------------------------------------
# "4. Non-Dimensionalised Parameters"

"""! Non-dimensionalized electrolyte diffusion timescale. """
Cₑ = τₑ / τᵈ
"""! Non-dimensionalized anode diffusion timescale. """
Cₙ = τₙ / τᵈ
"""! Non-dimensionalized cathode diffusion timescale. """
Cₚ = τₚ / τᵈ
"""! Non-dimensionalized anode interface reaction timescale. """
Cᵣₙ = τᵣₙ / τᵈ
"""! Non-dimensionalized cathode interface reaction timescale. """
Cᵣₚ = τᵣₚ / τᵈ

"""! Non-dimensionalized reference electrolyte concentration. """
γₑ = cₑ_typ / cₚ
"""! Non-dimensionalized maximum anode charge concentration. """
γₙ = cₙ / cₚ
"""! Non-dimensionalized cathode charge concentration. """
γₚ = cₚ / cₚ

"""! Non-dimensionalized anode thickness. """
Lₙ = Lₙ_dim / L_dim
"""! Non-dimensionlized separator thickness. """
Lₛ = Lₛ_dim / L_dim
"""! Non-dimensionalized cathode thickness. """
Lₚ = Lₚ_dim / L_dim
"""! Non-dimensionalized thicknesses for the whole cell. """
Lₑ = pybamm.Concatenation(
    pybamm.PrimaryBroadcast(Lₙ, "negative electrode"),
    pybamm.PrimaryBroadcast(Lₛ, "separator"),
    pybamm.PrimaryBroadcast(Lₚ, "positive electrode")
)

"""! Non-dimensionalized electronic conductivity of the anode. """
σₙ = (thermal_voltage / (C / A * L_dim)) * σₙ_dim
"""! Non-dimensionalized electronic conductivity of the cathode. """
σₚ = (thermal_voltage / (C / A * L_dim)) * σₚ_dim

# ----------------------------------------------------------------------
# "5. Input current and voltage"

"""! Externally applied current for galvanostatic operation (in A).
     Please note that the variable name is important for PyBaMM comp.. """
I_extern_dim = pybamm.FunctionParameter(
    "Current function [A]",
    {"Time [s]": pybamm.t * timescale}
)
"""! Non-dimensionalized external current. """
I_extern = I_extern_dim / C

# These parameters exist only for compatibility with pybamm.Simulation.
"""! Current divider. """
n_electrodes_parallel = Parameter("Number of electrodes connected in parallel"
                                  " to make a cell")
"""! Voltage multiplier. """
n_cells = Parameter("Number of cells connected in series to make a battery")
"""! Copy of the C-rate of the battery for PyBaMM compatibility. """
I_typ = C
"""! Copy of the cross-section area for PyBaMM compatibility. """
A_cc = A
"""! Copy of the external current for PyBaMM compatibility. """
current_with_time = I_extern
"""! Copy of the dimensional external current for PyBaMM compatibility. """
dimensional_current_with_time = I_extern_dim
"""! Copy of the dimensional current density for PyBaMM compatibility. """
dimensional_current_density_with_time = I_extern_dim / A
"""! Copy of the lower voltage threshold for PyBaMM compatibility. """
voltage_low_cut = Uₗ
"""! Copy of the upper voltage threshold for PyBaMM compatibility. """
voltage_high_cut = Uᵤ
"""! Cell capacity for calculating amperage from C-rates (in Ah). """
capacity = Parameter("Nominal cell capacity [A.h]")
