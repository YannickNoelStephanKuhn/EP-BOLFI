"""
Comprehensive list of parameters of every model.

SI units are assumed unless stated otherwise. When imported, this
package turns into the list of variables contained in here.
The "syntactic sugar" with the lower and upper indexed letters is
treated by Python to be the same as their normal counterparts.
E.g., "aₙ" and "an" refer to the exact same variable.
Greek letters aren't converted, unless they are also indexed, in which
case the same as before applies: "εᵝ" is the same as "εβ".

Function parameters are defined in accordance with PyBaMM and noted with
their units in the code here, so for their documentation, look there.
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
R = Scalar(constants.R)
"""Gas constant."""
F = Scalar(constants.physical_constants["Faraday constant"][0])
"""Faraday constant."""
k_B = constants.physical_constants["Boltzmann constant"][0]
"""Boltzmann constant."""
qₑ = constants.physical_constants["electron volt"][0]
"""Electron volt."""
ɛ_0 = constants.physical_constants["vacuum electric permittivity"][0]
"""Vacuum permittivity."""

T_ref = Parameter("Reference temperature [K]")
"""Reference temperature for non-dimensionalization."""
T_init = Parameter("Initial temperature [K]")
"""Initial temperature."""
thermal_voltage = R * T_ref / F
ΔT = Scalar(1)
"""Typical temperature rise."""

Lₙ_dim = Parameter("Negative electrode thickness [m]")
"""Anode thickness."""
Lₛ_dim = Parameter("Separator thickness [m]")
"""Separator thickness."""
Lₚ_dim = Parameter("Positive electrode thickness [m]")
"""Cathode thickness."""
L_dim = Lₙ_dim + Lₛ_dim + Lₚ_dim
"""Cell thickness."""
A = Parameter("Current collector perpendicular area [m2]")
"""Cross-section area of the cell."""
V = Parameter("Cell volume [m3]")
"""Volume of the cell. Left as a separate Parameter for compatibility."""
L_x = L_dim
"""PyBaMM-compatible name for the cell thickness."""
L_y = Parameter("Electrode width [m]")
"""Width of the electrode for PyBaMM compatibility."""
L_z = Parameter("Electrode height [m]")
"""Height of the electrode for PyBaMM compatibility."""

C = Parameter("Typical current [A]")
"""C-rate of the battery (in A)."""
Uₗ = pybamm.Parameter("Lower voltage cut-off [V]")
"""Lower threshold for the cell voltage."""
Uᵤ = pybamm.Parameter("Upper voltage cut-off [V]")
"""Upper threshold for the cell voltage."""

cₑ_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
"""Reference electrolyte concentration for non-dimensionalization."""

cₙ = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
"""Maximum lithium concentration in the anode active material."""
cₚ = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
"""Maximum lithium concentration in the cathode active material."""
σₙ_dim = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
"""Electronic conductivity of the anode."""
σₚ_dim = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
"""Electronic conductivity of the cathode."""

aₙ_dim = Parameter("Negative electrode surface area to volume ratio [m-1]")
"""Specific surface area of the anode."""
aₚ_dim = Parameter("Positive electrode surface area to volume ratio [m-1]")
"""Specific surface area of the cathode."""
Rₙ = Parameter("Negative particle radius [m]")
"""Mean/Median radius of the anode particles."""
Rₚ = Parameter("Positive particle radius [m]")
"""Mean/Median radius of the cathode particles."""
αₙₙ = Parameter("Negative electrode anodic charge-transfer coefficient")
"""Anodic symmetry factor for the anode interface reaction."""
αₚₙ = Parameter("Negative electrode cathodic charge-transfer coefficient")
"""Cathodic symmetry factor for the anode interface reaction."""
αₙₚ = Parameter("Positive electrode anodic charge-transfer coefficient")
"""Anodic symmetry factor for the cathode interface reaction."""
αₚₚ = Parameter("Positive electrode cathodic charge-transfer coefficient")
"""Cathodic symmetry factor for the cathode interface reaction."""
βₑₙ_scalar = Parameter(
    "Negative electrode Bruggeman coefficient (electrolyte)"
)
"""Bruggeman coefficient for the electrolyte in the anode (scalar)."""
βₑₛ_scalar = Parameter("Separator Bruggeman coefficient (electrolyte)")
"""
Bruggeman coefficient for the electrolyte in the separator (scalar).
"""
βₑₚ_scalar = Parameter(
    "Positive electrode Bruggeman coefficient (electrolyte)"
)
"""Bruggeman coefficient for the electrolyte in the cathode (scalar)."""
βₑₙ = pybamm.PrimaryBroadcast(βₑₙ_scalar, "negative electrode")
"""Bruggeman coefficient for the electrolyte in the anode (vector)."""
βₑₛ = pybamm.PrimaryBroadcast(βₑₛ_scalar, "separator")
"""
Bruggeman coefficient for the electrolyte in the separator (vector).
"""
βₑₚ = pybamm.PrimaryBroadcast(βₑₚ_scalar, "positive electrode")
"""Bruggeman coefficient for the electrolyte in the cathode (vector)."""
βₛₙ_scalar = Parameter("Negative electrode Bruggeman coefficient (electrode)")
"""Bruggeman coefficient for the solid part in the anode."""
βₛₛ_scalar = Parameter("Separator Bruggeman coefficient (electrode)")
"""Bruggeman coefficient for the solid part in the seperator."""
βₛₚ_scalar = Parameter("Positive electrode Bruggeman coefficient (electrode)")
"""Bruggeman coefficient for the solid part in the cathode."""
βₑ = pybamm.Concatenation(βₑₙ, βₑₛ, βₑₚ)
"""Bruggeman coefficient for the electrolyte in the whole cell."""
εₙ_scalar = Parameter("Negative electrode porosity")
"""Porosity of the anode (scalar)."""
εₛ_scalar = Parameter("Separator porosity")
"""Porosity of the separator (scalar)."""
εₚ_scalar = Parameter("Positive electrode porosity")
"""Porosity of the cathode (scalar)."""
εₙ = pybamm.PrimaryBroadcast(εₙ_scalar, "negative electrode")
"""Porosity of the anode (vector)."""
εₛ = pybamm.PrimaryBroadcast(εₛ_scalar, "separator")
"""Porosity of the separator (vector)."""
εₚ = pybamm.PrimaryBroadcast(εₚ_scalar, "positive electrode")
"""Porosity of the cathode (vector)."""
ε = pybamm.Concatenation(εₙ, εₛ, εₚ)
"""Porosity of the whole cell."""
εᵝ = ε**βₑ
"""Tortuosity of the whole cell."""

# Example on how to make εₑₙ a Function Parameter:
# εₑₙ = pybamm.FunctionParameter(
#     "Porosity of negative electrode",
#     {"Through-cell distance (x_n) [m]":
#      pybamm.standard_spatial_vars.x_n}
# )

zₙ = Parameter("Negative electrode electrons in reaction")
"""Charge number of the anode interface reaction."""
zₚ = Parameter("Positive electrode electrons in reaction")
"""Charge number of the cathode interface reaction."""

zₙ_salt = Parameter("Charge number of anion in electrolyte salt dissociation")
"""Charge number of the salt dissociation (anion)."""
zₚ_salt = Parameter("Charge number of cation in electrolyte salt dissociation")
"""Charge number of the salt dissociation (cation)."""
nₙ = Parameter("Stoichiometry of anion in electrolyte salt dissociation")
"""Stoichiometric coefficient of the salt dissociation (anion)."""
nₚ = Parameter("Stoichiometry of cation in electrolyte salt dissociation")
"""Stoichiometric coefficient of the salt dissociation (cation)."""

cₑ_dim_init = Parameter("Initial concentration in electrolyte [mol.m-3]")
"""Initial electrolyte concentration."""
cₑ_init = cₑ_dim_init / cₑ_typ
"""Non-dimensionalized initial electrolyte concentration."""

ρₑ = Parameter("Mass density of electrolyte [kg.m-3]")
"""Electrolyte mass density."""
ρₑ_plus = Parameter("Mass density of cations in electrolyte [kg.m-3]")
"""Cation mass density in electrolyte."""
M_N = Parameter("Molar mass of electrolyte solvent [kg.mol-1]")
"""Solvent molar mass."""
c_N = Parameter("Solvent concentration [mol.m-3]")
"""Solvent concentration."""
ρ_N = M_N * c_N
"""Solvent mass density."""
v_N = Parameter("Partial molar volume of electrolyte solvent [m3.mol-1]")
"""Solvent partial molar volume."""
tilde_ρ_N = M_N / v_N
"""Solvent partial mass density (i.e., weight change per volume)."""

# Double-Layer parameters.
C_DLₙ = Parameter("Negative electrode double-layer capacity [F.m-2]")
"""Negative electrode double-layer capacity."""
C_DLₚ = Parameter("Positive electrode double-layer capacity [F.m-2]")
"""Positive electrode double-layer capacity."""

# SEI parameters. Only implemented for the negative electrode, as we
# neglect positive electrode materials with high enough potential.
L_SEI = Parameter("SEI thickness [m]")
"""Thickness of the SEI."""
permittivity_SEI = Parameter("SEI relative permittivity")
"""Relative permittivity of the SEI."""
C_SEI = ɛ_0 * permittivity_SEI / L_SEI
"""Capacitance of the SEI."""
t_SEI_minus = Parameter("Anion transference number in SEI")
"""
Transference number of anions in the SEI (with reference to the
center-of-mass velocity).
"""
ε_SEI = Parameter("SEI porosity")
"""Porosity of the SEI."""
β_SEI = Parameter("SEI Bruggeman coefficient")
"""Bruggeman coefficient determining the tortuosity of the SEI."""
κ_SEI = Parameter("SEI ionic conductivity [S.m-1]")
"""Ionic conductivity of the SEI."""
R_SEI = L_SEI / (ε_SEI**β_SEI * κ_SEI)
"""Resistance of the SEI."""
L_electrolyte_for_SEI_model = (((Lₙ_dim + Lₚ_dim) / 2) + Lₛ_dim) / 2
"""Length-scale for bulk electrolyte contribution in SEI model."""

# ----------------------------------------------------------------------
# "2. Function Parameters"


def SOCₙ_dim_init(x):
    """Initial SOC of the anode."""
    return FunctionParameter(
        "Initial concentration in negative electrode [mol.m-3]",
        {"Dimensionless through-cell position (x_n)": x}
    )


def SOCₙ_init(x):
    """Non-dimensionalized initial SOC of the anode."""
    return SOCₙ_dim_init(x) / cₙ


def SOCₚ_dim_init(x):
    """Initial SOC of the cathode."""
    return FunctionParameter(
        "Initial concentration in positive electrode [mol.m-3]",
        {"Dimensionless through-cell position (x_p)": x}
    )


def SOCₚ_init(x):
    """Non-dimensionalized initial SOC of the cathode."""
    return SOCₚ_dim_init(x) / cₚ


def Dₑ_dim(cₑ_dim, T_dim):
    """Electrolyte diffusivity."""
    return FunctionParameter(
        "Electrolyte diffusivity [m2.s-1]",
        {
            "Electrolyte concentration [mol.m-3]": cₑ_dim,
            "Temperature [K]": T_dim,
        }
    )


Dₑ_typ = Dₑ_dim(cₑ_typ, T_ref)
"""Reference electrolyte diffusivity for non-dimensionalization."""


def Dₑ(cₑ, T):
    """Non-dimensionalized electrolyte diffusivity."""
    return Dₑ_dim(cₑ_typ * cₑ, ΔT * T + T_ref) / Dₑ_typ


def κₑ_dim(cₑ_dim, T_dim):
    """Electrolyte conductivity."""
    return FunctionParameter(
        "Electrolyte conductivity [S.m-1]",
        {
            "Electrolyte concentration [mol.m-3]": cₑ_dim,
            "Temperature [K]": T_dim,
        }
    )


κₑ_typ = κₑ_dim(cₑ_typ, T_ref)
"""Reference electrolyte conductivity for non-dimensionalization."""


def κₑ(cₑ, T):
    """Non-dimensionalized electrolyte conductivity."""
    return κₑ_dim(cₑ_typ * cₑ, ΔT * T + T_ref) / κₑ_typ


κₑ_hat = (R * T_ref / F) / (C / A * L_dim / κₑ_typ)
"""Thermal voltage divided by ionic resistance."""


def t_plus_dim(cₑ_dim):
    """Transference number."""
    return FunctionParameter(
        "Cation transference number",
        {"Electrolyte concentration [mol.m3]": cₑ_dim}
    )


def t_plus(cₑ):
    """
    Non-dimensionalized (referring to the input) transference number.
    """
    return t_plus_dim(cₑ_typ * cₑ)


def one_plus_dlnf_dlnc_dim(cₑ_dim):
    """! Thermodynamic factor. """
    return FunctionParameter(
        "Thermodynamic factor",
        {"Electrolyte concentration [mol.m3]": cₑ_dim}
    )


def one_plus_dlnf_dlnc(cₑ):
    """
    Non-dimensionalized (referring to the input) thermodynamic factor.
    """
    return one_plus_dlnf_dlnc_dim(cₑ_typ * cₑ)


def Dₙ_dim(SOCₙ, T_dim):
    """Anode diffusivity."""
    return FunctionParameter(
        "Negative particle diffusivity [m2.s-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Temperature [K]": T_dim,
        }
    )


Dₙ_typ = Dₙ_dim(SOCₙ_init(0), T_ref)
"""Reference anode diffusivity for non-dimensionalization."""


def Dₙ(SOCₙ, T):
    """Non-dimensionalized anode diffusivity."""
    return Dₙ_dim(SOCₙ, ΔT * T + T_ref) / Dₙ_typ


def Dₚ_dim(SOCₚ, T_dim):
    """Cathode diffusivity."""
    return FunctionParameter(
        "Positive particle diffusivity [m2.s-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Temperature [K]": T_dim,
        }
    )


Dₚ_typ = Dₚ_dim(SOCₚ_init(1), T_ref)
"""Reference cathode diffusivity for non-dimensionalization."""


def Dₚ(SOCₚ, T):
    """Non-dimensionalized cathode diffusivity."""
    return Dₚ_dim(SOCₚ, ΔT * T + T_ref) / Dₚ_typ


def iₛₑₙ_0_dim(cₑₙ_dim, SOCₙ_surf_dim, cₙ_max, T_dim):
    """Anode exchange current density."""
    return FunctionParameter(
        "Negative electrode exchange-current density [A.m-2]",
        {
            "Electrolyte concentration [mol.m-3]": cₑₙ_dim,
            "Negative particle surface concentration [mol.m-3]": SOCₙ_surf_dim,
            "Maximum concentration in negative electrode [mol.m-3]": cₙ_max,
            "Temperature [K]": T_dim,
        }
    )


iₛₑₙ_0_ref = iₛₑₙ_0_dim(cₑ_typ, SOCₙ_init(0) * cₙ, cₙ, T_ref)
"""
Reference anode exchange current density for non-dimensionalization.
"""


def iₛₑₙ_0(cₑₙ, SOCₙ_surf, cₙ_max, T):
    """Non-dimensionalized anode exchange current density."""
    return (
        iₛₑₙ_0_dim(cₑ_typ * cₑₙ, SOCₙ_surf * cₙ, cₙ_max, ΔT * T + T_ref)
        / iₛₑₙ_0_ref
    )


def d_cₑₙ_iₛₑₙ_0_dim(cₑₙ_dim, SOCₙ_surf_dim, cₙ_max, T_dim):
    """
    ∂ anode exchange current density / ∂ electrolyte concentration.
    """
    return FunctionParameter(
        "Negative electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]",
        {
            "Electrolyte concentration": cₑₙ_dim,
            "Negative particle surface concentration": SOCₙ_surf_dim,
            "Maximum concentration in negative electrode [mol.m-3]": cₙ_max,
            "Temperature": T_dim,
        }
    )


def d_cₑₙ_iₛₑₙ_0(cₑₙ, SOCₙ_surf, cₙ_max, T):
    """
    The non-dimensionalized version of the prior variable.
    """
    return (
        d_cₑₙ_iₛₑₙ_0_dim(cₑ_typ * cₑₙ, SOCₙ_surf * cₙ, cₙ_max, ΔT * T + T_ref)
        * cₑ_typ
        / iₛₑₙ_0_ref
    )


def iₛₑₚ_0_dim(cₑₚ_dim, SOCₚ_surf_dim, cₚ_max, T_dim):
    """
    Cathode exchange current density.
    """
    return FunctionParameter(
        "Positive electrode exchange-current density [A.m-2]",
        {
            "Electrolyte concentration [mol.m-3]": cₑₚ_dim,
            "Positive particle surface concentration [mol.m-3]": SOCₚ_surf_dim,
            "Maximum concentration in positive electrode [mol.m-3]": cₚ_max,
            "Temperature [K]": T_dim,
        }
    )


iₛₑₚ_0_ref = iₛₑₚ_0_dim(cₑ_typ, SOCₚ_init(1) * cₚ, cₚ, T_ref)
"""
Reference cathode exchange current density for non-dimensionalization.
"""


def iₛₑₚ_0(cₑₚ, SOCₚ_surf, cₚ_max, T):
    """Non-dimensionalized cathode exchange current density."""
    return (
        iₛₑₚ_0_dim(cₑ_typ * cₑₚ, SOCₚ_surf * cₚ, cₚ_max, ΔT * T + T_ref)
        / iₛₑₚ_0_ref
    )


def d_cₑₚ_iₛₑₚ_0_dim(cₑₚ_dim, SOCₚ_surf_dim, cₚ_max, T_dim):
    """
    ∂ cathode exchange current density / ∂ electrolyte concentration.
    """
    return FunctionParameter(
        "Positive electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]",
        {
            "Electrolyte concentration": cₑₚ_dim,
            "Positive particle surface concentration": SOCₚ_surf_dim,
            "Maximum concentration in positive electrode [mol.m-3]": cₚ_max,
            "Temperature": T_dim,
        }
    )


def d_cₑₚ_iₛₑₚ_0(cₑₚ, SOCₚ_surf, cₚ_max, T):
    """The non-dimensionalized version of the prior variable."""
    return (
        d_cₑₚ_iₛₑₚ_0_dim(cₑ_typ * cₑₚ, SOCₚ_surf * cₚ, cₚ_max, ΔT * T + T_ref)
        * cₑ_typ
        / iₛₑₚ_0_ref
    )


def dOCVₙ_dT_dim(SOCₙ):
    """∂ anode OCV / ∂ temperature."""
    return FunctionParameter(
        "Negative electrode OCP entropic change [V.K-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Max negative particle concentration [mol.m-3]": cₙ,
        }
    )


def dOCVₙ_dT(SOCₙ):
    """Non-dimensionalized ∂ anode OCV / ∂ temperature."""
    return dOCVₙ_dT_dim(SOCₙ) * ΔT / thermal_voltage


def dOCVₙ_dT_dSOCₙ_dim(SOCₙ):
    """(∂ anode OCV / ∂ temperature) / ∂ anode SOC."""
    return FunctionParameter(
        "Negative electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]",
        {
            "Negative particle stoichiometry": SOCₙ,
            "Max negative particle concentration [mol.m-3]": cₙ,
        }
    )


def dOCVₙ_dT_dSOCₙ(SOCₙ):
    """
    Non-dimensionalized (∂ anode OCV / ∂ temperature) / ∂ anode SOC.
    """
    return dOCVₙ_dT_dSOCₙ_dim(SOCₙ) * ΔT / thermal_voltage


def dOCVₚ_dT_dim(SOCₚ):
    """∂ cathode OCV / ∂ temperature."""
    return FunctionParameter(
        "Positive electrode OCP entropic change [V.K-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Max positive particle concentration [mol.m-3]": cₚ,
        }
    )


def dOCVₚ_dT(SOCₚ):
    """Non-dimensionalized ∂ cathode OCV / ∂ temperature."""
    return dOCVₚ_dT_dim(SOCₚ) * ΔT / thermal_voltage


def dOCVₚ_dT_dSOCₚ_dim(SOCₚ):
    """(∂ cathode OCV / ∂ temperature) / ∂ cathode SOC."""
    return FunctionParameter(
        "Positive electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]",
        {
            "Positive particle stoichiometry": SOCₚ,
            "Max positive particle concentration [mol.m-3]": cₚ,
        }
    )


def dOCVₚ_dT_dSOCₚ(SOCₚ):
    """
    Non-dimensionalized (∂ cathode OCV / ∂ temperature) / ∂ cathode SOC.
    """
    return dOCVₚ_dT_dSOCₚ_dim(SOCₚ) * ΔT / thermal_voltage


def OCVₙ_dim(SOCₙ, T_dim):
    """Anode OCV."""
    OCVₙ_at_T_ref = FunctionParameter(
        "Negative electrode OCP [V]",
        {"Negative particle stoichiometry": SOCₙ}
    )
    return OCVₙ_at_T_ref + (T_dim - T_ref) * dOCVₙ_dT_dim(SOCₙ)


OCVₙ_ref = OCVₙ_dim(SOCₙ_init(0), T_ref)
"""Reference anode OCV for non-dimensionalization."""


def OCVₙ(SOCₙ, T):
    """Non-dimensionalized anode OCV."""
    return (OCVₙ_dim(SOCₙ, ΔT * T + T_ref) - OCVₙ_ref) / thermal_voltage


def dOCVₙ_dim_dSOCₙ(SOCₙ, T_dim):
    """∂ anode OCV / ∂ anode SOC."""
    dOCVₙ_dSOCₙ_at_T_ref = FunctionParameter(
        "Negative electrode OCP derivative by SOC [V]",
        {"Negative particle stoichiometry": SOCₙ}
    )
    return dOCVₙ_dSOCₙ_at_T_ref + (T_dim - T_ref) * dOCVₙ_dT_dSOCₙ_dim(SOCₙ)


def dOCVₙ_dSOCₙ(SOCₙ, T):
    """Non-dimensionalized ∂ anode OCV / ∂ anode SOC."""
    return dOCVₙ_dim_dSOCₙ(SOCₙ, ΔT * T + T_ref) / thermal_voltage


def OCVₚ_dim(SOCₚ, T_dim):
    """Cathode OCV."""
    OCVₚ_at_T_ref = FunctionParameter(
        "Positive electrode OCP [V]",
        {"Positive particle stoichiometry": SOCₚ}
    )
    return OCVₚ_at_T_ref + (T_dim - T_ref) * dOCVₚ_dT_dim(SOCₚ)


OCVₚ_ref = OCVₚ_dim(SOCₚ_init(1), T_ref)
"""Reference cathode OCV for non-dimensionalization."""


def OCVₚ(SOCₚ, T):
    """Non-dimensionalized cathode OCV."""
    return (OCVₚ_dim(SOCₚ, ΔT * T + T_ref) - OCVₚ_ref) / thermal_voltage


def dOCVₚ_dim_dSOCₚ(SOCₚ, T_dim):
    """∂ cathode OCV / ∂ cathode SOC."""
    dOCVₚ_dSOCₚ_at_T_ref = FunctionParameter(
        "Positive electrode OCP derivative by SOC [V]",
        {"Positive particle stoichiometry": SOCₚ}
    )
    return dOCVₚ_dSOCₚ_at_T_ref + (T_dim - T_ref) * dOCVₚ_dT_dSOCₚ_dim(SOCₚ)


def dOCVₚ_dSOCₚ(SOCₚ, T):
    """Non-dimensionalized ∂ cathode OCV / ∂ cathode SOC."""
    return dOCVₚ_dim_dSOCₚ(SOCₚ, ΔT * T + T_ref) / thermal_voltage


# ----------------------------------------------------------------------
# "3. Scales"

aₙ = aₙ_dim * Rₙ
"""Non-dimensionalized specific surface area of the anode."""
aₚ = aₚ_dim * Rₚ
"""Non-dimensionalized specific surface area of the cathode."""
# For the capacity Q, apply the function np.min
# Q = pybamm.Function(
#     np.min,
#         (1 - εₙ_scalar) * Lₙ_dim * cₙ * zₙ * F * A,
#         (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A
# )
# The above doesn't work; the cathode is usually smaller anyways.
Q = (1 - εₚ_scalar) * Lₚ_dim * cₚ * zₚ * F * A

τᵈ = F * cₚ * L_dim / (C / A)
"""Discharge timescale."""
τₑ = L_dim**2 / Dₑ_typ
"""Electrolyte diffusion timescale."""
τₙ = Rₙ**2 / Dₙ_typ
"""Anode diffusion timescale."""
τₚ = Rₚ**2 / Dₚ_typ
"""Cathode diffusion timescale."""
τᵣₙ = F * cₙ / (iₛₑₙ_0_ref * aₙ_dim)
"""Anode interface reaction timescale."""
τᵣₚ = F * cₚ / (iₛₑₚ_0_ref * aₚ_dim)
"""Cathode interface reaction timescale."""

timescale = τᵈ
"""Choose the discharge timescale for non-dimensionalization."""
# timescale = Parameter("Typical timescale [s]")

# ----------------------------------------------------------------------
# "4. Non-Dimensionalised Parameters"

Cₑ = τₑ / τᵈ
"""Non-dimensionalized electrolyte diffusion timescale."""
Cₙ = τₙ / τᵈ
"""Non-dimensionalized anode diffusion timescale."""
Cₚ = τₚ / τᵈ
"""Non-dimensionalized cathode diffusion timescale."""
Cᵣₙ = τᵣₙ / τᵈ
"""Non-dimensionalized anode interface reaction timescale."""
Cᵣₚ = τᵣₚ / τᵈ
"""Non-dimensionalized cathode interface reaction timescale."""

γₑ = cₑ_typ / cₚ
"""Non-dimensionalized reference electrolyte concentration."""
γₙ = cₙ / cₚ
"""Non-dimensionalized maximum anode charge concentration."""
γₚ = cₚ / cₚ
"""Non-dimensionalized cathode charge concentration."""

Lₙ = Lₙ_dim / L_dim
"""Non-dimensionalized anode thickness."""
Lₛ = Lₛ_dim / L_dim
"""Non-dimensionlized separator thickness."""
Lₚ = Lₚ_dim / L_dim
"""Non-dimensionalized cathode thickness."""
Lₑ = pybamm.Concatenation(
    pybamm.PrimaryBroadcast(Lₙ, "negative electrode"),
    pybamm.PrimaryBroadcast(Lₛ, "separator"),
    pybamm.PrimaryBroadcast(Lₚ, "positive electrode")
)
"""Non-dimensionalized thicknesses for the whole cell."""

σₙ = (thermal_voltage / (C / A * L_dim)) * σₙ_dim
"""Non-dimensionalized electronic conductivity of the anode."""
σₚ = (thermal_voltage / (C / A * L_dim)) * σₚ_dim
"""Non-dimensionalized electronic conductivity of the cathode."""

# ----------------------------------------------------------------------
# "5. Input current and voltage"

I_extern_dim = pybamm.FunctionParameter(
    "Current function [A]",
    {"Time [s]": pybamm.t * timescale}
)
"""
Externally applied current for galvanostatic operation (in A).
Please note that the variable name is important for PyBaMM comp..
"""
I_extern = I_extern_dim / C
"""Non-dimensionalized external current."""

# These parameters exist only for compatibility with pybamm.Simulation.
n_electrodes_parallel = Parameter(
    "Number of electrodes connected in parallel to make a cell"
)
"""Current divider."""
n_cells = Parameter("Number of cells connected in series to make a battery")
"""Voltage multiplier."""
I_typ = C
"""Copy of the C-rate of the battery for PyBaMM compatibility."""
A_cc = A
"""Copy of the cross-section area for PyBaMM compatibility."""
current_with_time = I_extern
"""Copy of the external current for PyBaMM compatibility."""
dimensional_current_with_time = I_extern_dim
"""Copy of the dimensional external current for PyBaMM compatibility."""
dimensional_current_density_with_time = I_extern_dim / A
"""Copy of the dimensional current density for PyBaMM compatibility."""
voltage_low_cut = Uₗ
"""Copy of the lower voltage threshold for PyBaMM compatibility."""
voltage_high_cut = Uᵤ
"""Copy of the upper voltage threshold for PyBaMM compatibility."""
capacity = Parameter("Nominal cell capacity [A.h]")
"""Cell capacity for calculating amperage from C-rates (in Ah)."""
