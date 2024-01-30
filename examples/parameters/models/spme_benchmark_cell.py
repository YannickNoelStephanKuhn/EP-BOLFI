"""!@file
Parameter file for the SPMe benchmark. Aitio2020 mentioned Moura2016,
which mentions the parameters of DUALFOIL for LiCoO₂ / graphite.
"""

from pybamm import cosh, exp, tanh


def graphite(soc):
    return (
        0.194
        + 1.5 * exp(-120.0 * soc)
        + 0.0351 * tanh((soc - 0.286) / 0.083)
        - 0.0045 * tanh((soc - 0.849) / 0.119)
        - 0.035 * tanh((soc - 0.9233) / 0.05)
        - 0.0147 * tanh((soc - 0.5) / 0.034)
        - 0.102 * tanh((soc - 0.194) / 0.142)
        - 0.022 * tanh((soc - 0.9) / 0.0164)
        - 0.011 * tanh((soc - 0.124) / 0.0226)
        + 0.0155 * tanh((soc - 0.105) / 0.029)
    )


def dsoc_graphite(soc):
    return (
        1.5 * -120.0 * exp(-120.0 * soc)
        + 0.0351 / (0.083 * cosh((soc - 0.286) / 0.083)**2)
        - 0.0045 / (0.119 * cosh((soc - 0.849) / 0.119)**2)
        - 0.035 / (0.05 * cosh((soc - 0.9233) / 0.05)**2)
        - 0.0147 / (0.034 * cosh((soc - 0.5) / 0.034)**2)
        - 0.102 / (0.142 * cosh((soc - 0.194) / 0.142)**2)
        - 0.022 / (0.0164 * cosh((soc - 0.9) / 0.0164)**2)
        - 0.011 / (0.0226 * cosh((soc - 0.124) / 0.0226)**2)
        + 0.0155 / (0.029 * cosh((soc - 0.105) / 0.029)**2)
    )


def LiCoO2(soc):
    return (
        2.16216
        + 0.07645 * tanh(30.834 - 54.4806 * soc)
        + 2.1581 * tanh(52.294 - 50.294 * soc)
        - 0.14169 * tanh(11.0923 - 19.8543 * soc)
        + 0.2051 * tanh(1.4684 - 5.4888 * soc)
        + 0.2531 * tanh((-soc + 0.56478) / 0.1316)
        - 0.02167 * tanh((soc - 0.525) / 0.006)
    )


def dsoc_LiCoO2(soc):
    return (
        0.07645 * -54.4806 / cosh(30.834 - 54.4806 * soc)**2
        + 2.1581 * -50.294 / cosh(52.294 - 50.294 * soc)**2
        - 0.14169 * -19.8543 / cosh(11.0923 - 19.8543 * soc)**2
        + 0.2051 * -5.4888 / cosh(1.4684 - 5.4888 * soc)**2
        - 0.2531 / (0.1316 * cosh((-soc + 0.56478) / 0.1316)**2)
        - 0.02167 / (0.006 * cosh((soc - 0.525) / 0.006)**2)
    )


"""! Anodic symmetry factor at the anode. """
αₙₙ = 0.5
"""! Cathodic symmetry factor at the anode. """
αₚₙ = 0.5
"""! Anodic symmetry factor at the cathode. """
αₙₚ = 0.5
"""! Cathodic symmetry factor at the cathode. """
αₚₚ = 0.5

"""!
Parameter dictionary. Initial conditions and the exchange current
densities will have been added after this file was loaded.
"""
parameters = {
    # Parameters that get estimated for the benchmark.
    "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
    "Cation transference number": 0.4,
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 1e-13,
    # Battery dimensions.
    "Negative electrode thickness [m]": 100e-6,
    "Separator thickness [m]": 25e-6,
    "Positive electrode thickness [m]": 100e-6,
    "Negative particle radius [m]": 10e-6,
    "Positive particle radius [m]": 10e-6,
    # These are superfluous scaling parameters. They are chosen such that
    # 1 C-rate is 1 A.
    "Current collector perpendicular area [m2]": 1 / 24,
    "Electrode width [m]": (1 / 24) ** 0.5,
    "Electrode height [m]": (1 / 24) ** 0.5,
    "Cell volume [m3]": 225e-6 / 24,
    # Material properties.
    "Negative electrode surface area to volume ratio [m-1]": 180000,
    "Positive electrode surface area to volume ratio [m-1]": 150000,
    "Maximum concentration in negative electrode [mol.m-3]": 24983,
    "Maximum concentration in positive electrode [mol.m-3]": 51218,
    "Negative electrode conductivity [S.m-1]": 100,
    "Positive electrode conductivity [S.m-1]": 10,
    "Electrolyte conductivity [S.m-1]": 1.1,
    "Negative electrode OCP [V]": graphite,
    "Negative electrode OCP derivative by SOC [V]": dsoc_graphite,
    "Positive electrode OCP [V]": LiCoO2,
    "Positive electrode OCP derivative by SOC [V]": dsoc_LiCoO2,
    "Negative electrode anodic charge-transfer coefficient": αₙₙ,
    "Negative electrode cathodic charge-transfer coefficient": αₚₙ,
    "Positive electrode anodic charge-transfer coefficient": αₙₚ,
    "Positive electrode cathodic charge-transfer coefficient": αₚₚ,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrode)": 1.5,
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Separator Bruggeman coefficient (electrode)": 1.5,
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Positive electrode Bruggeman coefficient (electrode)": 1.5,
    "Negative electrode porosity": 0.3,
    "Negative electrode active material volume fraction": 1 - 0.3,
    "Separator porosity": 1.0,
    "Positive electrode porosity": 0.3,
    "Positive electrode active material volume fraction": 1 - 0.3,
    # This parameter is from Danner2016.
    "Thermodynamic factor": 2.191,
    "Negative electrode electrons in reaction": 1.0,
    "Positive electrode electrons in reaction": 1.0,
    "Typical electrolyte concentration [mol.m-3]": 1000.0,
    # Operating conditions.
    "Lower voltage cut-off [V]": 3.0,
    "Upper voltage cut-off [V]": 4.5,
    "Reference temperature [K]": 298.15,
    "Ambient temperature [K]": 298.15,
    "Initial temperature [K]": 298.15,
    "Typical current [A]": 1.0,
    "Nominal cell capacity [A.h]": 0.05,
    "Current function [A]": 0.1,
    "Negative electrode OCP entropic change [V.K-1]": 0,
    "Negative electrode OCP entropic change partial derivative by SOC [V.K-1]":
        0,
    "Positive electrode OCP entropic change [V.K-1]": 0,
    "Positive electrode OCP entropic change partial derivative by SOC [V.K-1]":
        0,
    "Number of electrodes connected in parallel to make a cell": 1,
    "Number of cells connected in series to make a battery": 1,
}

"""! Maximum charge concentration in the anode active material. """
cₙ_max = parameters["Maximum concentration in negative electrode [mol.m-3]"]
"""! Maximum charge concentration in the cathode active material. """
cₚ_max = parameters["Maximum concentration in positive electrode [mol.m-3]"]

# Initial conditions.
parameters["Initial concentration in electrolyte [mol.m-3]"] = 1000.0
parameters["Initial concentration in negative electrode [mol.m-3]"] = (
    0.97 * cₙ_max
)
parameters["Initial concentration in positive electrode [mol.m-3]"] = (
    0.41 * cₚ_max
)

parameters["Negative electrode exchange-current density [A.m-2]"] = (
    lambda cₑ, cₙ, cₙ_max, T: (
        2e-5 * cₑ ** αₚₙ * cₙ ** αₙₙ * (cₙ_max - cₙ) ** αₚₙ
    )
)
parameters["Positive electrode exchange-current density [A.m-2]"] = (
    lambda cₑ, cₚ, cₚ_max, T: (
        6e-7 * cₑ ** αₙₚ * cₚ ** αₚₚ * (cₚ_max - cₚ) ** αₙₚ
    )
)

parameters[
    "Negative electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₙ, cₙ_max, T: (
        2e-5 * αₚₚ * cₑ ** (αₚₚ - 1) * cₙ ** αₙₚ * (cₙ_max - cₙ) ** αₚₚ
    )
)
parameters[
    "Positive electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₚ, cₚ_max, T: (
        6e-7 * αₚₚ * cₑ ** (αₚₚ - 1) * cₚ ** αₙₚ * (cₚ_max - cₚ) ** αₚₚ
    )
)
