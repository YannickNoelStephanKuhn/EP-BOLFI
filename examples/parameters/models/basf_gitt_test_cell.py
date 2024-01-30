"""!@file
Parameter file for the pouch cell that was measured by BASF.
"""

from .ocv_curves import OCV_graphite_precise, OCV_BASF_GITT_Test_cathode

###########################################################
# Assumptions made without justification by measurements. #
# Note: some of them are at the bottom of this file.      #
###########################################################

"""! Anodic symmetry factor at the anode. """
αₙₙ = 0.5
"""! Cathodic symmetry factor at the anode. """
αₚₙ = 0.5
"""! Anodic symmetry factor at the cathode. """
αₙₚ = 0.5
"""! Cathodic symmetry factor at the cathode. """
αₚₚ = 0.5

"""!
Parameter dictionary. The exchange current densities will have been
added after this file was loaded.
"""
parameters = {
    "Negative electrode anodic charge-transfer coefficient": αₙₙ,
    "Negative electrode cathodic charge-transfer coefficient": αₚₙ,
    "Positive electrode anodic charge-transfer coefficient": αₙₚ,
    "Positive electrode cathodic charge-transfer coefficient": αₚₚ,
    "Negative particle radius [m]": 12e-6,
    "Positive particle radius [m]": 5.5e-6,
    # These are copied from a similar battery (see Danner2016).
    "Negative electrode conductivity [S.m-1]": 10.67,
    "Positive electrode conductivity [S.m-1]": 1.07,
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 2e-15,

    ###########################################################
    # Parameters taken from measurements by hand (for now).   #
    ###########################################################

    # The effective surface area was adjusted to match a = 3 * (1 - ε) / R.
    # This ensures lithium conservation.
    "Negative electrode surface area to volume ratio [m-1]": 176991.0,
    "Positive electrode surface area to volume ratio [m-1]": 387019.0,
    # Voltage windows taken from CC-CV-cycle windows.
    "Lower voltage cut-off [V]": 2.6,  # 2.7,
    "Upper voltage cut-off [V]": 4.3,  # 4.2,
    # The measurement files state 25 °C (at the end of the header line).
    "Reference temperature [K]": 298.15,
    "Ambient temperature [K]": 298.15,
    "Initial temperature [K]": 298.15,
    # This value was adjusted to match 81.3% of the negative electrode
    # capacity to cell capacity (that's the balancing given below).
    "Maximum concentration in negative electrode [mol.m-3]": 21063.0,
    # This value was adjusted to match the positive electrode capacity to
    # cell capacity (they are one and the same at the start of cycling).
    # Slight additional adjustment to make the OCV curves fit perfectly.
    "Maximum concentration in positive electrode [mol.m-3]": 31168.0,
    ########################################################
    # Parameters extracted from experiments automatically. #
    ########################################################
    # See "ocv_from_cccv_and_gitt.py".
    "Positive electrode OCP [V]": OCV_BASF_GITT_Test_cathode,
    # These are from the 7-parameter estimation at GITT pulses 66+67.
    "Cation transference number": 0.3490477117897215,
    "Negative electrode Bruggeman coefficient (electrolyte)":
        2.723578079404257,
    "Negative electrode Bruggeman coefficient (electrode)":
        2.723578079404257,
    "Positive electrode Bruggeman coefficient (electrolyte)":
        3.06149286969267,
    "Positive electrode Bruggeman coefficient (electrode)":
        3.06149286969267,

    #######################################
    # Parameters taken to be for granted. #
    #######################################

    # This current was set as 1C in the measurement protocols.
    "Typical current [A]": 0.030083,
    # Set the base current to 0.
    "Current function [A]": 0.0,
    # Current-collector areas: 50x50 mm² cathode, 52x52 mm² anode
    "Current collector perpendicular area [m2]": 25e-4,
    "Electrode width [m]": 5e-2,
    "Electrode height [m]": 5e-2,
    "Cell volume [m3]": 25e-4 * 95e-6,
    # electrode densities:
    # anode 1.6 g/cm³ ̂= 1600 kg/m³   (mostly graphite 2.26 g/cm³),
    # cathode 3.2 g/cm³ ̂= 3200 kg/m³ (CAM+binder+add. 4.36 g/cm³)
    "Negative electrode porosity": 1.0 - 1.6 / 2.26,
    "Negative electrode active material volume fraction": 1.6 / 2.26 * 0.957,
    "Positive electrode porosity": 1.0 - 3.2 / 4.36,
    "Positive electrode active material volume fraction": 3.2 / 4.36 * 0.94,
    # Use weight, density and current-collector areas to get the lengths.
    # Cathode ACB440, ACB441: 200 mg => 8 mg/cm² ̂= 0.080 kg/m² ✓
    # Cathode ACB442, ACB443: 240 mg => 9.6 mg/cm² ̂= 0.096 kg/m²
    # Anode: 7,2 mg/cm² ̂= 0.072 kg/m² absolute
    "Negative electrode thickness [m]": 45e-6,
    "Positive electrode thickness [m]": 25e-6,
    # Separator: Celgard 2500 55x55 mm² (https://www.aotbattery.com/product/
    # Monolayer-PP-Membrane-Celgard-2500-Battery-Separator.html)
    # Tortuosity from DOI 10.1016/S0378-7753(03)00399-9
    "Separator porosity": 0.55,
    "Separator Bruggeman coefficient (electrolyte)": 3.6,
    "Separator Bruggeman coefficient (electrode)": 3.6,
    "Separator thickness [m]": 25e-6,
    # The anode consists of 95.7 % graphite.
    "Negative electrode OCP [V]": OCV_graphite_precise,
    # The electrolyte is "EC:DEC 3:7 Gew LiPF6 1 M 200 2 w%".
    # Nyman et al. (2008) report for EC:EMC 3:7 at 1 M at T ≈ 25 °C:
    # (1 - t₊) * (1 + dlnf/dlnc) ≈ 1.475
    # t₊ ≈ 0.3 ± 0.1 (replaced by EP-BOLFI estimation)
    # 1 + dlnf/dlnc has to be determined from t₊
    # Dₑ ≈ 3.69e-10 m/s²
    # κₑ ≈ 0.950 (Ωm)⁻¹
    "Thermodynamic factor": 1.475 / (1 - 0.3490477117897215),
    "Electrolyte diffusivity [m2.s-1]": 3.69e-10,
    "Electrolyte conductivity [S.m-1]": 0.950,
    "Typical electrolyte concentration [mol.m-3]": 1000.0,
    "Initial concentration in electrolyte [mol.m-3]": 1000.0,
    # The intercalation reactions most likely carry one electron.
    "Negative electrode electrons in reaction": 1.0,
    "Positive electrode electrons in reaction": 1.0,
    # The temperature is fixed.
    "Negative electrode OCP entropic change [V.K-1]": 0,
    "Negative electrode OCP entropic change partial derivative by SOC [V.K-1]":
        0,
    "Positive electrode OCP entropic change [V.K-1]": 0,
    "Positive electrode OCP entropic change partial derivative by SOC [V.K-1]":
        0,
    # The cell is a single pouch cell.
    "Number of electrodes connected in parallel to make a cell": 1,
    "Number of cells connected in series to make a battery": 1,
}

# Cell capacity matched to CC-cycle data gives 0.03965 Ah.
parameters["Nominal cell capacity [A.h]"] = 0.03965
# Cell capacity from cathode capacity.
"""
parameters["Cell capacity [A.h]"] = (
    (1 - parameters["Positive electrode porosity"])
    * parameters["Positive electrode thickness [m]"]
    * parameters["Maximum concentration in positive electrode [mol.m-3]"]
    * parameters["Positive electrode electrons in reaction"]
    * 96485.33212
    * parameters["Current collector perpendicular area [m2]"]
    / 3600
)
"""

###########################################################
# Assumptions made without justification by measurements. #
###########################################################

# The exchange-current densities are taken from a similar battery.

"""! Maximum charge concentration in the anode active material. """
cₙ_max = parameters["Maximum concentration in negative electrode [mol.m-3]"]
"""! Maximum charge concentration in the cathode active material. """
cₚ_max = parameters["Maximum concentration in positive electrode [mol.m-3]"]

parameters["Negative electrode exchange-current density [A.m-2]"] = (
    lambda cₑ, cₙ, cₙ_max, T: (
        3.67e-6 * cₑ ** αₚₙ * cₙ ** αₙₙ * (cₙ_max - cₙ) ** αₚₙ
    )
)
parameters["Positive electrode exchange-current density [A.m-2]"] = (
    lambda cₑ, cₚ, cₚ_max, T: (
        5.06e-6 * cₑ ** αₚₙ * cₚ ** αₙₙ * (cₚ_max - cₚ) ** αₚₙ
    )
)

parameters[
    "Negative electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₙ, cₙ_max, T: 3.67e-6
    * αₚₚ
    * cₑ ** (αₚₚ - 1)
    * cₙ**αₙₚ
    * (cₙ_max - cₙ) ** αₚₚ
)
parameters[
    "Positive electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₚ, cₚ_max, T: 5.06e-6
    * αₚₚ
    * cₑ ** (αₚₚ - 1)
    * cₚ**αₙₚ
    * (cₚ_max - cₚ) ** αₚₚ
)

###########################################################
# Parameters taken from measurements by hand (for now).   #
###########################################################


def negative_SOC_from_cell_SOC(cell_SOC):
    """!
    Estimated relationship between cell SOC and negative electrode SOC.
    Note: "cell SOC = 0" is defined as total delithiation of the
    positive electrode. This is more commonly denoted as "SOD".
    (0.07, 0.67) is the range of the graphite SOC that we fitted by hand
    in "ocv_from_cccv_and_gitt.py" to the (0.0, 1.0) range of the data.
    (0.232, 0.97) is the range of that data (the GITT pulses) within the
    estimated SOC range of the cell, which is 0.03965 Ah (from CC data).
    Finally, consider the sign conventions, which explain 0.97 -> 0.03.
    ([0.67, 0.07] - 0.03) / (0.97 - 0.232) = [0.867, 0.054]
    Note: cell_SOC is 0 for the fully charged cell and 1 for the fully
    discharged cell.
    """
    return 0.867 - cell_SOC * (0.867 - 0.054)


def cell_SOC_from_negative_SOC(negative_SOC):
    """! Estimated relationship between negative electrode SOC and cell SOC.
    """
    return (0.867 - negative_SOC) / (0.867 - 0.054)


########################################################
# Parameters extracted from experiments automatically. #
########################################################


def positive_SOC_from_cell_SOC(cell_SOC):
    """!
    Estimated relationship between cell SOC and positive electrode SOC.
    ([0.18073791, 0.96526985] - 0.232) / (0.97 - 0.232) = [-0.069, 0.994]
    """
    return -0.069 + cell_SOC * (0.994 - -0.069)


def cell_SOC_from_positive_SOC(positive_SOC):
    """! Estimated relationship between positive electrode SOC and cell SOC.
    """
    return (positive_SOC - -0.069) / (0.994 - -0.069)


# These are the fit parameters for the positive electrode OCV curve.
E_0 = [
    3.9863272907672287,
    3.9770716502364394,
    3.908636968226356,
    3.7234749637590765,
    3.673179452264688,
    3.6408841257157816,
]
a = [
    -26.627154259595873,
    -1.292777296589606,
    -0.1638198787655928,
    -0.777064236034688,
    -4.268952015290985,
    -2.8557091327913606,
]
Δx = [
    0.024115021125137043,
    0.041854701383275045,
    0.7000584168144725,
    0.0990843265896007,
    0.01814215130772722,
    0.11674538277978741,
]

"""! Fit parameters of the OCV of the positive electrode. """
positive_electrode_OCV_fit = [p[i] for i in range(6) for p in [E_0, a, Δx]]
