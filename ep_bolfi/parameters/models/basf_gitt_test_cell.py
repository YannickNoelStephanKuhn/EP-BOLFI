# Copyright (c): German Aerospace Center (DLR)
"""!@file
Parameter file for the pouch cell that was measured by BASF.
"""


def OCV_graphite_precise(SOC):
    """! OCV curve for graphite as seen in "A Parametric OCV Model". """
    return (
        (SOC < 0.0049839679358717436)
        * (
            (1871.6054394858402 * SOC + -40.372835438366714) * SOC
            + 0.8294391674436975
        )
        + (SOC >= 0.0049839679358717436)
        * (SOC < 0.008967935871743486)
        * (
            (1029.906653054506 * SOC + -31.982835911894924) * SOC
            + 0.80853142313274
        )
        + (SOC >= 0.008967935871743486)
        * (SOC < 0.016935871743486975)
        * (
            (306.9773407111552 * SOC + -19.016468486097367) * SOC
            + 0.7503906473507321
        )
        + (SOC >= 0.016935871743486975)
        * (SOC < 0.03287174348697395)
        * (
            (52.808351615653464 * SOC + -10.40732168511104) * SOC
            + 0.6774889443295539
        )
        + (SOC >= 0.03287174348697395)
        * (SOC < 0.04880761523046093)
        * (
            (55.64682404789602 * SOC + -10.59393276048607) * SOC
            + 0.6805560600303321
        )
        + (SOC >= 0.04880761523046093)
        * (SOC < 0.0647434869739479)
        * (
            (118.49687813361328 * SOC + -16.729055274544773) * SOC
            + 0.8302764095592897
        )
        + (SOC >= 0.0647434869739479)
        * (SOC < 0.09661523046092185)
        * (
            (11.767461240166142 * SOC + -2.9089860497888793) * SOC
            + 0.3828966736432695
        )
        + (SOC >= 0.09661523046092185)
        * (SOC < 0.12649498997995992)
        * (
            (3.1136066538851708 * SOC + -1.236797739331184) * SOC
            + 0.3021172441488046
        )
        + (SOC >= 0.12649498997995992)
        * (SOC < 0.18824649298597196)
        * (
            (0.6521212178052025 * SOC + -0.6140665881856844) * SOC
            + 0.26273105878662495
        )
        + (SOC >= 0.18824649298597196)
        * (SOC < 0.24999799599198397)
        * (
            (-1.4320456098983918 * SOC + 0.1706076040401161) * SOC
            + 0.18887497637507134
        )
        + (SOC >= 0.24999799599198397)
        * (SOC < 0.28186973947895794)
        * (
            (6.931135110518376 * SOC + -4.010949236405864) * SOC
            + 0.7115653914941045
        )
        + (SOC >= 0.28186973947895794)
        * (SOC < 0.31374148296593185)
        * (
            (0.6481543997291652 * SOC + -0.46898496420292446) * SOC
            + 0.21237911816929333
        )
        + (SOC >= 0.31374148296593185)
        * (SOC < 0.3754929859719439)
        * (
            (0.24514999263576875 * SOC + -0.21610656355634106) * SOC
            + 0.17270989595483877
        )
        + (SOC >= 0.3754929859719439)
        * (SOC < 0.49899599198396793)
        * (
            (-0.040095365277850625 * SOC + -0.0018913012011005925) * SOC
            + 0.13249173170357287
        )
        + (SOC >= 0.49899599198396793)
        * (SOC < 0.5627394789579159)
        * (
            (-0.41742320080259887 * SOC + 0.37467885398057277) * SOC
            + 0.03853823263535627
        )
        + (SOC >= 0.5627394789579159)
        * (SOC < 0.5786753507014029)
        * (
            (-24.43492887370718 * SOC + 27.405876110458735) * SOC
            + -7.567222697224246
        )
        + (SOC >= 0.5786753507014029)
        * (SOC < 0.5946112224448898)
        * (
            (13.556627748031701 * SOC + -16.563678593095148) * SOC
            + 5.154826046907559
        )
        + (SOC >= 0.5946112224448898)
        * (SOC < 0.6244909819639278)
        * (
            (6.0108065297982805 * SOC + -7.590018635246373) * SOC
            + 2.4869065882369474
        )
        + (SOC >= 0.6244909819639278)
        * (SOC < 0.747993987975952)
        * (
            (0.17721101798876493 * SOC + -0.3039630561458102) * SOC
            + 0.21186858661881147
        )
        + (SOC >= 0.747993987975952)
        * (SOC < 0.871496993987976)
        * (
            (-0.03345688538172764 * SOC + 0.011193594215444413) * SOC
            + 0.09400094674838266
        )
        + (SOC >= 0.871496993987976)
        * (SOC < 0.933248496993988)
        * (
            (-0.2560572477469343 * SOC + 0.3991846875392646) * SOC
            + -0.07506558901452465
        )
        + (SOC >= 0.933248496993988)
        * (SOC < 0.9491843687374749)
        * (
            (-1.328542399333486 * SOC + 2.400974999072332) * SOC
            + -1.0091494887821852
        )
        + (SOC >= 0.9491843687374749)
        * (SOC < 0.9651202404809619)
        * (
            (-0.77508838431595 * SOC + 1.3503151993330675) * SOC
            + -0.510514559395574
        )
        + (SOC >= 0.9651202404809619)
        * (SOC < 0.9730881763527054)
        * (
            (-13.420826248978187 * SOC + 25.759630337337285) * SOC
            + -12.289476607378674
        )
        + (SOC >= 0.9730881763527054)
        * (SOC < 0.9810561122244489)
        * (
            (11.438142448151552 * SOC + -22.620306693661178) * SOC
            + 11.249495741397709
        )
        + (SOC >= 0.9810561122244489)
        * (SOC < 0.9890240480961924)
        * (
            (-77.51096883174966 * SOC + 151.90783190249897) * SOC
            + -74.36145282106139
        )
        + (SOC >= 0.9890240480961924)
        * (
            (-1536.829709500849 * SOC + 3038.5104886208787) * SOC
            + -1501.821175217479
        )
    )


def OCV_BASF_GITT_Test_cathode(SOC):
    """!
    Estimated OCV for the cathode in the "BASF GITT Test"-data.
    E₀: [3.9863272907672287, 3.9770716502364394, 3.908636968226356,
         3.7234749637590765, 3.673179452264688, 3.6408841257157816]
    a: [-26.627154259595873, -1.292777296589606, -0.1638198787655928,
        -0.777064236034688, -4.268952015290985, -2.8557091327913606]
    Δx: [0.024115021125137043, 0.041854701383275045, 0.7000584168144725,
         0.0990843265896007, 0.01814215130772722, 0.11674538277978741]
    """
    try:
        print(SOC._storage)
    except:
        pass
    return (
        (SOC < 0.01984924623115578)
        * (
            (333.0496938075703 * SOC + -21.080897932496384) * SOC
            + 4.7498873861616815
        )
        + (SOC >= 0.01984924623115578)
        * (SOC < 0.029698492462311557)
        * (
            (124.59263988527528 * SOC + -12.805467148646812) * SOC
            + 4.667756854512925
        )
        + (SOC >= 0.029698492462311557)
        * (SOC < 0.04447236180904523)
        * (
            (60.383121401385324 * SOC + -8.991615347241918) * SOC
            + 4.611124030024724
        )
        + (SOC >= 0.04447236180904523)
        * (SOC < 0.07402010050251256)
        * (
            (22.48743795919836 * SOC + -5.620994257157776) * SOC
            + 4.536174289705013
        )
        + (SOC >= 0.07402010050251256)
        * (SOC < 0.1035678391959799)
        * (
            (8.69988759914122 * SOC + -3.5798825304881348) * SOC
            + 4.4606326421325395
        )
        + (SOC >= 0.1035678391959799)
        * (SOC < 0.13311557788944725)
        * (
            (6.005413185517341 * SOC + -3.021760744912285) * SOC
            + 4.431730908462402
        )
        + (SOC >= 0.13311557788944725)
        * (SOC < 0.2562311557788945)
        * (
            (2.8038871012608126 * SOC + -2.169414755244347) * SOC
            + 4.375000643974197
        )
        + (SOC >= 0.2562311557788945)
        * (SOC < 0.275929648241206)
        * (
            (4.80302402342204 * SOC + -3.193897083495358) * SOC
            + 4.5062527894956474
        )
        + (SOC >= 0.275929648241206)
        * (SOC < 0.2907035175879397)
        * (
            (17.6320482510273 * SOC + -10.27371336829674) * SOC
            + 5.483018398034346
        )
        + (SOC >= 0.2907035175879397)
        * (SOC < 0.32025125628140705)
        * (
            (-9.89808376503629 * SOC + 5.732499065162187) * SOC
            + 3.1564872692012074
        )
        + (SOC >= 0.32025125628140705)
        * (SOC < 0.3793467336683417)
        * (
            (-1.2425282366077681 * SOC + 0.18859400157708706) * SOC
            + 4.04420854986023
        )
        + (SOC >= 0.3793467336683417)
        * (SOC < 0.4433668341708543)
        * (
            (-0.9961339960939881 * SOC + 0.0016563009099854753) * SOC
            + 4.0796656529339685
        )
        + (SOC >= 0.4433668341708543)
        * (SOC < 0.502462311557789)
        * (
            (0.05432724724954596 * SOC + -0.9298230508509846) * SOC
            + 4.286159178576881
        )
        + (SOC >= 0.502462311557789)
        * (SOC < 0.6255778894472362)
        * (
            (0.9843501493071471 * SOC + -1.864425965190037) * SOC
            + 4.5209605489406
        )
        + (SOC >= 0.6255778894472362)
        * (SOC < 0.6600502512562815)
        * (
            (-0.35737862629639494 * SOC + -0.18571425388472562) * SOC
            + 3.9958780842663373
        )
        + (SOC >= 0.6600502512562815)
        * (SOC < 0.6895979899497487)
        * (
            (-0.4251901558748159 * SOC + -0.09619621961178382) * SOC
            + 3.966334883759373
        )
        + (SOC >= 0.6895979899497487)
        * (SOC < 0.7486934673366834)
        * (
            (2.26117752072264 * SOC + -3.801223719706968) * SOC
            + 5.24382464214645
        )
        + (SOC >= 0.7486934673366834)
        * (SOC < 0.812713567839196)
        * (
            (1.361184402706499 * SOC + -2.4535857834938497) * SOC
            + 4.739340782557406
        )
        + (SOC >= 0.812713567839196)
        * (SOC < 0.8718090452261307)
        * (
            (-0.4806551482113264 * SOC + 0.5401902021335445) * SOC
            + 3.52279960126225
        )
        + (SOC >= 0.8718090452261307)
        * (SOC < 0.9013567839195981)
        * (
            (-10.332189042607297 * SOC + 17.71750291910621) * SOC
            + -3.964868698404871
        )
        + (SOC >= 0.9013567839195981)
        * (SOC < 0.9309045226130653)
        * (
            (-27.056661031497242 * SOC + 47.86693548842322) * SOC
            + -17.552566487245713
        )
        + (SOC >= 0.9309045226130653)
        * (SOC < 0.9456783919597991)
        * (
            (-15.956196407296375 * SOC + 27.199990044877268) * SOC
            + -7.933089996246963
        )
        + (SOC >= 0.9456783919597991)
        * (SOC < 0.9604522613065327)
        * (
            (-39.52119531019889 * SOC + 71.76981058294041) * SOC
            + -29.007448104433934
        )
        + (SOC >= 0.9604522613065327)
        * (SOC < 0.9752261306532664)
        * (
            (-68.66564304507119 * SOC + 127.75351204591425) * SOC
            + -55.892284437646595
        )
        + (SOC >= 0.9752261306532664)
        * (SOC < 0.9850753768844221)
        * (
            (-231.3874904454642 * SOC + 445.1347072719873) * SOC
            + -210.65150191886642
        )
        + (SOC >= 0.9850753768844221)
        * (
            (-393.56737895228434 * SOC + 764.6535368598998) * SOC
            + -368.0265676578274
        )
    )


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
    "1 + dlnf/dlnc": 1.475 / (1 - 0.3490477117897215),
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
    lambda cₑ, cₙ, T: 3.67e-6 * cₑ**αₚₙ * cₙ**αₙₙ * (cₙ_max - cₙ) ** αₚₙ
)
parameters["Positive electrode exchange-current density [A.m-2]"] = (
    lambda cₑ, cₚ, T: 5.06e-6 * cₑ**αₚₙ * cₚ**αₙₙ * (cₚ_max - cₚ) ** αₚₙ
)

parameters[
    "Negative electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₙ, T: 3.67e-6
    * αₚₚ
    * cₑ ** (αₚₚ - 1)
    * cₙ**αₙₚ
    * (cₙ_max - cₙ) ** αₚₚ
)
parameters[
    "Positive electrode exchange-current density partial derivative "
    "by electrolyte concentration [A.m.mol-1]"
] = (
    lambda cₑ, cₚ, T: 5.06e-6
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
