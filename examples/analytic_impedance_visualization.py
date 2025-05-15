"""!@file
Example usage of the analytic impedance models.
"""

import matplotlib.pyplot as plt
import numpy as np

from ep_bolfi.models.analytic_impedance import AnalyticImpedance
from ep_bolfi.utility.visualization import (
    interactive_impedance_model, nyquist_plot
)

from parameters.models.spme_benchmark_cell import parameters

AI = AnalyticImpedance(parameters)
nop = 200
# ω = np.exp(np.linspace(np.log(0.0031623), np.log(1.0), nop))
ω = np.exp(np.linspace(np.log(1e-4), np.log(1e10), nop))
s = 1j * ω

# Z_SPM = AI.Z_SPM(s) - AI.Z_SPM_offset()
# Z_SPMe_1 = AI.Z_SPMe_1(s) - AI.Z_SPMe_1_offset()
# Z_SPMe = AI.Z_SPMe(s) - AI.Z_SPMe_offset()
# Z_SPM_halfcell = AI.Z_SPM_halfcell(s) - AI.Z_SPM_offset_halfcell()
# Z_SPMe_halfcell = AI.Z_SPMe_halfcell(s) - AI.Z_SPMe_offset_halfcell()
# Z_SPMe_1_halfcell = AI.Z_SPMe_1_halfcell(s) - AI.Z_SPMe_1_offset_halfcell()
Z_SPMe_with_double_layer_and_SEI = (
    AI.Z_SPMe_with_double_layer_and_SEI(s)
)

fig, ax = plt.subplots(figsize=(5 * 2**0.5, 5))
nyquist_plot(fig, ax, ω, Z_SPMe_with_double_layer_and_SEI, equal_aspect=False)
plt.show()

interactive_impedance_model(
    ω,
    Z_SPMe_with_double_layer_and_SEI,
    parameters,
    {
        "Electrolyte diffusivity [m2.s-1]": (2.8e-12, 2.8e-8),
        "Mass density of electrolyte [kg.m-3]": (1.5e2, 1.5e4),
        "Mass density of cations in electrolyte [kg.m-3]": (6.853, 685.3),
        "Molar mass of electrolyte solvent [kg.mol-1]": (8.908e-3, 890.8e-3),
        "Solvent concentration [mol.m-3]": (1.317e3, 131.7e3),
        "Partial molar volume of electrolyte solvent [m3.mol-1]": (
            7.593e-6, 759.3e-6
        ),
        "Negative electrode double-layer capacity [F.m-2]": (0.002, 20),
        "Positive electrode double-layer capacity [F.m-2]": (0.002, 20),
        "SEI thickness [m]": (6.7e-9, 670e-9),
        "SEI ionic conductivity [S.m-1]": (2.45e-11, 2.45e5),
        "SEI relative permittivity": (13.1, 1310),
        "Anion transference number in SEI": (0, 1),
        "SEI porosity": (0, 1),
        "SEI Bruggeman coefficient": (1, 6),
        "Negative electrode exchange-current density [A.m-2]": (1e-2, 1e2),
        "Positive electrode exchange-current density [A.m-2]": (1e-2, 1e2),
    },
    {
        "Electrolyte diffusivity [m2.s-1]": 'log',
        "Mass density of cations in electrolyte [kg.m-3]": 'log',
        "Mass density of electrolyte [kg.m-3]": 'log',
        "Molar mass of electrolyte solvent [kg.mol-1]": 'log',
        "Solvent concentration [mol.m-3]": 'log',
        "Partial molar volume of electrolyte solvent [m3.mol-1]": 'log',
        "Negative electrode double-layer capacity [F.m-2]": 'log',
        "Positive electrode double-layer capacity [F.m-2]": 'log',
        "SEI thickness [m]": 'log',
        "SEI ionic conductivity [S.m-1]": 'log',
        "SEI relative permittivity": 'log',
        "Negative electrode exchange-current density [A.m-2]": 'log',
        "Positive electrode exchange-current density [A.m-2]": 'log',

    },
    three_electrode=None,
    dimensionless_reference_electrode_location=0.5,
    with_dl_and_sei=True,
)
