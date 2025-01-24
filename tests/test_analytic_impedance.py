import matplotlib.pyplot as plt
import numpy as np
import pybamm
import unittest
from copy import deepcopy
from multiprocessing import Pool
from scipy.fft import rfft, rfftfreq
from scipy.integrate import quad

from pybamm import cosh, exp, tanh
from pybamm.models.full_battery_models.lithium_ion.spm import SPM
from pybamm.models.full_battery_models.lithium_ion.spme import SPMe
from ep_bolfi.models.analytic_impedance import AnalyticImpedance
from ep_bolfi.models.solversetup import (
    solver_setup, spectral_mesh_pts_and_method
)
from ep_bolfi.utility.preprocessing import find_occurrences
from ep_bolfi.utility.visualization import nyquist_plot

plt.style.use("default")


"""! Anodic symmetry factor at the anode. """
αₙₙ = 0.5
"""! Cathodic symmetry factor at the anode. """
αₚₙ = 0.5
"""! Anodic symmetry factor at the cathode. """
αₙₚ = 0.5
"""! Cathodic symmetry factor at the cathode. """
αₚₚ = 0.5


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


def neg_exc_cur(cₑ, cₙ, cₙ_max, T):
    return 2e-5 * cₑ ** αₚₙ * cₙ ** αₙₙ * (cₙ_max - cₙ) ** αₚₙ


def pos_exc_cur(cₑ, cₚ, cₚ_max, T):
    return 6e-7 * cₑ ** αₙₚ * cₚ ** αₚₚ * (cₚ_max - cₚ) ** αₙₚ


def neg_exc_cur_der(cₑ, cₙ, cₙ_max, T):
    return 2e-5 * αₚₚ * cₑ ** (αₚₚ - 1) * cₙ ** αₙₚ * (cₙ_max - cₙ) ** αₚₚ


def pos_exc_cur_der(cₑ, cₚ, cₚ_max, T):
    return 6e-7 * αₚₚ * cₑ ** (αₚₚ - 1) * cₚ ** αₙₚ * (cₚ_max - cₚ) ** αₚₚ


def spme_benchmark_cell():
    """!
    Parameter dictionary. Initial conditions and the exchange current
    densities will have been added after this file was loaded.
    """
    parameters = {
        # Parameters that get estimated for the benchmark.
        "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
        "Cation transference number": 0.4,
        "Negative particle diffusivity [m2.s-1]": 3.9e-14,
        "Positive particle diffusivity [m2.s-1]": 1e-13,
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
        "Negative electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]": 0,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        "Positive electrode OCP entropic change partial derivative by SOC "
        "[V.K-1]": 0,
        "Number of electrodes connected in parallel to make a cell": 1,
        "Number of cells connected in series to make a battery": 1,
        # Taken from Single2019.
        "Charge number of anion in electrolyte salt dissociation": -1,
        "Charge number of cation in electrolyte salt dissociation": 1,
        "Stoichiometry of anion in electrolyte salt dissociation": 1,
        "Stoichiometry of cation in electrolyte salt dissociation": 1,
        "Mass density of electrolyte [kg.m-3]": 1.5e3,  # Wikipedia, 20 °C
        "Mass density of cations in electrolyte [kg.m-3]": 68.53,  # Li+ PF6-
        "Molar mass of electrolyte solvent [kg.mol-1]": 89.08e-3,
        "Solvent concentration [mol.m-3]": 13.17e3,
        "Partial molar volume of electrolyte solvent [m3.mol-1]": 75.93e-6,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "SEI thickness [m]": 67e-9,
        "SEI ionic conductivity [S.m-1]": 2.45e-10,
        "SEI relative permittivity": 131,
        # t_plus is known as 0.063
        "Anion transference number in SEI": 1 - 0.063,
        "SEI porosity": 0.1,
        "SEI Bruggeman coefficient": 4.54,
    }

    """! Maximum charge concentration in the anode active material. """
    cₙ_max = parameters[
        "Maximum concentration in negative electrode [mol.m-3]"
    ]
    """! Maximum charge concentration in the cathode active material. """
    cₚ_max = parameters[
        "Maximum concentration in positive electrode [mol.m-3]"
    ]

    # Initial conditions.
    parameters["Initial concentration in electrolyte [mol.m-3]"] = 1000.0
    parameters["Initial concentration in negative electrode [mol.m-3]"] = (
        0.97 * cₙ_max
    )
    parameters["Initial concentration in positive electrode [mol.m-3]"] = (
        0.41 * cₚ_max
    )

    parameters["Negative electrode exchange-current density [A.m-2]"] = (
        neg_exc_cur
    )
    parameters["Positive electrode exchange-current density [A.m-2]"] = (
        pos_exc_cur
    )

    parameters[
        "Negative electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]"
    ] = neg_exc_cur_der
    parameters[
        "Positive electrode exchange-current density partial derivative "
        "by electrolyte concentration [A.m.mol-1]"
    ] = pos_exc_cur_der

    return parameters, cₙ_max, cₚ_max


def freq_func(
    i_amplitude,
    frequency,
    parameters,
    n_waves_total,
    n_waves_dft,
    t_fidelity,
    sim_model
):
    def current(t):
        return i_amplitude * pybamm.sin(frequency * 2 * np.pi * t)

    parameters.update({"Current function [A]": current})
    t_eval = np.linspace(
        0.0, (n_waves_total / frequency), n_waves_total * t_fidelity
    )
    # , options={'working electrode': 'positive'})
    solver = solver_setup(
        sim_model(),
        parameters,
        *spectral_mesh_pts_and_method(
            16, 16, 16, 1, 1, 1, halfcell=False
        ),
        verbose=False,
        reltol=1e-12,
        abstol=1e-12,
        root_tol=1e-9,
    )

    experimental_data = solver(t_eval)[
        "Terminal voltage [V]"
    ].entries
    # You could take only the last complete wave.
    # This way, the first freqency bin will be the
    # desired frequency, i.e., it has maximum spacing.
    i_eval = i_amplitude * np.sin(
        frequency * 2 * np.pi * t_eval[
            -len(t_eval) // n_waves_total * n_waves_dft:
        ]
    )
    # The minus sign corrects for the sign convention of the
    # overpotential.
    u_eval = -experimental_data[-len(t_eval) // n_waves_total * n_waves_dft:]
    dft_frequencies = rfftfreq(
        len(u_eval),
        d=(n_waves_total / frequency) / (n_waves_total * t_fidelity)
    )
    index = find_occurrences(dft_frequencies, frequency)[0]
    # The normalizations cancel out; each rfft is the
    # actual ft multiplied by 2 / len.
    return rfft(u_eval)[index] / rfft(i_eval)[index]


class TestAnalyticImpedanceConsistency(unittest.TestCase):
    """
    Test suite for ep_bolfi.models.analytic_impedance.
    Tests the internal consistency of the equations.
    """

    def setUp(self):
        """Takes the place of an __init__ in unittest."""
        parameters, cₙ_max, cₚ_max = spme_benchmark_cell()
        # Alter the parameters a bit, since they are too similar for the
        # two electrodes.
        parameters.update({
            "Positive electrode Bruggeman coefficient (electrolyte)": 2.25,
            "Positive electrode Bruggeman coefficient (electrode)": 2.25,
            "Positive eletrode porosity": 0.6,
        })
        self.parameters = parameters
        self.cₙ_max = cₙ_max
        self.cₚ_max = cₚ_max
        self.model = AnalyticImpedance(parameters, catch_warnings=True)
        self.f_eval = np.logspace(-3, 2, 30)
        self.s_eval_dim = 1j * self.f_eval
        self.s_eval = self.s_eval_dim * self.model.timescale

    def test_electrode_mean(self, debug=False):
        analytic_mean_neg = np.array(self.model.bar_cₑₙ_1(self.s_eval))
        integrated_mean_neg = [
            quad(
                lambda x: complex(self.model.cₑₙ_1(s_point, x)[0]),
                0.0,
                self.model.Lₙ,
                complex_func=True,
            )[0]
            for s_point in self.s_eval
        ]
        analytic_mean_pos = np.array(self.model.bar_cₑₚ_1(self.s_eval))
        integrated_mean_pos = [
            quad(
                lambda x: complex(self.model.cₑₚ_1(s_point, x)[0]),
                1.0 - self.model.Lₚ,
                1.0,
                complex_func=True,
            )[0]
            for s_point in self.s_eval
        ]
        if debug:
            print("Warning: test_electrode_mean disabled.")
            fig_neg, ax_neg = plt.subplots(figsize=(4 * 2**0.5, 4))
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                analytic_mean_neg,
                title_text="Integral check at negative electrode",
                legend_text="Analytic mean",
                equal_aspect=False
            )
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                integrated_mean_neg,
                ls='-.',
                title_text="Integral check at negative electrode",
                legend_text="Integrated mean",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                analytic_mean_pos,
                title_text="Integral check at positive electrode",
                legend_text="Analytic mean",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                integrated_mean_pos,
                ls='-.',
                title_text="Integral check at positive electrode",
                legend_text="Integrated mean",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                np.abs(np.sum(
                    (analytic_mean_neg - integrated_mean_neg)**2
                ))**0.5
            )
            self.assertAlmostEqual(
                0,
                np.abs(np.sum(
                    (analytic_mean_pos - integrated_mean_pos)**2
                ))**0.5
            )

    def test_continuity(self, debug=False):
        electrode_neg = self.model.cₑₙ_1(self.s_eval, self.model.Lₙ)
        separator_neg = self.model.cₑₛ_1(self.s_eval, self.model.Lₙ)
        separator_pos = self.model.cₑₛ_1(
            self.s_eval, self.model.Lₙ + self.model.Lₛ
        )
        electrode_pos = self.model.cₑₚ_1(
            self.s_eval, self.model.Lₙ + self.model.Lₛ
        )
        if debug:
            print("Warning: test_continuity disabled.")
            fig_neg, ax_neg = plt.subplots(figsize=(4 * 2**0.5, 4))
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                separator_neg,
                title_text="Continuity check at negative-separator interface",
                legend_text="Separator concentration at negative",
                equal_aspect=False,
            )
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                electrode_neg,
                ls='-.',
                title_text="Continuity check at negative-separator interface",
                legend_text="Negative electrolyte concentration",
                add_frequency_colorbar=False,
                equal_aspect=False,
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                separator_pos,
                title_text="Continuity check at separator-positive interface",
                legend_text="Separator concentration at positive",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                electrode_pos,
                ls='-.',
                title_text="Continuity check at separator-positive interface",
                legend_text="Positive electrolyte concentration",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                float(np.abs(np.sum([
                    (sep - ele)**2
                    for sep, ele in zip(separator_neg, electrode_neg)
                ]))**0.5),
                places=1
            )
            self.assertAlmostEqual(
                0,
                float(np.abs(np.sum([
                    (sep - ele)**2
                    for sep, ele in zip(separator_pos, electrode_pos)
                ]))**0.5),
                places=1
            )

    def test_gradient_continuity(self, debug=False):
        electrode_neg = (
            self.model.εₙ ** self.model.βₙ
            * np.array([
                complex(entry) for entry in
                self.model.d_dx_cₑₙ_1(self.s_eval, self.model.Lₙ)
            ])
        )
        separator_neg = (
            self.model.εₛ ** self.model.βₛ
            * np.array([
                complex(entry) for entry in
                self.model.d_dx_cₑₛ_1(self.s_eval, self.model.Lₙ)
            ])
        )
        separator_pos = (
            self.model.εₛ ** self.model.βₛ * np.array([
                complex(entry) for entry in self.model.d_dx_cₑₛ_1(
                    self.s_eval, self.model.Lₙ + self.model.Lₛ
                )
            ])
        )
        electrode_pos = (
            self.model.εₚ ** self.model.βₚ * np.array([
                complex(entry) for entry in self.model.d_dx_cₑₚ_1(
                    self.s_eval, self.model.Lₙ + self.model.Lₛ
                )
            ])
        )
        if debug:
            print("Warning: test_gradient_continuity disabled.")
            fig_neg, ax_neg = plt.subplots(figsize=(4 * 2**0.5, 4))
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                separator_neg,
                title_text="Gradient check at negative-separator interface",
                legend_text="Separator gradient at negative",
                equal_aspect=False,
            )
            nyquist_plot(
                fig_neg,
                ax_neg,
                self.f_eval,
                electrode_neg,
                ls='-.',
                title_text="Gradient check at negative-separator interface",
                legend_text="Negative electrolyte gradient",
                add_frequency_colorbar=False,
                equal_aspect=False,
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                separator_pos,
                title_text="Gradient check at separator-positive interface",
                legend_text="Separator gradient at positive",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                electrode_pos,
                ls='-.',
                title_text="Gradient check at separator-positive interface",
                legend_text="Positive electrolyte gradient",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                np.abs(np.sum((separator_neg - electrode_neg)**2))**0.5,
                places=0
            )
            self.assertAlmostEqual(
                0,
                np.abs(np.sum((separator_pos - electrode_pos)**2))**0.5,
                places=0
            )

    def test_electrode_mean_metal_counter(self, debug=False):
        analytic_mean_pos = np.array(
            self.model.bar_cₑₚ_1_metal_counter(self.s_eval)
        )
        integrated_mean_pos = [
            quad(
                lambda x: complex(
                    self.model.cₑₚ_1_metal_counter(s_point, x)[0]
                ),
                1.0 - self.model.Lₚ,
                1.0,
                complex_func=True,
            )[0]
            for s_point in self.s_eval
        ]
        if debug:
            print("Warning: test_electrode_mean_metal_counter disabled.")
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                analytic_mean_pos,
                title_text="Integral check at positive electrode (Li metal)",
                legend_text="Analytic mean",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                integrated_mean_pos,
                ls='-.',
                title_text="Integral check at positive electrode (Li metal)",
                legend_text="Integrated mean",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                np.abs(np.sum(
                    (analytic_mean_pos - integrated_mean_pos)**2
                ))**0.5
            )

    def test_continuity_metal_counter(self, debug=True):
        separator_pos = np.array([
            complex(entry) for entry in
            self.model.cₑₛ_1_metal_counter(self.s_eval, self.model.Lₛ)
        ])
        electrode_pos = np.array([
            complex(entry) for entry in
            self.model.cₑₚ_1_metal_counter(self.s_eval, self.model.Lₛ)
        ])
        if debug:
            print("Warning: test_continuity_metal_counter disabled.")
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                separator_pos,
                title_text="Continuity check sep.-pos. interface (Li metal)",
                legend_text="Separator concentration at positive",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                electrode_pos,
                ls='-.',
                title_text="Continuity check sep.-pos. interface (Li metal)",
                legend_text="Positive electrolyte concentration",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                np.abs(np.sum((separator_pos - electrode_pos)**2))**0.5,
                places=2
            )

    def test_gradient_continuity_metal_counter(self, debug=True):
        separator_pos = (
            self.model.εₛ ** self.model.βₛ * np.array([
                complex(entry) for entry in
                self.model.d_dx_cₑₛ_1_metal_counter(self.s_eval, self.model.Lₛ)
            ])
        )
        electrode_pos = (
            self.model.εₚ ** self.model.βₚ * np.array([
                complex(entry) for entry in
                self.model.d_dx_cₑₚ_1_metal_counter(self.s_eval, self.model.Lₛ)
            ])
        )
        if debug:
            print("Warning: test_gradient_continuity_metal_counter disabled.")
            fig_pos, ax_pos = plt.subplots(figsize=(4 * 2**0.5, 4))
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                separator_pos,
                title_text="Gradient check at sep.-pos. interface (Li metal)",
                legend_text="Separator gradient at positive",
                equal_aspect=False
            )
            nyquist_plot(
                fig_pos,
                ax_pos,
                self.f_eval,
                electrode_pos,
                ls='-.',
                title_text="Gradient check at sep.-pos. interface (Li metal)",
                legend_text="Positive electrolyte gradient",
                add_frequency_colorbar=False,
                equal_aspect=False
            )
            plt.show()
        else:
            self.assertAlmostEqual(
                0,
                np.abs(np.sum((separator_pos - electrode_pos)**2))**0.5,
                places=0
            )

    def test_consistency_with_time_model(self, debug=True):
        parameters, cₙ_max, cₚ_max = spme_benchmark_cell()
        # Recommendation:
        # Unimodal excitation (scaled to an 8 mV amplitude):
        # 10 nA sine, 8 waves sampled at frequency * 2**6.
        # At 1 µA and above, the response is nonlinear.
        i_amplitude = 1e-6
        n_waves_total = 8
        n_waves_dft = 4
        t_fidelity = 2**4

        # SOC scaling between negative and positive electrode: 0.590
        # Initial conditions for the unimodal sinusoidal excitations.
        soc_offsets = [0.17, 0.24, 0.30, 0.36, 0.42,
                       0.48, 0.54, 0.60, 0.66, 0.72, 0.78]
        offset_indices = [10]  # [0, 2, 5, 10]

        freq_eval = np.logspace(-3, 2, 30)

        for i, soc_offset_index in enumerate(offset_indices):

            soc_offset = soc_offsets[soc_offset_index]
            initial_socs = {
                "Initial concentration in negative electrode [mol.m-3]":
                    (0.97 - soc_offset) * cₙ_max,
                "Initial concentration in positive electrode [mol.m-3]":
                    (0.41 + soc_offset * 0.590) * cₚ_max,
            }
            local_parameters = deepcopy(parameters)
            local_parameters.update(initial_socs)

            ana_model_base = AnalyticImpedance(
                local_parameters, catch_warnings=False
            )

            for ana_model, sim_model, title in zip(
                (ana_model_base.Z_SPM, ana_model_base.Z_SPMe),
                (SPM, SPMe),
                ("SPM impedance", "SPMe impedance")
            ):
                parallel_arguments = [
                    [
                        i_amplitude,
                        freq,
                        deepcopy(local_parameters),
                        n_waves_total,
                        n_waves_dft,
                        t_fidelity,
                        sim_model
                    ]
                    for freq in freq_eval
                ]

                with Pool() as p:
                    simulated_impedance = p.starmap(
                        freq_func, parallel_arguments
                    )

                simulated_impedance = np.array(simulated_impedance)
                s = 1j * freq_eval
                local_parameters = deepcopy(parameters)
                local_parameters.update(initial_socs)
                analytic_impedance = ana_model(s)

                if debug:
                    print(
                        "Warning: test_consistency_with_time_model disabled."
                    )
                    fig, ax = plt.subplots(figsize=(5 * 2**0.5, 5))
                    nyquist_plot(
                        fig,
                        ax,
                        freq_eval,
                        simulated_impedance - simulated_impedance[-1],
                        ls=':',
                        legend_text="simulated",
                        equal_aspect=False,
                    )
                    nyquist_plot(
                        fig,
                        ax,
                        freq_eval,
                        analytic_impedance - analytic_impedance[-1],
                        ls='-',
                        legend_text="analytic",
                        add_frequency_colorbar=False,
                        equal_aspect=False,
                    )
                    ax.set_title(
                        title + "(ignores ohmic res., since it's arbitrary)"
                    )
                    plt.show()
                else:
                    self.assertAlmostEqual(
                        0,
                        np.abs(np.sum(
                            (
                                (simulated_impedance - simulated_impedance[-1])
                                - (analytic_impedance - analytic_impedance[-1])
                            )**2
                        ))**0.5,
                        places=1
                    )


if __name__ == '__main__':
    unittest.main()
