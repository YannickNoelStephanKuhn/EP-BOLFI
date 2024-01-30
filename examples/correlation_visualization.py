"""!@file
Example usage of the correlation matrix visualization tools.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

from ep_bolfi.utility.visualization import visualize_correlation

import matplotlib
fontsize = 15
matplotlib.rcParams.update({"font.size": fontsize})

"""
with open("../GITT estimation results/estimation_results.json") as f:
   estimation_results = json.load(f)
pulse_number = 66
pulse_correlation = estimation_results[pulse_number]['correlation']
names = [
    "$i_{se,n,0}$",
    "$i_{se,p,0}$",
    "$D_n$",
    "$D_p$",
]
"""

seven_par_corrs = []
for seed in [0, 1, 2]:
    with open(
        "../GITT estimation results/seven_parameter_estimation_seed_"
        + str(seed)
        + ".json",
        "r",
    ) as f:
        seven_par_corrs.append(np.array(json.load(f)['correlation']))
mean_corr = np.mean(seven_par_corrs, axis=0)
stdd_corr = np.sqrt(np.var(seven_par_corrs, axis=0, ddof=1))
stdd_corr[np.diag_indices(len(stdd_corr))] = 1

boosted_correlation = mean_corr / stdd_corr
boosted_correlation[np.diag_indices(len(boosted_correlation))] = 1

names = [
    "$i_{se,n,0}$",
    "$i_{se,p,0}$",
    "$D_n$",
    "$D_p$",
    "$t_+$",
    r"$\beta_n$",
    r"$\beta_p$",
]

title = "Estimation from GITT"

cmap = plt.get_cmap("BrBG")
entry_color = "black"

fig0, ax0 = plt.subplots(figsize=(2**0.5 * 7, 7))
visualize_correlation(
    fig0, ax0, seven_par_corrs[0], names, title, cmap, entry_color
)
fig1, ax1 = plt.subplots(figsize=(2**0.5 * 7, 7))
visualize_correlation(
    fig1,
    ax1,
    boosted_correlation,
    names,
    "Boosted correlation matrix",
    cmap,
    entry_color
)
plt.show()
