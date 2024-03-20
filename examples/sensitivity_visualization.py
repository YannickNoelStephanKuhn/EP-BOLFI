import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

matplotlib.rcParams.update({'font.size': 14})

with open('../GITT estimation results/estimation_results.json', 'r') as f:
    estimation_results = json.load(f)
with open('../GITT estimation results/simulated_features_at_each_pulse.json',
          'r') as f:
    evaluations = json.load(f)
with open('../GITT estimation results/boundaries_of_simulated_features.json',
          'r') as f:
    sensitivities = json.load(f)

labels = {
    "discharge relaxation time": "τᵣ",
    "ohmic voltage drop": "R",
    "GITT square root slope": "GITT-sqrt",
    "concentration overpotential": "η",
    "ICI square root slope": "ICI-sqrt",
}

p_diff = {
    p_name: [0.0 for i in range(len(estimation_results))]
    for p_name in estimation_results[0]['inferred parameters'].keys()
}
p_normalization = {
    p_name: [0.0 for i in range(len(estimation_results))]
    for p_name in estimation_results[0]['inferred parameters'].keys()
}
for i, estimation_result in enumerate(estimation_results):
    errorbars = estimation_result['error bounds']
    for p_name, bounds in errorbars.items():
        e_diff = np.abs(
            np.log(bounds[1]) - np.log(bounds[0])
        ) / chi2(4).ppf(0.95)**0.5
        e_aver = 0.5 * (np.log(bounds[0]) + np.log(bounds[1]))
        p_diff[p_name][i] = e_diff
        p_normalization[p_name][i] = e_aver

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(6 * 2**0.5, 6),
                                             nrows=2, ncols=2)
trans = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig.dpi_scale_trans
)
lines = []
names = []
for (p_name, f_dictionary), ax in zip(
            sensitivities.items(), (ax0, ax1, ax2, ax3)
        ):
    ax.set_xlabel("Pulse number")
    if p_name.find("Negative") != -1:
        if p_name.find("exchange") != -1:
            desc = "(a) $i_{se,n,0}^*$"
        elif p_name.find("diffusivity") != -1:
            desc = "(c) $D_n^*$"
    elif p_name.find("Positive") != -1:
        if p_name.find("exchange") != -1:
            desc = "(b) $i_{se,p,0}^*$"
        elif p_name.find("diffusivity") != -1:
            desc = "(d) $D_p^*$"
    ax.text(0.0, 1.0, desc, transform=ax.transAxes + trans,
            verticalalignment='top')
    for i, (f_name, f_sensitivities) in enumerate(f_dictionary.items()):
        f_normalization = np.max(np.abs(f_sensitivities))
        f_differences = np.abs(np.array(f_sensitivities).T[:][1]
                               - np.array(f_sensitivities).T[:][0])
        features = evaluations[f_name]
        line, = ax.plot(
            range(len(f_sensitivities)),
            np.abs(np.array(p_normalization[p_name]) * f_differences
                   / (np.array(p_diff[p_name]) * features)) + 4 - i,
            label=labels[f_name[:-11]]
        )
        lines.append(line)
        names.append(labels[f_name[:-11]])
    ax.yaxis.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0], labels=["0"] * 5)
    ax.set_ylim([-0.1, 5.1])
fig.legend(handles=lines[:5], labels=names[:5], loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.05))
ax0.xaxis.set_ticks([])
ax0.set_xlabel("")
ax1.xaxis.set_ticks([])
ax1.set_xlabel("")
fig.tight_layout()
fig.savefig('./sensitivities_in_features.pdf',
            bbox_inches='tight', pad_inches=0.0)
plt.show()
