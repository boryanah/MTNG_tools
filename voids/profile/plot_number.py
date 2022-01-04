import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import plotparams
plotparams.buba()

label_dic = {'TNG': r"{\rm TNG300-1}",
             'MTNG': r"{\rm MTNG \ (DM \ particles)}",
             'MTNG_DM': r"{\rm MTNG-DM \ (DM \ particles)}",
             'Gals': r"{\rm MTNG \ Subhalo \ (stellar \ mass)}",
             'Mtot_fp': r"{\rm MTNG \ Subhalo \ (total \ mass)}",
             'Mtot_dm': r"{\rm MTNG-DM \ Subhalo \ (total \ mass)}",
             'Gals_true': r"{\rm True LRGs}",
             'Gals_hod': r"{\rm Mass-only LRGs}",
             'Gals_env': r"{\rm Mass and Env. LRGs}"}

sim_type = "MTNG"
#sim_type = "MTNG_DM"
#sim_type = "TNG"
#sim_type = "Gals"
#sim_types = ["MTNG", "MTNG_DM", "Gals", "Mtot_fp"]
sim_types = ["Gals_true", "Gals_hod", "Gals_env"]
color_dic = {'MTNG': 'dodgerblue', 'MTNG_DM': '#4477AA', 'Gals': '#6699CC', 'Mtot_fp': '#AA4466', 'Mtot_dm': '#DDCC77', 'Gals_true': 'black', 'Gals_env': '#CC6677', 'Gals_hod': '#88CCEE'}
# ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
#  '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
#  '#4477AA']

data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/", 'MTNG_DM': "data_mtng_dm/", 'Gals': "data_gals/", 'Mtot_fp': "data_mtot_fp/", 'Mtot_dm': "data_mtot_dm/", 'Gals_true': "data_true/", 'Gals_env': "data_env/", 'Gals_hod': "data_hod/"}

# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
ax_histx = plt.axes(rect_histx)

#ref_sim = "MTNG"
ref_sim = "Gals_true"
chist_ref = np.load(data_dir_dic[ref_sim]+"chist.npy")

ax_histx.axhline(y=1, color='k', ls='--')

for sim_type in sim_types:
    data_dir = data_dir_dic[sim_type]

    chist = np.load(data_dir+"chist.npy")
    binc = np.load(data_dir+"binc.npy")

    ax_scatter.plot(binc, chist, color=color_dic[sim_type], label=label_dic[sim_type])
    if ref_sim == sim_type: continue
    #ax_histx.plot(binc, chist/chist_ref, color=color_dic[sim_type], label=label_dic[sim_type])
    ax_histx.errorbar(binc, chist/chist_ref, yerr=np.sqrt(chist)/chist_ref, marker='o', capsize=4, ls='-', color=color_dic[sim_type], label=label_dic[sim_type])

ax_scatter.legend(frameon=False, loc='upper right')
ax_scatter.set_yscale('log')
#ax_scatter.set_ylabel(r"$N_{{\rm void}, > R}$")
ax_scatter.set_ylabel(r"$N_{{\rm void}}$")
ax_scatter.set_xlabel(r"$R \ [{\rm Mpc}/h]$")
#ax_scatter.set_xlim([2., 25.])
ax_scatter.set_xlim([2., 50.])

ax_histx.legend(frameon=False)
ax_histx.set_ylabel(r"${\rm Ratio}$")
ax_histx.set_xlabel(r"$R$")
#ax_histx.set_xlim([2., 25.])
ax_histx.set_xlim([2., 50.])

plt.savefig("figs/distn.png")
plt.show()
