import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/", 'MTNG_DM': "data_mtng_dm/", 'Gals': "data_gals/", 'Mtot_fp': "data_mtot_fp/", 'Mtot_dm': "data_mtot_dm/"}

label_dic = {'TNG': r"${\rm TNG300-1}$",
             'MTNG': r"${\rm MTNG \ (DM \ particles)}$",
             'MTNG_DM': r"${\rm MTNG-DM \ (DM \ particles)}$",
             'Gals': r"${\rm MTNG \ Subhalo \ (stellar \ mass)}$",
             'Mtot_fp': r"${\rm MTNG \ Subhalo \ (total \ mass)}$",
             'Mtot_dm': r"${\rm MTNG-DM \ Subhalo \ (total \ mass)}$"}

sim_types = ["MTNG", "MTNG_DM", "Gals", "Mtot_fp"]

plt.figure(figsize=(12, 10))

for i, sim_type in enumerate(sim_types):
    
    data_dir = data_dir_dic[sim_type]

    rs = np.load(data_dir+"r_bins.npy")
    rc = np.load(data_dir+"r_cents.npy")
    shells = np.load(data_dir+"shells_part.npy")

    vdiffs = 4*np.pi*(rs[1:]**3-rs[:-1]**3)/3.

    if i == 0:
        plt.plot(rc, np.ones(len(rc)), 'k--')
    plt.plot(rc, shells/vdiffs, label=label_dic[sim_type])

plt.legend()
plt.xlabel(r"$r/R_{\rm eff}$")
plt.ylabel(r"$\rho (r)/\overline{\rho}$")
plt.savefig("figs/rho.png")
plt.show()

