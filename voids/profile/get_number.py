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
sim_types = ["MTNG", "MTNG_DM", "Gals", "Mtot_fp", "Gals_true", "Gals_hod", "Gals_env"]
ovl = 0.2 # 0.5

void_dic = {'TNG': f"../finders/tracers.SVF_recen_ovl{ovl:.1f}", 'MTNG': f"../visualize/data/pos_down_10000_snap_179_fp.SVF_recen_ovl{ovl:.1f}", 'MTNG_DM': f"../visualize/data/pos_down_10000_snap_184_dm.SVF_recen_ovl{ovl:.1f}", 'Gals': f"../visualize/data/SubhaloPos_fp_179_1e8.SVF_recen_ovl{ovl:.1f}", 'Mtot_fp': f"../visualize/data/SubhaloPos_fp_179_totmass_1e8.SVF_recen_ovl{ovl:.1f}", 'Mtot_dm': f"../visualize/data/SubhaloPos_dm_totmass_1e8.SVF_recen_ovl{ovl:.1f}", 'Gals_true': f"../visualize/data/xyz_gals_fp_187500_mstar_179.SVF_recen_ovl{ovl:.1f}", 'Gals_env': f"../visualize/data/xyz_env_fp_187500_mstar_179.SVF_recen_ovl{ovl:.1f}", 'Gals_hod': f"../visualize/data/xyz_hod_fp_187500_mstar_179.SVF_recen_ovl{ovl:.1f}"}
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/", 'MTNG_DM': "data_mtng_dm/", 'Gals': "data_gals/", 'Mtot_fp': "data_mtot_fp/", 'Mtot_dm': "data_mtot_dm/", 'Gals_true': "data_true/", 'Gals_env': "data_env/", 'Gals_hod': "data_hod/"}

bins = np.linspace(2, 40, 51)

for sim_type in sim_types:
    data_dir = data_dir_dic[sim_type]
    void = np.loadtxt(void_dic[sim_type])

    pos_void = void[:, :3]
    size_void = void[:, 3]
    max_void = size_void.max()

    hist, bin_edges = np.histogram(size_void, bins)
    binc = (bins[1:] + bins[:-1]) * 0.5

    #chist = np.cumsum(hist)
    #chist = np.cumsum(hist[::-1])[::-1] # og
    chist = hist # after alice
    print(chist.shape, hist.shape, binc.shape)

    np.save(data_dir+"chist.npy", chist)
    np.save(data_dir+"binc.npy", binc)

