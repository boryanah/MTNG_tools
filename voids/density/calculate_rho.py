import os

import numpy as np
from scipy.spatial import cKDTree

#sim_type = "MTNG"
sim_type = "MTNG_DM"
#sim_type = "TNG"
#sim_type = "Gals"
#sim_type = "Mtot_fp"
#sim_type = "Mtot_dm"

tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/data_parts/", 'MTNG_DM': "/mnt/alan1/boryanah/MTNG/data_parts/", 'Gals': "/mnt/alan1/boryanah/MTNG/data_fp/", 'Mtot_fp': "/mnt/alan1/boryanah/MTNG/data_fp/", 'Mtot_dm': "/mnt/alan1/boryanah/MTNG/data_dm/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500., 'MTNG_DM': 500., 'Gals': 500., 'Mtot_fp': 500., 'Mtot_dm': 500.}
pos_part_dic = {'TNG': "pos_parts_down_5000_tng300-3.npy", 'MTNG': "pos_down_1000_snap_179_fp.npy", 'MTNG_DM': "pos_down_1000_snap_184_dm.npy", 'Gals': "SubhaloPos_fp_179_1e8.dat", 'Mtot_fp': "SubhaloPos_fp_179_totmass_1e8.dat", 'Mtot_dm': "SubhaloPos_dm_184_totmass_1e8.dat"}
void_dic = {'TNG': "../finders/tracers.SVF_recen_ovl0.5", 'MTNG': "../visualize/data/pos_down_10000_snap_179_fp.SVF_recen_ovl0.5", 'MTNG_DM': "../glamdring/data/pos_down_1000_snap_184_dm_ncentres_1000000_voids_overlap0.2.dat", 'Gals': "../visualize/data/SubhaloPos_fp_179_1e8.SVF_recen_ovl0.5", 'Mtot_fp': "../visualize/data/SubhaloPos_fp_179_totmass_1e8.SVF_recen_ovl0.5", 'Mtot_dm': "../visualize/data/SubhaloPos_dm_totmass_1e8.SVF_recen_ovl0.5"}
# ../visualize/data/pos_down_10000_snap_184_dm.SVF_recen_ovl0.5 MTNG_DM
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/", 'MTNG_DM': "data_mtng_dm/", 'Gals': "data_gals/", 'Mtot_fp': "data_mtot_fp/", 'Mtot_dm': "data_mtot_dm/"}


#pos_part = np.load(tng_dir+"parts_position_tng300-3_99.npy")/1000.
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
if 'npy' in pos_part_dic[sim_type]:
    pos_part = np.load(tng_dir+pos_part_dic[sim_type])%Lbox
else:
    pos_part = np.loadtxt(tng_dir+pos_part_dic[sim_type])%Lbox
data_dir = data_dir_dic[sim_type]
void = np.loadtxt(void_dic[sim_type])
print(pos_part.min(), pos_part.max())

os.makedirs(data_dir, exist_ok=True)

mean_dens = pos_part.shape[0]/Lbox**3.
#vol = 4/3.*np.pi*3.**3

pos_void = void[:, :3]
size_void = void[:, 3]
max_void = size_void.max()
#option = (size_void > max_void/2.) & (size_void < max_void)
#option = (size_void > 0.) & (size_void < max_void/2.)
#option = (size_void > 0.) & (size_void < 5.)
option = (size_void > 0.)
print("in selection = ", np.sum(option))
print("largest void = ", size_void.max())
print("smallest void = ", size_void.min())
print("number of voids = ", size_void.shape[0])
print("number of tracers = ", pos_part.shape[0])
size_void = size_void[option]
pos_void = pos_void[option]

tree = cKDTree(pos_part, boxsize=Lbox)
print("built the tree")

rs = np.linspace(0, 3, 101)
rc = (rs[1:]+rs[:-1])*0.5

num_part_old = np.zeros(pos_part.shape[0])
shells_part = np.zeros(rc.shape[0])
for i in range(len(rc)):
    rm = rs[i]*size_void
    rp = rs[i+1]*size_void
    lenm = tree.query_ball_point(pos_void, rm, return_length=True)
    lenp = tree.query_ball_point(pos_void, rp, return_length=True)
    shells_part[i] = np.mean((lenp-lenm)/(mean_dens*size_void**3))#(mean_dens*4/3.*np.pi*((rs[-1]*size_void)**3 - (rs[-2]*size_void)**3)))#(mean_dens*4/3.*np.pi*(rp**3-rm**3)))

print(shells_part)

np.save(data_dir+"shells_part.npy", shells_part)
np.save(data_dir+"r_cents.npy", rc)
np.save(data_dir+"r_bins.npy", rs)
