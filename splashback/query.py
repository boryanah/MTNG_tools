import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

# simulation parameters
sim_type = 'MTNG'; fp_dm = 'fp'; pos_unit = 1.e3; snapshot = 179; z = 1.
#sim_type = 'MTNG'; fp_dm = 'fp'; pos_unit = 1.e3; snapshot = 264; z = 0.
#sim_type = 'MTNG'; fp_dm = 'dm'; pos_unit = 1.e3; snapshot = 184; z = 1.
#sim_type = 'TNG'; fp_dm = 'dm'; pos_unit = 1.; snapshot = 55; z = 0.82
#sim_type = 'TNG'; fp_dm = 'fp'; pos_unit = 1.; snapshot = 55; z = 0.82
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': f"/mnt/alan1/boryanah/MTNG/data_{fp_dm:s}/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
tracer = sys.argv[1] #"LRG"
#tracer = "all_mhalo_2.0e11"
n_gal = "2.0e-03"

# simulation params
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]*1.e3 # ckpc/h
a = 1./(1+z)

# change these!
boundary = sys.argv[2] #'r200m', 'rsplash'

# load halo properties
GroupPos = np.load(tng_dir+f'GroupPos_{fp_dm:s}_{snapshot:d}.npy')*pos_unit # ckpc/h
if boundary == 'rsplash':
    M_splash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # splashback mass
    R_splash = np.load(tng_dir+f'Group_R_Splash_{fp_dm:s}_{snapshot:d}.npy')/a # ckpc/h # splashback radius (output in kpc/h)
elif boundary == 'r200m':
    M_splash = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # original mass
    R_splash = np.load(tng_dir+f'Group_R_Mean200_{fp_dm:s}_{snapshot:d}.npy')*pos_unit # ckpc/h # originalen radius
M200mean = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h

# define galaxies
if sim_type == "MTNG":
    gal_ind = np.load(f"../selection/data/index_{tracer}_{n_gal}_{snapshot:d}.npy")
    N_g = len(gal_ind)
else:
    N_g = 12000
    M_star = np.load(tng_dir+f'SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10 # Msun/h
    gal_ind = (np.argsort(M_star)[::-1])[:N_g]

# get positions of galaxies
SubPos = np.load(tng_dir+f'SubhaloPos_fp_{snapshot:d}.npy')*pos_unit # ckpc/h
gal_pos = SubPos[gal_ind]


# define the central subhalos
if sim_type == "MTNG":
    SubhaloGrNr = np.load(tng_dir+f'SubhaloGroupNr_fp_{snapshot:d}.npy')
else:
    SubhaloGrNr = np.load(tng_dir+f'SubhaloGrNr_fp_{snapshot:d}.npy')
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# which galaxies are centrals
gal_ind_cent = np.intersect1d(gal_ind, sub_inds_cent)
gal_ind_sats = gal_ind[~np.in1d(gal_ind, gal_ind_cent)]  
print("percentage satellites = ", len(gal_ind_sats)*100./len(gal_ind))

# cut lower mass halos
M_thresh = 2.e11
m_choice = M200mean > M_thresh
halo_ind = np.arange(len(M200mean), dtype=int)[m_choice]
halo_pos = GroupPos[m_choice]
halo_msp = M_splash[m_choice]
halo_rsp = R_splash[m_choice]

# arrange in order of halo mass
i_sort = np.argsort(halo_msp)[::-1]
halo_ind = halo_ind[i_sort]
halo_pos = halo_pos[i_sort]
halo_msp = halo_msp[i_sort]
halo_rsp = halo_rsp[i_sort]
N_h = np.sum(m_choice)

# create periodic ckdtree of the galaxies
print("min max gal pos = ", gal_pos.min(), gal_pos.max(), Lbox)
gal_pos = gal_pos%Lbox
tree = cKDTree(gal_pos, boxsize=Lbox)
array_lists = tree.query_ball_point(halo_pos, halo_rsp)
assert len(array_lists) == halo_pos.shape[0]

#GroupCount = np.zeros(M200mean, dtype=int)
# initialize array with galaxy parents
gal_par = np.zeros(N_g, dtype=int)-1

# loop over each halo
for i in range(N_h):
    if i%10000 == 0: print(i)
    gal_list = array_lists[i]
    if len(gal_list) == 0: continue

    for j in range(len(gal_list)):
        if gal_par[gal_list[j]] != -1: continue
        gal_par[gal_list[j]] = halo_ind[i]

#np.save(f'data/galaxy_parent_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_par)
#np.save(f'data/galaxy_subind_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_ind)
print("how many galaxies are missing = ", np.sum(gal_par == -1))
missing_gal_ind = np.intersect1d(gal_ind[gal_par == -1], gal_ind_sats)
print("how many of the missing galaxies are satellites = ", len(missing_gal_ind))

missing = gal_par == -1
n_miss = np.sum(missing)
miss_par = gal_par[missing]
miss_pos = gal_pos[missing]
tree = cKDTree(miss_pos, boxsize=Lbox)
array_lists = tree.query_ball_point(halo_pos, 5000.)#5 mpc
assert len(array_lists) == halo_pos.shape[0]

miss_dens = np.ones(np.sum(missing))*(-1.)
print(miss_par.shape, miss_pos.shape)

# loop over each halo
for i in range(N_h):
    if i%10000 == 0: print(i)
    miss_list = array_lists[i] # local indices
    if len(miss_list) == 0: continue

    for j in range(len(miss_list)):
        dist = halo_pos[i] - miss_pos[miss_list[j]]
        dist[dist > Lbox/2.] -= Lbox
        dist[dist < -Lbox/2.] += Lbox
        dens = halo_msp[i]/np.sqrt(np.sum(dist**2))**3.
        if dens > miss_dens[miss_list[j]]:
            miss_dens[miss_list[j]] = dens
            miss_par[miss_list[j]] = -halo_ind[i]

print(miss_par)
gal_par[missing] = miss_par
np.save(f'data/galaxy_parent_{tracer}_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_par)
np.save(f'data/galaxy_subind_{tracer}_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_ind)
print("how many galaxies are missing = ", np.sum(gal_par == -1))
missing_gal_ind = np.intersect1d(gal_ind[gal_par == -1], gal_ind_sats)
print("how many of the missing galaxies are satellites = ", len(missing_gal_ind))
