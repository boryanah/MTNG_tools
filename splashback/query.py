import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

# simulation parameters
Lbox = 205.*1.e3 # ckpc/h
snapshot = 55
z = 0.82
a = 1./(1+z)
#fp_dm = 'dm'
fp_dm = 'fp'

boundary = 'r200m'
#boundary = 'rsplash'

# load halo properties
tng_dir = '/mnt/gosling1/boryanah/TNG300/'
GroupPos = np.load(tng_dir+f'GroupPos_{fp_dm:s}_{snapshot:d}.npy') # ckpc/h
if boundary == 'rsplash':
    M_splash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # splashback mass
    R_splash = np.load(tng_dir+f'Group_R_Splash_{fp_dm:s}_{snapshot:d}.npy')/a # ckpc/h # splashback radius
elif boundary == 'r200m':
    M_splash = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # original mass
    R_splash = np.load(tng_dir+f'Group_R_Mean200_{fp_dm:s}_{snapshot:d}.npy') # ckpc/h # originalen radius
M200mean = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h

# define galaxies
N_g = 12000
M_star = np.load(tng_dir+f'SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10 # Msun/h
SubPos = np.load(tng_dir+f'SubhaloPos_fp_{snapshot:d}.npy') # ckpc/h
gal_ind = (np.argsort(M_star)[::-1])[:N_g]
gal_pos = SubPos[gal_ind]

# define the central subhalos
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

np.save(f'data/galaxy_parent_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_par)
np.save(f'data/galaxy_subind_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy', gal_ind)
print("how many galaxies are missing = ", np.sum(gal_par == -1))
missing_gal_ind = np.intersect1d(gal_ind[gal_par == -1], gal_ind_sats)
print("how many of the missing galaxies are satellites = ", len(missing_gal_ind))
