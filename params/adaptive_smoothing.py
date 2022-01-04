import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from numba import njit
import numpy.linalg as la

from tools import numba_tsc_3D

#sim_type = "MTNG"; fp_dm = 'fp' # no choice
#sim_type = "MTNG_DM"; fp_dm = 'dm' # no choice
sim_type = "TNG"; fp_dm = 'dm' # actual choice
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
pos_part_dic = {'TNG': "parts_position_tng300-2_99.npy", 'MTNG': "data_parts/pos_down_1000_snap_179_fp.npy"}
void_dic = {'TNG': "../finders/tracers.SVF_recen_ovl0.5", 'MTNG': "../visualize/data/pos_down_10000_snap_179_fp.SVF_recen_ovl0.5"}
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/"}
N_dim_dic = {'TNG': 512, 'MTNG':512}
sim_name_dic = {'TNG': "tng300-2", 'MTNG': "mtng"}

#pos_part = np.load(tng_dir+"parts_position_tng300-3_99.npy")/1000.
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
data_dir = data_dir_dic[sim_type]
N_dim = N_dim_dic[sim_type]
#pos_parts = np.load(tng_dir+pos_part_dic[sim_type])
sim_name = sim_name_dic[sim_type]

if sim_type == "TNG":
    snapshot = '99'
    #pos_parts /= 1000. # deps on file
    if fp_dm == 'fp':
        GroupPos = np.load(tng_dir+'GroupPos_fp.npy')/1.e3
        GroupR200 = np.load(tng_dir+'Group_R_TopHat200_fp.npy')/1.e3
    else:
        GroupPos = np.load(tng_dir+'GroupPos_dm.npy')/1.e3
        GroupR200 = np.load(tng_dir+'Group_R_TopHat200_dm.npy')/1.e3
else:
    snapshot = '179'
    GroupPos = np.load(tng_dir+'data_fp/GroupPos_fp_'+str(snapshot)+'.npy')
    GroupPos_dm = np.load(tng_dir+'data_dm/GroupPos_dm_'+str(snapshot)+'.npy')
    print(GroupPos_dm.max())

cell = Lbox/N_dim
GroupPos = (GroupPos/cell).astype(int)%N_dim
Rs = [1.1, 1.5, 2., 2.5] # Mpc/h
delta_s = 0.25
p = 2

GroupEnv0 = np.load(tng_dir+f'GroupEnv_R{Rs[0]:.1f}_{fp_dm:s}_{snapshot:s}.npy')
GroupEnv1 = np.load(tng_dir+f'GroupEnv_R{Rs[1]:.1f}_{fp_dm:s}_{snapshot:s}.npy')
GroupEnv2 = np.load(tng_dir+f'GroupEnv_R{Rs[2]:.1f}_{fp_dm:s}_{snapshot:s}.npy')
GroupEnv3 = np.load(tng_dir+f'GroupEnv_R{Rs[3]:.1f}_{fp_dm:s}_{snapshot:s}.npy')
print("loaded bs")
GroupEnv = np.zeros(len(GroupR200))
GroupEnv[GroupR200 < Rs[0]] = GroupEnv0[GroupR200 < Rs[0]]
GroupEnv[(GroupR200 >= Rs[0]) & (GroupR200 < Rs[1])] = GroupEnv1[(GroupR200 >= Rs[0]) & (GroupR200 < Rs[1])]
GroupEnv[(GroupR200 >= Rs[1]) & (GroupR200 < Rs[2])] = GroupEnv2[(GroupR200 >= Rs[1]) & (GroupR200 < Rs[2])]
GroupEnv[(GroupR200 >= Rs[2]) & (GroupR200 < Rs[3])] = GroupEnv3[(GroupR200 >= Rs[2]) & (GroupR200 < Rs[3])]
assert np.sum(GroupR200 >= Rs[3]) == 0
np.save(tng_dir+f'GroupEnvAdapt_{fp_dm:s}_{snapshot:s}.npy', GroupEnv)
