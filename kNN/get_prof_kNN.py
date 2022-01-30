import sys

import numpy as np
from scipy.spatial import cKDTree

from knn import compute_cdf, compute_jackknife_cdf, compute_jackknife_cdf_multi, compute_cdf_multi

dtype = np.int64

# simulation parameter
Lbox = 500 # Mpc/h
N_query = dtype(1.e7)
N_dim = 3#5

# k nearest neighbors
ks = np.array([1, 2, 4, 8], dtype=dtype)

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
fp_dm = 'fp'; snapshot = 179; snapshot_fp = 179;
fp_dm = 'dm'; snapshot = 184; snapshot_fp = 179;
#gal_type = 'ELG'
#gal_type = 'LRG'
gal_type = sys.argv[1]
#fit_type = 'ramp'
#fit_type = 'plane'
fit_type = sys.argv[2]
fun_types = ['linear']
fun_type_sats = 'linear'
#mode = 'bins' # fitting in bins
mode = 'all' # fitting once for all
print(fit_type)
#n_gal = '7.4e-04'
n_gal = '2.0e-03'
other_dir = '/home/boryanah/MTNG/assembly_bias/'

# rad vs not rad
params = ['GroupVirial', 'GroupConcRad', 'GroupVelDisp', 'GroupShear_R2', 'GroupEnv_R2', 'GroupMarkedEnv_R2_s0.25_p2']
n_combos = len(params)*(len(params)-1)//2
print("combos = ", n_combos)
if 'ramp' == fit_type:
    secondaries = params.copy()
    tertiaries = ['None']
else:
    secondaries = []
    tertiaries = []
    for i_param in range(len(params)):
        for j_param in range(len(params)):
            if i_param <= j_param: continue
            if params[i_param] < params[j_param]:
                secondaries.append(params[i_param])
                tertiaries.append(params[j_param])
                print(params[i_param], params[j_param])
            else:
                secondaries.append(params[j_param])
                tertiaries.append(params[i_param])
                print(params[j_param], params[i_param])
print("combos = ", len(secondaries))

# load other halo properties
SubhaloPos = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot_fp:d}.npy')
SubhaloGrNr = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot_fp:d}.npy')
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot_fp:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot_fp:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot_fp:d}.npy")
GroupCountSats = GroupCount-GroupCountCent
GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot_fp:d}.npy')*1.e10

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot_fp:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)


# bins for making plot
bins = np.geomspace(3., 50., 51)
binc = (bins[1:] + bins[:-1]) * 0.5
binc = np.vstack((binc, binc, binc, binc)).T

for i in range(len(fun_types)):
    fun_type = fun_types[i]
    
    for i_pair in range(len(secondaries)):
        secondary = secondaries[i_pair]
        if fit_type == 'plane':
            tertiary = tertiaries[i_pair]
        else:
            tertiary = 'None'
        print("param pair = ", i_pair, secondary, tertiary)

        # directory of ramp and plane
        if fit_type == 'plane':
            print(f"{gal_type:s}/pos_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
            print(f"{gal_type:s}/pos_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                pos_sats_pred = np.load(other_dir+f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(other_dir+f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
        else:
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                pos_sats_pred = np.load(other_dir+f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(other_dir+f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
        pos = np.vstack((pos_cent_pred, pos_sats_pred))

        pos_cent_true = SubhaloPos[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos[index_sats[:len(pos_sats_pred)]]
        pos_ref = np.vstack((pos_cent_true, pos_sats_true))

        pos %= Lbox
        pos_ref %= Lbox

        # compute kNN-CDF # TESTING
        #data_mean, data_ref_mean, data_rat_mean = compute_cdf_multi(pos_ref, pos, ks, N_query, boxsize=Lbox, bins=binc)
        #    compute_jackknife_cdf_multi(pos_ref, pos, ks, N_query, N_dim, boxsize=Lbox, bins=binc)
        data_mean, data_err, data_ref_mean, data_ref_err, data_rat_mean, data_rat_err = \
                compute_jackknife_cdf_multi(pos_ref, pos, ks, N_query, N_dim, boxsize=Lbox, bins=binc)
        
        if fit_type == 'plane':
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                np.save(f"{gal_type:s}/kNN_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy", data_mean)
                np.save(f"{gal_type:s}/kNN_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy", data_err)
                np.save(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy", data_rat_err)
        elif fit_type == 'ramp':
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                np.save(f"{gal_type:s}/kNN_mean_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy", data_mean)
                np.save(f"{gal_type:s}/kNN_err_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy", data_err)
                np.save(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy", data_rat_err)

quit()
data_mean, data_err = compute_jackknife_cdf(pos, ks, N_query, N_dim, boxsize=Lbox, bins=binc)
