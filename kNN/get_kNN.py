"""
Compute the correlation function of prediction and true for different halo properties
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Corrfunc
from astropy.cosmology import FlatLambdaCDM

from knn import compute_cdf, compute_jackknife_cdf, compute_jackknife_cdf_multi
import plotparams
plotparams.buba()

np.random.seed(100) 

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    z_dict[snaps[i]] = zs[i]

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
#tng_dir = "/mnt/alan1/boryanah/MTNG/"
tng_dir = "/mnt/alan1/boryanah/MTNG/dm_arepo/"
tng_dir_fp = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500. # Mpc/h
gal_type = sys.argv[1]
fit_type = sys.argv[2]
fun_types = ['linear']
fun_type_sats = 'linear'
#fp_dm = 'fp'
fp_dm = 'dm'
if "arepo" in tng_dir:
    tng_dir_fp = tng_dir
#mode = 'bins' # fitting in bins
mode = 'all' # fitting once for all
ass_dir = "/home/boryanah/MTNG/assembly_bias/"
if gal_type == 'ELG':
    want_drad = False #True # TESTING
    want_cond = False #True
    want_pseudo = False #True
else:
    want_drad = False
    want_cond = False
    want_pseudo = False
drad_str = "_drad" if want_drad else ""
cond_str = "_cond" if want_cond else ""
pseudo_str = "_pseudo" if want_pseudo else ""
want_vrad = False
vrad_str = "_vrad" if want_vrad else ""
want_fixocc = False
fixocc_str = "_fixocc" if want_fixocc else ""
want_splash = False
splash_str = "_splash" if want_splash else ""
if len(sys.argv) > 3:
    n_gal = sys.argv[3]
else:
    n_gal = '2.0e-03' # '7.0e-04'
if len(sys.argv) > 4:
    snapshot_fp = int(sys.argv[4])
    if fp_dm == 'dm' and "arepo" not in tng_dir:
        offset = 5
    else:
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = z_dict[snapshot_fp]
else:
    snapshot_fp = 179;
    if fp_dm == 'dm' and "arepo" not in tng_dir:
        offset = 5
    else:
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = z_dict[snapshot_fp]
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{pseudo_str}_{drad_str}_{fixocc_str}_{cond_str}_{fp_dm}_{snapshot:d}_{n_gal}")

# k nearest neighbors
dtype = np.int64
ks = np.array([1, 2, 4, 8], dtype=dtype)

# bins for making plot
bins = np.geomspace(3., 50., 51)
binc = (bins[1:] + bins[:-1]) * 0.5
binc = np.vstack((binc, binc, binc, binc)).T
np.save(f"{gal_type}/rbinc.npy", binc)

N_query = dtype(1.e7)
N_dim = 3 #5

params = ['GroupConc', 'SubhaloMass_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupVelAni']
#params = []
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
if fit_type == 'ramp':
    secondaries.append('None')

# load other halo properties
SubhaloPos_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot_fp:d}.npy')
SubhaloVel_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloVel_{fp_dm}_{snapshot_fp:d}.npy')
SubhaloGrNr_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloGroupNr_{fp_dm}_{snapshot_fp:d}.npy')

# indices of the galaxies
if 'dm_arepo' in tng_dir:
    index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}_dm_arepo.npy")
else:
    index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr_fp, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)

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
            print(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            print(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            if mode == 'bins':
                pos_sats_pred = np.load(ass_dir+f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(ass_dir+f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(ass_dir+f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(ass_dir+f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")

            elif mode == 'all':
                pos_sats_pred = np.load(ass_dir+f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(ass_dir+f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(ass_dir+f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(ass_dir+f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(ass_dir+f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(ass_dir+f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
        else:
            if mode == 'bins':            
                pos_sats_pred = np.load(ass_dir+f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(ass_dir+f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(ass_dir+f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(ass_dir+f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")

            elif mode == 'all':
                pos_sats_pred = np.load(ass_dir+f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(ass_dir+f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(ass_dir+f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(ass_dir+f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(ass_dir+f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(ass_dir+f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")

        
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        vel_pred = np.vstack((vel_cent_pred, vel_sats_pred))
        
        pos_cent_true = SubhaloPos_fp[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos_fp[index_sats[:len(pos_sats_pred)]]
        vel_cent_true = SubhaloVel_fp[index_cent[:len(pos_cent_pred)]]
        vel_sats_true = SubhaloVel_fp[index_sats[:len(pos_sats_pred)]]

        pos_true = np.vstack((pos_cent_true, pos_sats_true))
        vel_true = np.vstack((vel_cent_true, vel_sats_true))
        print("nans = ", np.sum(np.isnan(pos_pred)))
        print("nans = ", np.sum(np.isnan(pos_true)))
        print("nans = ", np.sum(np.isnan(vel_pred)))
        print("nans = ", np.sum(np.isnan(vel_true)))
        
        pos_true %= Lbox
        pos_pred %= Lbox
        w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
        w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)
        print("true and fake difference = ", len(w_pred)-len(w_true))
        print("fake number = ", len(w_pred))

        pos_true = pos_true.astype(np.float32)
        pos_pred = pos_pred.astype(np.float32)
        
        # compute kNN-CDF
        data_mean, data_err, data_ref_mean, data_ref_err, data_rat_mean, data_rat_err = \
            compute_jackknife_cdf_multi(pos_true, pos_pred, ks, N_query, N_dim, boxsize=Lbox, bins=binc)
        if secondary == 'None':
            np.save(f"{gal_type}/kNN_true_mean_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_ref_mean)
            np.save(f"{gal_type}/kNN_shuff_mean_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_mean)
            np.save(f"{gal_type}/kNN_rat_shuff_mean_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_mean)
            np.save(f"{gal_type}/kNN_rat_shuff_err_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_err)
            quit()
        

        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type}/kNN_rat_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type}//kNN_rat_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_err)

            elif mode == 'all':
                np.save(f"{gal_type}/kNN_rat_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type}//kNN_rat_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_err)

        else:
            if mode == 'bins':
                np.save(f"{gal_type}/kNN_rat_mean_all_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type}//kNN_rat_err_all_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_err)

            elif mode == 'all':
                np.save(f"{gal_type}/kNN_rat_mean_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type}//kNN_rat_err_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", data_rat_err)
    
