import sys

import numpy as np
from scipy.spatial import cKDTree

from knn import compute_cdf, compute_jackknife_cdf, compute_jackknife_cdf_multi

dtype = np.int64

# simulation parameter
Lbox = 500 # Mpc/h
N_query = dtype(1.e7)
N_dim = 3#5

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
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
n_gal = '7.4e-04'
mstar_elg_thresh = 5e7
other_dir = '/home/boryanah/MTNG/assembly_bias/'

params = ['GroupVirial', 'GroupConc', 'GroupVelDisp', 'GroupShear_R2', 'GroupEnv_R2', 'GroupMarkedEnv_R2_s0.25_p2']
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

def downsample_counts(GroupCount, GroupCountPred):
    GroupCountCopy = GroupCount.copy()
    diff = np.abs(np.sum(GroupCountPred) - np.sum(GroupCount))
    print("difference = ", diff)
    if np.sum(GroupCountPred) < np.sum(GroupCount):
        GroupCountChange = GroupCountCopy.copy()
    else:
        GroupCountChange = GroupCountPred.copy()
    index_halo = np.arange(len(GroupCountChange), dtype=int)
    index_halo = index_halo[GroupCountChange > 0]
    count = GroupCountChange[GroupCountChange > 0].astype(float)
    count /= np.sum(count)
    #index_all = np.repeat(index_halo, count)
    samples = np.random.choice(index_halo, diff, replace=False, p=count)
    GroupCountChange[samples] -= 1
    if np.sum(GroupCountPred) < np.sum(GroupCountCopy):
        GroupCountCopy = GroupCountChange
    else:
        GroupCountPred = GroupCountChange
    
    print("predicted and true total number of galaxies", np.sum(GroupCountPred), np.sum(GroupCountCopy))
    GroupCountPred = GroupCountPred.astype(GroupPos_fp.dtype)
    GroupCountCopy = GroupCountCopy.astype(GroupPos_fp.dtype)
    return GroupCountCopy, GroupCountPred

def get_pos_sats(hid_sats, num_sats):
    sum = 0
    nosubs = 0
    pos_sats = np.zeros((int(np.sum(num_sats)), 3), dtype=GroupPos_fp.dtype)
    for i in range(len(num_sats)):
        # todo: kinda wrong cause needs to actually be order by stellar mass rather than number of particles (I think DM)
        # which subhalo index does this halo start at (+1 to get rid of the central subhalo) 
        start = GroupFirstSub_fp[hid_sats[i]]
        nsubs = GroupNsubs_fp[hid_sats[i]]
        mstar = SubhaloMstar_fp[start:start+nsubs]
        poses = SubhaloPos_fp[start:start+nsubs]
        ssfr = SubhalosSFR_fp[start:start+nsubs]
        first_pos = poses[0]
        poses = poses[1:]
        mstar = mstar[1:]
        ssfr = ssfr[1:]
        if gal_type == 'LRG':
            i_sort = np.argsort(mstar)[::-1]
        elif gal_type == 'ELG':
            i_sort = np.argsort(ssfr)[::-1]
        poses = poses[i_sort]
        mstar = mstar[i_sort]
        poses = poses[mstar >= mstar_elg_thresh]
        nsubs = poses.shape[0]
        num = int(num_sats[i])
        if nsubs < num: # should be the case v rarely (put in center)
            #pos_sats[sum:sum+num] = first_pos+GrRcrit_fp[hid_sats[i]]*0.05 
            pos_sats[sum:sum+nsubs] = poses[:nsubs]
            pos_sats[sum+nsubs:sum+num] = first_pos+GrRcrit_fp[hid_sats[i]]*0.05
            nosubs += 1
        else:
            pos_sats[sum:sum+num] = poses[:num] 
        sum += num
    print("# halos with a single subhalo = ", nosubs)
    return pos_sats


# load other halo properties
GroupPos_fp = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
GroupFirstSub_fp = np.load(tng_dir+f'data_fp/GroupFirstSub_fp_{snapshot:d}.npy')
GroupNsubs_fp = np.load(tng_dir+f'data_fp/GroupNsubs_fp_{snapshot:d}.npy')
SubhaloPos_fp = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot:d}.npy')
SubhaloMstar_fp = np.load(tng_dir+f'data_fp/SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10
SubhaloSFR_fp = np.load(tng_dir+f'data_fp/SubhaloSFR_fp_{snapshot:d}.npy')
SubhalosSFR_fp = SubhaloSFR_fp/SubhaloMstar_fp
#SubhaloGrNr_fp = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy')
#GroupPos_fp = np.zeros((len(GroupNsubs_fp), 3))
#_, inds = np.unique(SubhaloGrNr_fp, return_index=True)
#GroupPos_fp[_] = SubhaloPos_fp[inds]
GrMcrit_fp = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
GrRcrit_fp = np.load(tng_dir+f'data_fp/Group_R_TopHat200_fp_{snapshot:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountSats = GroupCount-GroupCountCent

# halo indexing
index_halo = np.arange(len(GroupCount), dtype=int)

# k nearest neighbors
ks = np.array([1, 2, 4, 8], dtype=dtype)

# bins for making plot
bins = np.geomspace(3., 50., 51)
binc = (bins[1:] + bins[:-1]) * 0.5
binc = np.vstack((binc, binc, binc, binc)).T

for i in range(len(fun_types)):
    fun_type = fun_types[i]
    
    for i_pair in range(1, len(secondaries)): # TESTING
        secondary = secondaries[i_pair]
        if fit_type == 'plane':
            tertiary = tertiaries[i_pair]
        else:
            tertiary = 'None'
        print("param pair = ", i_pair, secondary, tertiary)

        # directory of ramp and plane
        if fit_type == 'plane':
            print(f"{gal_type:s}/count_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            print(f"{gal_type:s}/count_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                GroupCountSatsPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
        else:
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                GroupCountSatsPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type:s}_cent_{secondary:s}_fp_{snapshot:d}.npy")
        GroupCountSatsPred[GroupCountSatsPred < 0] = 0
        GroupCountCentPred[GroupCountCentPred > 1] = 1
        GroupCountCentPred[GroupCountCentPred < 0] = 0

        # binomial and poisson
        GroupCountSatsPred = np.random.poisson(GroupCountSatsPred, len(GroupCountSatsPred))
        GroupCountCentPred = (np.random.rand(len(GroupCountCentPred)) < GroupCountCentPred).astype(int)

        # get the new downsampled numbers
        GroupCountCentCopy, GroupCountCentPred = downsample_counts(GroupCountCent, GroupCountCentPred)
        GroupCountSatsCopy, GroupCountSatsPred = downsample_counts(GroupCountSats, GroupCountSatsPred)

        # it is a bit expensive, but let's get the subhalo positions for both predicted and true
        index_halo_copy = index_halo.copy()
        num_sats_pred = GroupCountSatsPred[GroupCountSatsPred > 0]
        hid_sats_pred = index_halo_copy[GroupCountSatsPred > 0]
        index_halo_copy = index_halo.copy()
        num_sats_true = GroupCountSatsCopy[GroupCountSatsCopy > 0]
        hid_sats_true = index_halo_copy[GroupCountSatsCopy > 0]

        # positions of the central and satellite galaxies
        pos_cent_pred = GroupPos_fp[GroupCountCentPred > 0]
        pos_cent_true = GroupPos_fp[GroupCountCentCopy > 0]
        pos_sats_pred = get_pos_sats(hid_sats_pred, num_sats_pred)
        pos_sats_true = get_pos_sats(hid_sats_true, num_sats_true)
        pos = np.vstack((pos_cent_pred, pos_sats_pred))
        pos_ref = np.vstack((pos_cent_true, pos_sats_true))

        # compute kNN-CDF
        data_mean, data_err, data_ref_mean, data_ref_err, data_rat_mean, data_rat_err = \
            compute_jackknife_cdf_multi(pos_ref, pos, ks, N_query, N_dim, boxsize=Lbox, bins=binc)

        if fit_type == 'plane':
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                np.save(f"{gal_type:s}/kNN_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", data_mean)
                np.save(f"{gal_type:s}/kNN_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", data_err)
                np.save(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", data_rat_err)
        elif fit_type == 'ramp':
            if mode == 'bins':
                print("not implemented!")
                quit()
            elif mode == 'all':
                np.save(f"{gal_type:s}/kNN_mean_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", data_mean)
                np.save(f"{gal_type:s}/kNN_err_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", data_err)
                np.save(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", data_rat_mean)
                np.save(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", data_rat_err)

np.save(f"{gal_type:s}/binc.npy", binc)
np.save(f"{gal_type:s}/kNN_mean_fp_{snapshot:d}.npy", data_ref_mean)
np.save(f"{gal_type:s}/kNN_err_fp_{snapshot:d}.npy", data_ref_err)

quit()
data_mean, data_err = compute_jackknife_cdf(pos, ks, N_query, N_dim, boxsize=Lbox, bins=binc)
