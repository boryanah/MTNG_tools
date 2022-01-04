import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import Corrfunc

from tools.halostats import get_jack_corr
import plotparams
plotparams.buba()

np.random.seed(100) 

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
Lbox = 500. # Mpc/h
#gal_type = 'ELG'
#gal_type = 'LRG'
gal_type = sys.argv[1]
#n_gal = ['5.9e-04', '7.4e-04', '9.7e-04']
n_gal = '7.4e-04'
#mstar_elg_thresh = 8.4e9
mstar_elg_thresh = 5e7
#mstar_elg_thresh = 0.
#fit_type = 'ramp'
#fit_type = 'plane'
fit_type = sys.argv[2]
#fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
#fun_types = ['erf', 'linear'] # for mode all, I currently have only these two
fun_types = []
#fun_types = ['linear'] # TESTING
fun_type_sats = 'linear'
#mode = 'bins' # fitting in bins
mode = 'all' # fitting once for all

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
secondaries = ['GroupEnv_R2'] # TESTING
#secondaries = ['GroupConc']
tertiaries = ['GroupConc']
#tertiaries = ['GroupShear_R2']
#tertiaries = ['GroupEnv_R2']

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

# max halo mass
print("max halo mass = %.1e"%GrMcrit_fp.max())

# mass bins  # notice slightly lower upper limit cause few halos
mbins = np.logspace(10, 14, 41)
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit_fp))
choice = (mbins[-1] >= GrMcrit_fp) & (mbins[0] < GrMcrit_fp)

rbins = np.logspace(-1, 1.5, 31)
#rbins = np.logspace(-2, np.log10(20.), 31)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:]+rbins[:-1])/2.
np.save(f"{gal_type:s}/rbinc.npy", rbinc)

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
            print(f"{gal_type:s}/count_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            print(f"{gal_type:s}/count_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            if mode == 'bins':
                GroupCountSatsPred = np.load(f"{gal_type:s}/count_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(f"{gal_type:s}/count_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            elif mode == 'all':
                GroupCountSatsPred = np.load(f"{gal_type:s}/count_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(f"{gal_type:s}/count_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
        else:
            if mode == 'bins':            
                GroupCountSatsPred = np.load(f"{gal_type:s}/count_pred_{fun_type_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(f"{gal_type:s}/count_pred_{fun_type:s}_cent_{secondary:s}_fp_{snapshot:d}.npy")
            elif mode == 'all':
                GroupCountSatsPred = np.load(f"{gal_type:s}/count_pred_all_{fun_type_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy")
                GroupCountCentPred = np.load(f"{gal_type:s}/count_pred_all_{fun_type:s}_cent_{secondary:s}_fp_{snapshot:d}.npy")
        GroupCountSatsPred[GroupCountSatsPred < 0] = 0
        GroupCountCentPred[GroupCountCentPred > 1] = 1
        GroupCountCentPred[GroupCountCentPred < 0] = 0

        # leads to constant shift up
        #GroupCountCentPred = GroupCountCent.copy()
        #GroupCountSatsPred = GroupCountSats.copy()
        
        # poisson and binomial
        GroupCountSatsPred = np.random.poisson(GroupCountSatsPred, len(GroupCountSatsPred)).astype(int)
        GroupCountCentPred = (np.random.rand(len(GroupCountCentPred)) < GroupCountCentPred).astype(int)
        # TESTING # makes marginal difference
        #GroupCountSatsPred[choice] = np.random.poisson(GroupCountSatsPred[choice], len(GroupCountSatsPred[choice]))
        #GroupCountSatsPred = GroupCountSatsPred.astype(int)
        print("should be integers and equal pred = ", GroupCountSatsPred[mbins[-1] < GrMcrit_fp])
        print("should be integers and equal true = ", GroupCountSats[mbins[-1] < GrMcrit_fp])
        
        # get the new downsampled numbers
        GroupCountCentCopy, GroupCountCentPred = downsample_counts(GroupCountCent, GroupCountCentPred)
        GroupCountSatsCopy, GroupCountSatsPred = downsample_counts(GroupCountSats, GroupCountSatsPred)
        #GroupCountCentCopy, GroupCountCentPred = (GroupCountCent, GroupCountCentPred)
        #GroupCountSatsCopy, GroupCountSatsPred = (GroupCountSats, GroupCountSatsPred)
        
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
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        pos_true = np.vstack((pos_cent_true, pos_sats_true))
        w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
        w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)


        # N_dim should maybe be 5
        rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos_true, w_true, pos_pred, w_pred, Lbox, N_dim=3, bins=rbins)


    
        print("xir2 dr TRUE", np.sum(corr_true_mean*rbinc**2*drbin))
        print("xir2 dr SHUFF", np.sum(corr_shuff_mean*rbinc**2*drbin))
        # remove
        plt.figure(figsize=(9, 7))
        plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
        plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
        plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
        plt.xscale('log')
        #plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')

        plt.figure(figsize=(9, 7))
        plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
        plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
        plt.xscale('log')
        #plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')
        plt.show()
        quit()
        

        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", rat_err)
        else:
            if mode == 'bins':
                np.save(f"{gal_type:s}/corr_rat_mean_{fit_type:s}_{secondary:s}_fp_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_{fit_type:s}_{secondary:s}_fp_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_fp_{snapshot:d}.npy", rat_err)

GroupCountShuff = GroupCount.copy()
GroupCountCentShuff = GroupCountCent.copy()
GroupCountSatsShuff = GroupCountSats.copy()
for i in range(len(mbins)-1):
    choice = ((mbins[i]) < GrMcrit_fp) & ((mbins[i+1]) >= GrMcrit_fp)
    nhalo = np.sum(choice)
    if nhalo == 0: continue
    
    cts = GroupCount[choice]
    cts_cent = GroupCountCent[choice]
    cts_sats = GroupCountSats[choice]
    ngal = np.sum(cts)
    if ngal == 0: continue
    
    pos = GroupPos_fp[choice]
    cts = cts.astype(pos.dtype)
    cts_sats = cts_sats.astype(pos.dtype)
    cts_cent = cts_cent.astype(pos.dtype)
    cts_shuff = cts.copy()
    cts_sats_shuff = cts_sats.copy()
    cts_cent_shuff = cts_cent.copy()
    np.random.shuffle(cts_shuff)
    np.random.shuffle(cts_sats_shuff)
    np.random.shuffle(cts_cent_shuff)

    GroupCountShuff[choice] = cts_shuff
    GroupCountCentShuff[choice] = cts_cent_shuff
    GroupCountSatsShuff[choice] = cts_sats_shuff

# get the new downsampled numbers # not needed because number is preserved
GroupCountCentCopy, GroupCountCentPred = (GroupCountCent, GroupCountCentShuff)
GroupCountSatsCopy, GroupCountSatsPred = (GroupCountSats, GroupCountSatsShuff)

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
pos_sats_true = get_pos_sats(hid_sats_true, num_sats_true) # not a problem with this
pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
pos_true = np.vstack((pos_cent_true, pos_sats_true))
w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)

# N_dim should maybe be 5
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos_true, w_true, pos_pred, w_pred, Lbox, N_dim=3, bins=rbins)


# remove
plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
#plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
#plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')
plt.show()
quit()


np.save(f"{gal_type:s}/corr_rat_mean_shuff_fp_{snapshot:d}.npy", rat_mean)
np.save(f"{gal_type:s}/corr_rat_err_shuff_fp_{snapshot:d}.npy", rat_err)
np.save(f"{gal_type:s}/corr_mean_shuff_fp_{snapshot:d}.npy", corr_shuff_mean)
np.save(f"{gal_type:s}/corr_err_shuff_fp_{snapshot:d}.npy", corr_shuff_err)
np.save(f"{gal_type:s}/corr_mean_true_fp_{snapshot:d}.npy", corr_true_mean)
np.save(f"{gal_type:s}/corr_err_true_fp_{snapshot:d}.npy", corr_true_err)

quit()

'''
# centrals and satellies
xi = Corrfunc.theory.xi(Lbox, 16, rbins, pos[cts > 0, 0], pos[cts > 0, 1], pos[cts > 0, 2], weights=cts[cts > 0], weight_type="pair_product")['xi']
xi_shuff = Corrfunc.theory.xi(Lbox, 16, rbins, pos[cts_shuff > 0, 0], pos[cts_shuff > 0, 1], pos[cts_shuff > 0, 2], weights=cts_shuff[cts_shuff > 0], weight_type="pair_product")['xi']
plt.plot(rbinc, xi_shuff/xi, color=hexcolors_bright[i], label=r'$\log M = %.1f$'%mbinc[i])

rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts > 0], cts[cts > 0], pos[cts_shuff > 0], cts_shuff[cts_shuff > 0], Lbox, N_dim=5, bins=rbins)

plt.errorbar(rbinc*(1.+0.03*i), rat_mean, yerr=rat_err, ls='-', capsize=4, color=hexcolors_bright[i], label=r'$\log M = %.1f$'%mbinc[i])
'''
'''
# centrals
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts_cent > 0], cts[cts_cent > 0], pos[cts_cent_shuff > 0], cts_cent_shuff[cts_cent_shuff > 0], Lbox, N_dim=5, bins=rbins)

#np.savez(f'{gal_type:s}/{gal_type:s}_corr_bin_{i:d}_cent_fp_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[i])
    
# satellites
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts_sats > 0], cts[cts_sats > 0], pos[cts_sats_shuff > 0], cts_sats_shuff[cts_sats_shuff > 0], Lbox, N_dim=5, bins=rbins)

#np.savez(f'{gal_type:s}/{gal_type:s}_corr_bin_{i:d}_sats_fp_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[i])
'''

"""
per weird idea
GroupCountCopy = GroupCount.copy()

GroupCountPred = GroupCountSatsPred+GroupCountCentPred
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
"""
