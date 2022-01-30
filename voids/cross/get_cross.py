import os

import numpy as np
import matplotlib.pyplot as plt
import Corrfunc
from Corrfunc.theory.DD import DD

# corr func
N_bin = 14
bins = np.logspace(np.log10(0.8),np.log10(50),N_bin)
bin_centers = (bins[:-1] + bins[1:])/2.

# parameter choices
Lbox = 500. # Mpc/h
N_dim = 512

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
str_snap = f"_{snapshot:d}"
#gal_type = 'ELG'
gal_type = 'LRG'
#fit_type = 'ramp'
fit_type = 'plane'
print(fit_type)
other_dir = '/home/boryanah/MTNG/assembly_bias/'
#mode = 'bins'
mode = 'all'
fun_type = 'erf'
combo = 'GroupVelDisp', 'GroupEnv_R2'

# void param
type_corr = 'cross'
n_random = 35

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
    
    pos_sats = np.zeros((int(np.sum(num_sats)), 3), dtype=GroupPos_fp.dtype)
    for i in range(len(num_sats)):
        # todo: kinda wrong cause needs to actually be order by stellar mass rather than number of particles (I think DM)
        # which subhalo index does this halo start at (+1 to get rid of the central subhalo) 
        start = GroupFirstSub_fp[hid_sats[i]]
        nsubs = GroupNsubs_fp[hid_sats[i]]
        mstar = SubhaloMstar_fp[start:start+nsubs]
        poses = SubhaloPos_fp[start:start+nsubs]
        i_sort = np.argsort(mstar)[::-1]
        poses = poses[i_sort]
        num = int(num_sats[i])
        pos_sats[sum:sum+num] = poses[1:num+1]
        sum += num
    return pos_sats

# load other halo properties
GroupPos_fp = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
GroupFirstSub_fp = np.load(tng_dir+f'data_fp/GroupFirstSub_fp_{snapshot:d}.npy')
GroupNsubs_fp = np.load(tng_dir+f'data_fp/GroupNsubs_fp_{snapshot:d}.npy')
SubhaloPos_fp = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot:d}.npy')
SubhaloMstar_fp = np.load(tng_dir+f'data_fp/SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_fp_{snapshot:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_fp_{snapshot:d}.npy")
GroupCountSats = GroupCount-GroupCountCent

# halo indexing
index_halo = np.arange(len(GroupCount), dtype=int)

def get_cross(pos1,pos1_r,pos2,pos2_r,n1,n2,n_thread=16,periodic=True):
    X_jack_g = pos1[:,0]
    Y_jack_g = pos1[:,1]
    Z_jack_g = pos1[:,2]

    X_jack_m = pos2[:,0]
    Y_jack_m = pos2[:,1]
    Z_jack_m = pos2[:,2]

    N_g = n1
    N_m = n2


    X_jack_r_m = pos2_r[:,0]
    Y_jack_r_m = pos2_r[:,1]
    Z_jack_r_m = pos2_r[:,2]
    
    X_jack_r_g = pos1_r[:,0]
    Y_jack_r_g = pos1_r[:,1]
    Z_jack_r_g = pos1_r[:,2]

    N_r_g = N_g*n_random
    N_r_m = N_m*n_random
    
    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    DD_gm = results['npairs'].astype(float)
    DD_gm /= (N_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    DR_gm = results['npairs'].astype(float)
    DR_gm /= (N_g*1.*N_r_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    RD_gm = results['npairs'].astype(float)
    RD_gm /= (N_r_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    RR_gm = results['npairs'].astype(float)
    RR_gm /= (N_r_g*1.*N_r_m)

    Corr_gm = (DD_gm-DR_gm-RD_gm+RR_gm)/RR_gm

    return Corr_gm

def get_pos(pos_g,xyz,size):
    pos_g_jack = pos_g.copy()
    bool_arr = np.prod((xyz == (pos_g/size).astype(int)),axis=1).astype(bool)
    pos_g_jack[bool_arr] = np.array([0.,0.,0.])
    pos_g_jack = pos_g_jack[np.sum(pos_g_jack,axis=1)!=0.]
    return pos_g_jack

def get_random(pos,dtype=np.float64):
    N = pos.shape[0]
    N_r = N*n_random
    pos_r = np.random.uniform(0.,Lbox,(N_r,3)).astype(dtype)
    return pos_r
    
def get_jack(pos1,pos2,pos3,pos4):
    pos1_r = get_random(pos1,pos1.dtype)
    pos2_r = get_random(pos2,pos2.dtype)
    pos3_r = get_random(pos3,pos3.dtype)
    pos4_r = get_random(pos4,pos4.dtype)

    # TESTING
    N1 = pos1.shape[0]
    N2 = pos2.shape[0]
    N3 = pos3.shape[0]
    N4 = pos4.shape[0]
        
    
    N_dim = 5
    size = Lbox/N_dim
    corr12 = np.zeros((N_bin-1,N_dim**3))
    corr34 = np.zeros((N_bin-1,N_dim**3))
    rat = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                print(xyz)                
                xyz1_jack = get_pos(pos1,xyz,size)
                xyz2_jack = get_pos(pos2,xyz,size)
                xyz_r1_jack = get_pos(pos1_r,xyz,size)
                xyz_r2_jack = get_pos(pos2_r,xyz,size)

                xyz3_jack = get_pos(pos3,xyz,size)
                xyz4_jack = get_pos(pos4,xyz,size)
                xyz_r3_jack = get_pos(pos3_r,xyz,size)
                xyz_r4_jack = get_pos(pos4_r,xyz,size)

                Corr12 = get_cross(xyz1_jack,xyz_r1_jack,xyz2_jack,xyz_r2_jack,n1=N1,n2=N2)
                corr12[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12
                Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack,n1=N3,n2=N4)
                corr34[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34

                rat[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34/Corr12
                
    Corr12_mean = np.mean(corr12,axis=1)
    Corr12_error = np.sqrt(N_dim**3-1)*np.std(corr12,axis=1)
    Corr34_mean = np.mean(corr34,axis=1)
    Corr34_error = np.sqrt(N_dim**3-1)*np.std(corr34,axis=1)
    Rat_mean = np.mean(rat,axis=1)
    Rat_error = np.sqrt(N_dim**3-1)*np.std(rat,axis=1)
    return Corr12_mean, Corr12_error, Corr34_mean, Corr34_error, Rat_mean, Rat_error

for i_pair in range(len(secondaries)):
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]

    if fit_type == 'ramp':
        other = f'{fit_type:s}_{secondary:s}'
    elif fit_type == 'plane':
        other = f'{fit_type:s}_{secondary:s}_{tertiary:s}'

    # only for plane
    if (combo[0] != secondary) or (combo[1] != tertiary): continue

    # todo load voids
    pos_void = void[:, :3].astype(np.float32)
    size_void = void[:, 3]
    max_void = size_void.max()
    
    void_ref = np.loadtxt(void_dic[sim_type_ref])
    pos_void_ref = void_ref[:, :3].astype(np.float32)
    size_void_ref = void_ref[:, 3]
    max_void_ref = size_void_ref.max()
        
    print("param pair = ", i_pair, other)
    # directory of ramp and plane
    if fit_type == 'plane':
        if mode == 'bins':
            GroupCountSatsPred = np.load(other_dir+f"{gal_type:s}/count_pred_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            GroupCountCentPred = np.load(other_dir+f"{gal_type:s}/count_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
        elif mode == 'all':
            GroupCountSatsPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
            GroupCountCentPred = np.load(other_dir+f"{gal_type:s}/count_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
    else:
        GroupCountSatsPred = np.load(other_dir+f"{gal_type:s}/count_pred_{fit_type:s}_sats_{secondary:s}_fp_{snapshot:d}.npy")
        GroupCountCentPred = np.load(other_dir+f"{gal_type:s}/count_pred_{fit_type:s}_cent_{secondary:s}_fp_{snapshot:d}.npy")
    GroupCountSatsPred[GroupCountSatsPred < 0] = 0
    GroupCountCentPred[GroupCountCentPred > 1] = 1
    GroupCountCentPred[GroupCountCentPred < 0] = 0


    # poisson and binomial
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

    # needs to match pos_gal and pos_gal_ref
    # positions of the central and satellite galaxies
    pos_cent_pred = GroupPos_fp[GroupCountCentPred > 0]
    pos_cent_true = GroupPos_fp[GroupCountCentCopy > 0]
    pos_sats_pred = get_pos_sats(hid_sats_pred, num_sats_pred)
    pos_sats_true = get_pos_sats(hid_sats_true, num_sats_true)
    pos_g = np.vstack((pos_cent_pred, pos_sats_pred))
    pos_g_other = np.vstack((pos_cent_true, pos_sats_true))
    w_g = np.ones(pos_g.shape[0], dtype=pos_g.dtype)
    w_g_other = np.ones(pos_g_other.shape[0], dtype=pos_g_other.dtype)

    if type_corr == 'cross':
        cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(pos_void_ref, pos_gal_ref, pos_void, pos_gal)

    elif type_corr == 'auto-gg':
        cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(pos_gal_ref, pos_gal_ref, pos_gal, pos_gal)

    elif type_corr == 'auto-vv':
        cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(pos_void_ref, pos_void_ref, pos_void, pos_void)

    
    np.save("data/bin_cents.npy",bin_centers)
    np.save(f"data/rat_cross_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_mean.npy",rat_opt_mean)
    np.save(f"data/rat_cross_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_error.npy",rat_opt_err)
    np.save("data/cross_true_mean.npy",cross_mean)
    np.save("data/cross_true_error.npy",cross_err)
    np.save(f"data/cross_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_mean.npy",cross_opt_mean)
    np.save(f"data/cross_{fun_type:s}_sats_{secondary:s}_{tertiary:s}_error.npy",cross_opt_err)
