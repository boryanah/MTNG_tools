"""
Given halo property, predict halo occupancy and satellite distn and save galaxy distribution for DMO simulation
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import special
from scipy.interpolate import interp1d

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    z_dict[snaps[i]] = zs[i]

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_type = sys.argv[1] # 'LRG' # 'ELG'
fit_type = sys.argv[2] # 'ramp' # 'plane'
fun_cent = 'linear' # 'tanh' # 'erf' # 'gd' # 'abs' # 'arctan'
fun_sats = 'linear'
method = 'powell' # 'Nelder-Mead'
mode = 'all' #'all', 'bins'
subsamp_or_subhalos = 'subsamp' #'subhalos' # 'subsamp'
dm_str = "dm"
p0 = np.array([0., 0.]) 
Lbox = 500.
if len(sys.argv) > 3:
    want_vrad = int(sys.argv[3])
else:
    want_vrad = False
vrad_str = "_vrad" if want_vrad else ""
if len(sys.argv) > 4:
    want_splash = int(sys.argv[4])
else:
    want_splash = False
splash_str = "_splash" if want_splash else ""
if len(sys.argv) > 5:
    n_gal = sys.argv[5]
else:
    n_gal = '2.0e-03' # '7.0e-04'
if len(sys.argv) > 6:
    snapshot = int(sys.argv[6])
    if dm_str == 'dm':
        offset = 5
    elif dm_str == 'fp':
        offset = 0
    snapshot_dm = snapshot + offset
    redshift = z_dict[snapshot]
else:
    snapshot = 179;
    if dm_str == 'dm':
        offset = 5
    elif dm_str == 'fp':
        offset = 0
    snapshot_dm = snapshot + offset
    redshift = 1.
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{snapshot:d}_{n_gal}")

def downsample_counts(GroupCount, GroupCountPred):
    GroupCountCopy = GroupCount.copy()
    diff = (np.sum(GroupCountPred) - np.sum(GroupCount))
    print("difference = ", diff)
    if diff < 0:
        GroupCountChange = GroupCountCopy.copy()
        halo_index = np.arange(len(GroupCountChange), dtype=int)
        halo_index = halo_index[GroupCountChange > 0]
        count = GroupCountChange[GroupCountChange > 0].astype(float)
        count /= np.sum(count)
        #index_all = np.repeat(halo_index, count)
        samples = np.random.choice(halo_index, np.abs(diff), replace=False, p=count)
        GroupCountChange[samples] -= 1
        GroupCountCopy = GroupCountChange
    else:
        GroupCountChange = GroupCountPred.copy()
        halo_index_dm = np.arange(len(GroupCountChange), dtype=int)
        halo_index_dm = halo_index_dm[GroupCountChange > 0]
        count = GroupCountChange[GroupCountChange > 0].astype(float)
        count /= np.sum(count) # normalize so it sums to 1
        #index_all = np.repeat(halo_index, count)
        samples = np.random.choice(halo_index_dm, np.abs(diff), replace=False, p=count)
        GroupCountChange[samples] -= 1
        GroupCountPred = GroupCountChange
    
    assert np.sum(GroupCountPred) == np.sum(GroupCountCopy), "must be equal after changing"
    GroupCountPred = GroupCountPred.astype(GroupPos.dtype)
    GroupCountCopy = GroupCountCopy.astype(GroupPos.dtype)
    return GroupCountCopy, GroupCountPred

def like_cent(pars):
    # predicted a and b values
    a = pars[0]
    b = pars[1]
    if np.abs(a)+np.abs(b) > 2.: return np.inf
    
    # array with the probability of each halo to possess a central galaxy
    p = prob_cent(a, b)

    # individual log likelihoods
    tol = 1.e-6
    p[p < tol] = tol
    p[p > 1.-tol] = 1.-tol
    ln_like_i = GroupCountCent[choice]*np.log(p) + (1.-GroupCountCent[choice])*np.log(1.-p)
    
    # compute binomial likelihood
    ln_like = np.sum(ln_like_i)
    #print("logLike, a, b = ", ln_like, a, b)
    ln_like *= -1.
    return ln_like


def prob_linear_cent(a, b):
    p = (1. + (a*x[choice] + b*y[choice])*(1.-cts_cent_mean[choice])) * cts_cent_mean[choice]
    return p

def prob_linear_cent_dm(a, b):
    p = (1. + (a*x_dm[choice_dm] + b*y_dm[choice_dm])*(1.-cts_cent_mean_dm[choice_dm])) * cts_cent_mean_dm[choice_dm]
    return p

def prob_erf_cent(a, b):
    z = special.erfinv(2.*cts_cent_mean[choice] - 1.)
    p = 0.5*(1. + special.erf(z + a*x[choice] + b*y[choice]))
    return p

def prob_tanh_cent(a, b):
    z = np.arctanh(2.*cts_cent_mean[choice] - 1.)
    p = 0.5*(1. + np.tanh(z + a*x[choice] + b*y[choice]))
    return p

def prob_abs_cent(a, b):
    arg = (2.*cts_cent_mean[choice] - 1.)
    z = np.zeros(np.sum(choice))
    z[arg > 0.] = arg[arg > 0.]/(1-arg[arg > 0.])
    z[arg <= 0.] = arg[arg <= 0.]/(1+arg[arg <= 0.])
    z_mod = z + a*x[choice] + b*y[choice]
    p = 0.5*(1. + z_mod/(1+np.abs(z_mod)))
    return p

def prob_arctan_cent(a, b):
    arg = (2.*cts_cent_mean[choice] - 1.)
    z = np.tan(arg*np.pi/2.)*2/np.pi
    z_mod = z + a*x[choice] + b*y[choice]
    p = 0.5*(1. + 2./np.pi*np.arctan(np.pi/2.*z_mod))
    return p

def prob_gd_cent(a, b):
    arg = (2.*cts_cent_mean[choice] - 1.)
    z = np.pi/2.*np.log(np.abs(np.tan(np.pi/2.*arg/2.+np.pi/4.)))
    z_mod = z + a*x[choice] + b*y[choice]
    p = 0.5*(1. + 2./np.pi*np.arcsin(np.tanh(z_mod)))
    return p

def like_sats(pars):
    # predicted a and b values
    a = pars[0]
    b = pars[1]
    if np.abs(a)+np.abs(b) > 2.: return np.inf
    
    # array with the probability of each halo to possess a satsral galaxy
    lamb = prob_sats(a, b)

    # individual log likelihoods
    tol = 1.e-6
    lamb[lamb < tol] = tol
    ln_like_i = GroupCountSats[choice]*np.log(lamb) - lamb
    
    # compute binomial likelihood
    ln_like = np.sum(ln_like_i)
    ln_like *= -1.
    #print("logLike, a, b = ", ln_like, a, b)
    return ln_like

def prob_linear_sats(a, b):
    p = (1. + a*x[choice] + b*y[choice]) * cts_sats_mean[choice]
    return p

def prob_linear_sats_dm(a, b):
    p = (1. + a*x_dm[choice_dm] + b*y_dm[choice_dm]) * cts_sats_mean_dm[choice_dm]
    return p

# gamma 184 and 179
# vani 179
#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt']#, 'GroupMarkedEnv_s0.25_p2']#, 'GroupGamma'] # 'GroupConcRad'
params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupEnv_R1.5', 'GroupShear_R1.5', 'GroupConcRad', 'GroupVirial', 'GroupSnap_peak', 'GroupVelDisp', 'GroupPotential', 'Group_M_Splash', 'Group_R_Splash', 'GroupNsubs', 'GroupSnap_peak', 'GroupMarkedEnv_R2.0_s0.25_p2', 'GroupHalfmassRad']
n_combos = len(params)*(len(params)-1)//2
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
#secondaries = ['None'] # important
#secondaries = ['GroupEnv_R2']
#secondaries = ['GroupEnvAdapt']
#secondaries = ['GroupMarkedEnv_R2_s0.25_p2']
#secondaries = ['GroupGamma']
#secondaries = ['GroupConc']
#secondaries = ['GroupHalfmassRad']
#secondaries = ['GroupVelDispSqR']
#secondaries = ['GroupVelAni']
#secondaries = ['GroupPotentialCen'] # helps the most I think
#tertiaries = ['GroupConc']

if fun_cent == 'linear':
    prob_cent = prob_linear_cent
    prob_cent_dm = prob_linear_cent_dm
elif fun_cent == 'erf':
    prob_cent = prob_erf_cent
elif fun_cent == 'gd':
    prob_cent = prob_gd_cent
elif fun_cent == 'abs':
    prob_cent = prob_abs_cent
elif fun_cent == 'arctan':
    prob_cent = prob_arctan_cent
elif fun_cent == 'tanh':
    prob_cent = prob_tanh_cent
if fun_sats == 'linear':
    prob_sats = prob_linear_sats
    prob_sats_dm = prob_linear_sats_dm

# load other halo properties
SubhaloGrNr = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy')
SubhaloPos = np.load(tng_dir+f"data_fp/SubhaloPos_fp_{snapshot:d}.npy")
SubhaloVel = np.load(tng_dir+f"data_fp/SubhaloVel_fp_{snapshot:d}.npy")
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
GroupVel = np.load(tng_dir+f'data_fp/GroupVel_fp_{snapshot:d}.npy')*(1.+redshift) # peculiar velocity
GroupVelDisp = np.load(tng_dir+f'data_fp/GroupVelDisp_fp_{snapshot:d}.npy') # 1D velocity dispersion
GroupVelDisp *= np.sqrt(3.) # should make it 3D
GroupFirstSub = np.load(tng_dir+f'data_fp/GroupFirstSub_fp_{snapshot:d}.npy')
GroupNsubs = np.load(tng_dir+f'data_fp/GroupNsubs_fp_{snapshot:d}.npy')
if want_splash:
    GrMcrit = np.load(tng_dir+f'data_fp/Group_M_Splash_fp_{snapshot:d}.npy')
else:
    GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
GrRcrit = np.load(tng_dir+f'data_fp/Group_R_TopHat200_fp_{snapshot:d}.npy')
index_halo = np.arange(len(GrMcrit), dtype=int)

# TESTING! needs to be regenerated for splash
# load galaxy sample info
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")
if want_splash:
    print("make pretty")
    gal_par = np.abs(np.load(f'/home/boryanah/MTNG/splashback/data/galaxy_parent_rsplash_fp_{snapshot:d}.npy'))
    assert len(gal_par) == len(index)
    index_cent, comm1, comm2 = np.intersect1d(index, GroupFirstSub[gal_par], return_indices=True) # I think only one element reported for the intersection
    parent_cent = gal_par[comm2]
    index_sats = index[~np.in1d(index, index_cent)]
    parent_sats = gal_par[~np.in1d(index, index_cent)]
    print("cent frac = ", len(index_cent)/len(index))

    # count unique halo repetitions
    par_uni, cts = np.unique(gal_par, return_counts=True)
    GroupCount = np.zeros(len(GrMcrit), dtype=int)
    GroupCount[par_uni] = cts
    """
    # leads to assertion error
    par_cent_uni, cts = np.unique(parent_cent, return_counts=True)
    GroupCountCent = np.zeros(len(GrMcrit), dtype=int)
    GroupCountCent[par_cent_uni] = cts
    GroupCountSats = GroupCount-GroupCountCent
    """
    par_sats_uni, cts = np.unique(parent_sats, return_counts=True)
    GroupCountSats = np.zeros(len(GrMcrit), dtype=int)
    GroupCountSats[par_sats_uni] = cts
    GroupCountCent = GroupCount-GroupCountSats
    print("GroupCount", GroupCountCent.min(), GroupCountCent.max(), GroupCountSats.max())
else:
    # identify central subhalos
    _, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

    # which galaxies are centrals
    index_cent = np.intersect1d(index, sub_inds_cent)
    index_sats = index[~np.in1d(index, index_cent)]
    parent_sats = SubhaloGrNr[index_sats]

    # group info (from get_hod.py)
    GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
    GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
    GroupCountSats = GroupCount-GroupCountCent

print(f"minimum halo mass with any counts = {np.min(GrMcrit[GroupCount > 0]):.2e}")

    
# load dm halo properties
# TESTING! splashback is different not sure what this means maybe indexing?
GroupPos_dm = np.load(tng_dir+f'data_{dm_str}/GroupPos_{dm_str}_{snapshot_dm:d}.npy')
GrMcrit_dm = np.load(tng_dir+f'data_{dm_str}/Group_M_TopHat200_{dm_str}_{snapshot_dm:d}.npy')*1.e10
GrRcrit_dm = np.load(tng_dir+f'data_{dm_str}/Group_R_TopHat200_{dm_str}_{snapshot_dm:d}.npy')
GroupVelDisp_dm = np.load(tng_dir+f'data_{dm_str}/GroupVelDisp_{dm_str}_{snapshot_dm:d}.npy')
GroupVel_dm = np.load(tng_dir+f'data_{dm_str}/GroupVel_{dm_str}_{snapshot_dm:d}.npy')
index_halo_dm = np.arange(len(GrMcrit_dm), dtype=int) # could sort this to preserve the original order
i_sort = np.argsort(GrMcrit_dm)[::-1]
GroupPos_dm = GroupPos_dm[i_sort]
GrMcrit_dm = GrMcrit_dm[i_sort]
GroupVel_dm = GroupVel_dm[i_sort]
GroupVelDisp_dm = GroupVelDisp_dm[i_sort]
GrRcrit_dm = GrRcrit_dm[i_sort]
index_halo_dm = index_halo_dm[i_sort]
print("Note that because of the sorting, you can't use SubhaloGrNr")
    
# TESTING! needs to be regenerated and sorted
if subsamp_or_subhalos == 'subsamp':
    # load particle subsamples for some halos
    """
    GroupSubsampIndex = np.load(f"../hod_subsamples/data/subsample_halo_index_fp_{snapshot:d}.npy")
    GroupSubsampFirst = np.zeros(len(GroupCount), dtype=int)
    GroupSubsampSize = np.zeros(len(GroupCount), dtype=int)
    GroupSubsampFirst[GroupSubsampIndex] = np.load(f"../hod_subsamples/data/subsample_nstart_fp_{snapshot:d}.npy")
    GroupSubsampSize[GroupSubsampIndex] = np.load(f"../hod_subsamples/data/subsample_nsize_fp_{snapshot:d}.npy")
    PartSubsampPos = np.load(f"../hod_subsamples/data/subsample_pos_fp_{snapshot:d}.npy")
    PartSubsampVel = np.load(f"../hod_subsamples/data/subsample_vel_fp_{snapshot:d}.npy")
    """
    print("missing subsampling files!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    GroupSubsampFirst = np.zeros(len(GrMcrit_dm), dtype=int)
    GroupSubsampIndex = np.zeros(len(GrMcrit_dm), dtype=int)
    GroupSubsampSize = np.zeros(len(GrMcrit_dm), dtype=int)
    PartSubsampPos = np.zeros((100, 3))
    PartSubsampVel = np.zeros((100, 3))
    
# galaxy properties
mcrit_sats = GrMcrit[parent_sats]
rcrit_sats = GrRcrit[parent_sats]
print(f"mcrit_sats = {mcrit_sats.min():.2e}, {mcrit_sats.max():.2e}")
vcrit_sats = GroupVelDisp[parent_sats]
xdiff_sats = SubhaloPos[index_sats]-GroupPos[parent_sats]
vdiff_sats = SubhaloVel[index_sats]-GroupVel[parent_sats]
xdiff_sats[xdiff_sats > Lbox/2.] -= Lbox
xdiff_sats[xdiff_sats < -Lbox/2.] += Lbox
sbdnm_sats = np.linalg.norm(xdiff_sats, axis=1)
sbvnm_sats = np.linalg.norm(vdiff_sats, axis=1)
hat_r_sats = xdiff_sats/sbdnm_sats[:, None]
v_rad_sats = np.sum(vdiff_sats*hat_r_sats, axis=1)/sbvnm_sats
v_rad_sats = np.nan_to_num(v_rad_sats)
assert len(v_rad_sats) == len(mcrit_sats)
    
print("max_dist = ", sbdnm_sats.max()) # 7.9 Mpc/h
print("min_dist =", sbdnm_sats.min()) # 0.0 Mpc/h probably sometimes there is a larger subhalo that's not first just cause it's not where the min of the potential is?
print("min_dist (no zeros) =", sbdnm_sats[sbdnm_sats > 0.].min()) # 7.e-6 Mpc/h 
sbdnm_sats /= rcrit_sats # no NaNs
sbvnm_sats /= vcrit_sats # no NaNs
print("v dist = ", sbvnm_sats.min(), sbvnm_sats.max())
print("d dist = ", sbdnm_sats.min(), sbdnm_sats.max())
print("vr dist = ", v_rad_sats.min(), v_rad_sats.max())

# define radial and velocity bins
if want_splash:
    vbins = np.linspace(0., 10.2, 51)
    many_vs = np.linspace(0., 10.2, 1000)

    rbins = np.logspace(-3, 1.45, 61)
    many_rs = np.logspace(-3, 1.45, 1000)
else:
    vbins = np.linspace(0., 3., 51)
    many_vs = np.linspace(0., 3., 1000)

    rbins = np.logspace(-3, 1., 61)
    many_rs = np.logspace(-3, 1., 1000)
rbinc = (rbins[1:] + rbins[:-1])*0.5
vbinc = (vbins[1:] + vbins[:-1])*0.5
vrbins = np.linspace(-1.3, 1.3, 41)
vrbinc = (vrbins[1:]+vrbins[:-1])*.5
many_vrs = np.linspace(-1., 1., 100)

# need to add Lbox/4. to all the dm positions because of offset
print("NOTICE THAT WE ARE CORRECTING FOR BOX/4 SHIFT") # checked
GroupPos_dm += np.array([Lbox/4., 0., 0.])
GroupPos_dm %= Lbox

# bins for env and conc
cbins = np.linspace(-0.5, 0.5, 5)
ebins = np.linspace(-0.5, 0.5, 5)
ebins[0] -= 0.1 # fixes the assertion error
cbins[0] -= 0.1
ebins[-1] += 0.1
cbins[-1] += 0.1 

# max halo mass
print("max halo mass = %.1e"%GrMcrit.max())

# mass bins # notice slightly lower upper limit cause few halos
n_top = 400
m_top = np.sort(GrMcrit)[::-1][n_top]
print(f"maximum halo mass = {m_top:.2e}")
mbins = np.logspace(11, np.log10(m_top), 31) # with splashback
mbinc = (mbins[1:]+mbins[:-1])*0.5
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit))
mbins_dm = np.zeros(len(mbins))
assert np.min(GrMcrit[GroupCount > 0]) > mbins[0]

# halos that are never moved
n_top = np.sum(mbins[-1] < GrMcrit)
print("number of halos above the last mass bin = ", n_top)
mbins_dm[-1] = GrMcrit_dm[n_top]
print("total occupancy of the top halos = ", np.sum(GroupCountCent[(mbins[-1] < GrMcrit)])+np.sum(GroupCountSats[(mbins[-1] < GrMcrit)]))

for i_pair in range(len(secondaries)):
    # read secondary and tertiary property names
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]
    else:
        tertiary = 'None'
    print("param pair = ", i_pair, secondary, tertiary)

    # array with predicted counts 
    GroupCountCentPred_dm = np.zeros(len(GrMcrit_dm))
    GroupCountSatsPred_dm = np.zeros(len(GrMcrit_dm))
    GroupCountSatsPred_dm[:n_top] = GroupCountSats[(mbins[-1] < GrMcrit)]
    GroupCountCentPred_dm[:n_top] = GroupCountCent[(mbins[-1] < GrMcrit)]

    # load secondary and tertiary property
    if secondary != 'None':
        GroupEnv = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
        GroupEnv_dm = np.load(tng_dir+f'data_{dm_str}/{secondary:s}_{dm_str}_{snapshot_dm:d}.npy')[i_sort] # crucial to sort!!!
    else:
        GroupEnv = np.zeros(len(GrMcrit))
        GroupEnv_dm = np.zeros(len(GrMcrit_dm))
    if fit_type == 'ramp':
        GroupConc = np.zeros(len(GroupEnv))
        GroupConc_dm = np.zeros(len(GroupEnv_dm))
    else:
        GroupConc = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')
        GroupConc_dm = np.load(tng_dir+f'data_{dm_str}/{tertiary:s}_{dm_str}_{snapshot_dm:d}.npy')[i_sort]

    """
    plt.scatter(GrMcrit_dm[mbins[10] < GrMcrit_dm], GroupEnv_dm[mbins[10] < GrMcrit_dm], s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    """
    # concentrations, environments of the satellite galaxies
    conc_sats = GroupConc[SubhaloGrNr[index_sats]]
    env_sats = GroupEnv[SubhaloGrNr[index_sats]]

    # initialize ranking arrays and mean number of gals in a halo per mass bin
    x = np.zeros_like(GrMcrit)
    y = np.zeros_like(GrMcrit)
    cts_sats_mean = np.zeros_like(GrMcrit)
    cts_cent_mean = np.zeros_like(GrMcrit)

    # initialize same arrays for dmo
    x_dm = np.zeros_like(GrMcrit_dm)
    y_dm = np.zeros_like(GrMcrit_dm)
    cts_sats_mean_dm = np.zeros_like(GrMcrit_dm)
    cts_cent_mean_dm = np.zeros_like(GrMcrit_dm)

    # only used if getting a and b per halo mass bin
    a_arr_cent = np.zeros(len(mbins)-1)
    b_arr_cent = np.zeros(len(mbins)-1)
    a_arr_sats = np.zeros(len(mbins)-1)
    b_arr_sats = np.zeros(len(mbins)-1)
    
    # initialize counter for each mass bin (needed for abundance matching)
    sum_halo = 0
    sum_halo += n_top

    # looping over each mass bin
    for i in range(len(mbins)-1)[::-1]: # crucial to invert cause of abundance matching
        # mass bin choice
        choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        # equivalent choice for dmo
        choice_dm = np.arange(sum_halo, sum_halo+nhalo, dtype=int) # TESTING (equiv?)
        #choice_dm = (mbins_dm[i] < GrMcrit_dm) & (mbins_dm[i+1] >= GrMcrit_dm) # og
        mbins_dm[i] = GrMcrit_dm[sum_halo+nhalo] # low bound
        sum_halo += nhalo # incrementing for abundance matching

        # counts and params for this bin
        cts = GroupCount[choice]
        cts_cent = GroupCountCent[choice]
        cts_sats = GroupCountSats[choice]

        # if no galaxies in that mass bin, skip to next mass bin
        if np.sum(cts) == 0: continue

        # otherwise select sec and tert property for halos in mass bin
        env = GroupEnv[choice]
        conc = GroupConc[choice]
        env_dm = GroupEnv_dm[choice_dm]
        conc_dm = GroupConc_dm[choice_dm]
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)
        rank_env_dm = np.argsort(np.argsort(env_dm))/(len(env_dm)-1)-0.5
        if fit_type == 'plane':
            rank_conc_dm = np.argsort(np.argsort(conc_dm))/(len(conc_dm)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc_dm = np.zeros_like(rank_env_dm)

        # record mean central and satellite occupation
        cts_sats_mean[choice] = np.mean(cts_sats)
        cts_cent_mean[choice] = np.mean(cts_cent)
        cts_sats_mean_dm[choice_dm] = np.mean(cts_sats)
        cts_cent_mean_dm[choice_dm] = np.mean(cts_cent)

        # record ranked arrays into x and y
        x[choice] = rank_env
        y[choice] = rank_conc
        x_dm[choice_dm] = rank_env_dm
        y_dm[choice_dm] = rank_conc_dm

        # if we are fitting a and b parameters individually for each mass bin
        if mode == 'bins':
            
            # then we minimize for a and b
            if secondary != 'None':
                res_cent = minimize(like_cent, p0, method=method)
                res_sats = minimize(like_sats, p0, method=method)
                a_cent, b_cent = res_cent['x']
                a_sats, b_sats = res_sats['x']
                print(f"a_cent, b_cent = {a_cent:.4f}, {b_cent:.4f}")
                print(f"a_sats, b_sats = {a_sats:.4f}, {b_sats:.4f}")
            else:
                a_cent, b_cent = 0., 0.
                a_sats, b_sats = 0., 0.
                
            # compute prediction for occupancy given best-fit a and b and save into counts arrays (equiv to true counts)
            GroupCountCentPred_dm[choice_dm] = prob_cent_dm(a_cent, b_cent)
            GroupCountSatsPred_dm[choice_dm] = prob_sats_dm(a_sats, b_sats)

            # save the best-fit a and b values
            a_arr_cent[i], b_arr_cent[i] = a_cent, b_cent
            a_arr_sats[i], b_arr_sats[i] = a_sats, b_sats

    # if we are fitting a single a and b parameter for all mass bins (for cent and sats)
    if mode == 'all':
        # use only the halos within the mass range of interest (speeds up)
        choice = (mbins[0] < GrMcrit) & (mbins[-1] >= GrMcrit)
        
        # then we minimize for a and b
        if secondary != 'None':
            res_cent = minimize(like_cent, p0, method=method)
            res_sats = minimize(like_sats, p0, method=method)
            a_cent, b_cent = res_cent['x']
            a_sats, b_sats = res_sats['x']
            print(f"a_cent, b_cent = {a_cent:.4f}, {b_cent:.4f}")
            print(f"a_sats, b_sats = {a_sats:.4f}, {b_sats:.4f}")
        else:
            a_cent, b_cent = 0., 0.
            a_sats, b_sats = 0., 0.
        
        # select equivalent halos in dmo
        #choice_dm = (mbins_dm[0] < GrMcrit_dm) & (mbins_dm[-1] >= GrMcrit_dm) # og        
        choice_dm = np.arange(n_top, sum_halo, dtype=int) # TESTING (equiv?)
        
        # compute prediction for occupancy and save into a new array for centrals and satellites
        GroupCountCentPred_dm[choice_dm] = prob_cent_dm(a_cent, b_cent)
        GroupCountSatsPred_dm[choice_dm] = prob_sats_dm(a_sats, b_sats)
    
    # print out number of galaxies
    print("pred satellites = ", np.sum(GroupCountSatsPred_dm))
    print("true satellites = ", np.sum(GroupCountSats))
    print("pred centrals = ", np.sum(GroupCountCentPred_dm))
    print("true centrals = ", np.sum(GroupCountCent))

    # make sure we don't get negative counts or values larger than one for the centrals (doesn't make a difference)
    GroupCountSatsPred_dm[GroupCountSatsPred_dm < 0] = 0
    GroupCountCentPred_dm[GroupCountCentPred_dm > 1] = 1
    GroupCountCentPred_dm[GroupCountCentPred_dm < 0] = 0

    # draw from a poisson and a binomial distribution for the halos in the mass range of interest
    choice = (mbins[-1] >= GrMcrit) & (mbins[0] < GrMcrit)
    choice_dm = (mbins_dm[-1] >= GrMcrit_dm) & (mbins_dm[0] < GrMcrit_dm)
    GroupCountSatsPred_dm[choice_dm] = np.random.poisson(GroupCountSatsPred_dm[choice_dm], len(GroupCountSatsPred_dm[choice_dm]))
    GroupCountCentPred_dm = (np.random.rand(len(GroupCountCentPred_dm)) < GroupCountCentPred_dm)
    GroupCountSatsPred_dm = GroupCountSatsPred_dm.astype(int)
    GroupCountCentPred_dm = GroupCountCentPred_dm.astype(int)
    print("pred poisson satellites = ", np.sum(GroupCountSatsPred_dm), np.sum(GroupCountSats))
    print("pred binomial centrals = ", np.sum(GroupCountCentPred_dm), np.sum(GroupCountCent))
    print("-------------------")
    
    # downsample the galaxies in order for the pred to have the same number of satellites and centrals as the truth
    GroupCountCentCopy = GroupCountCent.copy()
    GroupCountSatsCopy = GroupCountSats.copy() 
    GroupCountCentCopy[choice], GroupCountCentPred_dm[choice_dm] = downsample_counts(GroupCountCent[choice], GroupCountCentPred_dm[choice_dm])
    GroupCountSatsCopy[choice], GroupCountSatsPred_dm[choice_dm] = downsample_counts(GroupCountSats[choice], GroupCountSatsPred_dm[choice_dm])
    GroupCountSatsPred_dm = GroupCountSatsPred_dm.astype(int)
    GroupCountCentPred_dm = GroupCountCentPred_dm.astype(int)
    print("after downsampling, sats cent = ", np.sum(GroupCountSatsPred_dm), np.sum(GroupCountCentPred_dm))

    """
    # TESTING!!!!!!!!!!!!
    import Corrfunc
    GroupCount = GroupCountSats + GroupCountCent
    w_true = GroupCount[GroupCount > 0].astype(np.float32)
    x_true = GroupPos[GroupCount > 0].astype(np.float32)
    GroupCountPred_dm = GroupCountCentPred_dm + GroupCountSatsPred_dm
    w_shuff = GroupCountPred_dm[GroupCountPred_dm > 0].astype(x_true.dtype)
    x_shuff = GroupPos_dm[GroupCountPred_dm > 0].astype(x_true.dtype)
    rbins = np.logspace(-1, 1.5, 31)
    rbinc = (rbins[1:]+rbins[:-1])*.5
    xi = Corrfunc.theory.xi(Lbox, 16, rbins, *x_true.T, weights=w_true, weight_type="pair_product")['xi']
    xi_shuff = Corrfunc.theory.xi(Lbox, 16, rbins, *x_shuff.T, weights=w_shuff, weight_type="pair_product")['xi']
    plt.plot(rbinc, np.ones(len(rbinc)), color='black', ls='--')
    plt.plot(rbinc, xi_shuff/xi)
    plt.xscale('log')
    plt.ylim([0.0, 2.0])
    plt.show()
    quit()
    """
    
    # record pos and vel of centrals and parent halo index
    pos_pred_sats = np.zeros((np.sum(GroupCountSatsPred_dm), 3))
    vel_pred_sats = np.zeros((np.sum(GroupCountSatsPred_dm), 3))
    ind_pred_sats = np.zeros(np.sum(GroupCountSatsPred_dm), dtype=int)

    pos_pred_cent = GroupPos_dm[GroupCountCentPred_dm > 0]
    vel_pred_cent = GroupVel_dm[GroupCountCentPred_dm > 0]
    ind_pred_cent = index_halo_dm[GroupCountCentPred_dm > 0]
    
    # counter over satellites given
    sum_sats = 0
    
    # initialize counter for each mass bin (needed for abundance matching)
    for i in range(len(mbins)-1):
        mchoice = ((mbins_dm[i]) < GrMcrit_dm) & ((mbins_dm[i+1]) >= GrMcrit_dm)
        nhalo = np.sum(mchoice)
        if nhalo == 0: continue

        # abundance matched fp halo selection (only needed for setting thresholds for the satellite profile array)
        mchoice_fp = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        
        # predicted satellite occupancies in the mass bin (skip if none)
        cts_sats_pred = GroupCountSatsPred_dm[mchoice]
        if np.sum(cts_sats_pred) == 0: continue

        # values of sec and tert prop, pos, vel, radius, dispersion, first, nsub and halo inds in mass bin
        env = GroupEnv_dm[mchoice]
        conc = GroupConc_dm[mchoice]
        pos = GroupPos_dm[mchoice]
        vel = GroupVel_dm[mchoice]
        rcrit = GrRcrit_dm[mchoice]
        vcrit = GroupVelDisp_dm[mchoice]
        if subsamp_or_subhalos == 'subhalos':
            first = GroupFirstSub[mchoice] # tuks
            nsub = GroupNsubs[mchoice]
        elif subsamp_or_subhalos == 'subsamp':
            first = GroupSubsampFirst[mchoice]
            nsub = GroupSubsampSize[mchoice]
        index = index_halo_dm[mchoice]

        # env and conc for the fp halos (only needed for thresholds)
        env_fp = GroupEnv[mchoice_fp]
        conc_fp = GroupConc[mchoice_fp]
        
        # select the true satellites in mass bin (needed for drawing from radial distn)
        mchoice_sats = ((mbins[i]) < mcrit_sats) & ((mbins[i+1]) >= mcrit_sats)

        # select sec and tert prop of the satellite hosts and norm dist/vel to center
        conc_ms = conc_sats[mchoice_sats]
        env_ms = env_sats[mchoice_sats]
        sbdnm_ms = sbdnm_sats[mchoice_sats]
        sbvnm_ms = sbvnm_sats[mchoice_sats]
        sbvrnm_ms = v_rad_sats[mchoice_sats]

        # compute histogram of satellite dist/vel to center (needed only if no true sats at sec and tert prop)
        hist_sats, _ = np.histogram(sbdnm_ms, bins=rbins)
        p_sats = hist_sats/np.sum(hist_sats)
        hist_sats, _ = np.histogram(sbvnm_ms, bins=vbins)
        v_sats = hist_sats/np.sum(hist_sats)
        hist_sats, _ = np.histogram(sbvrnm_ms, bins=vrbins)
        vr_sats = hist_sats/np.sum(hist_sats)
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)

        # turn the secondary and tertiary parameter into ranked arrays
        rank_env_fp = np.argsort(np.argsort(env_fp))/(len(env_fp)-1)-0.5
        if fit_type == 'plane':
            rank_conc_fp = np.argsort(np.argsort(conc_fp))/(len(conc_fp)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc_fp = np.zeros_like(rank_env_fp)

        # identify the 0, 25, 50, 75 and 100th percentiles
        env_thresh = np.zeros(5)
        conc_thresh = np.zeros(5)
        for j in range(1, len(env_thresh)-1):
            env_thresh[j] = env_fp[np.argmin(np.abs(rank_env_fp - ebins[j]))]
            conc_thresh[j] = conc_fp[np.argmin(np.abs(rank_conc_fp - cbins[j]))]
        env_thresh[0] = env_fp.min()-0.1
        conc_thresh[0] = conc_fp.min()-0.1
        env_thresh[-1] = env_fp.max()+0.1
        conc_thresh[-1] = conc_fp.max()+0.1

        # initialize counter for number of times we didn't find true satellites at this bin of mass, c and e (used for rad prof)
        # and counter for number of predicted satellites that are close to an existing subhalo
        no_true_sats = 0
        close_to_exist = 0
        total_given = 0
        for j in range(len(cbins)-1):
            for k in range(len(ebins)-1):
                # select the halos in that bin of mass, c and e
                rchoice = (rank_env <= ebins[k+1]) & (rank_env > ebins[k])
                if fit_type == 'plane':
                    rchoice &= (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j])

                # number of pred satellites per halo and total
                ct = cts_sats_pred[rchoice]
                ng = np.sum(ct)
                if ng == 0: continue

                # select the satellites whose parents are part of this bin of mass, c and e
                rchoice_ms = (env_ms <= env_thresh[k+1]) & (env_ms > env_thresh[k])
                if fit_type == 'plane':
                    rchoice_ms &= (conc_ms <= conc_thresh[j+1]) & (conc_ms > conc_thresh[j])

                # if there are no examples of true satellites living in halos in this bin, then draw from the sat distn in the mass bin
                if np.sum(rchoice_ms) == 0:
                    radius = np.random.choice(rbinc, ng, p=p_sats)
                    velius = np.random.choice(vbinc, ng, p=v_sats)
                    velrad = np.random.choice(vrbinc, ng, p=vr_sats)
                    no_true_sats += ng
                else:
                    # otherwise compute the true rad and disp distn at this bin
                    hist_r, _ = np.histogram(sbdnm_ms[rchoice_ms], bins=rbins)
                    hist_v, _ = np.histogram(sbvnm_ms[rchoice_ms], bins=vbins)
                    hist_vr, _ = np.histogram(sbvrnm_ms[rchoice_ms], bins=vrbins)
                    
                    assert np.sum(hist_r) > 0 # assert we do have some objects in the rad bins
                    assert np.sum(hist_v) > 0 # assert we do have some objects in the disp bins
                    assert np.sum(hist_vr) > 0 # assert we do have some objects in the vrad bins

                    # if there is only one radial bin with information, just take whatever you are offered (can't interpolate with 1 point)
                    if np.sum(hist_r > 0.) == 1 or np.sum(hist_v > 0.) == 1 or np.sum(hist_vr > 0.) == 1:
                        pr = hist_r/np.sum(hist_r)
                        radius = np.random.choice(rbinc, ng, p=pr)
                        pv = hist_v/np.sum(hist_v)
                        velius = np.random.choice(vbinc, ng, p=pv)
                        pvr = hist_vr/np.sum(hist_vr)
                        velrad = np.random.choice(vrbinc, ng, p=pvr)
                    else:
                        # interpolate to get the rad and disp distn (must have two points for each)
                        pr = interp1d(rbinc[hist_r > 0.], hist_r[hist_r > 0.], bounds_error=False, fill_value=0.)(many_rs)
                        pr /= np.sum(pr)
                        radius = np.random.choice(many_rs, ng, p=pr)
                        pv = interp1d(vbinc[hist_v > 0.], hist_v[hist_v > 0.], bounds_error=False, fill_value=0.)(many_vs)
                        pv /= np.sum(pv)
                        velius = np.random.choice(many_vs, ng, p=pv)
                        pvr = interp1d(vrbinc[hist_vr > 0.], hist_vr[hist_vr > 0.], bounds_error=False, fill_value=0.)(many_vrs)
                        pvr /= np.sum(pvr)
                        velrad = np.random.choice(many_vrs, ng, p=pvr)


                # index of first subhalo and total number of subhalos for each halo in bin of mass, c and e
                start = first[rchoice]
                npout = nsub[rchoice]
                rcr = rcrit[rchoice] # halo rad
                vcr = vcrit[rchoice] # halo disp
                ind = index[rchoice] # halo index

                # starting index into `radius` and `velius` arrays (st and ct are a pair; start and npout are a pair)
                st = np.zeros(len(ct), dtype=int)
                st[1:] = np.cumsum(ct)[:-1]
                
                # loop over halos in this bin of mass, c and e
                for n in range(len(start)):
                    if ct[n] == 0: continue # if there are no satellites here, skip
                    
                    # scale the predicted `radii` and `velii` by halo rad and disp 
                    radii = radius[st[n]:st[n]+ct[n]] * rcr[n]
                    velii = velius[st[n]:st[n]+ct[n]] * vcr[n]
                    vradii = velrad[st[n]:st[n]+ct[n]]  # number between -1 and 1
                    
                    # select pos and vel of subhalos/pcles in this halo
                    if subsamp_or_subhalos == 'subhalos':
                        poses = SubhaloPos[start[n]:start[n]+npout[n]] # tuks
                        vels = SubhaloVel[start[n]:start[n]+npout[n]]
                    elif subsamp_or_subhalos == 'subsamp':
                        poses = PartSubsampPos[start[n]:start[n]+npout[n]]
                        vels = PartSubsampVel[start[n]:start[n]+npout[n]]
                        if npout[n] == 0:
                            #print(f"eh empty {ct[n]:d} {GrMcrit_dm[mchoice][rchoice][n]:.2e}");
                            poses = np.zeros((2,3)); vels = np.zeros((2,3)); # tuks # TESTING
                    
                    # compute distance to halo center of subhalos/pcles
                    diffs = poses - pos[rchoice][n] # pos[rchoice][n], SubhaloPos[start[n]]: same
                    diffs[diffs > Lbox/2.] -= Lbox
                    diffs[diffs < -Lbox/2.] += Lbox
                    dists = np.linalg.norm(diffs, axis=1)
                    #norms = diffs/dists[:, None] # 3d
                    #assert norms.shape[0] == len(radii)
                    
                    # loop over each pred satellite in this halo
                    for m in range(len(radii)):
                        # min arg of |dist to center of predicted halo - dists of all the subhalos in the halo|
                        i_min = np.argmin(np.abs(radii[m] - dists))
                        dist_min = dists[i_min]
                        # if we are far from any subhalos at the pred rad
                        if True:#dist_min > .1: # (TESTING)
                            # draw random 3d position
                            theta = np.arccos(2.*np.random.rand()-1.)
                            phi = np.random.rand()*2.*np.pi
                            x = np.cos(phi)*np.sin(theta)
                            y = np.sin(phi)*np.sin(theta)
                            z = np.cos(theta)
                            p = radii[m]*np.array([x, y, z])
                            
                            # add the offset from the center
                            pos_pred_sats[sum_sats] = p + pos[rchoice][n] #SubhaloPos[start[n]] # shouldn't matter much

                            if want_vrad:
                                # find perpendicular vector and normalize it
                                #perp = np.array([1., 1., -(x+y)/z])
                                #perp /= np.linalg.norm(perp)
                                #v = (np.sqrt(1.-vradii[m]**2)*perp + vradii[m]*p/radii[m]) * velii[m]
                                u = np.array([1., 1., 0.,])/np.sqrt(2)
                                v1 = np.array([x, y, z])
                                if np.all(np.isclose(u, v1)):
                                    u = np.array([1., 1., 1.,])/np.sqrt(3)
                                v2 = np.cross(v1, u)
                                v2 /= np.linalg.norm(v2)
                                v3 = np.cross(v1, v2)
                                v3 /= np.linalg.norm(v3)
                                phi = np.random.rand()*2.*np.pi
                                v = vradii[m]*v1 + np.sqrt(1.-vradii[m]**2)*(np.cos(phi)*v2 + np.sin(phi)*v3)
                                v *= velii[m]
                                if np.any(np.isnan(v)): print("nan"); v = velii[m]*vradii[m]*v1
                            else:
                                # draw random 3d velocity
                                theta = np.arccos(2.*np.random.rand()-1.)
                                phi = np.random.rand()*2.*np.pi
                                x = np.cos(phi)*np.sin(theta)
                                y = np.sin(phi)*np.sin(theta)
                                z = np.cos(theta)
                                v = velii[m]*np.array([x, y, z])
                                
                            # add the offset from the center
                            vel_pred_sats[sum_sats] = v + vel[rchoice][n]# I think this is less noisy than SubhaloVel[start[n]]
                        else:
                            # if we are close, just take the info from the closest subhalo
                            pos_pred_sats[sum_sats] = poses[i_min]
                            vel_pred_sats[sum_sats] = vels[i_min]
                            close_to_exist += 1

                        # record halo index of newly assigned satellite and increment counter
                        ind_pred_sats[sum_sats] = ind[n]
                        sum_sats += 1
                        total_given += 1
                
            # if ramp you need only to cycle through environments
            if fit_type == 'ramp':
                break

        # report numbers for this mass bin
        print("no true galaxies at this mass bin to get profile from = ", no_true_sats)
        print("number of pred satellites close to an existing subhalo = ", close_to_exist)
        print("percentage of pred satellites close to an existing subhalo = ", close_to_exist*100./total_given)

    # ensure there are no satellites below that threshold (would still work if only centrals below (since we record them before)) if we want this just add below mass bin
    assert np.sum(mcrit_sats <= mbins[0]) == 0

    # take true satellite pos, vel and halo ind of the halos above the mass threshold
    pos_new = SubhaloPos[index_sats[mcrit_sats > mbins[-1]]]
    vel_new = SubhaloVel[index_sats[mcrit_sats > mbins[-1]]]
    ind_new = parent_sats[mcrit_sats > mbins[-1]]

    # ensure that the number of remaining satellites is equal to the expected number of remaining satellites
    assert pos_new.shape[0] == pos_pred_sats[sum_sats:].shape[0], f"{pos_new.shape[0]:d}, {pos_pred_sats[sum_sats:].shape[0]:d}"

    # record satellite pos, vel and halo index of remaining pred sats
    pos_pred_sats[sum_sats:sum_sats+pos_new.shape[0]] = pos_new
    vel_pred_sats[sum_sats:sum_sats+pos_new.shape[0]] = vel_new
    ind_pred_sats[sum_sats:sum_sats+pos_new.shape[0]] = ind_new
    
    # number of satellites we have assigned so far (i.e. excluding most massive halos)
    sum_sats += pos_new.shape[0]

    # final number of pred satellites vs expected number of satellites
    assert sum_sats == pos_pred_sats.shape[0] # dmo case should be fine if abundance matching
    pos_pred_sats = pos_pred_sats[:sum_sats]
    vel_pred_sats = vel_pred_sats[:sum_sats]
    ind_pred_sats = ind_pred_sats[:sum_sats]

    # record all the information (pos, vel and halo inds of pred cent and sats) 
    if fit_type == 'plane':
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_cent)
    else:
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{dm_str}_{snapshot_dm:d}.npy", ind_pred_cent)

