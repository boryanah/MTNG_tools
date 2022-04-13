"""
Given halo property, predict halo occupancy and satellite distn and save galaxy distribution
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

from pseudo_poisson import pseudo_poisson

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
want_pseudo_poisson = True # currently only turned on for ELGs
mode = 'all'#'all', 'bins'
sat_type = 'subhalos' #'subsamp' #'subhalos' #'subsamp' #'subhalos' # 'subsamp' 
fp_dm = 'fp'
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
    if fp_dm == 'dm':
        offset = 5
    elif fp_dm == 'fp':
        offset = 0
    snapshot_dm = snapshot + offset
    redshift = z_dict[snapshot]
else:
    snapshot = 179;
    if fp_dm == 'dm':
        offset = 5
    elif fp_dm == 'fp':
        offset = 0
    snapshot_dm = snapshot + offset
    redshift = 1.
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{fp_dm}_{snapshot:d}_{n_gal}")

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

#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupEnv_R1.5', 'GroupShear_R1.5', 'GroupConcRad', 'GroupVirial', 'GroupSnap_peak', 'GroupVelDisp', 'GroupPotential', 'Group_M_Splash', 'Group_R_Splash', 'GroupNsubs', 'GroupSnap_peak', 'GroupMarkedEnv_R2.0_s0.25_p2', 'GroupHalfmassRad']
params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupNsubs']
params = []
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

if fun_cent == 'linear':
    prob_cent = prob_linear_cent
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

# load other halo properties
SubhaloGrNr = np.load(tng_dir+f'data_{fp_dm}/SubhaloGroupNr_{fp_dm}_{snapshot:d}.npy')
SubhaloPos = np.load(tng_dir+f"data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot:d}.npy")
SubhaloVel = np.load(tng_dir+f"data_{fp_dm}/SubhaloVel_{fp_dm}_{snapshot:d}.npy")
GroupPos = np.load(tng_dir+f'data_{fp_dm}/GroupPos_{fp_dm}_{snapshot:d}.npy')
GroupVel = np.load(tng_dir+f'data_{fp_dm}/GroupVel_{fp_dm}_{snapshot:d}.npy')*(1.+redshift) # peculiar velocity
GroupVelDisp = np.load(tng_dir+f'data_{fp_dm}/GroupVelDisp_{fp_dm}_{snapshot:d}.npy') # 1D velocity dispersion
GroupVelDisp *= np.sqrt(3.) # should make it 3D
GroupFirstSub = np.load(tng_dir+f'data_{fp_dm}/GroupFirstSub_{fp_dm}_{snapshot:d}.npy')
# TESTING!!!!!!!!!!
#GroupPos = SubhaloPos[GroupFirstSub]
GroupNsubs = np.load(tng_dir+f'data_{fp_dm}/GroupNsubs_{fp_dm}_{snapshot:d}.npy')
if want_splash:
    GrMcrit = np.load(tng_dir+f'data_{fp_dm}/Group_M_Splash_{fp_dm}_{snapshot:d}.npy')
else:
    GrMcrit = np.load(tng_dir+f'data_{fp_dm}/Group_M_TopHat200_{fp_dm}_{snapshot:d}.npy')*1.e10
Group_M_Crit200 = np.load(tng_dir+f'data_{fp_dm}/Group_M_Crit200_{fp_dm}_{snapshot:d}.npy')
#GrRcrit = np.load(tng_dir+f'data_{fp_dm}/Group_R_TopHat200_{fp_dm}_{snapshot:d}.npy')
GrRcrit = np.ones_like(GrMcrit) # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
index_halo = np.arange(len(GrMcrit), dtype=int)

# load galaxy sample info
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")
if want_splash:
    print("make pretty")
    gal_par = np.abs(np.load(f'/home/boryanah/MTNG/splashback/data/galaxy_parent_rsplash_{fp_dm}_{snapshot:d}.npy'))
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
    GroupCount = np.load(tng_dir+f"data_{fp_dm}/GroupCount{gal_type:s}_{n_gal:s}_{fp_dm}_{snapshot:d}.npy")
    GroupCountCent = np.load(tng_dir+f"data_{fp_dm}/GroupCentsCount{gal_type:s}_{n_gal:s}_{fp_dm}_{snapshot:d}.npy")
    GroupCountSats = GroupCount-GroupCountCent

print(f"minimum halo mass with any counts = {np.min(GrMcrit[GroupCount > 0]):.2e}")

    
if sat_type == 'subsamp':
    # load particle subsamples for some halos
    
    GroupSubsampIndex = np.load(f"../hod_subsamples/data/subsample_halo_index_{fp_dm}_{snapshot:d}.npy")
    GroupSubsampFirst = np.zeros(len(GroupCount), dtype=int)
    GroupSubsampSize = np.zeros(len(GroupCount), dtype=int)
    GroupSubsampFirst[GroupSubsampIndex] = np.load(f"../hod_subsamples/data/subsample_nstart_{fp_dm}_{snapshot:d}.npy")
    GroupSubsampSize[GroupSubsampIndex] = np.load(f"../hod_subsamples/data/subsample_nsize_{fp_dm}_{snapshot:d}.npy")
    PartSubsampPos = np.load(f"../hod_subsamples/data/subsample_pos_{fp_dm}_{snapshot:d}.npy")
    PartSubsampVel = np.load(f"../hod_subsamples/data/subsample_vel_{fp_dm}_{snapshot:d}.npy")
    """
    print("not using subsampling files!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    GroupSubsampFirst = np.zeros(len(GrMcrit), dtype=int)
    GroupSubsampIndex = np.zeros(len(GrMcrit), dtype=int)
    GroupSubsampSize = np.zeros(len(GrMcrit), dtype=int)
    PartSubsampPos = np.zeros((100, 3))
    PartSubsampVel = np.zeros((100, 3))
    """
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

    #rbins = np.logspace(-3, 1., 61) # og
    #rbins = np.logspace(-3, 1., 10000) # og
    #many_rs = np.logspace(-3, 1., 10000) # og
    rbins = np.geomspace(sbdnm_sats.min()*0.99999, 10., 91) # TESTING!!!!!!!!!!!!!!!
    #rbins[0] = 0. # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    many_rs = np.geomspace(sbdnm_sats.min()*0.99999, 10., 1000)
    #many_rs = np.logspace(-5, 1., 1000) # TESTING!!!!!!!!!!!!!!!!!!!
vrbins = np.linspace(-1.3, 1.3, 41)
many_vrs = np.linspace(-1., 1., 100)
rbinc = (rbins[1:] + rbins[:-1])*0.5
vbinc = (vbins[1:] + vbins[:-1])*0.5
vrbinc = (vrbins[1:]+vrbins[:-1])*.5


"""
hist_sats, _ = np.histogram(sbdnm_sats, bins=rbins)
p_sats = hist_sats/np.sum(hist_sats)
plt.plot(rbinc, hist_sats, label='sats')
plt.legend()
plt.show()
vcrit_cent = GroupVelDisp[SubhaloGrNr[index_cent]]
vdiff_cent = SubhaloVel[index_cent]-GroupVel[SubhaloGrNr[index_cent]]
sbvnm_cent = np.sqrt(np.sum((vdiff_cent)**2, axis=1))/vcrit_cent
hist_cent, _ = np.histogram(sbvnm_cent, bins=vbins, density=True)
hist_sats, _ = np.histogram(sbvnm_sats, bins=vbins, density=True)
plt.plot(vbinc, hist_cent, label='cent')
plt.plot(vbinc, hist_sats, label='sats')
plt.legend()
plt.show()
quit()
"""

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
mbins = np.logspace(11, np.log10(m_top), 31)
mbinc = (mbins[1:]+mbins[:-1])*0.5
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit))

for i_pair in range(len(secondaries)):
    # read secondary and tertiary property names
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]
    else:
        tertiary = 'None'
    print("param pair = ", i_pair, secondary, tertiary)

    # array with predicted counts
    GroupCountCentPred = (GroupCountCent.copy()).astype(GrMcrit.dtype)
    GroupCountSatsPred = (GroupCountSats.copy()).astype(GrMcrit.dtype)

    # load secondary and tertiary property
    if 'None' != secondary: 
        GroupEnv = np.load(tng_dir+f'data_{fp_dm}/{secondary:s}_{fp_dm}_{snapshot:d}.npy')
    else:
        GroupEnv = np.zeros(len(GrMcrit))
    if fit_type == 'ramp':
        GroupConc = np.zeros(len(GroupEnv))
    else:
        GroupConc = np.load(tng_dir+f'data_{fp_dm}/{tertiary:s}_{fp_dm}_{snapshot:d}.npy')
            
    # concentrations of the satellite galaxies
    conc_sats = GroupConc[parent_sats]
    env_sats = GroupEnv[parent_sats]

    # create empty arrays for the mean counts per halo and ranked sec/tert prop
    x = np.zeros_like(GrMcrit)
    y = np.zeros_like(GrMcrit)
    cts_sats_mean = np.zeros_like(GrMcrit)
    cts_cent_mean = np.zeros_like(GrMcrit)

    # in case we are saving all the fitted a and b params for sec and tert prop
    a_arr_cent = np.zeros(len(mbins)-1)
    b_arr_cent = np.zeros(len(mbins)-1)
    a_arr_sats = np.zeros(len(mbins)-1)
    b_arr_sats = np.zeros(len(mbins)-1)
    
    # looping over each mass bin
    for i in range(len(mbins)-1):
        # mass bin choice
        choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        # true counts per halo
        cts = GroupCount[choice]
        cts_cent = GroupCountCent[choice]
        cts_sats = GroupCountSats[choice]

        # if no galaxies in that mass bin, skip to next mass bin
        if np.sum(cts) == 0: continue

        # otherwise select sec and tert property for halos in mass bin
        env = GroupEnv[choice]
        conc = GroupConc[choice]
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)

        # record ranked arrays into x and y, and mean central and satellite occupations for the mass bin
        cts_sats_mean[choice] = np.mean(cts_sats)
        cts_cent_mean[choice] = np.mean(cts_cent)
        x[choice] = rank_env
        y[choice] = rank_conc

        # if we are fitting a and b parameters individually for each mass bin
        if mode == 'bins':
            # we minimize for a and b in each mass bin
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
            GroupCountCentPred[choice] = prob_cent(a_cent, b_cent)
            GroupCountSatsPred[choice] = prob_sats(a_sats, b_sats)

            # save the best-fit a and b values
            a_arr_cent[i] = a_cent
            b_arr_cent[i] = b_cent
            a_arr_sats[i] = a_sats
            b_arr_sats[i] = b_sats

    # if we are fitting a single a and b parameter for all mass bins (for cent and sats)
    if mode == 'all':
        # use only the halos within the mass range of interest (speeds up)
        choice = (mbins[0] < GrMcrit) & (mbins[-1] >= GrMcrit)

        # we minimize for a and b in all mass bins
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
        
        # compute prediction for occupancy and save into a new array for centrals and satellites
        GroupCountCentPred[choice] = prob_cent(a_cent, b_cent)
        GroupCountSatsPred[choice] = prob_sats(a_sats, b_sats)

    # print out number of galaxies
    print("pred satellites = ", np.sum(GroupCountSatsPred))
    print("true satellites = ", np.sum(GroupCountSats))
    print("pred centrals = ", np.sum(GroupCountCentPred))
    print("true centrals = ", np.sum(GroupCountCent))

    # make sure we don't get negative counts or values larger than one for the centrals (doesn't make a difference)
    GroupCountSatsPred[GroupCountSatsPred < 0.] = 0.
    GroupCountCentPred[GroupCountCentPred > 1.] = 1.
    GroupCountCentPred[GroupCountCentPred < 0.] = 0.

    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! tuks (and comment out until l. 550) (can play with bin size)
    GroupCountSatsPred = GroupCountSats.copy()
    GroupCountCentPred = GroupCountCent.copy()

    """
    # draw from a poisson and a binomial distribution for the halos in the mass range of interest
    choice = (mbins[-1] >= GrMcrit) & (mbins[0] < GrMcrit)
    if want_pseudo_poisson and gal_type == 'ELG':
        #choice &= (GroupCountSatsPred > 1.e-6)
        choice_poisson = choice & (GrMcrit <= 1.e13)
        GroupCountSatsPred[choice_poisson] = np.random.poisson(GroupCountSatsPred[choice_poisson], len(GroupCountSatsPred[choice_poisson]))
        choice_pseudo = choice & (GrMcrit > 1.e13)
        count_sats = GroupCountSatsPred[choice_pseudo]
        alpha = 0.8 # should be different
        tmp = np.zeros(len(count_sats))
        print("maximum satellites = ", np.max(count_sats))
        for i_c in range(len(count_sats)):
            #if count_sats[i_c] < 1.e-4: continue
            # maximum satellites we can give to this halo
            if i_c%10000 == 0: print(i_c, len(count_sats))
            n_max = count_sats[i_c] #np.max(GroupCountSatsPred[choice])
            n_max = int(n_max+10.*np.sqrt(n_max))
            n_max = np.max([n_max, 1])
            #if n_max == 0: print("too tiny = ", count_sats[i_c]); continue
            #print("pseudo poisson n_max = ", n_max)
            ints = np.arange(0, n_max+1)
            ps = pseudo_poisson(ints, alpha, count_sats[i_c])
            ps /= np.sum(ps)
            tmp[i_c] = np.random.choice(ints, p=ps)
        GroupCountSatsPred[choice_pseudo] = tmp
    else:
        GroupCountSatsPred[choice] = np.random.poisson(GroupCountSatsPred[choice], len(GroupCountSatsPred[choice]))
    GroupCountCentPred = (np.random.rand(len(GroupCountCentPred)) < GroupCountCentPred)
    GroupCountSatsPred = GroupCountSatsPred.astype(int)
    GroupCountCentPred = GroupCountCentPred.astype(int)
    print("pred poisson satellites = ", np.sum(GroupCountSatsPred), np.sum(GroupCountSats))
    print("pred binomial centrals = ", np.sum(GroupCountCentPred), np.sum(GroupCountCent))
    print("-------------------")
    
    # downsample the galaxies in order for the pred to have the same number of satellites and centrals as the truth
    GroupCountCentCopy = GroupCountCent.copy()
    GroupCountSatsCopy = GroupCountSats.copy()
    GroupCountCentCopy[choice], GroupCountCentPred[choice] = downsample_counts(GroupCountCent[choice], GroupCountCentPred[choice])
    GroupCountSatsCopy[choice], GroupCountSatsPred[choice] = downsample_counts(GroupCountSats[choice], GroupCountSatsPred[choice])
    GroupCountSatsPred = GroupCountSatsPred.astype(int)
    GroupCountCentPred = GroupCountCentPred.astype(int)
    """
    
    # initialize arrays for storing the satellite info
    pos_pred_sats = np.zeros((np.sum(GroupCountSatsPred), 3))
    vel_pred_sats = np.zeros((np.sum(GroupCountSatsPred), 3))
    ind_pred_sats = np.zeros(np.sum(GroupCountSatsPred), dtype=int)

    # record pos and vel of centrals and parent halo index
    pos_pred_cent = GroupPos[GroupCountCentPred > 0]
    vel_pred_cent = GroupVel[GroupCountCentPred > 0]
    ind_pred_cent = index_halo[GroupCountCentPred > 0]

    # counter over satellites given
    sum_sats = 0
    
    # loop over mass bins
    for i in range(len(mbins)-1):
        # select mass bin and skip if empty of halos
        mchoice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(mchoice)
        if nhalo == 0: continue

        # predicted satellite occupancies in the mass bin (skip if none)
        cts_sats_pred = GroupCountSatsPred[mchoice]
        if np.sum(cts_sats_pred) == 0: continue
        
        # values of sec and tert prop, pos, vel, radius, dispersion, first, nsub and halo inds in mass bin
        env = GroupEnv[mchoice]
        conc = GroupConc[mchoice]
        pos = GroupPos[mchoice]
        vel = GroupVel[mchoice]
        rcrit = GrRcrit[mchoice]
        vcrit = GroupVelDisp[mchoice]
        if sat_type == 'subhalos':
            first = GroupFirstSub[mchoice]
            nsub = GroupNsubs[mchoice]
        elif sat_type == 'subsamp':
            first = GroupSubsampFirst[mchoice]
            nsub = GroupSubsampSize[mchoice]
        index = index_halo[mchoice]
        
        # select the true satellites in mass bin (needed for drawing from radial distn)
        choice_sats = ((mbins[i]) < mcrit_sats) & ((mbins[i+1]) >= mcrit_sats)

        # TESTING!!!!!!!!!!!!!!!!!!! only true because we are cheating
        print("count stats (min, max, mean, med) = ", np.min(cts_sats_pred), np.max(cts_sats_pred), np.mean(cts_sats_pred), np.median(cts_sats_pred))
        print("predicted number of satellites vs true = ", np.sum(cts_sats_pred), np.sum(choice_sats))
        assert np.sum(cts_sats_pred) == np.sum(choice_sats)
        
        # select sec and tert prop of the satellite hosts and norm dist/vel to center
        conc_ms = conc_sats[choice_sats]
        env_ms = env_sats[choice_sats]
        sbdnm_ms = sbdnm_sats[choice_sats]
        sbvnm_ms = sbvnm_sats[choice_sats]
        sbvrnm_ms = v_rad_sats[choice_sats]

        # compute histogram of satellite dist/vel to center (needed only if no true sats at sec and tert prop)
        hist_sats, _ = np.histogram(sbdnm_ms, bins=rbins)
        p_sats = hist_sats/np.sum(hist_sats)
        hist_sats, _ = np.histogram(sbvnm_ms, bins=vbins)
        v_sats = hist_sats/np.sum(hist_sats)
        hist_sats, _ = np.histogram(sbvrnm_ms, bins=vrbins)
        vr_sats = hist_sats/np.sum(hist_sats)
        
        # turn the secondary and tertiary parameter into ranked arrays for the halos
        rank_env = np.argsort(np.argsort(env))/(len(env)-1.)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1.)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)
        
        # identify the 0, 25, 50, 75 and 100th percentiles in these arrays
        env_thresh = np.zeros(5)
        conc_thresh = np.zeros(5)
        for j in range(1, len(env_thresh)-1):
            env_thresh[j] = env[np.argmin(np.abs(rank_env - ebins[j]))]
            conc_thresh[j] = conc[np.argmin(np.abs(rank_conc - cbins[j]))]
        env_thresh[0] = env.min()-0.1
        conc_thresh[0] = conc.min()-0.1
        env_thresh[-1] = env.max()+0.1
        conc_thresh[-1] = conc.max()+0.1

        # initialize counter for number of times we didn't find true satellites at this bin of mass, c and e (used for rad prof)
        # and counter for number of predicted satellites that are close to an existing subhalo
        no_true_sats = 0
        close_to_exist = 0
        total_given = 0
        # loop over each sec and tert prop bin
        for j in range(len(cbins)-1):
            for k in range(len(ebins)-1):
                # select the halos in that bin of mass, sec and tert prop
                if secondary != 'None':
                    rchoice = (rank_env <= ebins[k+1]) & (rank_env > ebins[k])
                else:
                    rchoice = np.ones(len(env), dtype=bool)
                if fit_type == 'plane':
                    rchoice &= (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j])

                # number of pred satellites per halo and total
                ct = cts_sats_pred[rchoice]
                ng = np.sum(ct)
                if ng == 0: continue

                # select the satellites whose parents are part of this bin of mass, c and e
                if secondary != 'None':
                    rchoice_ms = (env_ms <= env_thresh[k+1]) & (env_ms > env_thresh[k])
                else:
                    rchoice_ms = np.ones(len(env_ms), dtype=bool)
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

                    # TESTING!!!!!!!!!!!!!!!!
                    print("expected galaxies vs true galaxies = ", ng, np.sum(rchoice_ms))
                    assert ng == np.sum(rchoice_ms)
                    
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
                        pr = interp1d(rbinc[hist_r > 0.], hist_r[hist_r > 0.], bounds_error=False, fill_value=0.)(many_rs) # og
                        pr /= np.sum(pr) # og
                        radius = np.random.choice(many_rs, ng, p=pr) # og
                        #pr = hist_r/np.sum(hist_r) # TESTING
                        #radius = np.random.choice(rbinc, ng, p=pr) # TESTING
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
                    if sat_type == 'subhalos':
                        poses = SubhaloPos[start[n]:start[n]+npout[n]]
                        vels = SubhaloVel[start[n]:start[n]+npout[n]]
                    elif sat_type == 'subsamp':
                        poses = PartSubsampPos[start[n]:start[n]+npout[n]]
                        vels = PartSubsampVel[start[n]:start[n]+npout[n]]
                        if npout[n] == 0:
                            #print(f"eh empty {ct[n]:d} {GrMcrit[mchoice][rchoice][n]:.2e}");
                            poses = np.zeros((2,3)); vels = np.zeros((2,3)); # tuks
                    
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
                        if True: #False: #dist_min > .5: #True: #dist_min > .1: # (TESTING) # very bad to do False for some reason
                            # draw random 3d position
                            theta = np.arccos(2.*np.random.rand()-1.)
                            phi = np.random.rand()*2.*np.pi
                            x = np.cos(phi)*np.sin(theta)
                            y = np.sin(phi)*np.sin(theta)
                            z = np.cos(theta)
                            p = radii[m]*np.array([x, y, z])
                            
                            # add the offset from the center
                            pos_pred_sats[sum_sats] = p + pos[rchoice][n] #pos[rchoice][n], SubhaloPos[start[n]] # shouldn't matter

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

                # passed with flying colors for random vector assignment of pos and vel

                if np.log10(mbinc[i]) > 16.: # TESTING 
                    print(f"mass bin = {mbinc[i]:.2e}")
                    
                    # plot rad and disp distn in this bin of mass, c and e 
                    if ng == 0 or np.sum(rchoice_ms) == 0: continue
                    print("number of satellites pred and true = ", ng, np.sum(rchoice_ms))

                    # compute disp hist for pred and true satellites
                    diff = np.sqrt(np.sum((vel_pred_sats[sum_sats-ng: sum_sats]-np.repeat(vel[rchoice], ct, axis=0))**2, axis=1))/np.repeat(vcrit[rchoice], ct)
                    hist_pred, _ = np.histogram(diff, bins=vbins)
                    hist_true, _ = np.histogram(sbvnm_ms[rchoice_ms], bins=vbins)

                    plt.figure(1)
                    plt.axvline(x=np.sqrt(1.), ls='--', color='black', zorder=0)
                    plt.plot(vbinc, hist_true, ls='-', color='red', lw=2, label='true')
                    plt.plot(vbinc, hist_pred, ls='-', color='blue', lw=2, label='pred')
                    #plt.xscale('log')
                    plt.legend()

                    # compute rad hist for pred and true satellites
                    diff = np.sqrt(np.sum((pos_pred_sats[sum_sats-ng: sum_sats]-np.repeat(pos[rchoice], ct, axis=0))**2, axis=1))/np.repeat(rcrit[rchoice], ct)
                    hist_pred, _ = np.histogram(diff, bins=rbins)
                    # doesn't matter whether we use this or og:
                    #diff = np.sqrt(np.sum((((SubhaloPos[index_sats])[choice_sats])[rchoice_ms]-((GroupPos[parent_sats])[choice_sats])[rchoice_ms])**2, axis=1))/np.repeat(rcrit[rchoice], ct)
                    #hist_true, _ = np.histogram(diff, bins=rbins)
                    hist_true, _ = np.histogram(sbdnm_ms[rchoice_ms], bins=rbins) # og

                    plt.figure(2)
                    plt.axvline(x=1., ls='--', color='black', zorder=0)
                    plt.plot(rbinc, hist_true, ls='-', color='red', lw=2, label='true')
                    plt.plot(rbinc, hist_pred, ls='-', color='blue', lw=2, label='pred')
                    plt.xscale('log')
                    plt.legend()
                    
                    from utils import get_jack_corr
                    import Corrfunc
                    xyz_pred = pos_pred_sats[sum_sats-ng: sum_sats]
                    xyz_true = ((SubhaloPos[index_sats])[choice_sats])[rchoice_ms]
                    xyz_pred %= Lbox
                    xyz_true %= Lbox
                    w_pred = np.ones(xyz_pred.shape[0], dtype=xyz_pred.dtype)
                    w_true = np.ones(xyz_true.shape[0], dtype=xyz_true.dtype)
                    print("true and fake difference = ", len(w_pred)-len(w_true))
                    print("fake number = ", len(w_pred))
                    bins = np.logspace(-1, 1.5, 21)
                    binc = (bins[1:]+bins[:-1])*.5
                    corr_pred = Corrfunc.theory.xi(boxsize=Lbox, nthreads=20, binfile=bins, X=xyz_pred[:, 0], Y=xyz_pred[:, 1], Z=xyz_pred[:, 2])['xi']
                    corr_true = Corrfunc.theory.xi(boxsize=Lbox, nthreads=20, binfile=bins, X=xyz_true[:, 0], Y=xyz_true[:, 1], Z=xyz_true[:, 2])['xi']
                    #rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(xyz_true, w_true, xyz_pred, w_pred, Lbox, N_dim=3, bins=bins)

                    plt.figure(3)
                    plt.plot(binc, corr_pred*binc**2, label="pred", color='blue')
                    plt.plot(binc, corr_true*binc**2, label="true", color='red')
                    plt.legend()
                    plt.xscale('log')
                    plt.show()
                    
                if secondary == 'None': break # since we only need to go through the halos once (there is no division by env)
            # if ramp, you need only to cycle through sec property, so just exit loop once we've cycled through `k` loop
            if fit_type == 'ramp':
                break
        # report numbers for this mass bin
        print("no true galaxies at this mass bin to get profile from = ", no_true_sats)
        print("number of pred satellites close to an existing subhalo/particle = ", close_to_exist)
        print("percentage of pred satellites close to an existing subhalo/particle = ", close_to_exist*100./total_given)

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
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_cent)
    else:
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_sats)
            np.save(f"{gal_type:s}/vel_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", vel_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ind_pred_cent)

