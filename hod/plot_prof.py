import os
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import special
from scipy.interpolate import interp1d

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
#gal_type = 'ELG'
#gal_type = 'LRG'
gal_type = sys.argv[1]
#fit_type = 'plane'
#fit_type = 'ramp'
fit_type = sys.argv[2]
#fun_cent = 'tanh'
#fun_cent = 'erf'
#fun_cent = 'gd'
#fun_cent = 'abs'
#fun_cent = 'arctan'
fun_cent = 'linear'
#n_gal = '7.4e-04'
#n_gal = '9.7e-04'
n_gal = '2.0e-03'
fun_sats = 'linear'
method = 'powell'
#method = 'Nelder-Mead'
#mode = 'bins'
mode = 'all'
p0 = np.array([0., 0.]) 
Lbox = 500.

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

def get_dist_sats(hid_sats, num_sats):
    sum = 0
    nosubs = 0
    dist_sats = np.zeros((int(np.sum(num_sats))), dtype=GroupPos.dtype)
    for i in range(len(num_sats)):
        # todo: kinda wrong cause needs to actually be order by stellar mass rather than number of particles (I think DM)
        # which subhalo index does this halo start at (+1 to get rid of the central subhalo) 
        start = GroupFirstSub[hid_sats[i]]
        nsubs = GroupNsubs[hid_sats[i]]
        mstar = SubhaloMstar[start:start+nsubs]
        poses = SubhaloPos[start:start+nsubs]
        #ssfr = SubhalosSFR[start:start+nsubs]
        vmax = SubhaloVmax[start:start+nsubs]
        first_pos = poses[0]
        poses = poses[1:]
        mstar = mstar[1:]
        #ssfr = ssfr[1:]
        vmax = vmax[1:]
        vmax = mstar # TESTING
        if gal_type == 'LRG':
            i_sort = np.argsort(vmax)[::-1]
        elif gal_type == 'ELG':
            i_sort = np.argsort(vmax)[::-1]
        poses = poses[i_sort]
        #mstar = mstar[i_sort]
        #poses = poses[mstar >= mstar_elg_thresh]
        nsubs = poses.shape[0]
        num = int(num_sats[i])
        if nsubs < num: # should be the case v rarely (put in center)
            if nsubs > 0:
                dist_sats[sum:sum+nsubs] = np.sqrt(np.sum((poses[:nsubs]-first_pos)**2, axis=1))
            dist_sats[sum+nsubs:sum+num] = 0.
            nosubs += 1
        else:
            dist_sats[sum:sum+num] = np.sqrt(np.sum((poses[:num]-first_pos)**2, axis=1))
        sum += num
    print("# halos with a single subhalo = ", nosubs)
    return dist_sats


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
#secondaries = ['GroupVelDisp']
#tertiaries = ['GroupConc']
tertiaries = ['GroupConcRad']
#tertiaries = ['GroupVelDisp']
#tertiaries = ['GroupEnv_R2'] 
#tertiaries = ['GroupShear_R2']

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
SubhaloGrNr = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy')
SubhaloPos = np.load(tng_dir+f"data_fp/SubhaloPos_fp_{snapshot:d}.npy")
SubhaloVel = np.load(tng_dir+f"data_fp/SubhaloVel_fp_{snapshot:d}.npy")
SubhaloMstar = np.load(tng_dir+f"data_fp/SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
GroupVel = np.load(tng_dir+f'data_fp/GroupVel_fp_{snapshot:d}.npy')
GroupVelDisp = np.load(tng_dir+f'data_fp/GroupVelDisp_fp_{snapshot:d}.npy')
GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
GrRcrit = np.load(tng_dir+f'data_fp/Group_R_TopHat200_fp_{snapshot:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountSats = GroupCount-GroupCountCent
index_halo = np.arange(len(GroupCount), dtype=int)

# for the satellite locations
GroupNsubs = np.load(tng_dir+f'data_fp/GroupNsubs_fp_{snapshot:d}.npy')
GroupFirstSub = np.load(tng_dir+f'data_fp/GroupFirstSub_fp_{snapshot:d}.npy')
SubhaloVmax = np.load(tng_dir+f"data_fp/SubhaloVmax_fp_{snapshot:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
#index_sats = np.array([index[i] if index[i] not in index_cent for i in range(len(index)])

# galaxy properties
mcrit_sats = GrMcrit[SubhaloGrNr[index_sats]]
rcrit_sats = GrRcrit[SubhaloGrNr[index_sats]]
vcrit_sats = GroupVelDisp[SubhaloGrNr[index_sats]]
xdiff_sats = SubhaloPos[index_sats]-GroupPos[SubhaloGrNr[index_sats]]
xdiff_sats[xdiff_sats > Lbox/2.] -= Lbox
xdiff_sats[xdiff_sats < -Lbox/2.] += Lbox
sbdnm_sats = np.sqrt(np.sum((xdiff_sats)**2, axis=1))/rcrit_sats
vdiff_sats = SubhaloVel[index_sats]-GroupVel[SubhaloGrNr[index_sats]]
sbvnm_sats = np.sqrt(np.sum((vdiff_sats)**2, axis=1))/vcrit_sats
#print(sbdnm_sats.max()) # 6.8 Mpc/h
rbins = np.logspace(-1.3, 1, 21)#np.logspace(-3, 1, 16)
many_rs = np.logspace(-1.3, 1, 100)
rbinc = (rbins[1:] + rbins[:-1])*0.5
vbins = np.logspace(-1.3, 1, 21)
many_vs = np.logspace(-1.3, 1, 100)
vbinc = (vbins[1:] + vbins[:-1])*0.5


# bins for env and conc
cbins = np.linspace(-0.5, 0.5, 5)
ebins = np.linspace(-0.5, 0.5, 5)


# max halo mass
print("max halo mass = %.1e"%GrMcrit.max())

# mass bins # notice slightly lower upper limit cause few halos
#mbins = np.logspace(10, 14, 21)
mbins = np.logspace(11, 14, 31)
mbinc = (mbins[1:]+mbins[:-1])*0.5
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit))

for i_pair in range(len(secondaries)):
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]
    else:
        tertiary = 'None'
    
    print("param pair = ", i_pair, secondary, tertiary)

    GroupEnv = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
    if fit_type == 'ramp':
        GroupConc = np.zeros(len(GroupEnv))
    else:
        GroupConc = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')

    # concentrations of the satellite galaxies
    conc_sats = GroupConc[SubhaloGrNr[index_sats]]
    env_sats = GroupEnv[SubhaloGrNr[index_sats]]
        
    x = np.zeros_like(GrMcrit)
    y = np.zeros_like(GrMcrit)

    for i in range(len(mbins)-1):
        choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        env = GroupEnv[choice]
        conc = GroupConc[choice]
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)

        # record x and y and mean central and satellite occupation
        x[choice] = rank_env
        y[choice] = rank_conc

        # select also the galaxies with this halo mass (only needed for drawing from here)
        choice_sats = ((mbins[i]) < mcrit_sats) & ((mbins[i+1]) >= mcrit_sats)
        if np.sum(choice_sats) < 100: continue
        print("total galaxy number = ", np.sum(choice_sats))
        conc_ms = conc_sats[choice_sats]
        env_ms = env_sats[choice_sats]
        sbdnm_ms = sbdnm_sats[choice_sats]
        sbvnm_ms = sbvnm_sats[choice_sats]

        # TESTING    
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env_ms = np.argsort(np.argsort(env_ms))/(len(env_ms)-1)-0.5
        if fit_type == 'plane':
            rank_conc_ms = np.argsort(np.argsort(conc_ms))/(len(conc_ms)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc_ms = np.zeros_like(rank_env_ms)

        # identify the 0, 25, 50, 75 and 100th percentiles
        env_thresh = np.zeros(5)
        conc_thresh = np.zeros(5)
        for j in range(1, len(env_thresh)-1):
            env_thresh[j] = env[np.argmin(np.abs(rank_env - ebins[j]))] # og
            #env_thresh[j] = env_ms[np.argmin(np.abs(rank_env_ms - ebins[j]))] # TESTING better
            conc_thresh[j] = conc[np.argmin(np.abs(rank_conc - cbins[j]))] # og
            #conc_thresh[j] = conc_ms[np.argmin(np.abs(rank_conc_ms - cbins[j]))] # TESTING
        env_thresh[0] = env.min()-0.1
        conc_thresh[0] = conc.min()-0.1
        env_thresh[-1] = env.max()+0.1
        conc_thresh[-1] = conc.max()+0.1
        ebins[0] -= 0.1 # fixes the assertion error
        cbins[0] -= 0.1
        ebins[-1] += 0.1
        cbins[-1] += 0.1 
        
        plots = 0
        for j in range(len(cbins)-1):
            for k in range(len(ebins)-1):
                # select the halos in that bin of mass, c and e
                rchoice = (rank_env <= ebins[k+1]) & (rank_env > ebins[k])
                if fit_type == 'plane':
                    rchoice &= (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j])

                # select the satellites whose parents are part of this bin of mass, c and e
                rchoice_ms = (env_ms <= env_thresh[k+1]) & (env_ms > env_thresh[k])
                if fit_type == 'plane':
                    rchoice_ms &= (conc_ms <= conc_thresh[j+1]) & (conc_ms > conc_thresh[j])

                if np.sum(rchoice_ms) == 0:
                    continue

                mean_sat = (np.sum(rchoice_ms)/np.sum(rchoice))
                print("mean number of satellites = ", mean_sat)
                mean_sat = np.ones(np.sum(rchoice))*np.ceil(mean_sat)
                mean_sat = mean_sat.astype(int)
                dist_sats = get_dist_sats((index_halo[choice])[rchoice], mean_sat)

                hist_gen, _ = np.histogram(dist_sats, bins=rbins)
                hist_gen = hist_gen.astype(np.float)/np.sum(hist_gen)/dvbin

                hist, _ = np.histogram(sbdnm_ms[rchoice_ms], bins=rbins)
                #hist = hist.astype(np.float)/np.sum(hist)/dvbin # see below

                print(hist)
                if np.sum(hist > 0.) <= 1: continue
                p_sats = interp1d(rbinc[hist > 0.], hist[hist > 0.], bounds_error=False, fill_value=0.)(many_rs)
                p_sats /= np.sum(p_sats)
                radius = np.random.choice(many_rs, np.sum(rchoice_ms), p=p_sats)

                hist_gen, _ = np.histogram(radius, bins=rbins)
                hist_gen = hist_gen.astype(np.float)/np.sum(hist_gen)/dvbin

                hist = hist.astype(np.float)/np.sum(hist)/dvbin
                
                ind_lo1 = 0
                ind_lo2 = 0
                ind_hi1 = len(cbins)-2
                ind_hi2 = len(cbins)-2
                if j in [ind_lo1, ind_lo2] and k in [ind_lo1, ind_lo2]:
                    plt.plot(rbinc, hist, ls='-', color='red', lw=2, label="lo env, lo conc")
                    plt.plot(rbinc, hist_gen, ls='--', color='red', lw=2, label="lo env, lo conc")
                    plots += 1
                if j in [ind_lo1, ind_lo2] and k in [ind_hi1, ind_hi2]:
                    plt.plot(rbinc, hist, ls='-', color='green', lw=2, label="hi env, lo conc")
                    plt.plot(rbinc, hist_gen, ls='--', color='green', lw=2, label="hi env, lo conc")
                    plots += 1
                elif j in [ind_hi1, ind_hi2] and k in [ind_lo1, ind_lo2]:
                    plt.plot(rbinc, hist, ls='-', color='blue', lw=2, label="lo env, hi conc")
                    plt.plot(rbinc, hist_gen, ls='--', color='blue', lw=2, label="lo env, hi conc")
                    plots += 1
                elif j in [ind_hi1, ind_hi2] and k in [ind_hi1, ind_hi2]:
                    plt.plot(rbinc, hist, ls='-', color='black', lw=2, label="hi env, hi conc")
                    plt.plot(rbinc, hist_gen, ls='--', color='black', lw=2, label="hi env, hi conc")
                    plots += 1

        if plots > 1:
            plt.title(f"logM = {np.log10(mbinc[i]):.1f}")
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.show()
        else:
            print("not enough stuff to plot")
quit()

for i_pair in range(len(secondaries)):
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]
    else:
        tertiary = 'None'
    
    print("param pair = ", i_pair, secondary, tertiary)

    # array with predicted counts
    GroupCountCentPred = (GroupCountCent.copy()).astype(GrMcrit.dtype)
    GroupCountSatsPred = (GroupCountSats.copy()).astype(GrMcrit.dtype)

    GroupEnv = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
    if fit_type == 'ramp':
        GroupConc = np.zeros(len(GroupEnv))
    else:
        GroupConc = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')

    # concentrations of the satellite galaxies
    conc_sats = GroupConc[SubhaloGrNr[index_sats]]
    env_sats = GroupEnv[SubhaloGrNr[index_sats]]
        
    x = np.zeros_like(GrMcrit)
    y = np.zeros_like(GrMcrit)
    cts_sats_mean = np.zeros_like(GrMcrit)
    cts_cent_mean = np.zeros_like(GrMcrit)

    a_arr_cent = np.zeros(len(mbins)-1)
    b_arr_cent = np.zeros(len(mbins)-1)
    a_arr_sats = np.zeros(len(mbins)-1)
    b_arr_sats = np.zeros(len(mbins)-1)
    for i in range(len(mbins)-1):
        choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        cts = GroupCount[choice]
        cts_cent = GroupCountCent[choice]
        cts_sats = GroupCountSats[choice]
        ngal = np.sum(cts)
        if ngal == 0: continue
        env = GroupEnv[choice]
        conc = GroupConc[choice]
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)

        # record x and y and mean central and satellite occupation
        cts_sats_mean[choice] = np.mean(cts_sats)
        cts_cent_mean[choice] = np.mean(cts_cent)
        x[choice] = rank_env
        y[choice] = rank_conc

        if mode == 'bins':
            
            # then we minimize for a and b
            res_cent = minimize(like_cent, p0, method=method)
            a_cent, b_cent = res_cent['x']
            print("centrals: a, b = ", a_cent, b_cent)
            res_sats = minimize(like_sats, p0, method=method)
            a_sats, b_sats = res_sats['x']
            print("satellites: a, b = ", a_sats, b_sats)
        
            # compute prediction for occupancy and save into a new array for centrals and satellites
            GroupCountCentPred[choice] = prob_cent(a_cent, b_cent)
            GroupCountSatsPred[choice] = prob_sats(a_sats, b_sats)

            a_arr_cent[i] = a_cent
            b_arr_cent[i] = b_cent
            a_arr_sats[i] = a_sats
            b_arr_sats[i] = b_sats


    if mode == 'all':
        # to spare us some time, concentrate on only the halos above the minimum mass threshold
        choice = (mbins[0] < GrMcrit) & (mbins[-1] >= GrMcrit)

        
        # then we minimize for a and b
        res_cent = minimize(like_cent, p0, method=method)
        a_cent, b_cent = res_cent['x']
        print("centrals: a, b = ", a_cent, b_cent)
        res_sats = minimize(like_sats, p0, method=method)
        a_sats, b_sats = res_sats['x']
        print("satellites: a, b = ", a_sats, b_sats)
        
        
        # for tert conc and sec env 7.4e-4 and sec envis 0.599
        #a_cent, b_cent =  0.6041735518063931, 1.3958260405470013
        #a_sats, b_sats =  0.8542338199867262, -1.0231410553515738

        # for tert conc and sec env 9.7e-4
        #a_cent, b_cent =  0.5835058889187836, 1.4164932429475072
        #a_sats, b_sats =  0.8024862313515366, -0.986720598710901
        #a_cent, b_cent =  0.5694688290575092, 0.
        #a_sats, b_sats =  0.9491890530406367, 0.
        
        # for tert conc and sec env 2.0e-3
        #a_cent, b_cent =  0.5477703384636718, 1.4522284872633087
        #a_sats, b_sats =  0.6140774140191954, -0.9207314275217406
        
        # compute prediction for occupancy and save into a new array for centrals and satellites
        GroupCountCentPred[choice] = prob_cent(a_cent, b_cent)
        GroupCountSatsPred[choice] = prob_sats(a_sats, b_sats)
    
    print("satellites = ", np.sum(GroupCountSatsPred))
    print("centrals = ", np.sum(GroupCountCentPred))
    print("true satellites = ", np.sum(GroupCountSats))
    print("true centrals = ", np.sum(GroupCountCent))
    print("-------------------")

    # normalizing
    GroupCountSatsPred[GroupCountSatsPred < 0] = 0
    GroupCountCentPred[GroupCountCentPred > 1] = 1
    GroupCountCentPred[GroupCountCentPred < 0] = 0
    
    # poisson and binomial
    choice = (mbins[-1] >= GrMcrit) & (mbins[0] < GrMcrit)
    GroupCountSatsPred[choice] = np.random.poisson(GroupCountSatsPred[choice], len(GroupCountSatsPred[choice]))
    GroupCountCentPred = (np.random.rand(len(GroupCountCentPred)) < GroupCountCentPred)
    GroupCountSatsPred = GroupCountSatsPred.astype(int)
    GroupCountCentPred = GroupCountCentPred.astype(int)

    print("poisson satellites = ", np.sum(GroupCountSatsPred), np.sum(GroupCountSats))
    print("binomial centrals = ", np.sum(GroupCountCentPred), np.sum(GroupCountCent))
    print("-------------------")
    
    # get the new downsampled numbers
    GroupCountCentCopy = GroupCountCent.copy()
    GroupCountSatsCopy = GroupCountSats.copy()
    GroupCountCentCopy[choice], GroupCountCentPred[choice] = downsample_counts(GroupCountCent[choice], GroupCountCentPred[choice])
    GroupCountSatsCopy[choice], GroupCountSatsPred[choice] = downsample_counts(GroupCountSats[choice], GroupCountSatsPred[choice])
    GroupCountSatsPred = GroupCountSatsPred.astype(int)
    GroupCountCentPred = GroupCountCentPred.astype(int)

    print("after downsampling, sats cent = ", np.sum(GroupCountSatsPred), np.sum(GroupCountCentPred))
    
    # where the new positions are to be stored
    pos_pred_sats = np.zeros((np.sum(GroupCountSatsPred), 3))
    ind_pred_sats = np.zeros(np.sum(GroupCountSatsPred), dtype=int)
    pos_pred_cent = GroupPos[GroupCountCentPred > 0]
    ind_pred_cent = index_halo[GroupCountCentPred > 0]

    sum = 0
    for i in range(len(mbins)-1):
        mchoice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(mchoice)
        if nhalo == 0: continue

        # satellite number
        cts_sats_pred = GroupCountSatsPred[mchoice]
        ngal_sats_pred = np.sum(cts_sats_pred)
        if ngal_sats_pred == 0: continue
        env = GroupEnv[mchoice]
        conc = GroupConc[mchoice]
        pos = GroupPos[mchoice]
        rcrit = GrRcrit[mchoice]
        index = index_halo[mchoice]
        
        # select also the galaxies with this halo mass (only needed for drawing from here)
        choice_sats = ((mbins[i]) < mcrit_sats) & ((mbins[i+1]) >= mcrit_sats)
        conc_ms = conc_sats[choice_sats]
        env_ms = env_sats[choice_sats]
        sbdnm_ms = sbdnm_sats[choice_sats]

        hist_sats, _ = np.histogram(sbdnm_ms, bins=rbins)
        p_sats = hist_sats/np.sum(hist_sats)
        
        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5
        if fit_type == 'plane':
            rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5
        elif fit_type == 'ramp':
            rank_conc = np.zeros_like(rank_env)

        # identify the 0, 25, 50, 75 and 100th percentiles
        env_thresh = np.zeros(5)
        conc_thresh = np.zeros(5)
        for j in range(1, len(env_thresh)-1):
            env_thresh[j] = env[np.argmin(np.abs(rank_env - ebins[j]))]
            conc_thresh[j] = conc[np.argmin(np.abs(rank_conc - cbins[j]))]
        env_thresh[0] = env.min()-0.1
        conc_thresh[0] = conc.min()-0.1
        env_thresh[-1] = env.max()+0.1
        conc_thresh[-1] = conc.max()+0.1
        ebins[0] -= 0.1 # fixes the assertion error
        cbins[0] -= 0.1
        ebins[-1] += 0.1
        cbins[-1] += 0.1 
        
        empty = 0
        for j in range(len(cbins)-1):
            for k in range(len(ebins)-1):
                # select the halos in that bin of mass, c and e
                rchoice = (rank_env <= ebins[k+1]) & (rank_env > ebins[k]) # og
                #rchoice = (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j])
                if fit_type == 'plane':
                    rchoice &= (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j]) # og
                    #pass

                # how many galaxies you need to draw    
                ct = cts_sats_pred[rchoice]
                ng = np.sum(ct)
                if ng == 0: continue

                # select the satellites whose parents are part of this bin of mass, c and e
                rchoice_ms = (env_ms <= env_thresh[k+1]) & (env_ms > env_thresh[k]) # og
                #rchoice_ms = (conc_ms <= conc_thresh[j+1]) & (conc_ms > conc_thresh[j])
                if fit_type == 'plane':
                    rchoice_ms &= (conc_ms <= conc_thresh[j+1]) & (conc_ms > conc_thresh[j]) # og
                    #pass

                if np.sum(rchoice_ms) == 0:
                    radius = np.random.choice(rbinc, ng, p=p_sats)
                    empty += ng
                else:
                    hist, _ = np.histogram(sbdnm_ms[rchoice_ms], bins=rbins)
                    p = hist/np.sum(hist)
                    radius = np.random.choice(rbinc, ng, p=p)
                    
                    
                # draw randomly on the sphere
                theta = np.arccos(1.-2.*np.random.rand(ng))
                phi = np.random.rand(ng)*2.*np.pi
                x = radius*np.cos(phi)*np.sin(theta)
                y = radius*np.sin(phi)*np.sin(theta)
                z = radius*np.cos(theta)
                pos_new = np.vstack((x, y, z)).T

                # multiply by virial radius and add the offset from the center
                pos_new = pos_new*np.repeat(rcrit[rchoice], ct)[:, None] + np.repeat(pos[rchoice], ct, axis=0)
                ind_new = np.repeat(index[rchoice], ct)
                
                pos_pred_sats[sum:sum+ng] = pos_new
                ind_pred_sats[sum:sum+ng] = ind_new
                sum += ng
            # if ramp you need only to cycle through environments
            #if True:
            if fit_type == 'ramp':
                break
        print("no true galaxies at this mass bin to get profile from = ", empty)

    # one question is what happens to the super massive dudes
    pos_new = SubhaloPos[index_sats[GrMcrit[SubhaloGrNr[index_sats]] > mbins[-1]]]
    ind_new = SubhaloGrNr[index_sats[GrMcrit[SubhaloGrNr[index_sats]] > mbins[-1]]]
    assert pos_new.shape[0] == pos_pred_sats[sum:].shape[0], f"OOPS {pos_new.shape[0]:d}, {pos_pred_sats[sum:].shape[0]:d}"
    pos_pred_sats[sum:sum+pos_new.shape[0]] = pos_new
    ind_pred_sats[sum:sum+pos_new.shape[0]] = ind_new
    sum += pos_new.shape[0]
    pos_pred_sats = pos_pred_sats[:sum]
    ind_pred_sats = ind_pred_sats[:sum]
    
    if fit_type == 'plane':
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", ind_pred_cent)
    else:
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", ind_pred_cent)

