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
snapshot = 179; snapshot_dm = 184
#gal_type = 'ELG'
#gal_type = 'LRG'
gal_type = sys.argv[1]
#want_show = True
#want_show = False
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
#secondaries = ['GroupEnv_R2'] # TESTING
secondaries = ['GroupPotential']
#secondaries = ['GroupVelDisp']
#tertiaries = ['GroupConcRad']
#tertiaries = ['GroupConc']
#tertiaries = ['GroupVelDisp']
#tertiaries = ['GroupEnv_R2'] 
#tertiaries = ['GroupShear_R2']

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
#SubhaloVel = np.load(tng_dir+f"data_fp/SubhaloVel_fp_{snapshot:d}.npy")
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
GrRcrit = np.load(tng_dir+f'data_fp/Group_R_TopHat200_fp_{snapshot:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountSats = GroupCount-GroupCountCent
index_halo = np.arange(len(GroupCount), dtype=int)

# load dm halo properties
#SubhaloPos_dm = np.load(tng_dir+f"data_dm/SubhaloPos_dm_{snapshot_dm:d}.npy")
GroupPos_dm = np.load(tng_dir+f'data_dm/GroupPos_dm_{snapshot_dm:d}.npy')
GrMcrit_dm = np.load(tng_dir+f'data_dm/Group_M_TopHat200_dm_{snapshot_dm:d}.npy')*1.e10
GrRcrit_dm = np.load(tng_dir+f'data_dm/Group_R_TopHat200_dm_{snapshot_dm:d}.npy')
index_halo_dm = np.arange(len(GrMcrit_dm), dtype=int) # could sort this to preserve the original order
i_sort = np.argsort(GrMcrit_dm)[::-1]
GroupPos_dm = GroupPos_dm[i_sort]
GrMcrit_dm = GrMcrit_dm[i_sort]
GrRcrit_dm = GrRcrit_dm[i_sort]
print("Note that because of the sorting, you can't use SubhaloGrNr")

# need to add Lbox/4. to all the dm positions because of offset
print("NOTICE THAT WE ARE CORRECTING FOR BOX/4 SHIFT")
GroupPos_dm += np.array([Lbox/4., 0., 0.])
GroupPos_dm %= Lbox

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
#index_sats = np.array([index[i] if index[i] not in index_cent for i in range(len(index)])

# satellite galaxy properties
mcrit_sats = GrMcrit[SubhaloGrNr[index_sats]]
rcrit_sats = GrRcrit[SubhaloGrNr[index_sats]]
xdiff_sats = SubhaloPos[index_sats]-GroupPos[SubhaloGrNr[index_sats]]
xdiff_sats[xdiff_sats > Lbox/2.] -= Lbox
xdiff_sats[xdiff_sats < -Lbox/2.] += Lbox
sbdnm_sats = np.sqrt(np.sum((xdiff_sats)**2, axis=1))/rcrit_sats
#print(sbdnm_sats.max()) # 6.8 Mpc/h
rbins = np.logspace(-3, 1, 41) # -3, 1, 16 og
many_rs = np.logspace(-3, 1, 1000)
rbinc = (rbins[1:] + rbins[:-1])*0.5
vbins = np.pi*4./3.*rbins**3
dvbin = vbins[1:]-vbins[:-1]
#hist_sats, _ = np.histogram(sbdnm_sats, bins=rbins)
#p_sats = hist_sats/np.sum(hist_sats)

# bins for env and conc
cbins = np.linspace(-0.5, 0.5, 5)
ebins = np.linspace(-0.5, 0.5, 5)

def get_lims(cts, param1, param2, percent=68.):

    cbins = np.arange(cts.min()-0.5, cts.max()+0.501, 1)
    
    cbinc = (cbins[1:]+cbins[:-1])/2.
    param1m = np.zeros(len(cbinc))
    param1l = np.zeros(len(cbinc))
    param1h = np.zeros(len(cbinc))
    param2l = np.zeros(len(cbinc))
    param2h = np.zeros(len(cbinc))
    for j in range(len(cbinc)):
        cchoice = (cbins[j] < cts) & (cts < cbins[j+1])
        if np.sum(cchoice) == 0: continue
        
        low2 = np.percentile(param2[cchoice], (100-percent)/2.)
        high2 = np.percentile(param2[cchoice], 100.-(100-percent)/2.)
        param2l[j] = np.median((param1[cchoice])[(param2[cchoice] < low2)])
        param2h[j] = np.median((param1[cchoice])[(param2[cchoice] > high2)])
        param1m[j] = np.median(param1[cchoice])
        param1l[j] = np.percentile(param1[cchoice], (100-percent)/2.)
        param1h[j] = np.percentile(param1[cchoice], 100.-(100-percent)/2.)
    return cbinc, param1m, param1l, param1h, param2l, param2h

# max halo mass
print("max halo mass = %.1e"%GrMcrit.max())

# mass bins # notice slightly lower upper limit cause few halos
mbins = np.logspace(11, 14, 31)
mbinc = (mbins[1:]+mbins[:-1])*0.5
mbins_dm = np.zeros(31)

# halos that are never moved
n_top = np.sum(mbins[-1] < GrMcrit)
print("number of halos above the last mass bin = ", n_top)
mbins_dm[-1] = GrMcrit_dm[n_top]
print("total occupancy of the top halos = ", np.sum(GroupCountCent[(mbins[-1] < GrMcrit)])+np.sum(GroupCountSats[(mbins[-1] < GrMcrit)]))

for i_pair in range(len(secondaries)):
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
    # can you look at the subhalo profiles of these halos? can you plot them? can you trace their history? SFR? weak lensing correction -- how do this??

    # load the secondary and tertiary bias parameter
    GroupEnv = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
    GroupEnv_dm = np.load(tng_dir+f'data_dm/{secondary:s}_dm_{snapshot_dm:d}.npy')[i_sort] # SUPER IMPORTANT BUGGGGGGGGGGGGGGGGGG
    if fit_type == 'ramp':
        GroupConc = np.zeros(len(GroupEnv))
        GroupConc_dm = np.zeros(len(GroupEnv_dm))
    else:
        GroupConc = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')
        GroupConc_dm = np.load(tng_dir+f'data_dm/{tertiary:s}_dm_{snapshot_dm:d}.npy')[i_sort]

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

    # only needed if getting a and b per halo mass bin
    a_arr_cent = np.zeros(len(mbins)-1)
    b_arr_cent = np.zeros(len(mbins)-1)
    a_arr_sats = np.zeros(len(mbins)-1)
    b_arr_sats = np.zeros(len(mbins)-1)
    
    # initialize counter for each mass bin (needed for abundance matching)
    sum_halo = 0
    sum_halo += n_top
    for i in range(len(mbins)-1)[::-1]: # needed to invert cause of abundance matching
        choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        nhalo = np.sum(choice)
        if nhalo == 0: continue
        choice_dm = np.arange(sum_halo, sum_halo+nhalo, dtype=int) # TESTING
        #choice_dm = (mbins_dm[i] < GrMcrit_dm) & (mbins_dm[i+1] >= GrMcrit_dm) # og
        mbins_dm[i] = GrMcrit_dm[sum_halo+nhalo] # low bound
        sum_halo += nhalo # incrementing for abundance matching

        # counts and params for this bin
        cts = GroupCount[choice]
        cts_cent = GroupCountCent[choice]
        cts_sats = GroupCountSats[choice]
        ngal = np.sum(cts)
        if ngal == 0: continue
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

        # record x and y and mean central and satellite occupation
        cts_sats_mean[choice] = np.mean(cts_sats)
        cts_cent_mean[choice] = np.mean(cts_cent)
        cts_sats_mean_dm[choice_dm] = np.mean(cts_sats)
        cts_cent_mean_dm[choice_dm] = np.mean(cts_cent)
        x[choice] = rank_env
        y[choice] = rank_conc
        x_dm[choice_dm] = rank_env_dm
        y_dm[choice_dm] = rank_conc_dm
        
        if mode == 'bins':
            
            # then we minimize for a and b
            res_cent = minimize(like_cent, p0, method=method)
            a_cent, b_cent = res_cent['x']
            print("centrals: a, b = ", a_cent, b_cent)
            res_sats = minimize(like_sats, p0, method=method)
            a_sats, b_sats = res_sats['x']
            print("satellites: a, b = ", a_sats, b_sats)
            
            # compute prediction for occupancy and save into a new array for centrals and satellites
            GroupCountCentPred_dm[choice] = prob_cent_dm(a_cent, b_cent)
            GroupCountSatsPred_dm[choice] = prob_sats_dm(a_sats, b_sats)

            a_arr_cent[i] = a_cent
            b_arr_cent[i] = b_cent
            a_arr_sats[i] = a_sats
            b_arr_sats[i] = b_sats

            
    if mode == 'all':
        # select all halos of interest
        choice = (mbins[0] < GrMcrit) & (mbins[-1] >= GrMcrit)

        
        # then we minimize for a and b
        res_cent = minimize(like_cent, p0, method=method)
        a_cent, b_cent = res_cent['x']
        print("centrals: a, b = ", a_cent, b_cent)
        res_sats = minimize(like_sats, p0, method=method)
        a_sats, b_sats = res_sats['x']
        print("satellites: a, b = ", a_sats, b_sats)
        
        #a_cent, b_cent = 6., 0.
        #a_sats, b_sats = 6., 0.
        
        # select equivalent halos in dmo
        #choice_dm = (mbins_dm[0] < GrMcrit_dm) & (mbins_dm[-1] >= GrMcrit_dm) # og        
        choice_dm = np.arange(n_top, sum_halo, dtype=int) # TESTING
        
        # compute prediction for occupancy and save into a new array for centrals and satellites
        GroupCountCentPred_dm[choice_dm] = prob_cent_dm(a_cent, b_cent)
        GroupCountSatsPred_dm[choice_dm] = prob_sats_dm(a_sats, b_sats)
    
    print("satellites = ", np.sum(GroupCountSatsPred_dm))
    print("centrals = ", np.sum(GroupCountCentPred_dm))
    print("true satellites = ", np.sum(GroupCountSats))
    print("true centrals = ", np.sum(GroupCountCent))
    print("-------------------")

    # normalizing
    GroupCountSatsPred_dm[GroupCountSatsPred_dm < 0] = 0
    GroupCountCentPred_dm[GroupCountCentPred_dm > 1] = 1
    GroupCountCentPred_dm[GroupCountCentPred_dm < 0] = 0
    
    # poisson and binomial
    choice = (mbins[-1] >= GrMcrit) & (mbins[0] < GrMcrit)
    choice_dm = (mbins_dm[-1] >= GrMcrit_dm) & (mbins_dm[0] < GrMcrit_dm)
    GroupCountSatsPred_dm[choice_dm] = np.random.poisson(GroupCountSatsPred_dm[choice_dm], len(GroupCountSatsPred_dm[choice_dm]))
    GroupCountCentPred_dm = (np.random.rand(len(GroupCountCentPred_dm)) < GroupCountCentPred_dm)
    GroupCountSatsPred_dm = GroupCountSatsPred_dm.astype(int)
    GroupCountCentPred_dm = GroupCountCentPred_dm.astype(int)

    print("poisson satellites = ", np.sum(GroupCountSatsPred_dm), np.sum(GroupCountSats))
    print("binomial centrals = ", np.sum(GroupCountCentPred_dm), np.sum(GroupCountCent))
    print("-------------------")
    
    # get the new downsampled numbers
    GroupCountCentCopy = GroupCountCent.copy()
    GroupCountSatsCopy = GroupCountSats.copy() 
    GroupCountCentCopy[choice], GroupCountCentPred_dm[choice_dm] = downsample_counts(GroupCountCent[choice], GroupCountCentPred_dm[choice_dm])
    GroupCountSatsCopy[choice], GroupCountSatsPred_dm[choice_dm] = downsample_counts(GroupCountSats[choice], GroupCountSatsPred_dm[choice_dm])
    GroupCountSatsPred_dm = GroupCountSatsPred_dm.astype(int)
    GroupCountCentPred_dm = GroupCountCentPred_dm.astype(int)

    print("after downsampling, sats cent = ", np.sum(GroupCountSatsPred_dm), np.sum(GroupCountCentPred_dm))
    
    # where the new positions are to be stored
    pos_pred_sats = np.zeros((np.sum(GroupCountSatsPred_dm), 3))
    ind_pred_sats = np.zeros(np.sum(GroupCountSatsPred_dm), dtype=int)
    pos_pred_cent = GroupPos_dm[GroupCountCentPred_dm > 0]
    ind_pred_cent = index_halo_dm[GroupCountCentPred_dm > 0]

    # initiate galaxy sum
    sum = 0
    
    # initialize counter for each mass bin (needed for abundance matching)
    for i in range(len(mbins)-1):
        choice = ((mbins_dm[i]) < GrMcrit_dm) & ((mbins_dm[i+1]) >= GrMcrit_dm)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        # abundance matched fp halo selection (only needed for setting thresholds for the satellite profile array)
        choice_fp = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
        
        # satellite number
        cts_sats_pred = GroupCountSatsPred_dm[choice]
        ngal_sats_pred = np.sum(cts_sats_pred)
        if ngal_sats_pred == 0: continue
        env = GroupEnv_dm[choice]
        conc = GroupConc_dm[choice]
        pos = GroupPos_dm[choice]
        rcrit = GrRcrit_dm[choice]
        index = index_halo_dm[choice]

        # env and conc for the fp halos (only needed for thresholds)
        env_fp = GroupEnv[choice_fp]
        conc_fp = GroupConc[choice_fp]
        
        # select also the galaxies with this halo mass (only needed for drawing from here)
        choice_sats = ((mbins[i]) < mcrit_sats) & ((mbins[i+1]) >= mcrit_sats)
        conc_ms = conc_sats[choice_sats]
        env_ms = env_sats[choice_sats]
        sbdnm_ms = sbdnm_sats[choice_sats]

        # this probability will be used only in the event of no galaxies present in ebin and cbin
        hist_sats, _ = np.histogram(sbdnm_ms, bins=rbins)
        p_sats = hist_sats/np.sum(hist_sats)
        
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
        ebins[0] -= 0.1 # fixes the assertion error
        cbins[0] -= 0.1
        ebins[-1] += 0.1
        cbins[-1] += 0.1 
        
        empty = 0
        for j in range(len(cbins)-1):
            for k in range(len(ebins)-1):
                # select the halos in that bin of mass, c and e
                rchoice = (rank_env <= ebins[k+1]) & (rank_env > ebins[k])
                if fit_type == 'plane':
                    rchoice &= (rank_conc <= cbins[j+1]) & (rank_conc > cbins[j])

                # how many galaxies you need to draw    
                ct = cts_sats_pred[rchoice]
                ng = np.sum(ct)
                if ng == 0: continue

                # select the satellites whose parents are part of this bin of mass, c and e
                rchoice_ms = (env_ms <= env_thresh[k+1]) & (env_ms > env_thresh[k])
                if fit_type == 'plane':
                    rchoice_ms &= (conc_ms <= conc_thresh[j+1]) & (conc_ms > conc_thresh[j])

                if np.sum(rchoice_ms) == 0:
                    radius = np.random.choice(rbinc, ng, p=p_sats)
                    empty += ng
                else:
                    hist, _ = np.histogram(sbdnm_ms[rchoice_ms], bins=rbins)
                    if np.sum(hist > 0.) <= 1:
                        p = hist/np.sum(hist)
                        radius = np.random.choice(rbinc, ng, p=p)
                        empty += ng
                    else:
                        p = interp1d(rbinc[hist > 0.], hist[hist > 0.], bounds_error=False, fill_value=0.)(many_rs)
                        p /= np.sum(p)
                        radius = np.random.choice(many_rs, ng, p=p)                    
                
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
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_dm_{snapshot_dm:d}.npy", ind_pred_cent)
    else:
        if mode == 'bins':
            np.save(f"{gal_type:s}/pos_pred_{fun_sats:s}_sats_{secondary:s}_dm_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_{fun_cent:s}_cent_{secondary:s}_dm_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_{fun_sats:s}_sats_{secondary:s}_dm_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_{fun_cent:s}_cent_{secondary:s}_dm_{snapshot_dm:d}.npy", ind_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/pos_pred_all_{fun_sats:s}_sats_{secondary:s}_dm_{snapshot_dm:d}.npy", pos_pred_sats)
            np.save(f"{gal_type:s}/pos_pred_all_{fun_cent:s}_cent_{secondary:s}_dm_{snapshot_dm:d}.npy", pos_pred_cent)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_sats:s}_sats_{secondary:s}_dm_{snapshot_dm:d}.npy", ind_pred_sats)
            np.save(f"{gal_type:s}/ind_pred_all_{fun_cent:s}_cent_{secondary:s}_dm_{snapshot_dm:d}.npy", ind_pred_cent)

