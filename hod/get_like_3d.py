import os
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import special

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
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
#fun_cent = sys.argv[1]
n_gal = '7.4e-04'
fun_sats = 'linear'
method = 'powell'
#method = 'Nelder-Mead'
#mode = 'bins'
mode = 'all'
p0 = np.array([0., 0.]) 

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
#secondaries = ['GroupEnv_R2'] # TESTING
#tertiaries = ['GroupConc']
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
GrMcrit_fp = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy")
GroupCountSats = GroupCount-GroupCountCent

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
print("max halo mass = %.1e"%GrMcrit_fp.max())

# mass bins # notice slightly lower upper limit cause few halos
#mbins = np.logspace(10, 14, 21)
mbins = np.logspace(11, 14, 31)
mbinc = (mbins[1:]+mbins[:-1])*0.5
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit_fp))

for i_pair in range(len(secondaries)):
    secondary = secondaries[i_pair]
    if fit_type == 'plane':
        tertiary = tertiaries[i_pair]
    else:
        tertiary = 'None'
    
    print("param pair = ", i_pair, secondary, tertiary)

    # array with predicted counts
    count_pred_cent = (GroupCountCent.copy()).astype(GrMcrit_fp.dtype)
    count_pred_sats = (GroupCountSats.copy()).astype(GrMcrit_fp.dtype)

    GroupEnv_fp = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
    if fit_type == 'ramp':
        GroupConc_fp = np.zeros(len(GroupEnv_fp))
    else:
        GroupConc_fp = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')

    x = np.zeros_like(GrMcrit_fp)
    y = np.zeros_like(GrMcrit_fp)
    cts_sats_mean = np.zeros_like(GrMcrit_fp)
    cts_cent_mean = np.zeros_like(GrMcrit_fp)

    a_arr_cent = np.zeros(len(mbins)-1)
    b_arr_cent = np.zeros(len(mbins)-1)
    a_arr_sats = np.zeros(len(mbins)-1)
    b_arr_sats = np.zeros(len(mbins)-1)
    for i in range(len(mbins)-1):
        choice = ((mbins[i]) < GrMcrit_fp) & ((mbins[i+1]) >= GrMcrit_fp)
        nhalo = np.sum(choice)
        if nhalo == 0: continue

        cts = GroupCount[choice]
        cts_cent = GroupCountCent[choice]
        cts_sats = GroupCountSats[choice]
        ngal = np.sum(cts)
        if ngal == 0: continue
        env = GroupEnv_fp[choice]
        conc = GroupConc_fp[choice]

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
            count_pred_cent[choice] = prob_cent(a_cent, b_cent)
            count_pred_sats[choice] = prob_sats(a_sats, b_sats)

            a_arr_cent[i] = a_cent
            b_arr_cent[i] = b_cent
            a_arr_sats[i] = a_sats
            b_arr_sats[i] = b_sats


    if mode == 'all':
        # to spare us some time, concentrate on only the halos above the minimum mass threshold
        choice = (mbins[0] < GrMcrit_fp) & (mbins[-1] >= GrMcrit_fp)
        
        # then we minimize for a and b
        res_cent = minimize(like_cent, p0, method=method)
        a_cent, b_cent = res_cent['x']
        print("centrals: a, b = ", a_cent, b_cent)
        res_sats = minimize(like_sats, p0, method=method)
        a_sats, b_sats = res_sats['x']
        print("satellites: a, b = ", a_sats, b_sats)
        
        # compute prediction for occupancy and save into a new array for centrals and satellites
        count_pred_cent[choice] = prob_cent(a_cent, b_cent)
        count_pred_sats[choice] = prob_sats(a_sats, b_sats)

    # should not be necessary with the prior but just in case (esp. if you are experimenting)
    #count_pred_cent[count_pred_cent < 0.] = 0.
    #count_pred_cent[count_pred_cent > 1.] = 1.
    #count_pred_sats[count_pred_sats < 0.] = 0.
    
    print("satellites = ", np.sum(count_pred_sats))
    print("centrals = ", np.sum(count_pred_cent))
    print("true satellites = ", np.sum(GroupCountSats))
    print("true centrals = ", np.sum(GroupCountCent))
    print("-------------------")
    
    if fit_type == 'plane':
        if mode == 'bins':
            np.save(f"{gal_type:s}/count_pred_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_sats)
            np.save(f"{gal_type:s}/count_pred_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_cent)
            np.save(f"{gal_type:s}/a_arr_cent_{fun_cent:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", a_arr_cent)
            np.save(f"{gal_type:s}/b_arr_cent_{fun_cent:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", b_arr_cent)
            np.save(f"{gal_type:s}/a_arr_sats_{fun_sats:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", a_arr_sats)
            np.save(f"{gal_type:s}/b_arr_sats_{fun_sats:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", b_arr_sats)
        elif mode == 'all':
            np.save(f"{gal_type:s}/count_pred_all_{fun_sats:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_sats)
            np.save(f"{gal_type:s}/count_pred_all_{fun_cent:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_cent)
    else:
        if mode == 'bins':
            np.save(f"{gal_type:s}/count_pred_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", count_pred_sats)
            np.save(f"{gal_type:s}/count_pred_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", count_pred_cent)
        elif mode == 'all':
            np.save(f"{gal_type:s}/count_pred_all_{fun_sats:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", count_pred_sats)
            np.save(f"{gal_type:s}/count_pred_all_{fun_cent:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", count_pred_cent)

