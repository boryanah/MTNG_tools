import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
#gal_type = 'ELG'
gal_type = 'LRG'
want_show = True
#want_show = False
fit_type = 'plane'
#fit_type = 'ramp'
n_gal = '7.4e-04'

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
tertiaries = ['GroupConc']
#tertiaries = ['GroupShear_R2']

def plane(x, y, a, b, c):
    return x*a + y*b + c

def _plane(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr += plane(x, y, *args)
    return arr

def ramp(x, y, a, b):
    return y*a + b

def _ramp(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr += ramp(x, y, *args)
    return arr

if fit_type == 'ramp':
    p0 = [1., 0.]
    function = ramp
    _function = _ramp
elif fit_type == 'plane':
    p0 = [1., 1., 0.]
    function = plane
    _function = _plane

# load other halo properties
GroupPos_fp = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
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
mbins = np.logspace(10, 14, 21)
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
    count_pred_cent = (GroupCountCent.copy()).astype(GroupPos_fp.dtype)
    count_pred_sats = (GroupCountSats.copy()).astype(GroupPos_fp.dtype)

    GroupEnv_fp = np.load(tng_dir+f'data_fp/{secondary:s}_fp_{snapshot:d}.npy')
    if fit_type == 'ramp':
        GroupConc_fp = np.zeros(len(GroupEnv_fp))
    else:
        GroupConc_fp = np.load(tng_dir+f'data_fp/{tertiary:s}_fp_{snapshot:d}.npy')

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
        #print(f"halos and total gals in bin {mbins[i]:.1e}-{mbins[i+1]:.1e} = {nhalo:d}, {ngal:d}")

        pos = GroupPos_fp[choice]
        env = GroupEnv_fp[choice]
        conc = GroupConc_fp[choice]
        cts = cts.astype(pos.dtype)
        cts_sats = cts_sats.astype(pos.dtype)
        cts_cent = cts_cent.astype(pos.dtype)

        # turn the secondary and tertiary parameter into ranked arrays
        rank_env = np.argsort(np.argsort(env))/(len(env)-1)-0.5#/2. # TESTING X AND Y ARE SWITCHED FOR SOME FUCNIJGNDR REASON
        rank_conc = np.argsort(np.argsort(conc))/(len(conc)-1)-0.5

        # turn the counts into average occupation
        cts_sats_norm = cts_sats/np.mean(cts_sats)
        cts_cent_norm = cts_cent/np.mean(cts_cent)
        cts_sats_norm[cts_sats == 0.] = 0.
        cts_cent_norm[cts_cent == 0.] = 0.

        # define meshgrid (could probably be taken out)
        x, y = np.linspace(-0.5, 0.5, 6), np.linspace(-0.5, 0.5, 6)
        X, Y = np.meshgrid(x, y)
        bins = np.linspace(-0.5, 0.5, len(x)+1)
        hist_sats, _, _ = np.histogram2d(rank_env, rank_conc, bins=[bins, bins], weights=cts_sats_norm)
        hist_cent, _, _ = np.histogram2d(rank_env, rank_conc, bins=[bins, bins], weights=cts_cent_norm)
        hist, _, _ = np.histogram2d(rank_env, rank_conc, bins=[bins, bins])
        Z_sats = hist_sats/hist
        Z_cent = hist_cent/hist
        Z_sats[hist == 0.] = 0.
        Z_cent[hist == 0.] = 0.
        #print("satellites min, max = ", Z_sats.min(), Z_sats.max())
        #print("centrals min, max = ", Z_cent.min(), Z_cent.max())


        # ravel (flatten) the meshgrids of X, Y points to a pair of 1-D arrays.
        xdata = np.vstack((X.ravel(), Y.ravel()))

        # fit for centrals
        popt, pcov = curve_fit(_function, xdata, Z_cent.ravel(), p0)
        fit_cent = function(X, Y, *popt)
        print("cent params = ", popt)
        a_arr_cent[i] = popt[0]
        b_arr_cent[i] = popt[1]

        # compute prediction for occupancy and save into a new array # TESTING SWITCH
        count_pred_cent[choice] = function(rank_conc, rank_env, *popt)*np.mean(cts_cent)

        # fit for satellites
        popt, pcov = curve_fit(_function, xdata, Z_sats.ravel(), p0)
        fit_sats = function(X, Y, *popt)
        print("sats params = ", popt)
        a_arr_sats[i] = popt[0]
        b_arr_sats[i] = popt[1]

        # compute prediction for occupancy and save into a new array
        count_pred_sats[choice] = function(rank_conc, rank_env, *popt)*np.mean(cts_sats)

        print(f"logm = {mbinc[i]:.1e}")
        print("satellites = ", np.sum(cts_sats))
        print("centrals = ", np.sum(cts_cent))
        print("number of predicted gals vs. true", np.sum(count_pred_sats[choice]+count_pred_cent[choice]), np.sum(cts_sats+cts_cent))
        print("-------------------")
    
        if want_show:
            # plot 3d thing
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z_sats, cmap='plasma')
            ax.plot_surface(X, Y, fit_sats, cmap='magma')
            ax.set_zlim(0, np.max([Z_sats, fit_sats]))
            ax.set_ylabel(secondary)
            ax.set_xlabel(tertiary)
            ax.set_zlabel("N sats/<N sats>")
            plt.show()

            # plot 3d thing
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z_cent, cmap='plasma')
            ax.plot_surface(X, Y, fit_cent, cmap='magma')
            ax.set_ylabel(secondary)
            ax.set_xlabel(tertiary)
            ax.set_zlabel("N cent/<N cent>")
            ax.set_zlim(0, np.max([Z_cent, fit_cent]))
            plt.show()

        # do fitting of linear function or polynomial?
        
    if fit_type == 'plane':
        np.save(f"{gal_type:s}/count_pred_{fit_type:s}_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_sats)
        np.save(f"{gal_type:s}/count_pred_{fit_type:s}_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", count_pred_cent)
        np.save(f"{gal_type:s}/a_arr_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", a_arr_cent)
        np.save(f"{gal_type:s}/b_arr_cent_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", b_arr_cent)
        np.save(f"{gal_type:s}/a_arr_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", a_arr_sats)
        np.save(f"{gal_type:s}/b_arr_sats_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy", b_arr_sats)
        np.save(f"{gal_type:s}/mbinc.npy", mbinc)

    else:
        np.save(f"{gal_type:s}/count_pred_{fit_type:s}_sats_{secondary:s}_fp_{snapshot:d}.npy", count_pred_sats)
        np.save(f"{gal_type:s}/count_pred_{fit_type:s}_cent_{secondary:s}_fp_{snapshot:d}.npy", count_pred_cent)


'''
GroupConcRad
Group_R_TopHat200_fp_179.npy
GroupConc_fp_179.npy
GroupShear_R[1,2,5]_fp_179.npy
GroupEnv_R[1,2,5]_fp_179.npy
GroupMarkedEnv_R[1,2,5]_s0.25_p2_fp_179.npy
SubhaloVelDisp_fp_179.npy
SubhaloVmax_fp_179.npy
SubhaloHalfmassRadType_fp_179.npy
SubhaloVmaxRad_fp_179.npy
'''
