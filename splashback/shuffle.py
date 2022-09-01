import sys

import numpy as np
import matplotlib.pyplot as plt

from tools.halostats import get_jack_corr
import plotparams
plotparams.buba()

assignment = 'query' 
#assignment = 'FoF' # must do fp for this

# simulation parameters
sim_type = 'MTNG'; fp_dm = 'fp'; pos_unit = 1.e3; snapshot = 179; z = 1.
#sim_type = 'MTNG'; fp_dm = 'fp'; pos_unit = 1.e3; snapshot = 264; z = 0.
#sim_type = 'TNG'; fp_dm = 'dm'; pos_unit = 1.; snapshot = 55; z = 0.82
#sim_type = 'TNG'; fp_dm = 'fp'; pos_unit = 1.; snapshot = 55; z = 0.82
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': f"/mnt/alan1/boryanah/MTNG/data_{fp_dm:s}/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
tracer = sys.argv[1] # LRG ELG
n_gal = '2.0e-03'

# simulation params
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]*1.e3 # ckpc/h
a = 1./(1+z)

boundary = sys.argv[2] # 'r200m', 'rsplash'
if boundary == 'r200m': text = "Virial radius"
if boundary == 'rsplash': text = "Splashback"

# load halo properties
GroupPos = np.load(tng_dir+f'GroupPos_{fp_dm:s}_{snapshot:d}.npy')*pos_unit # ckpc/h
if boundary == 'rsplash':
    M_splash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # splashback mass
    R_splash = np.load(tng_dir+f'Group_R_Splash_{fp_dm:s}_{snapshot:d}.npy')/a # ckpc/h # splashback radius (output in kpc/h)
elif boundary == 'r200m':
    #M_splash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # splashback mass # TESTING the majority of the effect seems to come from using M splash
    M_splash = np.load(tng_dir+f'Group_M_TopHat200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h # original mass
    R_splash = np.load(tng_dir+f'Group_R_TopHat200_{fp_dm:s}_{snapshot:d}.npy')*pos_unit # ckpc/h # originalen radius
M200mean = np.load(tng_dir+f'Group_M_TopHat200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h

# define galaxies
if sim_type == "MTNG":
    gal_ind = np.load(f"../selection/data/index_{tracer}_{n_gal}_179.npy")
    N_g = len(gal_ind)
else:
    N_g = 12000
    M_star = np.load(tng_dir+f'SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10 # Msun/h
    gal_ind = (np.argsort(M_star)[::-1])[:N_g]

# define the central subhalos
if sim_type == "MTNG":
    SubGrNr = np.load(tng_dir+f'SubhaloGroupNr_fp_{snapshot:d}.npy')
else:
    SubGrNr = np.load(tng_dir+f'SubhaloGrNr_fp_{snapshot:d}.npy')
gal_grn = SubGrNr[gal_ind]
SubPos = np.load(tng_dir+f'SubhaloPos_fp_{snapshot:d}.npy')*pos_unit # ckpc/h
gal_pos = SubPos[gal_ind]

gal_par = np.load(f'data/galaxy_parent_{tracer}_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy')
missing = gal_par <= -1
print("missing galaxies number and fraction = ", np.sum(missing), np.sum(missing)*100./len(missing))
want_ignore = False
if want_ignore:
    print("ignore missing")
    gal_par = gal_par[~missing]
    ignore_str = "_ignore"
else:
    ignore_str = ""
print("we are just gonna add them based on enclosed density") # cause they occupy the negative indices
gal_par = np.abs(gal_par)

if boundary == 'rsplash':
    m_bins = np.logspace(11, 14.2, 21)
else:
    m_bins = np.logspace(11, 14, 21)
m_binc = (m_bins[1:] + m_bins[:-1])*.5

GroupCount = np.zeros(len(M_splash), dtype=int)
if assignment == 'query':
    un_par, cts = np.unique(gal_par, return_counts=True) # new assignment
elif assignment == 'FoF':
    assert  fp_dm == 'fp', "To use with DMO, need to have a bridge between FP (sub)halos and DMO (sub)halos"
    un_par, cts = np.unique(gal_grn, return_counts=True) # FoF assignment
GroupCount[un_par] = cts

plot_hod = False
if plot_hod:
    hist_gal, _ = np.histogram(M_splash[gal_par], m_bins)
    hist_halo, _ = np.histogram(M_splash, m_bins)

    plt.plot(m_binc, hist_gal/hist_halo)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

GroupCountShuff = GroupCount.copy()
for i in range(len(m_bins)-1):
    #choice = ((m_bins[i]) < M200mean) & ((m_bins[i+1]) >= M200mean)
    choice = ((m_bins[i]) < M_splash) & ((m_bins[i+1]) >= M_splash)
    nhalo = np.sum(choice)
    if nhalo == 0: continue
    
    cts = GroupCount[choice]
    ngal = np.sum(cts)
    if ngal == 0: continue
    
    pos = GroupPos[choice]
    cts = cts.astype(pos.dtype)

    cts_shuff = cts.copy()
    np.random.shuffle(cts_shuff)
    GroupCountShuff[choice] = cts_shuff


#x_true = gal_pos
#w_true = np.ones(x_true.shape[0], dtype=x_true.dtype)

# convert units
GroupPos /= 1.e3 # cMpc/h
Lbox /= 1.e3

w_true = GroupCount[GroupCount > 0].astype(gal_pos.dtype)
x_true = GroupPos[GroupCount > 0].astype(gal_pos.dtype)

w_shuff = GroupCountShuff[GroupCountShuff > 0].astype(x_true.dtype)
x_shuff = GroupPos[GroupCountShuff > 0].astype(x_true.dtype)
print(w_true.sum(), w_shuff.sum())

rbins = np.logspace(-1, 1.5, 31)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:]+rbins[:-1])/2.

# N_dim should maybe be 5
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(x_true, w_true, x_shuff, w_shuff, Lbox, N_dim=3, bins=rbins)
np.savez(f"data/corr_{tracer}_{boundary}_{snapshot:d}{ignore_str}.npz", rbinc=rbinc, rat_mean=rat_mean, rat_err=rat_err, corr_shuff_mean=corr_shuff_mean, corr_shuff_err=corr_shuff_err, corr_true_mean=corr_true_mean, corr_true_err=corr_true_err)

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
plt.ylabel(r'$\xi (r) r^2$')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
plt.savefig(f'figs/corr_{boundary}_{snapshot:d}{ignore_str}.png')
plt.close()

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.ylim([0.75, 1.25])
plt.xscale('log')
plt.ylabel(r'$\xi_{\rm shuff}/\xi_{\rm orig}$')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
ax = plt.gca()
plt.text(x=0.03, y=0.90, s=text, transform=ax.transAxes, color="black")
plt.savefig(f'figs/corr_rat_{boundary}_{snapshot:d}{ignore_str}.png')
#plt.show()
plt.close()
