import numpy as np
import matplotlib.pyplot as plt

from tools.halostats import get_jack_corr
import plotparams
plotparams.buba()

#Lbox = 500. # cMpc/h
Lbox = 205. # cMpc/h
snapshot = 55
z = 0.82
a = 1./(1+z)
#fp_dm = 'dm'
fp_dm = 'fp'

#boundary = 'r200m'
boundary = 'rsplash' # currently have it only for DMO
assignment = 'query' 
#assignment = 'FoF' # must do fp for this

tng_dir = '/mnt/gosling1/boryanah/TNG300/'
GroupPos = np.load(tng_dir+f'GroupPos_{fp_dm:s}_{snapshot:d}.npy') # ckpc/h
if boundary == 'r200m':
    M_splash = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h # orig mass
    R_splash = np.load(tng_dir+f'Group_R_Mean200_{fp_dm:s}_{snapshot:d}.npy') # ckpc/h # original radius
elif boundary == 'rsplash':
    M_splash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot:d}.npy') # Msun/h # splashback mass
    R_splash = np.load(tng_dir+f'Group_R_Splash_{fp_dm:s}_{snapshot:d}.npy')/a # ckpc/h # splashback radius
M200mean = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot:d}.npy')*1.e10 # Msun/h

N_g = 12000
M_star = np.load(tng_dir+f'SubhaloMassType_fp_{snapshot:d}.npy')[:, 4]*1.e10 # Msun/h
SubPos = np.load(tng_dir+f'SubhaloPos_fp_{snapshot:d}.npy') # ckpc/h
SubGrNr = np.load(tng_dir+f'SubhaloGrNr_fp_{snapshot:d}.npy')
ind_gal = (np.argsort(M_star)[::-1])[:N_g]
gal_grn = SubGrNr[ind_gal]
gal_pos = SubPos[ind_gal]

gal_par = np.load(f'data/galaxy_parent_{boundary:s}_{fp_dm:s}_{snapshot:d}.npy')
gal_par = gal_par[gal_par != -1]

m_bins = np.logspace(11, 14, 21)
m_binc = (m_bins[1:] + m_bins[:-1])*.5

GroupCount = np.zeros(len(M200mean), dtype=int)
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
    choice = ((m_bins[i]) < M200mean) & ((m_bins[i+1]) >= M200mean)
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

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
plt.ylabel(r'$\xi (r) r^2$')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
#plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.ylim([0.75, 1.25])
plt.xscale('log')
plt.ylabel(r'$\xi_{\rm shuff}/\xi_{\rm orig}$')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
#plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}_{snapshot:d}.png')
plt.show()
