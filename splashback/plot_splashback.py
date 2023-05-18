import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

cs = ['#CC3311','#EE7733','limegreen','#0077BB','#BBBBBB','#33BBEE','#EE3377','#0099BB']
#cs = ['#33BBEE','#EE7733','#CC3311','#0099BB','#0077BB','#EE3377','#BBBBBB']

#sim_type = 'MTNG'; fp_dm = 'fp'
sim_type = 'MTNG'; fp_dm = 'dm'
#sim_type = 'TNG'; fp_dm = 'dm'
#sim_type = 'TNG'; fp_dm = 'fp'
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
tng_dir_dic['MTNG'] = "/mnt/alan1/boryanah/MTNG/dm_arepo/"

# simulation params
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]

snapshot0 = 179; z0 = 1.0;
#snapshot0 = 264; z0 = 0.0;

if (fp_dm == 'fp' or "arepo" in tng_dir) and sim_type == 'MTNG':
    snapshot0 += 0    
elif fp_dm == 'dm' and sim_type == 'MTNG':
    snapshot0 += 5
if sim_type == 'TNG':
    snapshot0 = 55; z0 = 0.82; snapshot1 = 41; z1 = 1.41
    
if sim_type == 'MTNG':
    Group_M_Mean200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_M_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e10
    Group_R_Mean200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_R_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e3
    Rsplash = np.load(tng_dir+f'data_{fp_dm:s}/Group_R_Splash_{fp_dm:s}_{snapshot0:d}.npy')
    Msplash = np.load(tng_dir+f'data_{fp_dm:s}/Group_M_Splash_{fp_dm:s}_{snapshot0:d}.npy')
else:
    Group_M_Mean200 = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e10 # Msun/h
    Group_R_Mean200 = np.load(tng_dir+f'Group_R_Mean200_{fp_dm:s}_{snapshot0:d}.npy') # ckpc/h
    Rsplash = np.load(tng_dir+f'Group_R_Splash_{fp_dm:s}_{snapshot0:d}.npy')
    Msplash = np.load(tng_dir+f'Group_M_Splash_{fp_dm:s}_{snapshot0:d}.npy')
    print(Msplash)
Group_R_Mean200 = Group_R_Mean200*1./(1+z0)

logm_bins = np.linspace(11.75, 13.75, 5)
logm_binc = (logm_bins[1:] + logm_bins[:-1])*.5
dlogm = 0.5*(logm_bins[1]-logm_bins[0])
logm_bins = 10.**logm_bins
logm_binc = 10.**logm_binc

for i in range(len(logm_bins)-1):
    r_choice = (Group_M_Mean200 > logm_bins[i]) & (Group_M_Mean200 <= logm_bins[i+1])
    m_choice = (Group_M_Mean200 > logm_bins[i]) & (Group_M_Mean200 <= logm_bins[i+1])
    print("total number of halos = ", np.sum(r_choice), np.sum(m_choice))
    print(len(Rsplash), len(Group_R_Mean200))
    R_rat = (Rsplash/Group_R_Mean200)[r_choice]
    M_rat = (Msplash/Group_M_Mean200)[m_choice]

    print("min max R_rat = ", np.min(R_rat), np.max(R_rat))
    print("min max M rat = ", np.min(M_rat), np.max(M_rat))

    r_bins = np.linspace(0.5, 2., 31)
    r_binc = (r_bins[1:] + r_bins[:-1])*.5
    
    m_bins = np.linspace(0.5, 2., 31)
    m_binc = (m_bins[1:] + m_bins[:-1])*.5

    hist_r, _ = np.histogram(R_rat, r_bins, density=True)
    hist_m, _ = np.histogram(M_rat, m_bins, density=True)

    plt.figure(1, figsize=(9, 7))
    plt.axvline(x=np.median(R_rat), ls='--', color=cs[i])
    plt.plot(r_binc, hist_r, color=cs[i], label=rf'$\log M = {np.log10(logm_binc[i]):.1f} \pm {dlogm:.2f}$')
    
    plt.figure(2, figsize=(9, 7))
    plt.axvline(x=np.median(M_rat), ls='--', color=cs[i])
    plt.plot(m_binc, hist_m, color=cs[i], label=rf'$\log M = {np.log10(logm_binc[i]):.1f} \pm {dlogm:.2f}$')

plt.figure(1, figsize=(9, 7))
plt.ylabel(r'${\rm PDF}$')
plt.xlabel(r'$R_{\rm sp}/R_{\rm 200m}$')
plt.legend(fontsize=22)
plt.savefig(f"figs/r_splash.pdf", bbox_inches='tight', pad_inches=0.)

plt.figure(2, figsize=(9, 7))
plt.ylabel(r'${\rm PDF}$')
plt.xlabel(r'$M_{\rm sp}/M_{\rm 200m}$')
#plt.legend(fontsize=22)
plt.savefig(f"figs/m_splash.pdf", bbox_inches='tight', pad_inches=0.)
plt.show()
