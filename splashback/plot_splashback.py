import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

cs = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']

#sim_type = 'MTNG'; fp_dm = 'fp'
#sim_type = 'TNG'; fp_dm = 'dm'
sim_type = 'TNG'; fp_dm = 'fp'
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}

# simulation params
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]

snapshot0 = 55; z0 = 0.82

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

logm_bins = np.logspace(np.log10(2.e11), 14., 5)
logm_binc = (logm_bins[1:] + logm_bins[:-1])*.5

#(Msplash > 0.) & (Group_M_Mean200 > 0) &
for i in range(len(logm_bins)-1):
    r_choice = (Group_M_Mean200 > logm_bins[i]) & (Group_M_Mean200 <= logm_bins[i+1])
    m_choice = (Group_M_Mean200 > logm_bins[i]) & (Group_M_Mean200 <= logm_bins[i+1])
    print("total number of halos = ", np.sum(r_choice), np.sum(m_choice))
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
    plt.plot(r_binc, hist_r, color=cs[i], label=rf'$\log M = {np.log10(logm_binc[i]):.1f}$')
    plt.ylabel(r'${\rm PDF}$')
    plt.xlabel(r'$R_{\rm sp}/R_{\rm 200m}$')
    plt.legend()
    
    plt.figure(2, figsize=(9, 7))
    plt.plot(m_binc, hist_m, color=cs[i], label=rf'$\log M = {np.log10(logm_binc[i]):.1f}$')
    plt.axvline(x=np.median(M_rat), ls='--', color=cs[i])
    plt.ylabel(r'${\rm PDF}$')
    plt.xlabel(r'$M_{\rm sp}/M_{\rm 200m}$')
    plt.legend()
plt.show()
