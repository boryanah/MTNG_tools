import numpy as np
import matplotlib.pyplot as plt

tng_dir = "/mnt/gosling1/boryanah/TNG300/"
Lbox = 205.
snapshot = 55
#snapshot = 47
#snapshot = 41
SubhaloSFR = np.load(tng_dir+f"SubhaloSFR_fp_{snapshot:d}.npy")
SubhaloMstar = np.load(tng_dir+f"SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10
#IndsDESI = np.load(tng_dir+f"SubhaloIndsELG_DESI_fp_{snapshot:d}.npy")
IndsDESI = np.load(tng_dir+f"SubhaloIndsPreELG_DESI_fp_{snapshot:d}.npy")
#IndsDESI = np.load(tng_dir+f"SubhaloIndsPreELG_eBOSS_fp_{snapshot:d}.npy")


SubhalosSFR = SubhaloSFR/SubhaloMstar

mstar, ssfr = SubhaloMstar[IndsDESI], SubhalosSFR[IndsDESI]
if snapshot == 41:
    # 41 this gives 2400 (1500)
    mstar_thresh = 2.5e10
    ssfr_thresh = 7.5e-10
    mstar_lrg = 1.e11
elif snapshot == 47:
    # 47 this gives 8000 (4600)
    mstar_thresh = 8.44e9
    ssfr_thresh = 8e-10
    mstar_lrg = 6.5e10
elif snapshot == 55:
    # 55 this gives 26000 objects (8600)
    mstar_thresh = 6.44e9
    ssfr_thresh = 4.0e-10
    mstar_lrg = 5.2e10
IndsELG_superDESI = np.arange(len(SubhaloMstar), dtype=int)[(SubhaloMstar > mstar_thresh) & (SubhalosSFR > ssfr_thresh)]
print("number of galaxies = ", len(IndsELG_superDESI))
print("number of original galaxies = ", len(IndsDESI))
print("number density of galaxies = ", len(IndsELG_superDESI)/Lbox**3)
mstar_new, ssfr_new = SubhaloMstar[IndsELG_superDESI], SubhalosSFR[IndsELG_superDESI]
np.save(tng_dir+f"SubhaloIndsELG_superDESI_fp_{snapshot:d}.npy", IndsELG_superDESI)

choice = SubhaloMstar > 1.e9
SubhaloMstar, SubhalosSFR = SubhaloMstar[choice], SubhalosSFR[choice]
print("%.1e, %.1e" %(SubhalosSFR.min(), SubhalosSFR.max()))
print("%.1e, %.1e" %(SubhaloMstar.min(), SubhaloMstar.max()))

#ELG's are approximately y > 1.9-10, x > 6.44e9

index_LRG = np.argsort(SubhaloMstar)[::-1][:len(IndsDESI)]
print("min mass LRG = %.1e" %SubhaloMstar[index_LRG].min())

plt.figure(figsize=(9, 7))
plt.scatter(SubhaloMstar, SubhalosSFR, s=10, color="dodgerblue", alpha=0.1, marker="*")
plt.scatter(SubhaloMstar[index_LRG], SubhalosSFR[index_LRG], s=50, color="orangered", marker="*", alpha=0.2)
# ELGs
#plt.scatter(mstar, ssfr, s=50, color="orangered", marker="*", alpha=0.2)
#plt.scatter(mstar_new, ssfr_new, s=40, alpha=0.4, color="gray", marker="*")
plt.gca().axvline(x=mstar_thresh, color='k', ls='--')
plt.gca().axhline(y=ssfr_thresh, color='k', ls='--')
plt.ylim([1.e-14, 1.e-7])
plt.xscale('log')
plt.yscale('log')
plt.show()
