import numpy as np
import matplotlib.pyplot as plt

tng_dir = '/mnt/gosling1/boryanah/TNG300/'

R_jay = np.logspace(np.log10(0.4), np.log10(10), 20)
print(R_jay)
index = 6 # 6 corr to 1ish 10 to 2ish 12 to 3 15 to 5

# load fields
R = 1
GroupShear = np.load(tng_dir+f'GroupShear_R{R:d}_fp.npy')
#GroupMass = np.sum(np.load(tng_dir+"GroupMassType_fp.npy"), axis=1)*1.e10
GroupMass = np.load(tng_dir+"Group_M_Mean200_fp.npy")*1.e10
GroupShearJay = np.load("data_jay/GroupShear_qR_fp_1e11Mass.npy")[:, index]
print(GroupShearJay.shape)


GroupShear = GroupShear[GroupMass > 1.e11]
#GroupShear = GroupShear[:GroupShearJay.shape[0]]
assert GroupShear.shape == GroupShearJay.shape


plt.figure(figsize=(9, 7))
plt.scatter(GroupShearJay, GroupShear, s=0.1)
plt.show()
