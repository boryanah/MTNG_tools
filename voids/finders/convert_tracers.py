import numpy as np

pos = np.load("/mnt/gosling1/boryanah/TNG300/pos_parts_down_5000_tng300-3.npy")
#pos = np.load("/mnt/gosling1/boryanah/TNG300/parts_position_tng300-3_99.npy")

np.savetxt("tracers.dat", pos)
