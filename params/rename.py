import numpy as np
import os
import glob
import re

fns = glob.glob("/mnt/alan1/boryanah/MTNG/data_dm/*dm_179.npy")

old = 'dm_179'
new = 'dm_184'

for i in range(len(fns)):
    src = fns[i]
    dst = fns[i]
    dst = re.sub(old, new, dst)
    print("src, dst", src, dst)
    os.rename(src, dst)
