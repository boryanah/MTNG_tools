import os

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

def annulus_env(allpos, allmasses, allrvir, allsubpos, allsubmasses, r_outer, Lbox):
    allrvir = allrvir.astype(allpos.dtype)
    print(allrvir.shape, allpos.shape)

    
    # for KD tree
    allsubpos %= Lbox
    allpos %= Lbox
    #allsubpos_tree = KDTree(allsubpos)
    allsubpos_tree = cKDTree(allsubpos, boxsize=Lbox)
    print("built KD tree")
    
    allsubinds_inner = allsubpos_tree.query_ball_point(allpos, r=allrvir)
    allsubinds_outer = allsubpos_tree.query_ball_point(allpos, r=r_outer)
    #allsubinds_inner = allsubpos_tree.query_radius(allpos, r=allrvir)
    #allsubinds_outer = allsubpos_tree.query_radius(allpos, r=r_outer)
    print("queried tree")
    
    #Menv = np.array([np.sum(allsubmasses[allsubinds_outer[ind]]) - np.sum(allsubmasses[allsubinds_inner[ind]]) for ind in np.arange(len(allmasses))])
    Menv = np.array([(np.sum(allsubmasses[allsubinds_outer[ind]]) - np.sum(allsubmasses[allsubinds_inner[ind]]))/(r_outer**3-allrvir[ind]**3) for ind in np.arange(len(allmasses))])
    print("computed mass stacks")
    
    nbins = 100
    mbins = np.logspace(np.log10(3e10), 15.5, nbins+1)
    
    """
    fenv_rank = np.zeros(len(Menv))
    for ibin in range(nbins):
        mmask = (allmasses > mbins[ibin]) \
                & (allmasses < mbins[ibin + 1])
        if np.sum(mmask) > 0:
            fenv_rank[mmask] = Menv[mmask]/np.mean(Menv[mmask])
            '''
            if np.sum(mmask) == 1:
                fenv_rank[mmask] = 0
            else:
                new_fenv_rank = Menv[mmask].argsort().argsort()
                fenv_rank[mmask] = new_fenv_rank / np.max(new_fenv_rank) - 0.5
            '''
    return fenv_rank
    """
    return Menv
    

tng_dir = '/mnt/gosling1/boryanah/TNG300/'
type_tng = "_fp"
#type_tng = "_dm"
snap_str = "_55"
#snap_str = ""
SubhaloPos = np.load(tng_dir+'SubhaloPos'+type_tng+snap_str+'.npy')/1.e3
SubhaloMdm = np.load(tng_dir+'SubhaloMassType'+type_tng+snap_str+'.npy')[:, 1]*1.e10
SubhaloGrNr = np.load(tng_dir+'SubhaloGrNr'+type_tng+snap_str+'.npy')
GroupPos = np.load(tng_dir+'GroupPos'+type_tng+snap_str+'.npy')/1.e3
Group_R_Mean200 = np.load(tng_dir+'Group_R_Mean200'+type_tng+snap_str+'.npy')/1.e3 # PROBLEM for snap 99
#Group_R_Mean200 = np.load(tng_dir+'Group_R_Crit200'+type_tng+snap_str+'.npy')/1.e3
Group_M_Mean200 = np.load(tng_dir+'Group_M_Mean200'+type_tng+snap_str+'.npy')*1.e10
Lbox = 205.

r_outers = [3, 5, 7]
for r_outer in r_outers:
    #if os.path.exists(tng_dir+f'GroupAnnEnv_R{r_outer:d}'+type_tng+snap_str+'.npy'): continue
    print("r_outer = ", r_outer)

    # leave only subhalos satisfying a minimum threshold
    inds = np.arange(len(SubhaloMdm), dtype=int)[SubhaloMdm > 1.e11]
    
    fenv_rank = annulus_env(GroupPos[:], Group_M_Mean200[:], Group_R_Mean200[:], SubhaloPos[inds], SubhaloMdm[inds], r_outer, Lbox)
    np.save(tng_dir+f'GroupAnnEnv_R{r_outer:d}'+type_tng+snap_str+'.npy', fenv_rank)
