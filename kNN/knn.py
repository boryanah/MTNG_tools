import numpy as np
import sys, os
import scipy.spatial
from scipy import interpolate


def cdf_vol_knn(vol):
    '''
    Computes an interpolating function to evaluate CDF 
    at a given radius.
    
    Parameters
    ----------
    
    vol: float[:,:]
        List of nearest neighbor distances for each kNN.
        vol.shape[1] should be # of kNN
    
    Returns
    -------
    
    cdf: scipy interpolating function for each kNN
    '''
    
    cdf = []
    n = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, n) + 1) / (n*1.0))
    for c in range(l):
        ind = np.argsort(vol[:, c])
        s_vol= vol[ind, c]
        cdf.append(interpolate.interp1d(s_vol, gof, kind='linear', 
                                        bounds_error=False))
    return cdf

def compute_cdf(pos, kNN, nrandoms, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    boxsize: float
        Size of the simulation box
    bins: float[:, :]
        Bin centers for each kNN
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    xtree = scipy.spatial.cKDTree(pos, boxsize=boxsize)

    #Generate  nrandoms randoms on the same volume
    random_pos = np.random.rand(nrandoms, 3)*boxsize

    vol, disi = xtree.query(random_pos, k=kNN,
                            n_jobs=-1)

    bine = np.logspace(0, 1.9, 1000)
    binc = (bine[1:] + bine[:-1]) / 2


    #Now get the CDF
    data = (np.zeros((bins.shape[0], 
                      len(kNN))))
    cdfs = cdf_vol_knn(vol)
    for i in range(len(kNN)):
        dummycdf = cdfs[i](binc)
        dummycdf[np.isnan(dummycdf)] = 1.0
        cdf_interp = interpolate.interp1d(binc, dummycdf, 
                                          kind='linear', 
                                          bounds_error=False, 
                                          fill_value=(0., 0.))
        data[:, i] = cdf_interp(bins[:, i])
    return data

def compute_cdf_query(pos, kNN, random_pos, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    boxsize: float
        Size of the simulation box
    bins: float[:, :]
        Bin centers for each kNN
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    xtree = scipy.spatial.cKDTree(pos, boxsize=boxsize)

    # query the tree
    vol, disi = xtree.query(random_pos, k=kNN,
                            n_jobs=-1)

    bine = np.logspace(0, 1.9, 1000)
    binc = (bine[1:] + bine[:-1]) / 2


    #Now get the CDF
    data = (np.zeros((bins.shape[0], 
                      len(kNN))))
    cdfs = cdf_vol_knn(vol)
    for i in range(len(kNN)):
        dummycdf = cdfs[i](binc)
        dummycdf[np.isnan(dummycdf)] = 1.0
        cdf_interp = interpolate.interp1d(binc, dummycdf, 
                                          kind='linear', 
                                          bounds_error=False, 
                                          fill_value=(0., 0.))
        data[:, i] = cdf_interp(bins[:, i])
    return data


def compute_jackknife_cdf(pos, kNN, nrandoms, njackdim, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    njackdim: int
        Number of dimensions for the jackknifing
    boxsize: float
        Size of the simulation box
    bins: float[:, :]
        Bin centers for each kNN
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    #Generate  nrandoms randoms on the same volume
    pos_query = np.random.rand(nrandoms, 3)*boxsize
    
    data_all = np.zeros((bins.shape[0], njackdim**3, len(kNN)))
    for i in range(njackdim):
        for j in range(njackdim):
            for k in range(njackdim):
                print(i, j, k)
                
                ijk = np.array([i, j, k],dtype=int)
                size = boxsize/njackdim
                
                pos_jack = pos.copy()
                bool_arr = np.prod((ijk == (pos/size).astype(int)), axis=1).astype(bool)
                pos_jack[bool_arr] = np.array([0., 0., 0.])
                pos_jack = pos_jack[np.sum(pos_jack,axis=1)!=0.]

                pos_query_jack = pos_query.copy()
                bool_arr = np.prod((ijk == (pos_query/size).astype(int)), axis=1).astype(bool)
                pos_query_jack[bool_arr] = np.array([0., 0., 0.])
                pos_query_jack = pos_query_jack[np.sum(pos_query_jack,axis=1)!=0.]
                
                data = compute_cdf_query(pos_jack, kNN, pos_query_jack, boxsize, bins)
                data_all[:, i*njackdim**2+j*njackdim+k, :] = data
                
    data_mean = np.zeros((bins.shape[0], len(kNN)))
    data_err = np.zeros((bins.shape[0], len(kNN)))
    for i in range(len(kNN)):
        data_i = data_all[:, :, i]
        data_mean[:, i] = np.mean(data_i, axis=1)
        data_err[:, i] = np.sqrt(njackdim**3 - 1) * np.std(data_i, axis=1)

    return data_mean, data_err


def compute_jackknife_cdf_multi(pos1, pos2, kNN, nrandoms, njackdim, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    njackdim: int
        Number of dimensions for the jackknifing
    boxsize: float
        Size of the simulation box
    bins: float[:, :]
        Bin centers for each kNN
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    #Generate  nrandoms randoms on the same volume
    pos_query = np.random.rand(nrandoms, 3)*boxsize
    
    data1_all = np.zeros((bins.shape[0], njackdim**3, len(kNN)))
    data2_all = np.zeros((bins.shape[0], njackdim**3, len(kNN)))
    data21_all = np.zeros((bins.shape[0], njackdim**3, len(kNN)))
    for i in range(njackdim):
        for j in range(njackdim):
            for k in range(njackdim):
                print(i, j, k)
                
                ijk = np.array([i, j, k],dtype=int)
                size = boxsize/njackdim
                
                pos1_jack = pos1.copy()
                bool_arr = np.prod((ijk == (pos1/size).astype(int)), axis=1).astype(bool)
                pos1_jack[bool_arr] = np.array([0., 0., 0.])
                pos1_jack = pos1_jack[np.sum(pos1_jack,axis=1)!=0.]

                pos2_jack = pos2.copy()
                bool_arr = np.prod((ijk == (pos2/size).astype(int)), axis=1).astype(bool)
                pos2_jack[bool_arr] = np.array([0., 0., 0.])
                pos2_jack = pos2_jack[np.sum(pos2_jack,axis=1)!=0.]
                
                pos_query_jack = pos_query.copy()
                bool_arr = np.prod((ijk == (pos_query/size).astype(int)), axis=1).astype(bool)
                pos_query_jack[bool_arr] = np.array([0., 0., 0.])
                pos_query_jack = pos_query_jack[np.sum(pos_query_jack,axis=1)!=0.]
                
                data1 = compute_cdf_query(pos1_jack, kNN, pos_query_jack, boxsize, bins)
                data2 = compute_cdf_query(pos2_jack, kNN, pos_query_jack, boxsize, bins)
                data1_all[:, i*njackdim**2+j*njackdim+k, :] = data1
                data2_all[:, i*njackdim**2+j*njackdim+k, :] = data2
                data21_all[:, i*njackdim**2+j*njackdim+k, :] = data2/data1
                
    data1_mean = np.zeros((bins.shape[0], len(kNN)))
    data1_err = np.zeros((bins.shape[0], len(kNN)))
    data2_mean = np.zeros((bins.shape[0], len(kNN)))
    data2_err = np.zeros((bins.shape[0], len(kNN)))
    data21_mean = np.zeros((bins.shape[0], len(kNN)))
    data21_err = np.zeros((bins.shape[0], len(kNN)))
    for i in range(len(kNN)):
        data1_i = data1_all[:, :, i]
        data1_mean[:, i] = np.mean(data1_i, axis=1)
        data1_err[:, i] = np.sqrt(njackdim**3 - 1) * np.std(data1_i, axis=1)
        
        data2_i = data2_all[:, :, i]
        data2_mean[:, i] = np.mean(data2_i, axis=1)
        data2_err[:, i] = np.sqrt(njackdim**3 - 1) * np.std(data2_i, axis=1)

        data21_i = data21_all[:, :, i]
        data21_mean[:, i] = np.mean(data21_i, axis=1)
        data21_err[:, i] = np.sqrt(njackdim**3 - 1) * np.std(data21_i, axis=1)

    return data1_mean, data1_err, data2_mean, data2_err, data21_mean, data21_err


def compute_cdf_multi(pos1, pos2, kNN, nrandoms, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    njackdim: int
        Number of dimensions for the jackknifing
    boxsize: float
        Size of the simulation box
    bins: float[:, :]
        Bin centers for each kNN
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    #Generate  nrandoms randoms on the same volume
    pos_query = np.random.rand(nrandoms, 3)*boxsize                

    data1 = compute_cdf_query(pos1, kNN, pos_query, boxsize, bins)
    data2 = compute_cdf_query(pos2, kNN, pos_query, boxsize, bins)
    data21 = data2/data1

    return data1, data2, data21
