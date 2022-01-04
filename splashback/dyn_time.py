from colossus.halo import *
from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology('planck18')

z0s = [0., 0.25, 0.5, 0.82, 1., 1.5, 2.0]

for z0 in z0s:
    t_dyn = mass_so.dynamicalTime(z0, '200m')
    t0 = cosmo.age(z0)
    t1 = t0 - t_dyn
    z1 = cosmo.age(t1, inverse = True)

    print("t_dyn, z0, z1 = ", t_dyn, z0, z1)
