#models.py

from classy import Class
import numpy as np



def build_class(Omega_m, h, Omega_b=0.048):
    cosmo = Class()
    cosmo.set({
        'h': h,
        'Omega_b': Omega_b,           # baryons
        'Omega_cdm': Omega_m - Omega_b,         # CDM
        'Omega_k': 0.0,               # flat
        'Omega_fld': 1.0 - Omega_m,  # dark energy fraction
        'w0_fld': -1.0,              
        'wa_fld': 0.0
                })
    
    cosmo.compute()
    return cosmo

def mu_from_class(z_array, Omega_m, h):
    """
    Return distance modulus mu(z) only.
    """
    cosmo = build_class(Omega_m, h)
    
    #all distances in classy are calculated using Mpc
    dL = np.array([cosmo.luminosity_distance(z) for z in z_array])
    mu = 5.0 * np.log10(dL) + 25.0
    
    cosmo.struct_cleanup()
    cosmo.empty()
    return mu

def bao_model(z_bao, obs_q, Omega_m, h):
    cosmo = build_class(Omega_m, h)
    rd = cosmo.rs_drag()
    dv_model = []
    for z, qty in zip(z_bao, obs_q):
        D_M = cosmo.angular_distance(z) * (1.0 + z)  # Mpc
        H = cosmo.Hubble(z)                          # units depend on CLASS configuration; see notes
        D_H = 1.0 / H                                # Mpc if H is in 1/Mpc
        if qty == 'DM_over_rs':
            dv_model.append(D_M / rd)
        elif qty == 'DH_over_rs':
            dv_model.append(D_H / rd)
        elif qty == 'DV_over_rs':
            D_V = (D_M**2 * D_H * z)**(1.0/3.0)
            dv_model.append(D_V / rd)
        else:
            raise ValueError(f"Unknown BAO quantity: {qty}")
    cosmo.struct_cleanup()
    cosmo.empty()
    return np.array(dv_model)
