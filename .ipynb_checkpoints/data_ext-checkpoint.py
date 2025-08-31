#data_ext.py

import numpy as np
import pandas as pd

def load_pantheon(path):
    df = pd.read_csv(path, sep='\s+')
    z = df["zCMB"].values
    m = df["m_b_corr"].values
    merr = df["m_b_corr_err_DIAG"].values
    return z, m, merr

def load_desi(mean_path, cov_path):
    bao = np.loadtxt(mean_path, dtype={"names":("z","value","qty"),"formats":(float,float,"U10")})
    z = bao["z"]; obs_val = bao["value"]; obs_q = bao["qty"]
    cov = np.loadtxt(cov_path)
    inv_cov = np.linalg.inv(cov)
    return z, obs_q, obs_val, inv_cov
