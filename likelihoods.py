#likelihoods.py

from models import bao_model, mu_from_class
import numpy as np
from functools import lru_cache
from classy import CosmoSevereError

# caching mu_from_class and bao_model calls to avoid recomputation
@lru_cache(maxsize=12000)
def cached_mu(Omega_m, h, z_sne):
    """Cache mu_from_class results keyed by (Omega_m, h, z_sne tuple)."""
    return tuple(mu_from_class(list(z_sne), Omega_m, h))

@lru_cache(maxsize=12000)
def cached_bao_model(z_bao_tuple, obs_q_tuple, Om, h):
    return tuple(bao_model(np.array(z_bao_tuple), np.array(obs_q_tuple), Om, h))


def chi2_sne(Omega_m, h, M, z_sne, m_b, m_b_err):
    mu = np.array(cached_mu(Omega_m, h, tuple(z_sne))) #return to array instead of tuple
    m_model = M + mu
    chi2 = np.sum(((m_b - m_model) / m_b_err)**2)
    return chi2


def chi2_bao(Omega_m, h, z_bao, obs_q, obs_val, inv_cov):
    model = np.array(cached_bao_model(tuple(z_bao), tuple(obs_q), Omega_m, h))
    d = obs_val - model
    return float(d @ inv_cov @ d)


def chi2_H0prior(h, h_obs, h_sigma):
    return ((h - h_obs) / h_sigma) ** 2

class LogProb:
    """
    Callable class to be used as log_prob function for emcee.
    Avoids closures so it can be pickled and used with multiprocessing.
    """

    def __init__(self, z_sne, m_b, m_b_err,
                 z_bao=None, obs_q=None, obs_val=None, inv_cov=None,
                 use_bao=False, use_h0prior=False,
                 h_obs=None, h_sigma=None, vary_M=False,
                 bounds=((0.1, 1.0), (0.6, 0.8), (-21.5, -17.5))):

        self.z_sne = np.array(z_sne)
        self.m_b = np.array(m_b)
        self.m_b_err = np.array(m_b_err)

        self.z_bao = z_bao
        self.obs_q = obs_q
        self.obs_val = obs_val
        self.inv_cov = inv_cov

        self.use_bao = use_bao
        self.use_h0prior = use_h0prior
        self.h_obs = h_obs
        self.h_sigma = h_sigma
        self.vary_M = vary_M
        self.bounds = bounds

        self.fixed_M = -19.3
        if vary_M:
            self.use_h0prior = True  # force H0 prior if varying M

    def log_prior(self, theta):
        
        if self.vary_M:
            Om, h, M = theta
            (Om_lo, Om_hi), (h_lo, h_hi), (M_lo, M_hi) = self.bounds
            
            if not (Om_lo < Om < Om_hi and h_lo < h < h_hi and M_lo < M < M_hi):
                return -np.inf
            
        else:
            Om, h = theta
            (Om_lo, Om_hi), (h_lo, h_hi), _ = self.bounds
            if not (Om_lo < Om < Om_hi and h_lo < h < h_hi):
                return -np.inf

        # Extra physical guards
        if Om <= 0.048:            # ensures Omega_cdm = Om - 0.048 >= 0
            return -np.inf
        if Om >= 1.0:              # ensures Omega_fld = 1 - Om >= 0
            return -np.inf
        
        return 0.0

    def log_likelihood(self, theta):
        try:
            if self.vary_M:
                Om, h, M = theta
            else:
                Om, h = theta
                M = self.fixed_M

            chi2_tot = chi2_sne(Om, h, M, self.z_sne, self.m_b, self.m_b_err)

            if self.use_bao:
                if any(x is None for x in (self.z_bao, self.obs_q, self.obs_val, self.inv_cov)):
                    return -np.inf
                else:
                    chi2_tot += chi2_bao(Om, h, self.z_bao, self.obs_q, self.obs_val, self.inv_cov)

            if self.use_h0prior:
                if self.h_obs is None or self.h_sigma is None:
                    return -np.inf
                else:
                    chi2_tot += chi2_H0prior(h, self.h_obs, self.h_sigma)

            return -0.5 * chi2_tot

        except CosmoSevereError:
            return -np.inf

    def __call__(self, theta):
        """Make the class callable, so it acts like a function for emcee."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    
#REWRITE USING MONTEPYTHON
#https://baudren.github.io/montepython.html