import os
import json
import numpy as np
import time
import emcee
import random
from datetime import datetime
from multiprocessing import Pool

from likelihoods import LogProb
from data_ext import load_pantheon, load_desi


# ---------- Data Extraction ----------
z_sne, m_b, m_b_err = load_pantheon("data/Pantheon+SH0ES.dat")
z_bao, obs_q, obs_val, inv_cov = load_desi(
    "data/desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "data/desi_gaussian_bao_ALL_GCcomb_cov.txt"
)

outdir = 'output'
os.makedirs(outdir, exist_ok=True)

RANDOS = np.arange(0.66666667, 30.0, 0.333333334)

perturbation = np.arange(3 , 15, 0.75)
h_calc, h_err = 0.7304, 0.0104  #Reiss' paper


# ---------- Helpers ----------
def prepare_initial_pos(initial, ndim, nwalkers):
    """Generate jittered starting positions around `initial`."""
    random.shuffle(RANDOS)
    
    random.shuffle(perturbation)
    pert = random.choice(perturbation)
    
    scales = np.random.choice(RANDOS, size=nwalkers)
    return initial + (scales[:, None] *  (pert) * 1e-3 * np.random.randn(nwalkers, ndim))


def build_sampler(nwalkers, ndim, log_prob, pool, moves=None):
    """Construct EnsembleSampler with moves & multiprocessing pool."""
    if moves is None:
        moves = [
            (emcee.moves.StretchMove(a=3.8), 0.4),
            (emcee.moves.DEMove(),           0.45),
            (emcee.moves.DESnookerMove(),    0.20),
        ]
    return emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=moves, pool=pool)


def run_with_progress(sampler, pos, nsteps, label, print_every=1000):
    """Run sampler with progress messages."""
    start = datetime.now()
    print(f"\nStarting run: {label} with {nsteps} steps... [{start.strftime('%d-%m-%Y %H:%M:%S')}]", flush=True)

    for i, (pos, lnp, _) in enumerate(sampler.sample(pos, iterations=nsteps, progress=False)):
        if (i + 1) % print_every == 0:
            now = datetime.now()
            print(f"{label} run update: step {i+1}/{nsteps} completed [{now.strftime('%d-%m-%Y %H:%M:%S')}]", flush=True)

    done = datetime.now()
    print(f"Finished run: {label} [{done.strftime('%d-%m-%Y %H:%M:%S')}]", flush=True)
    print(f"Total time for this run: {done - start}")


def save_chain(sampler, label, discard, thin):
    """Save sampler chains, logprobs, acceptance, metadata."""
    prefix = os.path.join(outdir, label)

    chain = sampler.get_chain()
    np.save(prefix + "_chain_raw.npy", chain)
    flat = sampler.get_chain(discard=discard, flat=True, thin=thin)
    np.save(prefix + "_chain_flat.npy", flat)

    try:
        lp = sampler.get_log_prob()
        np.save(prefix + "_logprob.npy", lp)
    except Exception:
        pass

    acc_frac = sampler.acceptance_fraction
    np.save(prefix + "_acceptance.npy", acc_frac)

    meta = {
        "label": label,
        "nsteps": chain.shape[0],
        "nwalkers": chain.shape[1],
        "ndim": chain.shape[2],
        "discard": discard,
        "thin": thin,
        "acceptance_fraction_mean": float(np.mean(acc_frac)),
        "timestamp": time.asctime(),
    }
    
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {label}: raw shape {chain.shape}, flat shape {flat.shape}", flush=True)
    print(f"  Acceptance mean = {np.mean(acc_frac):.3f}, "
          f"min = {np.min(acc_frac):.3f}, max = {np.max(acc_frac):.3f}", flush=True)


def run_experiment(config):
    """Run a single MCMC experiment from config dict."""
    ndim, nwalkers, nsteps, discard, thin = (
        config["ndim"], config["nwalkers"], config["nsteps"], config["discard"], config["thin"]
    )
    pos = prepare_initial_pos(np.array(config["initial"]), ndim, nwalkers)
    log_prob = config["log_prob"]

    with Pool(config.get("ncores", 30)) as pool:
        sampler = build_sampler(nwalkers, ndim, log_prob, pool, moves=config.get("moves"))
        try:
            run_with_progress(sampler, pos, nsteps, config["label"])
        except Exception:
            np.save(os.path.join(outdir, config["label"] + "_chain_partial.npy"), sampler.get_chain())
            raise
        save_chain(sampler, config["label"], discard, thin)

        # quick diagnostics
        acc = sampler.acceptance_fraction
        print(f"Acceptance mean = {acc.mean():.3f}, median = {np.median(acc):.3f}, "
              f"min = {acc.min():.3f}, max = {acc.max():.3f}")
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print("Autocorr time per dim:", tau)
            
            chain = sampler.get_chain(discard=discard, flat=True) 
            N = chain.shape[0]
            # Effective sample size per parameter
            ess = N / tau
            
            print("ESS per parameter:", ess)
            print("Min ESS across parameters:", np.min(ess))
            
        except emcee.autocorr.AutocorrError:
            print("Autocorr time not reliable yet.")


# ---------- Configurations ----------
experiments = [
       {
        "label": "pantheon_only_fixedM",
        
        "ndim": 2, "nwalkers": 30, "nsteps": 35000, "discard": 4000, "thin": 1,
        
       "initial": [0.30, 0.72],
        
        "log_prob": LogProb(z_sne, m_b, m_b_err),
     }
    ,
    
    
    
       {
        "label": "pantheon_plus_bao_fixedM",
        
        "ndim": 2, "nwalkers": 30, "nsteps": 35000, "discard": 4000, "thin": 1,
        
        "initial": [0.30, 0.72],
        
        "log_prob": LogProb(z_sne, m_b, m_b_err,
                            z_bao=z_bao, obs_q=obs_q, obs_val=obs_val, inv_cov=inv_cov,
                            use_bao=True),
      },
    
    
    
     {
        "label": "joint_bao_h0_varyM",
        
        "ndim": 3, "nwalkers": 30, "nsteps": 35000, "discard": 4000, "thin": 1,
        
        "initial": [0.30, 0.72, -19],
        
        "log_prob": LogProb(z_sne, m_b, m_b_err,
                            z_bao=z_bao, obs_q=obs_q, obs_val=obs_val, inv_cov=inv_cov,
                            use_bao=True, use_h0prior=True,
                            h_obs=h_calc, h_sigma=h_err, vary_M=True),
        
     },
    
    
]


# ---------- Entry ----------
if __name__ == "__main__":
    print("I'm working fine... \n", flush=True)
    for experiment in experiments:
        run_experiment(experiment)
