# Cosmology MCMC Project

This repository contains the final version of a Markov Chain Monte Carlo (MCMC) code for cosmological parameter estimation.  
It originally started during the **5th Summer School of Physics – Cosmology Project Team** at the **BUE's Centre for Theoretical Physics (CTP)**, held at the British University in Egypt (Cairo) from July 7–17.

Under the supervision of **Dr. Mahmoud Hashim** and **Dr. Amr El-Zant**, I first built this MCMC framework from scratch to explore how **Type Ia supernovae (Pantheon+SH0ES)**, **BAO measurements (DESI)**, and **a local $H_0$ prior** can be combined to constrain cosmological parameters. I started the development in a Google Colab notebook (with detailed notes and references) before getting access to the CTP HPC:

- Colab notebook: [Google Colab Link](https://colab.research.google.com/drive/1mOKGvX3yLxajO0lVqBanl8AhrJMOP1wO#scrollTo=jBEBCltSxoH3)  
- Full project LaTeX overview with goals, references, notes and simple explanations of the cosmology (pre-HPC access, Highly recommended read for good understanding of what everything does and why): [PDF Overview](https://drive.google.com/file/d/1vgavDMfz3wDiCgNKeRXAqFBAWl22GVbJ/view?usp=sharing)

After gaining HPC access, I reorganized the Colab code into proper Python modules (what you see here), with the physics remaining (mostly) unchanged for cleaner structure and more robust runs that use HPC resources. Later, I transitioned the workflow to the **MontePython** ecosystem (which already provides standard likelihoods and data interfaces). Thus, this repo preserves my **standalone** implementation.

**Main science target:** set up and explore cosmological parameter inference with **Type Ia Supernovae (Pantheon+SH0ES dataset)** and **BAO (DESI 2025 data release)**, optionally with an **$H_0$ prior**.

---

## Project Overview

- `data/` — data files used for likelihood analysis  
  - Pantheon+SH0ES: [Dataset](https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat)  
  - DESI BAO (2025 release): [Dataset](https://github.com/LauraHerold/MontePython_desilike/tree/main/data/DESI_2025)  

- `models.py` — interfaces with [CLASS](http://class-code.net/) (via `classy`) to compute cosmological observables:  
  - Luminosity distance for SNe (used as Distance Modulus)  
  - BAO distance combinations ($ D_M/r_s, \, \, D_H/r_s, \, \, D_V/r_s $)  

- `data_ext.py` — utilities to load the Pantheon+SH0ES SNe and DESI BAO datasets  

- `likelihoods.py` — chi-square likelihoods for SNe, BAO, and an $H_0$ prior; includes a `LogProb` class callable for `emcee`.  

- `MCMC_script.py` — main driver defining experiments and running samplers, saving chains + metadata under `output/`  

- `analysis.ipynb` — quick analysis notebook that reads chains from `output/` and saves figures in `plots/`  

- `job_batch.sh` — SLURM batch script for HPC runs (set cores, walltime, memory, etc)  

The sampler is [`emcee`](https://emcee.readthedocs.io/en/stable/), using a mix of proposal moves (Stretch, DE, Snooker). Output includes:

- raw and flattened chains (`*_chain_raw.npy`, `*_chain_flat.npy`)  
- log-probabilities  
- acceptance fractions  
- JSON metadata with run details (walkers, steps, etc).

---

## Models & Runs (`MCMC_script.py`)

Three runs, defined in `experiments` dict are pre-wired:

### 1. `pantheon_only_fixedM`
- **Parameters:** ($\Omega_m$, h)  
- **Likelihoods:** Pantheon+SH0ES only  
- **M:** fixed to a fiducial value (set to -19.3 by default in `LogProb` in `likelihoods.py`) 

### 2. `pantheon_plus_bao_fixedM`
- **Parameters:** ($\Omega_m$, h)  
- **Likelihoods:** Pantheon+SH0ES + DESI BAO  
- **M:** fixed  

### 3. `joint_bao_h0_varyM`
- **Parameters:** ($\Omega_m$, h, M)  
- **Likelihoods:** Pantheon+SH0ES + DESI BAO + Gaussian $H_0$ prior  
- **Notes:** M varies, $H_0$ prior is applied to break the SNe absolute magnitude–$H_0$ degeneracy

---

## CLASS Setup (`models.py`)

- Flat $\omega$CDM model with  
  - `w0_fld = -1`  
  - `wa_fld = 0`  
  - fixed $\Omega_b = 0.048$ 
  - $\Omega_{cdm} = \Omega_m − \Omega_b$  
  - dark energy: $\Omega_{fld} = 1 − \Omega_m$ (flatness assumption)  

- Parameter space kept relatively small: just ($\Omega_m$, h) or ($\Omega_m$, h, M), still physically realistic. 
- Reduced Hubble constant **h** can be turned into $H_0$ by: $H_0$ = h × 100. 

---

## Usage

### Quick Start (Local)
If running on a laptop/desktop, reduce runtime:

- decrease `@lru_cache(maxsize=12000)` in `likelihoods.py` (e.g., 3k–5k)  
- lower `nsteps` (e.g., 2k–5k steps)  
- lower `nwalkers` (e.g., 12–20)  
- reduce burn-in steps proportionally  
- set `ncores` safely for your machine (e.g., 4–8)  

All parameters are in the `experiments` dict at the end of `MCMC_script.py`.  

Run:
```bash
python3 MCMC_script.py
```
This will run the experiments defined in `experiments` in **`MCMC_script.py`** and drop outputs into `output/`.

### HPC (SLURM)

This project is primarily geared for HPC. If you have access to one, just adjust `job_batch.sh` to your cluster (nodes/queue/time/mem/env), then run:

```bash
sbatch job_batch.sh
```

The `.sh` script provided requests 32 CPUs and runs:
```bash
python3 -u MCMC_script.py
```


## Requirements

### System

- Linux or macOS recommended (tested on HPC Linux, will probably work on venvs but I've found problems installing CLASS and Classy on VScode for example)
- **CLASS** compiled for the same Python you'll use here with the **classy** Python wrapper installed

### Python (runtime)
- Python 3.8+
- `numpy-`, `pandas`, `emcee`, `classy`
  
### Python (analysis, optional)
- `matplotlib`, `corner` , `seaborn` (optional) , `jupyter`/ `notebook`/ `jupyterlab`, `getdist`

#### Installation
For local machines, you can create a clean conda or venv environment to run the code and install the requirements, for HPC I'd recommend a regular python venv to be created if you don't want to run on the cluster directly.

## Data
- The data used is discussed in detail with the references of the data measurements and observations included are in the [overview report](https://drive.google.com/file/d/1vgavDMfz3wDiCgNKeRXAqFBAWl22GVbJ/view?usp=sharing)

## Outputs
The driver saves, per "experiment"

- `*_chain_raw.npy` (shape: nsteps × nwalkers × ndim)

- `*_chain_flat.npy` (post-burn-in flattened, shape: ( (nsteps x nwalkers) - (discard x nwalkers) ) x ndim)

- `*_logprob.npy`

- `*_acceptance.npy`

-  `*_meta.json` (small run metadata)

Figures from `analysis.ipynb` go into `plots/`.


Please don't hesitate to contact me for questions, corrections or suggestions.



<p align="center">
  <img src="https://github.com/user-attachments/assets/d88493d9-97cf-4a86-b1f8-055e5a8a34c8" width="300" style="background:white;padding:10px;"/>
  <img src="https://github.com/user-attachments/assets/181f0a4a-6a1d-43b1-ae5c-68df0e983f10" width="300" style="background:white;padding:10px;"/>
</p>

