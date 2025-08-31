#!/bin/bash
#
#SBATCH --job-name=mcmc_job
#SBATCH --output=log.txt
#SBATCH --nodelist=nut03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2500



## cmd to run
python3 -u MCMC_script.py
