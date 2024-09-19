#!/bin/bash
#SBATCH --job-name=cpu-use
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --output=./Output/%x-%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yzhang797@sheffield.ac.uk

module load Java/11.0.2
module load Anaconda3/2019.07
source activate medsam

# cd /mnt/fastdata/acs23yz/MedSAM/

python /home/acs23yz/code/stack_nnunet.py
# python preprocess1.py
# python preprocess2.py
# python preprocess3.py
# export PATH=$HOME/.local/bin:$PATH

# mednextv1_plan_and_preprocess -t 6 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

# spark-submit --driver-memory 20g --executor-memory 10g ./Code/Q4_code.py