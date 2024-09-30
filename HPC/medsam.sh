#!/bin/bash
#SBATCH --job-name=MedSAM-bias-predict  # Replace JOB_NAME with a name you like
#SBATCH --time=96:00:00  # Change this to a longer time if you need more time
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --output=./Output/%x-%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yzhang797@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2022.05

source activate medsam

echo "Hello, MedSAM"

cd /mnt/parscratch/users/acs23yz/MedSAM/

python preprocess.py

# python /mnt/parscratch/users/acs23yz/MedSAM/m_predict.py

# python attack.py

# python predict_evaluate.py

python bias_field.py
