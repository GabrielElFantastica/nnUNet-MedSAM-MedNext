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

# python MedSAM_Inference.py -task_name Task01_BrainTumour -batch_size 8 -num_workers 8
# python m_train.py -i /mnt/parscratch/users/acs23yz/MedSAM/data/npy/all_Brain/ -task_name Task01_BrainTumour -batch_size 8 -num_workers 8

# python MedSAM_Inference.py
# -i input_img -o output path

# echo "Hello, MedSAM"
# source activate nn-unet

# python /mnt/parscratch/users/acs23yz/MedSAM/m_predict.py

# python attack.py

# python predict_evaluate.py

python bias_field.py
