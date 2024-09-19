#!/bin/bash
#SBATCH --job-name=nn-Unet  # Replace JOB_NAME with a name you like
#SBATCH --time=96:00:00  # Change this to a longer time if you need more time
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=40G  # Request 40 gigabytes of real memory (mem)
#SBATCH --output=./Output/%x-%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yzhang797@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified

module load Java/17.0.4
module load Anaconda3/2022.05

source activate nn-unet

# cd nnUNet/
#cd /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/

# nnUNetv2_evaluate_folder Dataset006_Lung/pp/ Task06_Lung/labelsTs -djfile Dataset006_Lung/inferTs/dataset.json -pfile Dataset006_Lung/inferTs/plans.json

# nnUNetv2_evaluate_folder Dataset001_BrainTumour/pp/ Task01_BrainTumour/labelsTs -djfile Dataset001_BrainTumour/inferTs/dataset.json -pfile Dataset001_BrainTumour/inferTs/plans.json
