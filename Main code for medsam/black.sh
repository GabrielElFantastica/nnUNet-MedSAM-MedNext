#!/bin/bash
#SBATCH --job-name=black_nn_medsam_1  # Replace JOB_NAME with a name you like
#SBATCH --time=72:00:00  # Change this to a longer time if you need more time
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/home/acs23yz/Output/%x-%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yzhang797@sheffield.ac.uk

module load Java/11.0.2
module load Anaconda3/2019.07
source activate medsam

cd /mnt/fastdata/acs23yz/MedSAM/

echo "Hello, world!"

# python /home/acs23yz/code/preprocess_black.py -i $i

for i in {1..3}; do
    # python /home/acs23yz/code/preprocess_black.py -i $i
    python /home/acs23yz/code/predict_black_medsam.py -i $i
done