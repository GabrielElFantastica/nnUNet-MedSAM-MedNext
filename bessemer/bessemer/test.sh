#!/bin/bash
#SBATCH --job-name=bias_nn_1  # Replace JOB_NAME with a name you like
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
source activate besser

echo "Hello, world!"

# python /home/acs23yz/code/stack_nnunet.py

# nnUNetv2_convert_MSD_dataset -i /mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour/

# nnUNetv2_plan_and_preprocess -d 1

#nnUNetv2_predict -d 1 -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/imagesTs_pgd_40/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs_pgd_40 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans

# Define the associative array
declare -A datasets_dict
datasets_dict=(["1"]="1_BrainTumour" ["6"]="6_Lung")

# Define the attack method
attack_method='bias'

# Iterate over the dataset indices and epsilon values
# for i in 1 6; do
i=1
for j in {4..10}; do
    echo "Running for dataset ${datasets_dict[$i]} with epsilon $j"
    python /home/acs23yz/code/main.py -n $i -e $j -m $attack_method
    
    # Correctly form the directory path with variable expansion
    image_dir="/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset00${datasets_dict[$i]}/$attack_method/imagesTs_${attack_method}_$j/"
    output_dir="/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset00${datasets_dict[$i]}/$attack_method/inferTs_${attack_method}_$j"

    nnUNetv2_predict -d $i -i $image_dir -o $output_dir -f 0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
    
    eval_dir="/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset00${datasets_dict[$i]}/$attack_method/inferTs_${attack_method}_$j"
    labels_dir="/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task0${datasets_dict[$i]}/labelsTs/"
    nnUNetv2_evaluate_folder -djfile $eval_dir/dataset.json -pfile /mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset00${datasets_dict[$i]}/inferTs/plans.json $labels_dir $eval_dir
done
# done

#nnUNetv2_evaluate_folder  -djfile /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs_pgd/dataset.json -pfile /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs_pgd/plans.json /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour/labelsTs/ /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs_pgd_40
# nnUNetv2_predict -d 6 -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/imagesTs_pgd/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs_pgd -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
