#!/bin/bash
#SBATCH --job-name=MedNext-train_6_3d_5
#SBATCH --time=96:00:00
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

source activate nn-unet

export PATH=$HOME/.local/bin:$PATH

echo "Current environment variable nnUNet_n_proc_DA: $nnUNet_n_proc_DA"

which mednextv1_train

# mednextv1_plan_and_preprocess -t 6

echo "Training!!!"
# CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task001_BrainTumour 3 -p nnUNetPlansv2.1_trgSp_1x1x1
# CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task001_BrainTumour 4 -p nnUNetPlansv2.1_trgSp_1x1x1

# mednextv1_predict -i /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/imagesTs/ -o /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/inferTs -tr nnUNetTrainerV2_MedNeXt_S_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -f 0 1 2 3 4 -t 6 --save_npz -m 3d_fullres
# mednextv1_predict -i /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/imagesTs/ -o /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/inferTs -tr nnUNetTrainerV2_MedNeXt_S_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -f 0 1 2 3 4 -t 1 --save_npz -m 3d_fullres


#for i in {0..4}; do
#    CUDA_VISIBLE_DEVICES=0 mednextv1_train 2d nnUNetTrainerV2_MedNeXt_S_kernel3 Task006_Lung $i
#    CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task001_BrainTumour $i -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights /mnt/parscratch/users/acs23yz/MedNext/nnUNet_trained_models/nnUNet/3d_fullres/Task001_BrainTumour/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_$i/model_final_checkpoint.model -resample_weights
#    CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task001_BrainTumour $i --validation_only --npz
#    done


CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task001_BrainTumour 1 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights /mnt/parscratch/users/acs23yz/MedNext/nnUNet_trained_models/nnUNet/3d_fullres/Task001_BrainTumour/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_1/model_final_checkpoint.model -resample_weights

echo "Predicting!!!"
mednextv1_predict -i /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/imagesTs/ -o /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/inferTs -tr nnUNetTrainerV2_MedNeXt_S_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -f 1 -t 6 --save_npz -m 3d_fullres
mednextv1_predict -i /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/imagesTs/ -o /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/inferTs -tr nnUNetTrainerV2_MedNeXt_S_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -f 1 -t 1 --save_npz -m 3d_fullres

echo "Evaluating!!!"
mednextv1_evaluate_folder -pred /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/inferTs -ref /mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/labelsTs -l 1
   
