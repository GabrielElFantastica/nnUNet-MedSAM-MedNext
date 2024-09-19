#!/bin/bash
#SBATCH --job-name=pgd_nnunet_brat
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

# cd nnUNet/

#nnUNetv2_convert_MSD_dataset -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task03_Liver/
#nnUNetv2_convert_MSD_dataset -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task07_Pancreas/

#nnUNetv2_plan_and_preprocess -d 6
#nnUNetv2_plan_and_preprocess -d 7

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 0 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 1 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 2 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 3 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 4 --val --npz

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 1 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 2 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 3 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 4 --val --npz

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_lowres 0
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_lowres 1
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_lowres 2
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_lowres 3
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_lowres 4

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 0
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 1
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 2
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 3
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 4

# 定义模型类型数组
#model_types=("2d" "3d_lowres" "3d_fullres" "3d_cascade_fullres")

# 外层循环遍历模型类型
#for model_type in "${model_types[@]}"; do
    # 内层循环遍历索引
#    for i in {0..4}; do
#        nnUNetv2_train 6 "$model_type" $i --val --npz
#    done
#done

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 0 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 1 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 2 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 3 --val --npz
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 6 3d_cascade_fullres 4 --val --npz

#nnUNetv2_find_best_configuration 6
#nnUNetv2_find_best_configuration 1

#nnUNetv2_predict -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset007_Pancreas/imagesTs/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset007_Pancreas/inferTs -d 7 -c 2d -f 0

#nnUNetv2_predict -d Dataset006_Lung -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/imagesTs/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
#nnUNetv2_apply_postprocessing -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/pp -pp_pkl_file /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset006_Lung/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset006_Lung/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

#nnUNetv2_predict -d Dataset006_Lung -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/imagesTs/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs1 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities
#nnUNetv2_predict -d Dataset006_Lung -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/imagesTs/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs2 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_lowres -p nnUNetPlans --save_probabilities

#nnUNetv2_ensemble -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs1 /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs2 -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs -np 8

# nnUNetv2_apply_postprocessing -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/inferTs -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/pp -pp_pkl_file /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset006_Lung/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__3d_fullres___nnUNetTrainer__nnUNetPlans__3d_lowres___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset006_Lung/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__3d_fullres___nnUNetTrainer__nnUNetPlans__3d_lowres___0_1_2_3_4/plans.json

#nnUNetv2_predict -d Dataset001_BrainTumour -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/imagesTs/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
#nnUNetv2_apply_postprocessing -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/pp -pp_pkl_file /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

# attack
nnUNetv2_predict -d Dataset001_BrainTumour -i /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/imagesTs_pgd/ -o /mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/inferTs_pgd -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
