# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
export nnUNet_raw="/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/"
export nnUNet_preprocessed="/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_results"
export nnUNet_n_proc_DA=32
export nnUNet_def_n_proc=32

# MedNext
export nnUNet_raw_data_base="/mnt/parscratch/users/acs23yz/MedNext/nnUNet_raw_data_base"
export RESULTS_FOLDER="/mnt/parscratch/users/acs23yz/MedNext/nnUNet_trained_models"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
