import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from typing import Tuple, Union, List, Optional
import numpy as np
from tqdm import tqdm
import itertools

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


import os
import json
from batchgenerators.utilities.file_and_folder_operations import join

from useful import PGD_Attacker

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def initialize_model(checkpoint_path):
    checkpoint = torch.load(join(nnUNet_results, plans, 'fold_0/checkpoint_final.pth'),
                                    map_location=torch.device('cpu'))
    print(checkpoint.keys())
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
        'inference_allowed_mirroring_axes' in checkpoint.keys() else None
    parameters = []
    parameters.append(checkpoint['network_weights'])

def attack_nnUNet(dataset_num_name, epsilon, attack_method='pgd'):
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    
    assert attack_method in ['pgd', 'fgsm', 'bias'], 'Method only accepts fgsm, pgd and bias field'

    plans = f'Dataset00{dataset_num_name}/nnUNetTrainer__nnUNetPlans__3d_fullres'
    
    alpha = 30 / 255

    attacker = PGD_Attacker(
        epsilon=epsilon,
        alpha=alpha,
        attack_iterations=10,
        attack_method=attack_method,
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    attacker.initialize_from_trained_model_folder(
        join(nnUNet_results, plans),
        use_folds='0',
        checkpoint_name='checkpoint_final.pth',
    )

    attacker.attack_from_files(join(nnUNet_raw, f'Dataset00{dataset_num_name}/imagesTs'),
                                join(nnUNet_raw, f'Task0{dataset_num_name}/labelsTs'),
                                 join(nnUNet_raw, f'Dataset00{dataset_num_name}/{attack_method}_new/imagesTs_{attack_method}_{epsilon}'),
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # input_path = '/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset006_Lung/imagesTs/'
    # label_path = '/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task06_Lung/labelsTs/'
    # pgd_path = '/mnt/parscratch/users/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task06_Lung/pgdTs/'

    import argparse
    parser = argparse.ArgumentParser(description='Implementing attack on given medical image dataset and epsilon and alpha')

    parser.add_argument('-n', type=int, required=True,
                        help='index for datasets')
    parser.add_argument('-e', type=int, required=True,
                        help='epsilon to limit the pertubation')
    # parser.add_argument('-a', type=float, required=True,
    #                     help='AKA step_size')
    parser.add_argument('-m', type=str, required=False, default='pgd')

    args = parser.parse_args()

    datasets_dict = {1:'1_BrainTumour', 6:'6_Lung'}
    
    attack_nnUNet(dataset_num_name=datasets_dict[args.n], epsilon=args.e, attack_method=args.m)
