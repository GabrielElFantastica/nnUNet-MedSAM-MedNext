import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle
import SimpleITK as sitk

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def normalize(image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
    # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
    # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
    # reduced the image size.
    mask = image >= 0
    mean = image[mask].mean()
    std = image[mask].std()
    image[mask] = (image[mask] - mean) / (max(std, 1e-8))
    image[mask] -=  image[mask].min()
    image[~mask] = 0

    return image

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]

    # print(f'current_spacing : {current_spacing}')
    # print(f'predicted_logits: {predicted_logits.shape}')
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    print(f'after configuration: predicted_logits: {predicted_logits.shape}')
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    print(f'predicted_probabilities: {predicted_probabilities.shape}')
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    # print(f'segmentation: {segmentation.shape}')

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    print(f'segmentation_reverted_cropping: {segmentation_reverted_cropping.shape}')
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    print(f'segmentation_reverted_cropping: {segmentation_reverted_cropping.shape}')


    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    
    rw = plans_manager.image_reader_writer_class()
    # print(f'Shape about to save {predicted_array_or_file.shape}')
    if predicted_array_or_file.ndim > 3:
        num_modalities = predicted_array_or_file.shape[0]
        # print(f'The number of modalities: {num_modalities}')
        
        for i in range(num_modalities):
            print(f'Now the modality number is {i}: {predicted_array_or_file[i].shape}')
            
            modality_data = predicted_array_or_file[i]
            
            
            if isinstance(modality_data, torch.Tensor):
                attacked_image = modality_data.cpu().numpy()
            else:
                attacked_image = modality_data

            # rw.write_seg(attacked_image, output_file_truncated + f'_000{i}_b' + dataset_json_dict_or_file['file_ending'],
            #     properties_dict)
            # attacked_image = normalize(attacked_image)
            threshold_value = np.percentile(attacked_image, 5)
            attacked_image -= threshold_value

            print(f'Shape before convert: {attacked_image.shape}, {attacked_image.max()}, {attacked_image.min()}, {np.percentile(attacked_image, 25)}, {np.percentile(attacked_image, 75)}')
            # put segmentation in bbox (revert cropping)
            attacked_final = np.zeros(properties_dict['shape_before_cropping'])
            
            slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
            print(f'slicer is {slicer}')
            attacked_final[slicer] = attacked_image

            print(f'Shape right before writing: {attacked_final.shape}, {attacked_final.max()},{attacked_final.min()}')
            
            # rw.write_seg(attacked_final, output_file_truncated + f'_000{i}' + dataset_json_dict_or_file['file_ending'],
            #              properties_dict)

            itk_image = sitk.GetImageFromArray(attacked_final)
            itk_image.SetSpacing(properties_dict['sitk_stuff']['spacing'])
            itk_image.SetOrigin(properties_dict['sitk_stuff']['origin'])
            itk_image.SetDirection(properties_dict['sitk_stuff']['direction'])
            sitk.WriteImage(itk_image, output_file_truncated + f'_000{i}' + dataset_json_dict_or_file['file_ending'])

            print(f'Finish writing the modality {i}')
        
        del predicted_array_or_file
    else:    
    
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
            return_probabilities=save_probabilities
        )
        del predicted_array_or_file
        
        segmentation_final = ret
        rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)

    # # save
    # if save_probabilities:
    #     segmentation_final, probabilities_final = ret
    #     np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
    #     save_pickle(properties_dict, output_file_truncated + '.pkl')
    #     del probabilities_final, ret
    # else:
    #     segmentation_final = ret
    #     del ret


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)