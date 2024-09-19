import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager
from time import sleep
from typing import Union, List
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from scipy.ndimage import binary_fill_holes

# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice
from scipy.ndimage import binary_fill_holes

# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox

def crop_to_nonzero_label(data, label):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    assert data[0].shape == label[0].shape, f'{data[0].shape} != {label[0].shape}'
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    label = label[slicer]

    return label

def normalize(image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
    # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
    # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
    # reduced the image size.
    mask = seg >= 0
    mean = image[mask].mean()
    std = image[mask].std()
    image[mask] = (image[mask] - mean) / (max(std, 1e-8))
    image[mask] -=  image[mask].min()
    image[~mask] = 0

    return image

def run_case(image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
    rw = plans_manager.image_reader_writer_class()

    # load image(s)
    data, data_properties = rw.read_images(image_files)
    # if possible, load seg
    if seg_file is not None:
        seg, _ = rw.read_seg(seg_file)
    else:
        seg = None

    data, seg = run_case_npy(data, seg, data_properties, plans_manager, configuration_manager, dataset_json)
    return data, seg, data_properties

def run_case_npy(data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
     # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        print(f'Before normalization: {data.shape}')
        for c in range(data.shape[0]):
            data[c] = normalize(data[c], seg[0])

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)

        # print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
        #         f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg

def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_labels: List[str],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_lists)):
            data, seg, data_properties = preprocessor.run_case(list_of_lists[idx],
                                                list_of_segs_from_prev_stage_files[
                                                    idx] if list_of_segs_from_prev_stage_files is not None else None,
                                                plans_manager,
                                                configuration_manager,
                                                dataset_json)
            if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            rw = plans_manager.image_reader_writer_class()
            label, label_prop = rw.read_seg(list_of_labels[idx])
            label = torch.from_numpy(label).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            # print('Hello', data.shape, label.shape)

            item = {'data': data, 'label':label,'data_properties': data_properties,
                    'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                    list_of_labels: List[str],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_labels[i::num_processes],
                         list_of_segs_from_prev_stage_files[
                         i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        # if pin_memory:
        #     [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]