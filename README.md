# nnUNet-MedSAM-MedNext

Robust experiements on LFM's ability of medial segmentation

<p align="center">
  <img align="center" src="assets/advchain_logo.png" width="500">
  </a>
</p>

# Adversarial Data Augmentation with Chained Transformations (AdvChain)

This repo is based on the code of 3 large foundation models.
[nnUNet](https://github.com/MIC-DKFZ/nnUNet)
[MedSAM](https://github.com/bowang-lab/MedSAM)
[MedNext](https://github.com/MIC-DKFZ/MedNeXt)
while the code of implementing adversarial attack is based on the work of these 2 papers.
[PGD-MNIST](https://github.com/MadryLab/mnist_challenge)
[advchain](https://github.com/cherise215/advchain)

## Set up

Mainly used two environments

- According to the installation instruction of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) set the environment nn-unet
- According to the [MedSAM](https://github.com/bowang-lab/MedSAM) set the environment medsam
- Install the work of [MedNext](https://github.com/MIC-DKFZ/MedNeXt) under the environment nn-unet. Change the default variables if necessary.

## Training Models and collecting raw results

For models nnUnet and MedNext, use prewritten code from nnunet and MedNext through shell files in HPC/. See nnunet.sh and MedNext.sh for more details

For model MedSAM, the code used is written in python files and run by HPC. See medsam.sh and train_one_gpu.py for more details.

## White Box Attack

Use attack.py to perform white attack on nnUNet and MedNext.

Use attack.py to perform white attack on MedSAM.

## Black Box Attack

Use attack_black.py to perform white attack on nnUNet and MedNext.

Use attack_black.py to perform white attack on MedSAM.

## Example





## Example Code

First let's see a sample slice preprocessed by nnUNet:

<p align="center">
  <img align="center" src="assets/advchain_logo.png" width="500">
  </a>
</p>

Assuming that you already have the environment and the checkpoint files.

Define the function of fgsm and pgd attack. For bias attack, use code from [advchain](https://github.com/cherise215/advchain)
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from advchain.augmentor import AdvBias  # AdvBias class from advchain

# PGD Attack: A multi-step adversarial attack that iteratively applies perturbations to maximize the impact while staying within a specified epsilon bound.
def pgd_attack(model, images, labels, epsilon=0.1, alpha=2/255, iterations=10, device='cuda'):
    images = images.to(device)
    labels = torch.squeeze(labels, dim=1).to(device)

    loss_fn = nn.CrossEntropyLoss()
    original_images = images.data.clone()

    for i in range(iterations):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        loss = loss_fn(outputs, labels).to(device)
        loss.backward()

        # Apply perturbation and clamp within epsilon bounds
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images

# FGSM Attack: A single-step adversarial attack that applies the sign of the gradient of the loss with respect to the input to create perturbations.
def fgsm_attack(model, images, labels, epsilon=0.03, device='cuda'):
    images = images.to(device)
    labels = torch.squeeze(labels, dim=1).to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Enable gradient calculation for images
    images.requires_grad = True
    outputs = model(images)

    model.zero_grad()
    loss = loss_fn(outputs, labels).to(device)
    loss.backward()

    # Create adversarial images by applying the sign of the gradient
    pertubation = epsilon * images.grad.sign()
    adv_images = images + pertubation
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

# Bias Field Attack: Introduces spatial variations in intensity (bias) across the image, simulating real-world artifacts in medical images.
def bias_field_attack(model, images, labels, epsilon=0.1, control_point_spacing=[64, 64, 64], device='cuda'):
    images = images.to(device)
    labels = torch.squeeze(labels, dim=1).to(device)

    # Create an instance of the bias field attack (assuming AdvBias is a defined class)
    augmentor_bias = AdvBias(
        spatial_dims=3,
        config_dict={
            'epsilon': epsilon,
            'control_point_spacing': control_point_spacing,
            'downscale': 2,
            'data_size': images.shape,
            'interpolation_order': 3,
            'init_mode': 'random',
            'space': 'log'
        },
        debug=False  # Set to True for intermediate outputs
    )

    # Apply bias field perturbation
    adv_images = augmentor_bias.forward(images).detach()

    return adv_images
```

Calculate the dice score and success rate

```python
def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = np.logical_not(ignore_mask)  # Use logical_not for NumPy arrays
    
    tp = np.sum(np.logical_and(mask_ref, mask_pred) & use_mask)  # Use logical_and for TP
    fp = np.sum(np.logical_and(np.logical_not(mask_ref), mask_pred) & use_mask)  # Logical for FP
    fn = np.sum(np.logical_and(mask_ref, np.logical_not(mask_pred)) & use_mask)  # Logical for FN
    tn = np.sum(np.logical_and(np.logical_not(mask_ref), np.logical_not(mask_pred)) & use_mask)  # Logical for TN
    
    return tp, fp, fn, tn

def compute_metrics(segmentation: np.ndarray, label: np.ndarray):
    batch_size = segmentation.shape[0]
    dice_list = []
    success_list = []
    
    # Convert label tensor to NumPy array if it's a tensor
    mask_ref = label.cpu().numpy() if hasattr(label, 'cpu') else label
    mask_pred = segmentation

    for i in range(batch_size):
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref[i], mask_pred[i])
        
        if tp + fp + fn == 0:
            dice = np.nan
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
        
        dice_list.append(dice)
        success_list.append(1 if dice >= 0.5 else 0)

    return dice_list, success_list
```

Store the attacked images with epsilon. e.g. The image attacked by pgd at epsilon 0.3 shall be named image_pgd_3

<p align="center">
  <img align="center" src="assets/advchain_logo.png" width="500">
  </a>
</p>

Now implement the attacked image on another model to collect the result for black attack.

```python
# Initialize a new model.

# Predict with attacked image
segmentation = model(adv_image)

# Calculate metrics on new predictions
dice, success = compute_metrics(segmentation, label)
```

## Citation

For more

```
@ARTICLE{Chen_2021_Enhancing,
  title  = "Enhancing {MR} Image Segmentation with Realistic Adversarial Data Augmentation",
  journal = {Medical Image Analysis},
  author = "Chen, Chen and Qin, Chen and Ouyang, Cheng and Wang, Shuo and Qiu,
            Huaqi and Chen, Liang and Tarroni, Giacomo and Bai, Wenjia and
            Rueckert, Daniel",
    year = 2022,
    note = {\url{https://authors.elsevier.com/sd/article/S1361-8415(22)00230-4}}
}

```