# nnUNet-MedSAM-MedNext

Robust experiements on LFM's ability of medial segmentation

<p align="center">
  <img align="center" src="assets/advchain_logo.png" width="500">
  </a>
</p>

# Adversarial Data Augmentation with Chained Transformations (AdvChain)

This repo contains the pytorch implementation of adversarial data augmentation, which supports to perform adversarial training on a chain of image photometric transformations and geometric transformations for improved consistency regularization.
Please cite our work if you find it useful in your work.

[Full Paper](<https://authors.elsevier.com/sd/article/S1361-8415(22)00230-4>)

## License:

All rights reserved.

## Citation

If you find this useful for your work, please consider citing

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

@INPROCEEDINGS{Chen_MICCAI_2020_Realistic,
  title     = "Realistic Adversarial Data Augmentation for {MR} Image
               Segmentation",
  booktitle = "Medical Image Computing and Computer Assisted Intervention --
               {MICCAI} 2020",
  author    = "Chen, Chen and Qin, Chen and Qiu, Huaqi and Ouyang, Cheng and
               Wang, Shuo and Chen, Liang and Tarroni, Giacomo and Bai, Wenjia
               and Rueckert, Daniel",
  publisher = "Springer International Publishing",
  pages     = "667--677",
  year      =  2020
}

```

## Introduction

AdvChain is a **differentiable** data augmentation library, which supports to augment 2D/3D image tensors with _optimized_ data augmentation parameters. It takes both image information and network's current knowledge into account, and utilizes these information to find effective transformation parameters that are beneficial for the downstream segmentation task. Specifically, the underlying image transformation parameters are optimized so that the dissimilarity/inconsistency between the network's output for clean data and the output for perturbed/augmented data is maximized.

<img align="center" src="assets/advchain.png" width="800">

As shown below, the learned adversarial data augmentation focuses more on deforming/attacking region of interest, generating realistic adversarial examples that the network is sensitive at. In our experiments, we found that augmenting the training data with these adversarial examples are beneficial for enhancing the segmentation network's generalizability.
<img align="center" src="assets/cardiac_example.png" width="750">

## Requirements

- matplotlib>=2.0
- seaborn>=0.10.0
- numpy>=1.13.3
- SimpleITK>=2.1.0
- skimage>=0.0
- torch>=1.9.0

## Set Up

1.  Upgrade pip to the latest:
    ```
    pip install --upgrade pip
    ```
1.  Install PyTorch and other required python libraries with:
    ```
    pip install -r requirements.txt
    ```
1.  Play with the provided jupyter notebook to check the enviroments, see `example/adv_chain_data_generation_cardiac_2D_3D.ipynb` to find example usage.

## Usage

2. Import the library and then add it to your training codebase. Please refer to examples under the `example/` folder for more details.

### Example Code

First set up a set of transformation functions:

```python


```

We can then compose them by putting them in a list with a specified order and initialize a solver to perform random/adversarial data augmentation

```python

```

To perform random data augmentation, simply initialize transformation parameters and call `solver.forward`

```python

```

To perform adversarial data augmentation for adversarial training, a 2D/3D segmentation model `model` is needed.

```python

```
