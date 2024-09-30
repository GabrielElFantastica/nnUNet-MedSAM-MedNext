import nibabel as nib
import numpy as np
import os

folders = [f"/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/pgd/imagesTs_pgd_{i}" for i in range(3, 6)]

for folder in folders:
    print(f"Processing folder: {folder}")

    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.nii.gz')])

    base_filenames = set('_'.join(f.split('_')[:-1]) for f in filenames)

    for base_filename in base_filenames:
        print(base_filename)
        input_filenames = [
            os.path.join(folder, f"{base_filename}_0000.nii.gz"),
            os.path.join(folder, f"{base_filename}_0001.nii.gz"),
            os.path.join(folder, f"{base_filename}_0002.nii.gz"),
            os.path.join(folder, f"{base_filename}_0003.nii.gz")
        ]

        images = [nib.load(filename) for filename in input_filenames]

        data_arrays = [img.get_fdata() for img in images]
        stacked_data = np.stack(data_arrays, axis=0)

        affine = images[0].affine
        header = images[0].header

        new_img = nib.Nifti1Image(stacked_data, affine, header)

        output_dir = os.path.join(folder, "stacked_image")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{base_filename}.nii.gz")
        nib.save(new_img, output_filename)

        print(f"Saved combined image to {output_filename}")
