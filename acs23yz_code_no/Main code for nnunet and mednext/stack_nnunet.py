import nibabel as nib
import numpy as np
import os

# 定义你的文件夹路径
folders = [f"/mnt/fastdata/acs23yz/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_BrainTumour/pgd/imagesTs_pgd_{i}" for i in range(3, 6)]

# 遍历每个文件夹
for folder in folders:
    print(f"Processing folder: {folder}")

    # 获取该文件夹中的所有文件名
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.nii.gz')])

    # 提取基础文件名，去掉后缀 '_0000.nii.gz' 等
    base_filenames = set('_'.join(f.split('_')[:-1]) for f in filenames)

    # 遍历每个基础文件名，重新组合成多模态图像
    for base_filename in base_filenames:
        print(base_filename)
        input_filenames = [
            os.path.join(folder, f"{base_filename}_0000.nii.gz"),
            os.path.join(folder, f"{base_filename}_0001.nii.gz"),
            os.path.join(folder, f"{base_filename}_0002.nii.gz"),
            os.path.join(folder, f"{base_filename}_0003.nii.gz")
        ]

        # 加载每个模态
        images = [nib.load(filename) for filename in input_filenames]

        # 获取数据并堆叠成一个四维数组 (4, l, w, h)
        data_arrays = [img.get_fdata() for img in images]
        stacked_data = np.stack(data_arrays, axis=0)

        # 使用第一个模态的affine和header（通常他们是相同的）
        affine = images[0].affine
        header = images[0].header

        # 创建一个新的NIfTI图像
        new_img = nib.Nifti1Image(stacked_data, affine, header)

        # 保存回原来的文件名
        output_dir = os.path.join(folder, "stacked_image")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{base_filename}.nii.gz")
        nib.save(new_img, output_filename)

        print(f"Saved combined image to {output_filename}")
