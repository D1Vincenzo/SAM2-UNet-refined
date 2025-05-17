import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def nii_to_png_split(img_dir, label_dir, output_base_dir, test_ratio=0.1):
    os.makedirs(output_base_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    assert len(img_files) == len(label_files), "图像和标签数量不一致"

    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        img_files, label_files, test_size=test_ratio, random_state=42
    )

    def process_subset(img_list, label_list, subset_name):
        img_output_dir = os.path.join(output_base_dir, subset_name, "images")
        mask_output_dir = os.path.join(output_base_dir, subset_name, "masks")
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)

        idx = 0
        for img_file, label_file in tqdm(zip(img_list, label_list), total=len(img_list), desc=f"Processing {subset_name}"):
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, label_file)

            img_nii = nib.load(img_path).get_fdata()
            label_nii = nib.load(label_path).get_fdata()
            assert img_nii.shape == label_nii.shape, f"维度不一致: {img_file}"

            for i in range(img_nii.shape[2]):
                img_slice = img_nii[:, :, i]
                label_slice = label_nii[:, :, i]

                # Normalize 图像（保留灰度值）
                img_norm = ((img_slice - np.min(img_slice)) / (np.ptp(img_slice) + 1e-8) * 255).astype(np.uint8)

                # 保留标签的原始类别（如 0,1,2,3...），最多支持 255 类
                label_class = label_slice.astype(np.uint8)

                img_pil = Image.fromarray(img_norm).convert('L')  # 灰度图
                label_pil = Image.fromarray(label_class, mode='L')  # 单通道、像素值表示类别

                img_pil.save(os.path.join(img_output_dir, f"{subset_name}_img_{idx:05d}.png"))
                label_pil.save(os.path.join(mask_output_dir, f"{subset_name}_mask_{idx:05d}.png"))

                idx += 1

    process_subset(train_imgs, train_labels, "train")
    process_subset(test_imgs, test_labels, "test")


if __name__ == "__main__":
    img_dir = "/home/hxy/Documents/RawData/Training/img"
    label_dir = "/home/hxy/Documents/RawData/Training/label"
    output_base_dir = "./processed_dataset_labels"
    nii_to_png_split(img_dir, label_dir, output_base_dir, test_ratio=0.1)
