import os
import random
from PIL import Image
from pathlib import Path
import shutil


def binarize_mask(mask: Image.Image, threshold=128) -> Image.Image:
    """将 mask 二值化为 0/255"""
    return mask.convert("L").point(lambda p: 255 if p >= threshold else 0)


def convert_to_png_and_binarize(input_dir, output_dir, is_mask=False):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            continue
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB" if not is_mask else "L")
        if is_mask:
            img = binarize_mask(img)
        name = os.path.splitext(fname)[0] + ".png"
        img.save(os.path.join(output_dir, name))


def split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=1024):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    
    assert len(image_files) == len(mask_files), "图像和掩码数量不匹配！"

    paired = list(zip(image_files, mask_files))
    random.seed(seed)
    random.shuffle(paired)

    total = len(paired)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": paired[:train_end],
        "val": paired[train_end:val_end],
        "test": paired[val_end:]
    }

    for split, items in splits.items():
        img_out = os.path.join(output_dir, split, "images")
        mask_out = os.path.join(output_dir, split, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for img_name, mask_name in items:
            shutil.copy(os.path.join(images_dir, img_name), os.path.join(img_out, img_name))
            shutil.copy(os.path.join(masks_dir, mask_name), os.path.join(mask_out, mask_name))
        print(f"{split}: {len(items)} samples")


def preprocess_dataset(input_image_dir, input_mask_dir, output_root):
    tmp_image_dir = os.path.join(output_root, "temp_images")
    tmp_mask_dir = os.path.join(output_root, "temp_masks")

    print("🔧 正在转换图片为 .png 格式...")
    convert_to_png_and_binarize(input_image_dir, tmp_image_dir, is_mask=False)
    print("🔧 正在二值化并转换掩码为 .png 格式...")
    convert_to_png_and_binarize(input_mask_dir, tmp_mask_dir, is_mask=True)

    print("📂 正在划分数据集（train / val / test）...")
    split_dataset(tmp_image_dir, tmp_mask_dir, output_root)

    print("🧹 清理临时文件...")
    shutil.rmtree(tmp_image_dir)
    shutil.rmtree(tmp_mask_dir)

    print("✅ 数据处理完成！")


# 示例调用
if __name__ == "__main__":

    datasets = ["datasets/ClinicDB", "datasets/CVC-ColonDB", "datasets/ETIS", "datasets/Kvasir-SEG"]
    output_dir = datasets
    for dataset in datasets:
        preprocess_dataset(os.path.join(dataset, "images"), os.path.join(dataset, "masks"), dataset)
