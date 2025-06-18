import os
import random
from PIL import Image
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


def split_dataset_fixed(images_dir, masks_dir, output_dir, dataset_name, split_info, seed=1024):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    
    assert len(image_files) == len(mask_files), "图像和掩码数量不匹配！"

    paired = list(zip(image_files, mask_files))
    random.seed(seed)
    random.shuffle(paired)

    test_count = split_info.get("test", 0)
    train_count = split_info.get("train", len(paired) - test_count)

    assert train_count + test_count <= len(paired), f"{dataset_name} 分配的样本总数超过实际数量"

    train = paired[:train_count]
    test = paired[train_count:train_count + test_count]

    splits = {
        "train": train,
        "test": test
    }

    for split, items in splits.items():
        img_out = os.path.join(output_dir, split, "images")
        mask_out = os.path.join(output_dir, split, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for img_name, mask_name in items:
            shutil.copy(os.path.join(images_dir, img_name), os.path.join(img_out, img_name))
            shutil.copy(os.path.join(masks_dir, mask_name), os.path.join(mask_out, mask_name))
        print(f"{dataset_name} - {split}: {len(items)} samples")


def preprocess_dataset(input_image_dir, input_mask_dir, output_root, dataset_name, split_info):
    tmp_image_dir = os.path.join(output_root, "temp_images")
    tmp_mask_dir = os.path.join(output_root, "temp_masks")

    print(f"\n🔧 [{dataset_name}] 正在转换图片为 .png 格式...")
    convert_to_png_and_binarize(input_image_dir, tmp_image_dir, is_mask=False)
    print(f"🔧 [{dataset_name}] 正在二值化并转换掩码为 .png 格式...")
    convert_to_png_and_binarize(input_mask_dir, tmp_mask_dir, is_mask=True)

    print(f"📂 [{dataset_name}] 正在按照固定数量划分数据集...")
    split_dataset_fixed(tmp_image_dir, tmp_mask_dir, output_root, dataset_name, split_info)

    print(f"🧹 [{dataset_name}] 清理临时文件...")
    shutil.rmtree(tmp_image_dir)
    shutil.rmtree(tmp_mask_dir)

    print(f"✅ [{dataset_name}] 数据处理完成！")


# 示例调用
if __name__ == "__main__":
    custom_splits = {
        "Kvasir-SEG":  {"train": 900, "test": 100},
        "ClinicDB":    {"train": 550, "test": 62},
        "CVC-ColonDB": {"test": 380},
        "CVC-300":     {"test": 60},
        "ETIS":        {"test": 196},
    }

    datasets = [
        "datasets/Kvasir-SEG",
        "datasets/ClinicDB",
        "datasets/CVC-ColonDB",
        "datasets/CVC-300",
        "datasets/ETIS"
    ]

    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path)
        split_info = custom_splits.get(dataset_name)
        if split_info is None:
            print(f"⚠️ 未提供 {dataset_name} 的分配信息，跳过")
            continue
        preprocess_dataset(
            input_image_dir=os.path.join(dataset_path, "images"),
            input_mask_dir=os.path.join(dataset_path, "masks"),
            output_root=dataset_path,
            dataset_name=dataset_name,
            split_info=split_info
        )
