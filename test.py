import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--test_image_path", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--target_labels", nargs="+", type=int, required=True,
                    help="测试时指定参与训练的标签类别（与训练时一致）")
args = parser.parse_args()

# === 模型设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_labels = sorted(set(args.target_labels))
num_classes = len(target_labels)

target_labels = sorted(set(args.target_labels))
num_classes = len(target_labels)
label_mapping = {raw: i for i, raw in enumerate(target_labels)}
reverse_mapping = {v: k for k, v in label_mapping.items()}  # ✅ 用于反映射


model = SAM2UNet(checkpoint_path=args.checkpoint, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

# === 测试集加载 ===
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
os.makedirs(args.save_path, exist_ok=True)

# === 推理 ===
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        image = image.to(device)

        output, _, _ = model(image)  # [1, C, H, W]
        output = F.interpolate(output, size=gt.shape, mode='bilinear', align_corners=False)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]
        # pred: shape [H, W], 值为 0 ~ num_classes-1
        pred_remapped = np.vectorize(reverse_mapping.get)(pred).astype(np.uint8)

        print(f"✅ Saving prediction: {name}")
        imageio.imwrite(os.path.join(args.save_path, name[:-4] + ".png"), pred_remapped)

