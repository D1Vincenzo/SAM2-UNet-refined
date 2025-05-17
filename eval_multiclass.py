import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description="Multi-class segmentation evaluation")
    parser.add_argument("--gt_dir", required=True, help="Ground truth mask directory")
    parser.add_argument("--pred_dir", required=True, help="Predicted mask directory")
    parser.add_argument("--target_labels", nargs="+", type=int, required=True,
                        help="List of labels to evaluate, e.g. 0 6 7 8 9")
    return parser.parse_args()

def load_mask(path):
    return np.array(Image.open(path)).astype(np.uint8)

def compute_confusion_matrix(gt_list, pred_list, num_classes, ignore_label=255):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(gt_list, pred_list):
        mask = (gt != ignore_label)
        hist += confusion_matrix(
            gt[mask].flatten(),
            pred[mask].flatten(),
            labels=list(range(num_classes))
        )
    return hist

def compute_metrics(hist):
    acc = np.diag(hist).sum() / (hist.sum() + 1e-8)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0) + 1e-8)

    precision = np.diag(hist) / (hist.sum(0) + 1e-8)
    recall = np.diag(hist) / (hist.sum(1) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Specificity 和 FPR 需要总样本数
    total = hist.sum()
    TP = np.diag(hist)
    FP = hist.sum(0) - TP
    FN = hist.sum(1) - TP
    TN = total - (TP + FP + FN)
    specificity = TN / (TN + FP + 1e-8)
    fpr = FP / (FP + TN + 1e-8)

    return {
        'pixel_acc': acc,
        'mIoU': np.nanmean(iou),
        'mDice': np.nanmean(dice),
        'IoU_per_class': iou,
        'Dice_per_class': dice,
        'Precision_per_class': precision,
        'Recall_per_class': recall,
        'F1_per_class': f1,
        'Specificity_per_class': specificity,
        'FPR_per_class': fpr
    }

def main():
    args = get_args()
    label_map = {raw: idx for idx, raw in enumerate(sorted(args.target_labels))}
    reverse_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith(".png")])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(".png")])
    assert len(gt_files) == len(pred_files), "预测与标签文件数不一致"

    gt_list, pred_list = [], []

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Evaluating"):
        gt = load_mask(os.path.join(args.gt_dir, gt_file))
        pred = load_mask(os.path.join(args.pred_dir, pred_file))
        # print(f"\n🧪 Checking: {gt_file}")
        # print(f"  GT unique labels:   {np.unique(gt)}")
        # print(f"  Pred unique labels: {np.unique(pred)}")


        # 映射 GT → 连续 label
        gt_remapped = np.full_like(gt, 255)
        for raw, mapped in label_map.items():
            gt_remapped[gt == raw] = mapped

        # ✅ 预测图已是原始标签 ID，不需 remap，只要转换为连续 index
        pred_remapped = np.full_like(pred, 255)
        for raw, mapped in label_map.items():
            pred_remapped[pred == raw] = mapped

        gt_list.append(gt_remapped)
        pred_list.append(pred_remapped)

    hist = compute_confusion_matrix(gt_list, pred_list, num_classes)
    results = compute_metrics(hist)

    print("\n🎯 Evaluation Results:")
    print(f"Pixel Accuracy: {results['pixel_acc']:.4f}")
    print(f"Mean IoU:       {results['mIoU']:.4f}")
    print(f"Mean Dice:      {results['mDice']:.4f}\n")

    for i, raw_label in enumerate(label_map.keys()):
        print(f"Class {raw_label:>2} → IoU: {results['IoU_per_class'][i]:.4f}  |  "
              f"Dice: {results['Dice_per_class'][i]:.4f}  |  "
              f"Prec: {results['Precision_per_class'][i]:.4f}  |  "
              f"Rec: {results['Recall_per_class'][i]:.4f}  |  "
              f"F1: {results['F1_per_class'][i]:.4f}  |  "
              f"Spec: {results['Specificity_per_class'][i]:.4f}  |  "
              f"FPR: {results['FPR_per_class'][i]:.4f}")


if __name__ == "__main__":
    main()
