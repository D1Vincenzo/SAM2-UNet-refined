import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import FullDataset
from SAM2UNet import SAM2UNet
from py_sod_metrics import FmeasureV2, DICEHandler
from PIL import Image
import imageio


def build_combined_dataset(train_roots, size=352, mode='train'):
    datasets = []
    for train_root in train_roots:
        image_path = os.path.join(train_root, "images")
        mask_path = os.path.join(train_root, "masks")
        dataset = FullDataset(image_path, mask_path, size, mode=mode)
        print(f"[{mode}] 数据集路径: {train_root} - 样本数: {len(dataset)}")
        datasets.append(dataset)
    return ConcatDataset(datasets)


def evaluate_dice_with_metric(model, dataloader, device, save_path=None):
    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    FMv2 = FmeasureV2(metric_handlers={"dice": DICEHandler(**sample_gray)})

    model.eval()
    os.makedirs(save_path, exist_ok=True) if save_path else None

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            gts = batch['label'].to(device)

            preds = model(images)[0]
            preds = torch.sigmoid(preds).cpu().numpy()
            gts = gts.cpu().numpy()

            for i, (pred, gt) in enumerate(zip(preds, gts)):
                pred = (pred[0] * 255).astype(np.uint8)
                gt = (gt[0] * 255).astype(np.uint8)
                FMv2.step(pred=pred, gt=gt)

    results = FMv2.get_results()
    return results["dice"]["dynamic"].mean()


def main(args):
    val_dataset = build_combined_dataset([root.replace("train", "test") for root in args.test_image_path], size=352, mode='val') # No validation dataset for now, using test datasets as validation
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"验证集样本数: {len(val_dataset)}")
    
    device = torch.device("cuda")
    # model = SAM2UNet(args.hiera_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # Load checkpoint
    print(f"=> Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 读取 LoRA 参数（优先使用 ckpt 中的值）
    lora_rank = checkpoint.get('lora_rank', args.lora_rank)
    lora_alpha = checkpoint.get('lora_alpha', args.lora_alpha)
    name = checkpoint.get('name', 'SAM2UNet')

    # 构建模型
    model = SAM2UNet(args.hiera_path, lora_rank=lora_rank, lora_alpha=lora_alpha)

    model.to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_val_dice = checkpoint.get('best_val_dice', 0.0)
    print(f"=> Loading epoch {epoch} with best loss {best_val_loss} and best DICE {best_val_dice}")

    num_saved = 0
    
    os.makedirs(args.save_path, exist_ok=True)
    with torch.no_grad():
    #     for batch in val_loader:
    #         x = batch['image'].to(device)
    #         target = batch['label'].to(device)
    #         pred0, pred1, pred2 = model(x)
            
    #         ### Visulize predictions
    #         if num_saved < 10:
    #             pred = torch.sigmoid(pred0)
    #             pred_bin = (pred > 0.5).float()

    #             vis_pred = pred_bin[0][0].cpu().numpy() * 255
    #             vis_gt = target[0][0].cpu().numpy() * 255

    #             imageio.imwrite(os.path.join(args.save_path, f"pred{num_saved+1}.png"), vis_pred.astype(np.uint8))
    #             imageio.imwrite(os.path.join(args.save_path, f"gt{num_saved+1}.png"), vis_gt.astype(np.uint8))
                
    #             num_saved += 1
    #         ###

        # DICE 评估
        dice_score = evaluate_dice_with_metric(model, val_loader, device)
        print(f"model: {name} | DICE Score: {dice_score:.4f}")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2UNet Evaluation")
    parser.add_argument('--hiera_path', type=str, required=True, help="Path to pretrained hiera model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument('--test_image_path', nargs='+', required=True, help="Validation dataset root(s)")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save results")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    args = parser.parse_args()

    main(args)
