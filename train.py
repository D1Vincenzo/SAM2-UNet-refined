import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import imageio
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from py_sod_metrics import FmeasureV2, DICEHandler
from peft import MoELinear
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument('--resume', type=str, default=None,
                    help="path to the model checkpoint (.pth) to resume from")
parser.add_argument('--name', type=str, default='moe',
                    help="name of the model, used for logging and saving")
parser.add_argument('--train_roots', nargs='+', required=True,
                    help="List of dataset train roots, e.g., datasets/ClinicDB/train datasets/CVC-ColonDB/train")
parser.add_argument("--lora_rank", type=int, default=8,
                    help="LoRA rank (default: 8)")
parser.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha (default: 32)")
parser.add_argument('--moe_loss_weight', type=float, default=0.01, 
                    help='Weight for MoE auxiliary loss')
parser.add_argument('--conv_lora_expert_num', type=int, default=4, 
                    help='')
parser.add_argument('--conv_lora_topk', type=int, default=1, 
                    help='')

args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def evaluate_dice_with_metric(model, dataloader, device):
    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    FMv2 = FmeasureV2(
        metric_handlers={
            "dice": DICEHandler(**sample_gray),
        }
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            gts = batch['label'].to(device)

            preds = model(images)[0]
            preds = torch.sigmoid(preds).cpu().numpy()
            gts = gts.cpu().numpy()

            for pred, gt in zip(preds, gts):
                pred = (pred[0] * 255).astype(np.uint8)  # H×W, uint8
                gt = (gt[0] * 255).astype(np.uint8)
                FMv2.step(pred=pred, gt=gt)

    results = FMv2.get_results()
    return results["dice"]["dynamic"].mean()


def build_combined_dataset(train_roots, size=352, mode='train'):
    datasets = []
    for train_root in train_roots:
        image_path = os.path.join(train_root, "images")
        mask_path = os.path.join(train_root, "masks")
        dataset = FullDataset(image_path, mask_path, size, mode=mode)
        print(f"[{mode}] 数据集路径: {train_root} - 样本数: {len(dataset)}")
        datasets.append(dataset)
    return ConcatDataset(datasets)


def main(args):    
    train_dataset = build_combined_dataset(args.train_roots, size=352)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(f"训练集样本数: {len(train_dataset)}")

    val_dataset = build_combined_dataset([root.replace("train", "test") for root in args.train_roots], size=352, mode='val') # No validation dataset for now, using test datasets as validation
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"验证集样本数: {len(val_dataset)}")

    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, conv_lora_expert_num=args.conv_lora_expert_num, conv_lora_topk=args.conv_lora_topk)
    model.to(device)
    
    print(f"Training name: {args.name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params/1e6:.2f}M, 可训练参数: {trainable_params/1e6:.2f}M")
    
    optimizer = opt.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},  # 只包含LoRA参数
        {"params": model.rfb1.parameters(), "lr": args.lr},
        {"params": model.rfb2.parameters(), "lr": args.lr},
        {"params": model.rfb3.parameters(), "lr": args.lr},
        {"params": model.rfb4.parameters(), "lr": args.lr},
        {"params": model.up1.parameters(), "lr": args.lr},
        {"params": model.up2.parameters(), "lr": args.lr},
        {"params": model.up3.parameters(), "lr": args.lr},
        {"params": model.up4.parameters(), "lr": args.lr},
        {"params": model.side1.parameters(), "lr": args.lr},
        {"params": model.side2.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_dice = 0.0
    vis_save_path = os.path.join(args.save_path, "val_visuals")
    os.makedirs(vis_save_path, exist_ok=True)

    # Load checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复 MoE 配置
        args.moe_loss_weight = checkpoint.get('moe_loss_weight', 0.01)  # 🔥
        if 'conv_lora_expert_num' in checkpoint:
            args.conv_lora_expert_num = checkpoint['conv_lora_expert_num']
        if 'conv_lora_topk' in checkpoint:
            args.conv_lora_topk = checkpoint['conv_lora_topk']
    
    else:
        print("=> No checkpoint loaded. Training from scratch.")

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'runs'))

    scaler = GradScaler()  # ✅ 添加
    accumulation_steps = 2  # ✅ 添加：根据需要设置累积步数

    for epoch in range(start_epoch, args.epoch):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()  # ✅ 添加：把 zero_grad 移到循环外（配合累积）

        for i, batch in enumerate(train_loader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            # ✅ 添加：AMP 混合精度上下文
            with autocast():
                pred0, pred1, pred2 = model(x)
                loss0 = structure_loss(pred0, target)
                loss1 = structure_loss(pred1, target)
                loss2 = structure_loss(pred2, target)
                main_loss = loss0 + loss1 + loss2

                moe_loss = 0.0
                for module in model.modules():
                    if isinstance(module, MoELinear) and module.current_moe_loss is not None:
                        moe_loss += module.current_moe_loss
                        module.current_moe_loss = None

                total_batch_loss = main_loss + moe_loss * args.moe_loss_weight

            # ✅ 修改：除以累积步数再反向传播
            scaled_loss = total_batch_loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            # ✅ 添加：每隔 accumulation_steps 才执行一次 optimizer.step()
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += total_batch_loss.item()

            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, total_batch_loss.item()))



        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        model.eval()
        val_loss = 0.0
        num_saved = 0
        
        # 重置所有 MoE 损失
        for module in model.modules():
            if isinstance(module, MoELinear):
                module.current_moe_loss = None
        
        with torch.no_grad():
            counter = 0
            for batch in val_loader:
                x = batch['image'].to(device)
                target = batch['label'].to(device)
                pred0, pred1, pred2 = model(x)
                loss = structure_loss(pred0, target) + structure_loss(pred1, target) + structure_loss(pred2, target)
                val_loss += loss.item()
                counter += 1
                
                ### Visulize predictions
                if num_saved < 10:
                    pred = torch.sigmoid(pred0)
                    pred_bin = (pred > 0.5).float()

                    vis_pred = pred_bin[0][0].cpu().numpy() * 255
                    vis_gt = target[0][0].cpu().numpy() * 255

                    imageio.imwrite(os.path.join(vis_save_path, f"epoch{epoch+1}_pred{num_saved+1}.png"), vis_pred.astype(np.uint8))
                    imageio.imwrite(os.path.join(vis_save_path, f"epoch{epoch+1}_gt{num_saved+1}.png"), vis_gt.astype(np.uint8))
                    
                    num_saved += 1
                ###
            
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        scheduler.step()
        
        # DICE 评估
        dice_score = evaluate_dice_with_metric(model, val_loader, device)
        writer.add_scalar('Metric/DICE', dice_score, epoch)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | DICE Score: {dice_score:.4f}")

        # 是否为当前最优模型
        is_best = dice_score > best_val_dice
        if is_best:
            best_val_dice = dice_score
            best_val_loss = avg_val_loss

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'name': args.name,
            # 添加 MoE 相关配置
            'moe_loss_weight': args.moe_loss_weight,  # 🔥 新增
        }

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LR', current_lr, epoch)

        if is_best:
            best_model_path = os.path.join(args.save_path, f'{args.name}-best-model.pth')
            torch.save(checkpoint_dict, best_model_path)
            print(f"=> ✅ Best model (DICE ↑) saved to {best_model_path}")
            
        latest_model_path = os.path.join(args.save_path, f'{args.name}-latest-model.pth')
        torch.save(checkpoint_dict, latest_model_path)
        print(f"[Saving latest model:] {latest_model_path}")


    writer.close()

def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1024)
    main(args)