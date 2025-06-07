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


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
# parser.add_argument("--train_image_path", type=str, required=True, 
#                     help="path to the image that used to train the model")
# parser.add_argument("--train_mask_path", type=str, required=True,
#                     help="path to the mask file for training")
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
parser.add_argument('--n_experts', type=int, default=4)

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


def build_combined_dataset(train_roots, size=352):
    datasets = []
    for train_root in train_roots:
        image_path = os.path.join(train_root, "images")
        mask_path = os.path.join(train_root, "masks")
        datasets.append(FullDataset(image_path, mask_path, size, mode='train'))
    return ConcatDataset(datasets)

def denormalize(tensor, mean, std):
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    return TF.normalize(tensor, inv_mean, inv_std).clamp(0, 1)


def main(args):    
    # 加载训练集
    # train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_dataset = build_combined_dataset(args.train_roots, size=352)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    
    # 加载验证集（目录命名为 val）
    # val_dataset = FullDataset(args.train_image_path.replace("train", "val"), 
    #                           args.train_mask_path.replace("train", "val"),
    #                           352, mode='val')
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dataset = build_combined_dataset([root.replace("train", "val") for root in args.train_roots], size=352)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path, n_experts=args.n_experts)
    model.to(device)
    
    optim = opt.AdamW([{"params": model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    start_epoch = 0
    best_val_loss = float('inf')  # 用于记录最优模型
    best_val_dice = 0.0  # 新增：用于保存最佳 DICE
    vis_save_path = os.path.join(args.save_path, "val_visuals")
    os.makedirs(vis_save_path, exist_ok=True)


    if args.resume is not None and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 是 checkpoint dict
            model.load_state_dict(checkpoint['model_state_dict'])
            optim_state_dict = checkpoint.get('optimizer_state_dict', None)
            sched_state_dict = checkpoint.get('scheduler_state_dict', None)
            if optim_state_dict:
                optim.load_state_dict(optim_state_dict)
            if sched_state_dict:
                scheduler.load_state_dict(sched_state_dict)
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_dice = checkpoint.get('best_val_dice', 0.0)
            print(f"=> Resuming training from epoch {start_epoch}")
        else:
            # 兼容旧模型：只有 model weights
            model.load_state_dict(checkpoint)
            print("=> Loaded model weights only. Training will restart from epoch 0.")
    else:
        print("=> No checkpoint loaded. Training from scratch.")

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'runs'))

    for epoch in range(start_epoch, args.epoch):
        model.train()
        total_loss = 0.0
        total_loss_aux = 0.0
        for i, batch in enumerate(train_loader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            ### 计算辅助损失
            aux_loss_total = 0.0
            for m in model.modules():
                if hasattr(m, 'aux_loss'):
                    aux_loss_total += m.aux_loss
            loss += aux_loss_total
            total_loss_aux += aux_loss_total.item()
            ###
            loss.backward()
            optim.step()
            total_loss += loss.item()
            
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/aux_loss', aux_loss_total.item(), epoch)


        # 验证阶段（不涉及梯度）
        model.eval()

        val_loss = 0.0
        num_saved = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['image'].to(device)
                target = batch['label'].to(device)
                pred0, pred1, pred2 = model(x)
                loss = structure_loss(pred0, target) + structure_loss(pred1, target) + structure_loss(pred2, target)
                val_loss += loss.item()
                
                # 保存预测图、GT、输入图像（最多前10张）
                if num_saved < 10:
                    pred = torch.sigmoid(pred0)
                    pred_bin = (pred > 0.5).float()

                    # 取 batch 中的第 1 张图（0索引）
                    vis_pred = pred_bin[0][0].cpu().numpy() * 255
                    vis_gt = target[0][0].cpu().numpy() * 255

                    imageio.imwrite(os.path.join(vis_save_path, f"epoch{epoch+1}_pred{num_saved+1}.png"), vis_pred.astype(np.uint8))
                    imageio.imwrite(os.path.join(vis_save_path, f"epoch{epoch+1}_gt{num_saved+1}.png"), vis_gt.astype(np.uint8))
                    
                    num_saved += 1
            
            
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        # DICE 评估
        dice_score = evaluate_dice_with_metric(model, val_loader, device)
        writer.add_scalar('Metric/DICE', dice_score, epoch)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | DICE Score: {dice_score:.4f}")

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
        }
        
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LR', current_lr, epoch)

        # 保存最佳模型
        if dice_score > best_val_dice:
            best_val_dice = dice_score
            best_model_path = os.path.join(args.save_path, f'{args.name}-best-model.pth')
            torch.save(checkpoint_dict, best_model_path)
            print(f"=> Best model (DICE ↑) saved to {best_model_path}")
            
        # 保存当前模型
        latest_model_path = os.path.join(args.save_path, f'{args.name}-latest-model.pth')
        torch.save(checkpoint_dict, latest_model_path)
        print(f"[Saving latest model:] {latest_model_path}")

        scheduler.step()

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