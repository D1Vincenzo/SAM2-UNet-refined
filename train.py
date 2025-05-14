import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter  # ✅ TensorBoard

from dataset import FullDataset
from SAM2UNet import SAM2UNet


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
# parser.add_argument("--num_classes", default=4, type=int)  # ✅ 添加类别数参数
parser.add_argument("--target_labels", nargs="+", type=int, required=True,
                    help="标签中需要训练的类别 ID 列表，例如: 0 2 5 7")


args = parser.parse_args()
target_labels = sorted(set(args.target_labels))
num_classes = len(target_labels)
label_mapping = {raw_label: i for i, raw_label in enumerate(target_labels)}


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main(args):
    # dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataset = FullDataset(
        args.train_image_path, args.train_mask_path, size=352, mode='train',
        label_mapping=label_mapping, ignore_label=255
    )

    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path, num_classes=num_classes)  # ✅ 传入 num_classes
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = opt.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'runs'))

    # print("✅ Trainable parameters (should be LoRA only):")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(" -", name, param.shape)

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            x = batch['image'].to(device)                 # [N, 3, H, W]
            target = batch['label'].to(device)            # [N, H, W] — 每像素为 int 类别 ID
            optim.zero_grad()

            pred0, pred1, pred2 = model(x)                # 输出应为 [N, C, H, W]
            loss0 = criterion(pred0, target)
            loss1 = criterion(pred1, target)
            loss2 = criterion(pred2, target)
            loss = loss0 + loss1 + loss2

            loss.backward()
            optim.step()
            total_loss += loss.item()

            if i % 50 == 0:
                print(f"epoch:{epoch + 1}-{i + 1}: loss:{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            ckpt_path = os.path.join(args.save_path, f'SAM2-UNet-{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print('[Saving Snapshot:]', ckpt_path)

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
