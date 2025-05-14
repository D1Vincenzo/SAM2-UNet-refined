import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class ToTensorMultiClass:
    def __init__(self, label_mapping, ignore_label):
        self.label_mapping = label_mapping
        self.ignore_label = ignore_label

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.to_tensor(image)
        label_np = np.array(label).astype(np.uint8)

        mapped = np.full_like(label_np, self.ignore_label, dtype=np.uint8)
        for raw, mapped_id in self.label_mapping.items():
            mapped[label_np == raw] = mapped_id

        label_tensor = torch.from_numpy(mapped).long()
        return {'image': image, 'label': label_tensor}



class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.resize(image, self.size)
        label = F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)  # 不插值类别
        return {'image': image, 'label': label}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = F.vflip(image)
            label = F.vflip(label)
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, label_mapping=None, ignore_label=255):
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensorMultiClass(label_mapping, ignore_label),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensorMultiClass(label_mapping, ignore_label),
                Normalize()
            ])
        
        self.label_mapping = label_mapping or {}
        self.ignore_label = ignore_label

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.label_loader(self.gts[idx])
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return img.copy()  # ✅ 复制图像内容，解除文件绑定

    def label_loader(self, path):
        with open(path, 'rb') as f:
            label = Image.open(f)  # 不加 convert()
            return label.copy()    # ✅ 复制图像内容，防止文件关闭后报错



class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()  # 可以用 numpy 处理替代

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.label_loader(self.gts[self.index])
        gt = np.array(gt).astype(np.uint8)

        name = os.path.basename(self.images[self.index])
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB').copy()  # ✅ 复制后解除文件绑定

    def label_loader(self, path):
        with open(path, 'rb') as f:
            label = Image.open(f)
            return label.copy()  # ✅ 保留类别信息，同时解除文件依赖

