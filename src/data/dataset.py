import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    """
    Custom dataset for abnormal event detection with support for YOLO features and advanced augmentations.
    Includes MixUp and CutMix for improved generalization.
    """
    def __init__(self, processed_dir: str, yolo_dir: str = None, train: bool = True, use_mixup: bool = False, use_cutmix: bool = False):
        self.files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.h5')]
        self.yolo_files = [os.path.join(yolo_dir, f) for f in os.listdir(yolo_dir) if f.endswith('.h5')] if yolo_dir else None
        self.train = train
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        if train:
            self.strong_transform = A.Compose([
                A.Resize(224, 224, always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
                A.GaussianBlur(blur_limit=(3, 9), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-30, 30), shear=(-20, 20), p=0.7),
                A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.4),
                A.RandomRain(blur_value=5, p=0.4),
                A.RandomShadow(p=0.4),
                A.GaussNoise(var_limit=(20, 60), p=0.4),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.OpticalDistortion(p=0.3),
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),
                ToTensorV2()
            ])
            self.light_transform = A.Compose([
                A.Resize(224, 224, always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HorizontalFlip(p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224, always_apply=True),
                ToTensorV2()
            ])

        self.labels = []
        self.lengths = []
        for f in self.files:
            with h5py.File(f, 'r') as hf:
                self.lengths.append(hf['sequences'].shape[0])
                self.labels.extend(hf['labels'][:].tolist())
        self.cumulative_lengths = []
        cum_sum = 0
        for l in self.lengths:
            cum_sum += l
            self.cumulative_lengths.append(cum_sum)

    def mixup(self, x1, y1, x2, y2, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def cutmix(self, x1, y1, x2, y2, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x1.shape, lam)
        x = x1.clone()
        x[..., bby1:bby2, bbx1:bbx2] = x2[..., bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.shape[-1] * x1.shape[-2]))
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def rand_bbox(self, size, lam):
        W = size[-1]
        H = size[-2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx):
        """
        Returns:
            seq_tensor: [seq_len, 3, H, W] (torch.Tensor)
            binary_label: 0 for Normal, 1 for Abnormal (torch.LongTensor)
            multiclass_label: 0-13 (torch.LongTensor)
            yolo_features: [seq_len, 80] (torch.Tensor)
        """
        file_idx = 0
        while idx >= self.cumulative_lengths[file_idx]:
            file_idx += 1
        offset = idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)
        with h5py.File(self.files[file_idx], 'r') as hf:
            seq = hf['sequences'][offset]
        seq = np.transpose(seq, (0, 3, 1, 2))
        seq_len = 16
        # Temporal augmentations
        if self.train and np.random.rand() < 0.5:
            drop_num = np.random.randint(0, seq_len // 4)
            keep_indices = np.random.choice(seq_len, seq_len - drop_num, replace=False)
            keep_indices.sort()
            seq = seq[keep_indices]
            if len(seq) < seq_len:
                seq = np.pad(seq, ((0, seq_len - len(seq)), (0, 0), (0, 0), (0, 0)), mode='edge')
            if np.random.rand() < 0.3:
                speed = np.random.uniform(0.8, 1.2)
                new_len = int(seq_len / speed)
                indices = np.linspace(0, seq_len - 1, new_len).astype(int)
                seq = seq[np.clip(indices, 0, seq_len - 1)]
                if len(seq) < seq_len:
                    seq = np.pad(seq, ((0, seq_len - len(seq)), (0, 0), (0, 0), (0, 0)), mode='edge')
                elif len(seq) > seq_len:
                    seq = seq[:seq_len]
        # Frame augmentations
        augmented_seq = []
        transform = self.strong_transform if self.train and np.random.rand() < 0.6 else self.light_transform
        for frame in seq:
            augmented = transform(image=frame.transpose(1, 2, 0)) if self.train else self.transform(image=frame.transpose(1, 2, 0))
            augmented_seq.append(augmented['image'].float() / 255.0)
        seq_tensor = torch.stack(augmented_seq)
        # YOLO features
        yolo_features = torch.zeros(seq_len, 80, dtype=torch.float32) if self.yolo_files is None else None
        if self.yolo_files:
            with h5py.File(self.yolo_files[file_idx], 'r') as hf_yolo:
                yolo_features = torch.tensor(hf_yolo['yolo_features'][offset], dtype=torch.float32)
        # Hierarchical labels
        multiclass_label = torch.tensor(self.labels[idx], dtype=torch.long)
        binary_label = torch.tensor(0 if multiclass_label == 7 else 1, dtype=torch.long)  # 0: Normal, 1: Abnormal
        return seq_tensor, binary_label, multiclass_label, yolo_features