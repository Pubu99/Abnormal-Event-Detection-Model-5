import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, processed_dir: str, yolo_dir: str = None, train: bool = True):
        self.files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.h5')]
        self.yolo_files = [os.path.join(yolo_dir, f) for f in os.listdir(yolo_dir) if f.endswith('.h5')] if yolo_dir else None
        self.train = train
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
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),  # Added for regularization
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
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.cumulative_lengths[file_idx]:
            file_idx += 1
        
        offset = idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)
        
        with h5py.File(self.files[file_idx], 'r') as hf:
            seq = hf['sequences'][offset]  # Shape: [seq_len, height, width, channels]
        
        seq = np.transpose(seq, (0, 3, 1, 2))  # [seq_len, 3, height, width]
        
        seq_len = 16
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
        
        augmented_seq = []
        transform = self.strong_transform if self.train and np.random.rand() < 0.6 else self.light_transform
        for frame in seq:
            augmented = transform(image=frame.transpose(1, 2, 0)) if self.train else self.transform(image=frame.transpose(1, 2, 0))
            augmented_seq.append(augmented['image'].float() / 255.0)
        
        seq_tensor = torch.stack(augmented_seq)  # Shape: [seq_len, 3, height, width]
        
        yolo_features = torch.zeros(seq_len, 80, dtype=torch.float32) if self.yolo_files is None else None
        if self.yolo_files:
            with h5py.File(self.yolo_files[file_idx], 'r') as hf_yolo:
                yolo_features = torch.tensor(hf_yolo['yolo_features'][offset], dtype=torch.float32)
        
        return seq_tensor, torch.tensor(self.labels[idx], dtype=torch.long), yolo_features