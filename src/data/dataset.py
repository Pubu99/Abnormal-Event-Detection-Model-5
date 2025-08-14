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
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
                A.GaussianBlur(blur_limit=(3, 9), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-30, 30), shear=(-20, 20), p=0.7),
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.4),
                A.RandomRain(p=0.4, blur_value=5),
                A.RandomShadow(p=0.4),
                A.GaussNoise(p=0.4),  # fixed here
                A.MotionBlur(blur_limit=7, p=0.3),
                A.OpticalDistortion(p=0.3),
                ToTensorV2()
            ])


            self.light_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HorizontalFlip(p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = ToTensorV2()
        
        self.lengths = []
        for f in self.files:
            with h5py.File(f, 'r') as hf:
                self.lengths.append(hf['sequences'].shape[0])
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
            seq = hf['sequences'][offset]
            label = hf['labels'][offset]
        
        # Temporal augmentation
        if self.train and np.random.rand() < 0.5:
            seq_len = len(seq)
            drop_num = np.random.randint(0, seq_len // 4)  # Drop up to 25% frames
            keep_indices = np.random.choice(seq_len, seq_len - drop_num, replace=False)
            keep_indices.sort()
            seq = seq[keep_indices]
            if len(seq) < seq_len:  # Pad with last frame
                seq = np.pad(seq, ((0, seq_len - len(seq)), (0, 0), (0, 0), (0, 0)), mode='edge')
            # Speed perturbation: subsample or stretch
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
            augmented = transform(image=frame) if self.train else self.transform(image=frame)
            augmented_seq.append(augmented['image'])
        
        seq_tensor = torch.stack(augmented_seq).permute(1, 0, 2, 3)
        
        # Load YOLO features
        yolo_features = torch.zeros(16, 80) if self.yolo_files is None else None
        if self.yolo_files:
            with h5py.File(self.yolo_files[file_idx], 'r') as hf_yolo:
                yolo_features = torch.tensor(hf_yolo['yolo_features'][offset], dtype=torch.float32)
        
        return seq_tensor.float() / 255.0, torch.tensor(label, dtype=torch.long), yolo_features