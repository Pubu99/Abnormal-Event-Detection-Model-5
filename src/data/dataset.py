import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, processed_dir: str, train: bool = True):
        self.files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.h5')]
        self.train = train
        if train:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-15, 15), shear=(-10, 10), p=0.5),
                A.RandomFog(fog_coef_intensity=0.2, p=0.3),  # Fixed
                A.RandomRain(p=0.3),
                A.RandomShadow(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Fixed
                ToTensorV2()
            ])
        else:
            self.transform = ToTensorV2()
        
        # Precompute lengths for __len__ optimization
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
        
        augmented_seq = []
        for frame in seq:
            augmented = self.transform(image=frame)
            augmented_seq.append(augmented['image'])
        
        seq_tensor = torch.stack(augmented_seq).permute(1, 0, 2, 3)
        return seq_tensor.float() / 255.0, torch.tensor(label, dtype=torch.long)  # Fixed