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
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-15, 15), shear=(-10, 10), p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.RandomRain(p=0.3),
            A.RandomShadow(p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            ToTensorV2()
        ]) if train else ToTensorV2()
    
    def __len__(self):
        return sum(h5py.File(f, 'r')['sequences'].shape[0] for f in self.files)
    
    def __getitem__(self, idx):
        cumsum = 0
        for f in self.files:
            with h5py.File(f, 'r') as hf:
                seq_count = hf['sequences'].shape[0]
                if idx < cumsum + seq_count:
                    seq = hf['sequences'][idx - cumsum]
                    label = hf['labels'][idx - cumsum]
                    break
                cumsum += seq_count
        
        augmented_seq = []
        for frame in seq:
            augmented = self.transform(image=frame)
            augmented_seq.append(augmented['image'])
        
        seq_tensor = torch.stack(augmented_seq).permute(1, 0, 2, 3)
        return seq_tensor.float() / 255.0, torch.tensor(label)