"""
Synthetic Data Generation Utility
Augments rare classes using strong augmentations.
"""
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def augment_sample(image, n=5):
    aug = A.Compose([
        A.RandomBrightnessContrast(0.5, 0.5, p=0.8),
        A.GaussianBlur(3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.4),
        A.RandomRain(blur_value=5, p=0.4),
        A.RandomShadow(p=0.4),
        A.GaussNoise(var_limit=(20, 60), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.OpticalDistortion(p=0.3),
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),
        ToTensorV2()
    ])
    augmented = []
    for _ in range(n):
        aug_img = aug(image=image)['image']
        augmented.append(aug_img.numpy())
    return np.stack(augmented)
