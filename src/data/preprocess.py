import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from typing import List

def preprocess_data(raw_dir: str, processed_dir: str, seq_len: int = 16, img_size: tuple = (64, 64)):
    classes = os.listdir(raw_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    for cls in tqdm(classes, desc="Processing classes"):
        cls_path = os.path.join(raw_dir, cls)
        images = sorted([f for f in os.listdir(cls_path) if f.endswith('.png')])
        seqs = []
        labels = []
        class_idx = classes.index(cls)
        
        for i in tqdm(range(0, len(images) - seq_len, seq_len), desc=f"Building sequences for {cls}", leave=False):
            seq = []
            for j in range(seq_len):
                img_path = os.path.join(cls_path, images[i + j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, img_size)
                seq.append(img)
            seqs.append(np.array(seq))
            labels.append(class_idx)
        
        with h5py.File(os.path.join(processed_dir, f'{cls}_sequences.h5'), 'w') as hf:
            hf.create_dataset('sequences', data=np.array(seqs))
            hf.create_dataset('labels', data=np.array(labels))