import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import yaml
from collections import Counter

def preprocess_data(raw_dir: str, processed_dir: str, seq_len: int = 16, img_size: tuple = (64, 64)):
    # List class folders, ignore non-directories or hidden files
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"Found classes: {classes}")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize counter for label counts
    label_counts = Counter()
    
    for cls in tqdm(classes, desc="Processing classes", file=sys.stdout):
        cls_path = os.path.join(raw_dir, cls)
        images = sorted([f for f in os.listdir(cls_path) if f.endswith('.png')])
        print(f"Class '{cls}' - {len(images)} images found")
        
        seqs = []
        labels = []
        class_idx = classes.index(cls)
        
        if len(images) < seq_len:
            print(f"Warning: Not enough images in class '{cls}' to form one sequence of length {seq_len}. Skipping.")
            continue
        
        for i in tqdm(range(0, len(images) - seq_len + 1, seq_len), 
                      desc=f"Building sequences for {cls}", leave=False, file=sys.stdout):
            seq = []
            for j in range(seq_len):
                img_path = os.path.join(cls_path, images[i + j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {img_path}")
                img = cv2.resize(img, img_size)
                seq.append(img)
            seqs.append(np.array(seq))
            labels.append(class_idx)
            label_counts[class_idx] += 1
        
        if seqs:
            out_path = os.path.join(processed_dir, f'{cls}_sequences.h5')
            with h5py.File(out_path, 'w') as hf:
                hf.create_dataset('sequences', data=np.array(seqs))
                hf.create_dataset('labels', data=np.array(labels))
            print(f"Saved {len(seqs)} sequences for class '{cls}' to {out_path}")
        else:
            print(f"No sequences created for class '{cls}'")
    
    # Save label counts to YAML
    label_counts_path = os.path.join(processed_dir, 'label_counts.yaml')
    with open(label_counts_path, 'w') as f:
        yaml.dump(dict(label_counts), f)
    print(f"Saved label counts to {label_counts_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <raw_data_dir> <processed_data_dir>")
        sys.exit(1)
    raw_dir = sys.argv[1]
    processed_dir = sys.argv[2]
    preprocess_data(raw_dir, processed_dir)