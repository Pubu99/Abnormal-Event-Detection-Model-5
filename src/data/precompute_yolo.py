import h5py
import os
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm

def precompute_yolo_features(input_dir, output_dir, model_path='yolov8n.pt', batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(model_path).to(device)  # Move YOLO to GPU
    
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
        input_path = os.path.join(input_dir, h5_file)
        output_path = os.path.join(output_dir, h5_file.replace('.h5', '_yolo.h5'))
        
        with h5py.File(input_path, 'r') as hf_in:
            sequences = hf_in['sequences'][:]  # Shape: (N, 16, H, W, 3)
            labels = hf_in['labels'][:]
            num_samples = sequences.shape[0]
            
            with h5py.File(output_path, 'w') as hf_out:
                yolo_features = hf_out.create_dataset('yolo_features', (num_samples, 16, 80), dtype='float32')
                hf_out.create_dataset('labels', data=labels)
                
                # Process sequences in batches
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_seqs = sequences[start_idx:end_idx]  # Shape: (batch, 16, H, W, 3)
                    
                    # Prepare batch for YOLO
                    batch = torch.tensor(batch_seqs).permute(0, 1, 4, 2, 3).float() / 255.0  # Shape: (batch, 16, 3, H, W)
                    batch = batch.reshape(-1, 3, batch_seqs.shape[2], batch_seqs.shape[3])  # Shape: (batch*16, 3, H, W)
                    batch = batch.to(device)
                    
                    # YOLO inference
                    results = yolo(batch, imgsz=224)  # Set image size to 224
                    for i in range(end_idx - start_idx):
                        seq_features = []
                        for frame_idx in range(16):
                            res = results[i * 16 + frame_idx]
                            cls_probs = torch.zeros(80, device=device)  # Create on GPU
                            if res.boxes.cls is not None:
                                cls_probs[res.boxes.cls.long()] = res.boxes.conf  # Both on GPU
                            seq_features.append(cls_probs.cpu().numpy())  # Move to CPU for storage
                        yolo_features[start_idx + i] = seq_features

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed/train_yolo"
    precompute_yolo_features(input_dir, output_dir)