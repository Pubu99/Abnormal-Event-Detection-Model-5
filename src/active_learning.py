"""
Active Learning & Hard Negative Mining Utility
Selects low-confidence or misclassified samples for review and retraining.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import CustomDataset
from src.models.base_model import AnomalyModel

def select_hard_samples(model_path, data_dir, device, threshold=0.3, batch_size=32):
    model = AnomalyModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    dataset = CustomDataset(data_dir, train=False)
    loader = DataLoader(dataset, batch_size=batch_size)
    hard_samples = []
    for seqs, binary_labels, multiclass_labels, yolo_features in loader:
        seqs, binary_labels, multiclass_labels, yolo_features = seqs.to(device), binary_labels.to(device), multiclass_labels.to(device), yolo_features.to(device)
        with torch.no_grad():
            outputs = model(seqs, yolo_features)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)
            for i in range(seqs.size(0)):
                if conf[i] < threshold or preds[i] != multiclass_labels[i]:
                    hard_samples.append((seqs[i].cpu().numpy(), multiclass_labels[i].item()))
    np.save('hard_samples.npy', hard_samples)
    print(f"Saved {len(hard_samples)} hard/ambiguous samples for review.")
