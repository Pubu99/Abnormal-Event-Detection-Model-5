import torch
from src.models.base_model import AnomalyModel
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader

def batch_detect(model_path: str, data_dir: str, device: torch.device):
    model = AnomalyModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    dataset = CustomDataset(data_dir, train=False)
    loader = DataLoader(dataset, batch_size=32)
    
    results = []
    for seqs, _ in loader:  # Ignore labels for inference
        seqs = seqs.to(device)
        scores = model(seqs)
        for score in scores:
            anomaly = model.detect_anomaly(score.unsqueeze(0))
            conf = torch.max(score).item()
            results.append({"anomaly": anomaly, "score": conf})
    
    return results