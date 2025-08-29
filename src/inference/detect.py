import numpy as np
import torch.nn.functional as F

def grad_cam(model, input_tensor, target_class=None, layer_name='backbone.encoder.layers.11.output.dense'):
    """
    Compute Grad-CAM for a given input and model layer.
    Args:
        model: PyTorch model
        input_tensor: input batch [1, seq_len, 3, H, W]
        target_class: int or None (default: predicted class)
        layer_name: str, name of the layer to visualize
    Returns:
        cam: np.array, Grad-CAM heatmap
    """
    model.eval()
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    # Register hooks
    layer = model
    for attr in layer_name.split('.'):
        layer = getattr(layer, attr)
    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_backward_hook(backward_hook)
    # Forward
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    # Backward
    model.zero_grad()
    loss.backward(retain_graph=True)
    # Grad-CAM
    grads = gradients['value']  # [B, C, ...]
    acts = activations['value'] # [B, C, ...]
    weights = grads.mean(dim=[2, 3], keepdim=True) if grads.dim() == 4 else grads.mean(dim=1, keepdim=True)
    cam = (weights * acts).sum(dim=1)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_fwd.remove()
    handle_bwd.remove()
    return cam

def ensemble_predict(models, seq, yolo_features, device):
    """
    Ensemble inference: average logits from multiple models.
    Args:
        models: list of models
        seq: [seq_len, 3, H, W] tensor
        yolo_features: [seq_len, 80] tensor
        device: torch.device
    Returns:
        avg_logits: torch.Tensor
    """
    logits = []
    for m in models:
        m.eval()
        with torch.no_grad():
            l = m(seq.unsqueeze(0).to(device), yolo_features.unsqueeze(0).to(device))
            logits.append(l.cpu())
    avg_logits = torch.stack(logits).mean(dim=0)
    return avg_logits
import torch
from src.models.base_model import AnomalyModel
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader

from ultralytics import YOLO

def batch_detect(model_path: str, data_dir: str, device: torch.device, yolo_model_path: str = 'yolov8n.pt'):
    """
    Batch inference with anomaly override: if YOLO detects a weapon/suspicious object, override to 'Abnormal'.
    """
    model = AnomalyModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    dataset = CustomDataset(data_dir, train=False)
    loader = DataLoader(dataset, batch_size=32)
    yolo = YOLO(yolo_model_path).to(device)
    # Define suspicious classes (COCO): 32 = sports ball, 43 = knife, 44 = gun, 45 = baseball bat, 46 = baseball glove
    weapon_classes = {43, 44, 45, 46}
    results = []
    for seqs, _, _, _ in loader:
        seqs = seqs.to(device)
        # Run YOLO on each frame in the batch
        batch_size, seq_len, c, h, w = seqs.shape
        seqs_reshaped = seqs.view(-1, c, h, w)
        yolo_results = yolo(seqs_reshaped, imgsz=224)
        # For each sequence, check if any frame has a suspicious object
        for i in range(batch_size):
            suspicious = False
            for t in range(seq_len):
                res = yolo_results[i * seq_len + t]
                if res.boxes.cls is not None:
                    detected = set(res.boxes.cls.cpu().numpy().astype(int))
                    if weapon_classes & detected:
                        suspicious = True
                        break
            # Run anomaly model
            score = model(seqs[i].unsqueeze(0))
            anomaly = model.detect_anomaly(score)
            conf = torch.max(score).item()
            # Override: if suspicious object, force 'Abnormal'
            if suspicious:
                anomaly = 'Abnormal (Object Detected)'
                conf = 1.0
            results.append({"anomaly": anomaly, "score": conf, "suspicious": suspicious})
    return results