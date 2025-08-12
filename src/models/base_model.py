import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from ultralytics import YOLO

class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.feature_dim = 512
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        # Initialize YOLO lazily to avoid method conflicts
        self._yolo = None
    
    @property
    def object_detector(self):
        if self._yolo is None:
            self._yolo = YOLO('yolov8n.pt')
        return self._yolo
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.fc(features)
        scores = self.softmax(out)
        return scores
    
    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        pred = torch.argmax(scores, dim=1)
        conf = torch.max(scores, dim=1).values
        if pred == 7:
            return "No Anomaly"
        elif conf < thresh:
            return "Unknown Anomaly"
        else:
            classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
            return classes[pred.item()]