import torch
import torch.nn as nn
from torchvision.models.video import efficientnet_b0_3d
from ultralytics import YOLO

class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16):
        super().__init__()
        self.object_detector = YOLO('yolov8n.pt')
        self.backbone = efficientnet_b0_3d(num_classes=0)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.backbone.classifier.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.fc(features)
        scores = self.softmax(out)
        return scores
    
    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        pred = torch.argmax(scores, dim=1)
        conf = torch.max(scores, dim=1).values
        if pred == 7:  # 'Normal Videos'
            return "No Anomaly"
        elif conf < thresh:
            return "Unknown Anomaly"
        else:
            classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
            return classes[pred.item()]