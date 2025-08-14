import torch
import torch.nn as nn
from transformers import TimeSformerForVideoClassification

class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16):
        super().__init__()
        self.backbone = TimeSformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.backbone.gradient_checkpointing_enable()
        self.yolo_fc = nn.Linear(80, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(num_classes, num_classes)
    
    def forward(self, x: torch.Tensor, yolo_features: torch.Tensor) -> torch.Tensor:
        ts_logits = self.backbone(x).logits
        yolo_logits = self.yolo_fc(yolo_features.mean(dim=1))  # Average over frames
        gate = self.gate(torch.cat([ts_logits, yolo_logits], dim=1))
        fused_logits = self.fusion(gate * ts_logits + (1 - gate) * yolo_logits)
        return fused_logits
    
    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        scores = torch.softmax(scores, dim=1)
        pred = torch.argmax(scores, dim=1)
        conf = torch.max(scores, dim=1).values
        if pred == 7:
            return "No Anomaly"
        elif conf < thresh:
            return "Unknown Anomaly"
        else:
            classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
            return classes[pred.item()]