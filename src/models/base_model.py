import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel
import warnings
warnings.filterwarnings("ignore", message="Some weights.*were not used")
warnings.filterwarnings("ignore", message="Some weights.*were not initialized")
warnings.filterwarnings("ignore", message=".*resume_download.*")

class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16):
        super().__init__()
        self.config = TimesformerConfig.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_frames=seq_len
        )
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=self.config,
            ignore_mismatched_sizes=True
        )
        self.backbone.train()
        self.backbone.gradient_checkpointing_enable()
        
        self.ts_fc = nn.Sequential(  # Added dropout
            nn.Linear(self.config.hidden_size, num_classes),
            nn.Dropout(0.3)
        )
        
        self.yolo_fc = nn.Sequential(  # Added dropout
            nn.Linear(80, num_classes),
            nn.Dropout(0.3)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(  # Added dropout
            nn.Linear(num_classes, num_classes),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor, yolo_features: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(2) != 3:
            raise ValueError(f"Expected input shape [batch, seq_len, 3, height, width], got {x.shape}")
        if yolo_features.dim() != 3 or yolo_features.size(2) != 80:
            raise ValueError(f"Expected yolo_features shape [batch, seq_len, 80], got {yolo_features.shape}")
        
        ts_outputs = self.backbone(pixel_values=x).last_hidden_state
        ts_logits = self.ts_fc(ts_outputs[:, 0, :])
        
        yolo_logits = self.yolo_fc(yolo_features.mean(dim=1))
        
        gate = self.gate(torch.cat([ts_logits, yolo_logits], dim=1))
        fused_logits = self.fusion(gate * ts_logits + (1 - gate) * yolo_logits)
        
        return fused_logits

    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        scores = torch.softmax(scores, dim=1)
        pred = torch.argmax(scores, dim=1)
        conf = torch.max(scores, dim=1).values
        
        if pred.item() == 7:
            return "No Anomaly"
        elif conf.item() < thresh:
            return "Unknown Anomaly"
        else:
            classes = [
                'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
                'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery',
                'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
            ]
            return classes[pred.item()]