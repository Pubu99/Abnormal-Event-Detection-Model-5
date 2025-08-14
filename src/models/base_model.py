import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel  

class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16):
        super().__init__()
        
        # Load Timesformer configuration and model
        self.config = TimesformerConfig.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_frames=seq_len
        )
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=self.config
        )
        self.backbone.train()  # Ensure training mode
        self.backbone.gradient_checkpointing_enable()  # Save memory

        # Custom classification head for Timesformer
        self.ts_fc = nn.Linear(self.config.hidden_size, num_classes)

        # YOLO features head
        self.yolo_fc = nn.Linear(80, num_classes)

        # Gate for fusion
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid()
        )

        # Final fusion layer
        self.fusion = nn.Linear(num_classes, num_classes)

    def forward(self, x: torch.Tensor, yolo_features: torch.Tensor) -> torch.Tensor:
        """
        x: video frames tensor [batch, seq_len, channels, height, width]
        yolo_features: tensor [batch, seq_len, 80]
        """
        # Timesformer outputs (hidden states)
        ts_outputs = self.backbone(pixel_values=x).last_hidden_state  # [batch, seq_len, hidden_size]
        ts_logits = self.ts_fc(ts_outputs[:, 0, :])  # CLS token embedding

        # YOLO features
        yolo_logits = self.yolo_fc(yolo_features.mean(dim=1))  # Average over frames

        # Fusion using gate
        gate = self.gate(torch.cat([ts_logits, yolo_logits], dim=1))
        fused_logits = self.fusion(gate * ts_logits + (1 - gate) * yolo_logits)

        return fused_logits

    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        """
        Converts raw logits to class label with optional thresholding.
        """
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
