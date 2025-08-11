import torch
import torch.nn as nn
from typing import List
from .base_model import AnomalyModel

class MultiCameraFusion(nn.Module):
    def __init__(self, single_model: AnomalyModel, num_cameras: int = 3):
        super().__init__()
        self.single_models = nn.ModuleList([single_model for _ in range(num_cameras)])
        self.attention = nn.MultiheadAttention(embed_dim=14, num_heads=2)
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        scores = [model(input) for model, input in zip(self.single_models, inputs)]
        fused = torch.stack(scores, dim=1)
        fused, _ = self.attention(fused, fused, fused)
        aggregate_score = torch.mean(fused, dim=1)
        return aggregate_score