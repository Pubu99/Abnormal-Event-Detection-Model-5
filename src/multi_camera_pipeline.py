"""
Multi-Camera Fusion Example Pipeline
"""
import torch
from src.models.base_model import AnomalyModel
from src.models.fusion import MultiCameraFusion

def run_multi_camera_fusion(model_path, num_cameras=3):
    single_model = AnomalyModel()
    single_model.load_state_dict(torch.load(model_path))
    fusion_model = MultiCameraFusion(single_model, num_cameras=num_cameras)
    # Example: inputs = [seq1, seq2, seq3] from different cameras
    # outputs = fusion_model(inputs)
    print("Multi-camera fusion model ready.")
