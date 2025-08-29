"""
Explainability Utility: Save Grad-CAM heatmaps for UI integration.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.inference.detect import grad_cam

def save_gradcam_heatmap(model, input_tensor, save_path, target_class=None, layer_name='backbone.encoder.layers.11.output.dense'):
    cam = grad_cam(model, input_tensor, target_class, layer_name)
    cam_img = cam[0] if cam.ndim == 3 else cam
    plt.imshow(cam_img, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM heatmap to {save_path}")
