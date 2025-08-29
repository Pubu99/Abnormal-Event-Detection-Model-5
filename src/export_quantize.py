"""
Model Export & Quantization Utility
Exports model to ONNX and quantizes for edge deployment.
"""
import torch
from src.models.base_model import AnomalyModel

def export_onnx(model_path, out_path='model.onnx'):
    model = AnomalyModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dummy_seq = torch.randn(1, 16, 3, 224, 224)
    dummy_yolo = torch.randn(1, 16, 80)
    torch.onnx.export(model, (dummy_seq, dummy_yolo), out_path, input_names=['seq', 'yolo_features'], output_names=['output'], opset_version=17)
    print(f"Exported model to {out_path}")

def quantize_model(model_path, out_path='model_quantized.pth'):
    model = AnomalyModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    torch.save(model.state_dict(), out_path)
    print(f"Quantized model saved to {out_path}")
