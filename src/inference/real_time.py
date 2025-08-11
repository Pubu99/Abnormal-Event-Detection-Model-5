import cv2
import torch
from src.models.fusion import MultiCameraFusion
from src.models.base_model import AnomalyModel
from src.utils.alerts import send_alert

def real_time_detect(camera_urls: List[str], model_path: str):
    single_model = AnomalyModel()
    model = MultiCameraFusion(single_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = torch.quantization.quantize_dynamic(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    caps = [cv2.VideoCapture(url) for url in camera_urls]
    seq_buffers = [[] for _ in camera_urls]
    
    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (64, 64))
            seq_buffers[i].append(frame)
            if len(seq_buffers[i]) > 16:
                seq_buffers[i].pop(0)
            if len(seq_buffers[i]) == 16:
                seq_tensor = torch.tensor(np.array(seq_buffers[i])).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
                frames.append(seq_tensor.to(device))
        
        if len(frames) == len(camera_urls):
            scores = model(frames)
            anomaly = single_model.detect_anomaly(scores)  # Use base detect
            aggregate_score = torch.max(scores).item()
            
            print(f"Detection: {anomaly}, Score: {aggregate_score}")
            if aggregate_score > 0.7 and anomaly != "No Anomaly":
                send_alert(anomaly)
    
    for cap in caps: cap.release()