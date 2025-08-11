from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import asyncio
from src.inference.real_time import real_time_detect
from src.utils.alerts import send_alert

app = FastAPI()

@app.post("/detect")
async def detect_anomaly(data: dict):
    # Expect {"anomaly": str, "score": float} from model/inference
    anomaly = data.get("anomaly")
    score = data.get("score")
    
    if score > 0.7 and anomaly != "No Anomaly":
        send_alert(anomaly)
        return JSONResponse({"status": "alert_sent", "category": anomaly})
    return JSONResponse({"status": "no_alert"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Integrate real-time detection
    async def detect_task(camera_urls: list):
        # Run real_time_detect in thread, send results via WS
        def run_detection():
            real_time_detect(camera_urls, 'models_saved/best_model.pth')
        
        await asyncio.to_thread(run_detection)
        # On detection, send to WS and trigger alert via /detect API
    # Usage: Pass camera_urls via initial message