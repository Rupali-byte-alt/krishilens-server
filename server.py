from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="KrishiLens AI Server")

# Load model once at startup
model = YOLO("best.pt")

@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(image, conf=0.3)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "label": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in xyxy]
            })
        return {"detections": detections}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def health_check():
    return {"status": "KrishiLens AI Server is running!"}