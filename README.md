# KrishiLens Disease Detection Server

Plant disease detection API using YOLOv8.

## Classes
- Pepper Bell Bacterial Spot
- Potato Early Blight
- Potato Late Blight
- Tomato Yellow Leaf Curl Virus
- Tomato Bacterial Spot
- Tomato Septoria Leaf Spot

## Endpoints
- GET /health - Health check
- POST /detect - Detect diseases in image
- GET /classes - Get all classes
- GET /model_info - Get model information

## Local Testing
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Deployment
Deployed on Railway.app