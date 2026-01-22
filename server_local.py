# app.py
import os

# Disable telemetry and set safe config dir
os.environ["ULTRALYTICS_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["ULTRALYTICS_SETTINGS_DIR"] = "/tmp"

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
model = None
device = 'cpu'
class_names = []

# ✅ Use 'best.pt' from the models/ folder (relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'best.pt'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

def resize_image(image, max_size=640):
    """Resize image to max dimension while preserving aspect ratio"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def load_model():
    """Load YOLOv8 model at startup"""
    global model, class_names
    try:
        logger.info(f"Loading YOLOv8 model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = YOLO(MODEL_PATH)
        model.to(device)
        class_names = list(model.names.values())
        
        logger.info(f"✅ Model loaded on {device}")
        logger.info(f"Classes: {class_names}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

# Load model when app starts
load_model()

@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to Krishilens Plant Disease Detection API",
        "endpoints": ["/health", "/detect", "/classes", "/model_info"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'classes': class_names
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        confidence = float(request.form.get('confidence', CONFIDENCE_THRESHOLD))
        return_image = request.form.get('return_image', 'false').lower() == 'true'

        image = Image.open(image_file.stream).convert('RGB')
        image = resize_image(image, max_size=640)
        original_image = image.copy()

        results = model(image, conf=confidence)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class': class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}',
                'confidence': round(conf, 3),
                'box': [round(x, 2) for x in xyxy]
            })

        response_data = {
            'success': True,
            'detections': detections,
            'count': len(detections),
            'image_size': {'width': image.width, 'height': image.height}
        }

        if return_image and detections:
            annotated_image = draw_boxes(original_image, detections)
            response_data['annotated_image'] = image_to_base64(annotated_image)

        return jsonify(response_data)

    except Exception as e:
        logger.exception("💥 Detection failed with exception:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': class_names, 'count': len(class_names)})

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Object Detection - Plant Disease',
        'device': device,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'classes': class_names,
        'num_classes': len(class_names)
    })

def draw_boxes(image, detections):
    """Draw bounding boxes with safe default font"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']
        color = colors[hash(class_name) % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_name}: {confidence:.2f}"
        draw.text((x1, y1), label, fill=color, font=font)
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return f"image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)