from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


camera = None
detection_active = False
min_confidence = 0.2
model = None
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

def check_model_files():
    prototxt_path = r'C:\UNT\1. Fall 2024\CSCE 5214 SDAI - Ben Othmane (Sec 3)\Project\Version 5\models\ResNet-50-deploy.prototxt.txt'
    model_path = r'C:\UNT\1. Fall 2024\CSCE 5214 SDAI - Ben Othmane (Sec 3)\Project\Version 5\models\ResNet-50-model.caffemodel'

    
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return prototxt_path, model_path

def load_model():
    global model
    try:
        prototxt_path, model_path = check_model_files()
        print(f"Loading model from:\nProto: {prototxt_path}\nModel: {model_path}")
        model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def detect_objects(frame, confidence_threshold):
    if model is None:
        return frame
    
    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007, (300, 300), 130)
    model.setInput(blob)
    detected_objects = model.forward()
    
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_index = int(detected_objects[0, 0, i, 1])
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)
            
            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.rectangle(frame, (upper_left_x, upper_left_y), 
                         (lower_right_x, lower_right_y), 
                         colors[class_index].tolist(), 3)
            cv2.putText(frame, prediction_text, 
                       (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index].tolist(), 2)
    
    return frame

def gen_frames():
    global camera, detection_active
    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            if detection_active and model is not None:
                frame = detect_objects(frame, min_confidence)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'Dataset uploaded successfully'})

@app.route('/train_model', methods=['POST'])
def train_model():
    return jsonify({'message': 'Model training started'})

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    global min_confidence
    min_confidence = float(request.json['confidence'])
    return jsonify({'message': 'Confidence updated'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    detection_active = request.json['active']
    return jsonify({'message': 'Detection toggled'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'error': 'Failed to open camera'}), 500
        return jsonify({'message': 'Camera started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not load_model():
        print("Warning: Failed to load model. Starting server without model.")
    app.run(debug=True, host='0.0.0.0', port=5000)
