import os
import cv2
import numpy as np
import base64
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Ensure Haar Cascade XML exists locally on cloud deployments
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

def load_face_cascade():
    """Dynamically downloads and loads the Haar Cascade XML model if missing."""
    if not os.path.exists(CASCADE_FILENAME):
        try:
            print(f"Downloading {CASCADE_FILENAME} from official OpenCV repository...")
            response = requests.get(CASCADE_URL, timeout=10)
            if response.status_code == 200:
                with open(CASCADE_FILENAME, 'wb') as f:
                    f.write(response.content)
                print("Cascade file successfully downloaded.")
            else:
                print(f"Failed to download cascade file. Status code: {response.status_code}")
        except Exception as err:
            print(f"Error downloading face cascade file: {err}")

    if os.path.exists(CASCADE_FILENAME):
        cascade = cv2.CascadeClassifier(CASCADE_FILENAME)
        if not cascade.empty():
            return cascade
    
    # Fallback to internal OpenCV path if present
    try:
        internal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(internal_path)
        if not cascade.empty():
            return cascade
    except Exception:
        pass
        
    return None

# Initialize face detector
face_cascade = load_face_cascade()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for platform health checks."""
    return jsonify({
        'status': 'healthy',
        'cascade_loaded': face_cascade is not None and not face_cascade.empty()
    }), 200

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives base64 webcam frames, performs OpenCV face detection, and returns bounding coordinates."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Missing image payload'}), 400

        # Decode base64 image data
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Frame decoding failed'}), 400

        # Convert image to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_faces = []
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                })

        return jsonify({
            'status': 'success',
            'face_detected': len(detected_faces) > 0,
            'faces_count': len(detected_faces),
            'faces': detected_faces
        })

    except Exception as error:
        return jsonify({'status': 'error', 'message': str(error)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
