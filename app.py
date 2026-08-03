from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Strip base64 metadata

        # Decode base64 image bytes to OpenCV format
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'}), 400

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check face detection result
        face_detected = len(faces) > 0

        return jsonify({
            'status': 'success',
            'face_detected': face_detected,
            'faces_count': len(faces)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)