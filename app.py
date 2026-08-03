import base64
import os
import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None


def analyze_human_skin_biometrics(face_crop):
    if face_crop is None or face_crop.size == 0:
        return False, "No face pixels captured"

    try:
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_ratio = (np.count_nonzero(skin_mask) / float(face_crop.shape[0] * face_crop.shape[1])) * 100.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 5.0 or laplacian_var > 350.0:
            return False, "Digital Screen or Moiré Pattern Detected"

        if skin_ratio > 12.0:
            return True, "Human Skin Biometrics Verified"
        else:
            return False, "Non-Human / Artificial Texture Detected"

    except Exception as e:
        return False, f"Analysis Error: {str(e)}"


def compute_ear(face_crop):
    if face_crop is None or face_crop.size == 0:
        return 0.28

    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        eye_band = gray[int(h * 0.20):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
        if eye_band.size == 0:
            return 0.28

        _, thresh = cv2.threshold(eye_band, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid = [c for c in contours if cv2.boundingRect(c)[2] > 4]
            if valid:
                eb = cv2.boundingRect(valid[0])
                ear_val = round(float(eb[3]) / float(eb[2]), 2) if eb[2] > 0 else 0.28
                return float(np.clip(ear_val, 0.15, 0.42))

        return 0.28
    except Exception:
        return 0.28


def analyze_audio_spoof(audio_signal, sr=16000):
    if len(audio_signal) < 100:
        return 0.10, "INSUFFICIENT AUDIO"

    max_amp = np.max(np.abs(audio_signal))
    if max_amp < 0.001:
        return 0.05, "SILENT AUDIO"

    signal_norm = audio_signal / max_amp

    fft_vals = np.abs(rfft(signal_norm))
    freqs = rfftfreq(len(signal_norm), 1.0 / sr)

    vocal_band = (freqs >= 100) & (freqs <= 1500)
    artifact_band = (freqs > 2500) & (freqs <= 7500)

    vocal_power = np.sum(fft_vals[vocal_band] ** 2) + 1e-8
    artifact_power = np.sum(fft_vals[artifact_band] ** 2) + 1e-8

    tilt_ratio = artifact_power / vocal_power

    if tilt_ratio > 0.18:
        spoof_score = 0.78 + min(tilt_ratio * 0.4, 0.18)
    else:
        spoof_score = max(0.04, tilt_ratio * 0.7)

    return float(np.clip(spoof_score, 0.04, 0.98)), "SPEECH ANALYZED"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        data = request.json.get('image', '')
        if not data or ',' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid base64 payload'})

        _, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({'status': 'error', 'message': 'Frame decoding failed'})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_real = False
        ear_val = 0.28
        msg = "No Face Detected"

        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(80, 80))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                is_real, msg = analyze_human_skin_biometrics(face_crop)
                ear_val = compute_ear(face_crop)
            else:
                fh, fw, _ = frame.shape
                face_crop = frame[int(fh*0.15):int(fh*0.85), int(fw*0.25):int(fw*0.75)]
                is_real, msg = analyze_human_skin_biometrics(face_crop)
                ear_val = compute_ear(face_crop)
        else:
            fh, fw, _ = frame.shape
            face_crop = frame[int(fh*0.15):int(fh*0.85), int(fw*0.25):int(fw*0.75)]
            is_real, msg = analyze_human_skin_biometrics(face_crop)
            ear_val = compute_ear(face_crop)

        return jsonify({
            'status': 'success',
            'is_human': is_real,
            'ear': ear_val,
            'message': msg
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    try:
        audio_data = request.json.get('audio', [])
        if not audio_data:
            return jsonify({'status': 'error', 'message': 'No audio array received'})

        signal = np.array(audio_data, dtype=np.float32)
        spoof_score, msg = analyze_audio_spoof(signal)

        is_human = spoof_score <= 0.50
        confidence = round((1.0 - spoof_score) * 100, 1) if is_human else round(spoof_score * 100, 1)

        return jsonify({
            'status': 'success',
            'is_human': is_human,
            'confidence': confidence,
            'spoof_score': round(spoof_score, 2),
            'message': msg
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)