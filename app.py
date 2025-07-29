from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Optional: Slight denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return denoised

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_bytes = file.read()

    processed = preprocess_image(image_bytes)

    # Convert processed image to PNG for response
    _, img_encoded = cv2.imencode('.png', processed)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/png',
        as_attachment=False,
        download_name='processed.png'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
