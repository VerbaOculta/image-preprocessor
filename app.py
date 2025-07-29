from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Gaussian blur
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Adaptive Thresholding
    processed = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return processed

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
