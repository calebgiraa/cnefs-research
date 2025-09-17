"""
This code was produced in help with YouTube tutorials (freeCodeCamp.org), as well as AI tools.
@author Caleb Gira
"""

# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello, World!"

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Directory where images will be saved
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'received_images')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload directory: {UPLOAD_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', mathods=['POST'])
def upload_file():
    """
    Handles file uploads. Expects a file under the 'image' key in POST request
    """
    if 'image' not in request.files:
        logging.warning("No selected file in request.")
        return jsonify({"error": "No image file part"}), 400

    file = request.files['image']

    if file.filename == '':
        logging.warning("No selected file in request.")
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            logging.info(f"Successfully received and saved: {filename}")
            return jsonify({"message": "File uploaded successfully", "filename": filename}), 200
        except Exception as e:
            logging.error(f"Error saving file {filename}: {e}")
            return jsonify({"error": f"Failed to save file: {e}"}), 500
    else:
        logging.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/')
def index():
    """Homepage to confirm server is running."""
    return "Image Receiver Server is running. Send POST requests to /upload."

if __name__ == '__main__':
    logging.info(f"Starting Image Receiver Server on http://0.0.0.0:5000")
    logging.info(f"Images will be saved to: {UPLOAD_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=False)