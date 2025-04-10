import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
import logging
import traceback
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# History storage
HISTORY_FILE = 'processing_history.json'
FEEDBACK_FILE = 'recognition_feedback.json'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_processing_history():
    """Load processing history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_processing_history(history):
    """Save processing history to JSON file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def save_to_history(image_path, result_path, plate_text, confidence):
    history = get_processing_history()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        'timestamp': timestamp,
        'original_image': image_path,
        'processed_image': result_path,
        'plate_text': plate_text,
        'confidence': confidence
    }
    
    history.append(entry)
    save_processing_history(history)
    return True

def detect_license_plate(image):
    """Simplified license plate detection using OpenCV"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    potential_plates = []
    
    for contour in contours:
        # Approximate the contour as a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's a rectangle (4 corners)
        if len(approx) == 4:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(approx)
            
            # Check aspect ratio (license plates are typically wider than tall)
            aspect_ratio = w / float(h)
            
            if 1.5 <= aspect_ratio <= 6.0:
                potential_plates.append([y, x, y+h, x+w])
    
    # If no plates found, return a default region
    if not potential_plates:
        h, w = image.shape[:2]
        # Return a region in the center of the image
        center_y, center_x = h // 2, w // 2
        return [[max(0, center_y - h//5), max(0, center_x - w//3), 
                 min(h, center_y + h//5), min(w, center_x + w//3)]]
    
    return potential_plates

def recognize_text(plate_image):
    """Simplified text recognition. Returns placeholder text"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # For demonstration - return a placeholder
        return "DEMO-PLATE-123"
    except Exception as e:
        logger.error(f"Error in recognize_text: {str(e)}")
        return "Recognition Failed"

def enhance_plate_image(plate_img):
    """Enhance plate image for better OCR"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while preserving edges
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Invert if needed (black text on white background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
            
        # Convert back to BGR
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing plate image: {str(e)}")
        return plate_img  # Return original if enhancement fails

@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, processes the image, and returns results."""
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.error("Upload request missing 'file' part")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Upload request has empty filename")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Use a unique filename to avoid collisions
            unique_id = str(uuid.uuid4())[:8]
            base, ext = os.path.splitext(filename)
            unique_filename = f"{base}_{unique_id}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            logger.info(f"Saving uploaded file to: {file_path}")
            file.save(file_path)

            # Process the image
            try:
                # Load the image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    logger.error(f"Failed to load image using OpenCV: {file_path}")
                    try: os.remove(file_path)
                    except: pass
                    return jsonify({'error': 'Failed to read image file'}), 400

                logger.info(f"Image loaded successfully: {unique_filename}")

                # 1. Detect license plate
                logger.debug("Detecting license plate...")
                boxes = detect_license_plate(image)

                if not boxes:
                    logger.warning(f"No license plate detected in image: {unique_filename}")
                    return jsonify({'error': 'No license plate detected'}), 400

                # Process only the highest score detection
                best_box = boxes[0]
                best_score = 0.6  # Placeholder confidence score
                y1, x1, y2, x2 = best_box
                logger.info(f"Plate detected with score {best_score:.2f} at box: {best_box}")

                # Ensure box coordinates are valid before slicing
                h, w = image.shape[:2]
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(h, y2), min(w, x2)
                if y1 >= y2 or x1 >= x2:
                     logger.error(f"Invalid box dimensions after clipping: {best_box}")
                     return jsonify({'error': 'Internal error during plate extraction'}), 500

                plate_img = image[y1:y2, x1:x2]
                if plate_img.size == 0:
                    logger.error(f"Extracted plate image is empty for box: {best_box}")
                    return jsonify({'error': 'Internal error extracting plate region'}), 500

                # Save the extracted plate image
                plate_img_filename = f"plate_{unique_filename}"
                plate_img_path = os.path.join(app.config['RESULT_FOLDER'], plate_img_filename)
                cv2.imwrite(plate_img_path, plate_img)
                logger.debug(f"Saved extracted plate image to: {plate_img_path}")

                # Enhance plate image
                enhanced_plate = enhance_plate_image(plate_img)
                
                # Save the processed image
                processed_filename = f"processed_{unique_filename}"
                processed_path = os.path.join(app.config['RESULT_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, enhanced_plate)

                # 2. Recognize text
                logger.debug("Recognizing text...")
                plate_text = recognize_text(enhanced_plate)
                
                # Static confidence score for demo
                confidence_score = 85.0

                # 3. Save to history
                relative_upload_path = os.path.join('uploads', unique_filename)
                relative_result_path = os.path.join('results', processed_filename)
                
                success = save_to_history(
                    relative_upload_path,
                    relative_result_path,
                    plate_text,
                    confidence_score
                )
                
                if not success:
                    logger.warning("Failed to save processing history")

                # 4. Return results
                return jsonify({
                    'success': True,
                    'message': 'Image processed successfully',
                    'details': {
                        'id': unique_id,
                        'original_image': url_for('static', filename=relative_upload_path),
                        'processed_image': url_for('static', filename=relative_result_path),
                        'plate_text': plate_text,
                        'confidence': confidence_score
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': str(e)}), 500
                
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/history')
def view_history():
    history = get_processing_history()
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/correct', methods=['POST'])
def correct_text():
    """Handle text correction feedback from user"""
    data = request.json
    
    if not data or not all(k in data for k in ('id', 'original_text', 'corrected_text')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    logger.info("Text correction feedback received")
    
    # Simply log the correction for now
    logger.info(f"Correction: '{data['original_text']}' → '{data['corrected_text']}'")
    
    return jsonify({
        'success': True,
        'message': 'Feedback processed successfully'
    })

@app.route('/test')
def test():
    return "License Plate Recognition API is running!"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    from waitress import serve
    logger.info("Starting Waitress server on http://127.0.0.1:8080")
    print("Starting Waitress server on http://127.0.0.1:8080")
    print("You can access the application at:")
    print("  - Local: http://127.0.0.1:8080")
    print("  - Network: http://<your-ip-address>:8080")
    print("Press Ctrl+C to stop the server")
    serve(app, host='0.0.0.0', port=8080, threads=4) 