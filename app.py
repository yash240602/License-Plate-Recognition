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

# Configure logging (ensure this runs before any logger calls)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Decorator definitions (REMOVED - Moved to license_plate_model.py)
# def log_execution_time(func):
# ...
# def log_exceptions(func):
# ...

# Import necessary functions (after logging is configured)
try:
    # Apply decorators to key model functions if not already done
    from license_plate_model import (
        load_model, detect_license_plate, recognize_text,
        get_confidence_score, handle_edit_text_feedback,
        apply_feedback_correction, # save_feedback_data, load_feedback_data, <-- No longer imported here
        enhance_plate_image, process_video_stream,
        build_transfer_learning_model, train_character_recognition_model,
        preprocess_low_resolution
        # Decorators are now defined within license_plate_model.py
        # log_execution_time as model_log_time, # Import decorators if defined in model file
        # log_exceptions as model_log_exc
    )
    # No need to re-apply decorators here
    logger.info("Successfully imported model functions")
except Exception as e:
    logger.critical(f"FATAL: Error importing model functions: {str(e)}")
    logger.critical(traceback.format_exc())
    # Consider exiting if core model functions can't be imported
    raise

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# History storage
HISTORY_FILE = 'processing_history.json'
FEEDBACK_FILE = 'recognition_feedback.json'

try:
    # Initialize model
    logger.info("Loading model...")
    model = load_model()
    if model["detection"] is None or model["recognition"] is None:
        raise Exception("Model files not found in 'model/' directory")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    logger.warning("Consider downloading model files to the 'model/' directory")
    # Instructions to download models could be added here

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
                    # Clean up the saved file if loading fails
                    try: os.remove(file_path)
                    except: pass
                    return jsonify({'error': 'Failed to read image file. It might be corrupted or not a supported format.'}), 400

                logger.info(f"Image loaded successfully: {unique_filename}")

                # Preprocess for low resolution or common issues (Optional but good)
                # image = preprocess_low_resolution(image) # Add if needed

                # Load models (cached or loaded on demand)
                model_data = load_model() # Assumes this handles loading/caching

                # 1. Detect license plate
                logger.debug("Detecting license plate...")
                boxes, scores = detect_license_plate(image, model_data)

                if not boxes:
                    logger.warning(f"No license plate detected in image: {unique_filename}")
                    # Optionally save the original image for review
                    return jsonify({'error': 'No license plate detected'}), 400

                # For simplicity, process only the highest score detection
                best_box = boxes[0]
                best_score = scores[0] if scores else 0.0
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

                # Save the *extracted* plate image (before extensive OCR preprocessing)
                plate_img_filename = f"plate_{unique_filename}"
                plate_img_path = os.path.join(app.config['RESULT_FOLDER'], plate_img_filename)
                cv2.imwrite(plate_img_path, plate_img)
                logger.debug(f"Saved extracted plate image to: {plate_img_path}")

                # 2. Recognize text using the improved function
                logger.debug("Recognizing text...")
                plate_text = recognize_text(plate_img, model_data)
                plate_text = plate_text if plate_text else "Recognition Failed"
                logger.info(f"Recognized text: '{plate_text}'")

                # 3. Get confidence score (based on text or model score if available)
                # Assuming get_confidence_score is adapted or uses detection score
                confidence = get_confidence_score(plate_text) # Or pass best_score? Needs review.
                logger.info(f"Confidence score: {confidence:.2f}")

                # 4. Save the original image with bounding box overlay
                output_img = image.copy()
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{plate_text} ({confidence:.1f}%)"
                cv2.putText(output_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                processed_filename = f"processed_{unique_filename}"
                output_path = os.path.join(app.config['RESULT_FOLDER'], processed_filename)
                cv2.imwrite(output_path, output_img)
                logger.debug(f"Saved processed image with overlay to: {output_path}")

                # 5. Add to processing history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history = get_processing_history()
                needs_feedback_flag = confidence < 85.0 # Lower threshold for feedback prompt

                history.append({
                    'id': unique_id, # Add unique ID
                    'timestamp': timestamp,
                    'original_image': url_for('uploaded_file', filename=unique_filename), # Link to original upload
                    'plate_image': url_for('result_file', filename=plate_img_filename),
                    'processed_image': url_for('result_file', filename=processed_filename),
                    'plate_text': plate_text,
                    'confidence': float(confidence),
                    'needs_feedback': needs_feedback_flag
                })
                save_processing_history(history)
                logger.info(f"Processing history saved for ID: {unique_id}")

                # 6. Return JSON response for the frontend
                return jsonify({
                    'success': True,
                    'result_id': unique_id,
                    'original_image': url_for('uploaded_file', filename=unique_filename),
                    'plate_image': url_for('result_file', filename=plate_img_filename),
                    'processed_image': url_for('result_file', filename=processed_filename),
                    'plate_text': plate_text,
                    'confidence': float(confidence),
                    'needs_feedback': needs_feedback_flag
                })

            except Exception as e:
                logger.error(f"Error processing image '{unique_filename}': {str(e)}")
                logger.error(traceback.format_exc())
                # Clean up saved files on error
                try: os.remove(file_path) 
                except: pass
                return jsonify({'error': f'An error occurred during image processing: {str(e)}'}), 500
        else:
            logger.error(f"Upload failed: File type not allowed for '{file.filename}'")
            return jsonify({'error': 'File type not allowed'}), 400

    # Should not happen for POST, but handle anyway
    logger.warning("Upload endpoint accessed with method other than POST")
    return jsonify({'error': 'Method not allowed'}), 405

# Add route for serving result files
@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/history')
def view_history():
    logger.info("History page requested")
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    
    return render_template('history.html', history=history)

@app.route('/correct', methods=['POST'])
def correct_text():
    """Handles user feedback submission."""
    logger.info("Text correction feedback received")
    data = request.get_json()
    if not data:
        logger.error("Correction request has no JSON data")
        return jsonify({'success': False, 'error': 'Invalid request format'}), 400

    image_path = data.get('image_path') # Keep for logging/context if needed
    original_text = data.get('original_text')
    corrected_text = data.get('corrected_text')
    was_correct = data.get('was_correct', False)

    if original_text is None or (not was_correct and corrected_text is None):
        logger.error(f"Missing text data in correction request: {data}")
        return jsonify({'success': False, 'error': 'Missing text data'}), 400

    # Call feedback handler WITHOUT image_path
    success = handle_edit_text_feedback(
        original_text=original_text,
        corrected_text=corrected_text if not was_correct else original_text,
        was_correct=was_correct
    )

    if success:
        log_context = f"image related to: {image_path}" if image_path else "feedback submission"
        logger.info(f"Feedback processed successfully for {log_context}")
        return jsonify({'success': True})
    else:
        log_context = f"image related to: {image_path}" if image_path else "feedback submission"
        logger.error(f"Failed to save feedback for {log_context}")
        return jsonify({'success': False, 'error': 'Failed to save feedback'}), 500

@app.route('/test')
def test():
    return "<h1>License Plate Detection API is running!</h1>"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def create_app():
    """Function to create and configure the Flask app for Waitress"""
    return app

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    PORT = 8080  # Use standard Waitress default port
    logger.info(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    print(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=PORT, threads=4)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        print(f"Error starting server: {str(e)}") 