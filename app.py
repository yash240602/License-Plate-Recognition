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

# --- Model Function Imports --- #
# Import only the functions actually defined in the simplified license_plate_model.py
try:
    from license_plate_model import (
        load_model,
        detect_license_plate, # Main detection (might use TF if available)
        detect_license_plate_cv, # OpenCV fallback detection
        enhance_plate_image,
        recognize_text,
        get_confidence_score,
        apply_feedback_correction,
        handle_edit_text_feedback,
        preprocess_low_resolution
    )
    MODEL_FUNCTIONS_LOADED = True
    logger.info("Successfully imported model functions.")
except ImportError as import_err:
    logger.critical(f"FATAL: Error importing model functions: {import_err}")
    logger.critical(traceback.format_exc())
    MODEL_FUNCTIONS_LOADED = False
    # Define dummy functions if import fails, so the app can at least start
    def load_model(*args, **kwargs):
        logger.error("Dummy load_model called due to import error.")
        return {"detection": None, "recognition": None, "label_map": {}}
    def detect_license_plate(*args, **kwargs):
        logger.error("Dummy detect_license_plate called.")
        return None, 0.0
    def detect_license_plate_cv(*args, **kwargs):
        logger.error("Dummy detect_license_plate_cv called.")
        return None
    def enhance_plate_image(*args, **kwargs):
        logger.error("Dummy enhance_plate_image called.")
        return None
    def recognize_text(*args, **kwargs):
        logger.error("Dummy recognize_text called.")
        return "IMPORT_ERROR", 0.0, "import_error"
    def get_confidence_score(*args, **kwargs):
        logger.error("Dummy get_confidence_score called.")
        return 0.0
    def apply_feedback_correction(text, *args, **kwargs):
        logger.error("Dummy apply_feedback_correction called.")
        return text, False, {"reason": "Import Error"}
    def handle_edit_text_feedback(*args, **kwargs):
        logger.error("Dummy handle_edit_text_feedback called.")
        pass
    def preprocess_low_resolution(*args, **kwargs):
        logger.error("Dummy preprocess_low_resolution called.")
        return None

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

# Initialize model variable
model_data = None 

try:
    logger.info("Attempting to load models...")
    # load_model handles TF check internally
    model_data = load_model() 
    
    # Check the actual status based on load_model's logic
    if model_data.get("detection") is None and model_data.get("recognition") is None:
         # Only log warning if TF was actually available but models failed
         # load_model already logs if TF is missing, so just log general status here
         logger.warning("Placeholder models or no models loaded. Using CV/Tesseract only.")
    else:
        logger.info("Deep Learning models appear to be loaded.")

except Exception as e:
    # Catch any unexpected error during the loading process itself
    logger.error(f"Unexpected error during model loading attempt: {str(e)}")
    logger.error(traceback.format_exc())
    model_data = {"detection": None, "recognition": None, "label_map": {}} # Ensure safe default
    logger.warning("Proceeding without DL models due to loading error.")

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
    if not MODEL_FUNCTIONS_LOADED:
        return jsonify({'error': 'Server configuration error: Model functions failed to load.'}), 500

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
                # Detect function should return a tuple (boxes, scores) or None
                detection_result = detect_license_plate(image, model_data)

                # --- Check detection result BEFORE unpacking --- #
                if detection_result is None:
                    # If TF model failed, try OpenCV fallback
                    logger.warning("TF detection failed or returned None, trying OpenCV fallback...")
                    cv_boxes, cv_scores = detect_license_plate_cv(image)
                    if not cv_boxes:
                        logger.error(f"Both TF and CV detection failed for image: {unique_filename}")
                        # Clean up uploaded file
                        try: os.remove(file_path)
                        except: pass 
                        return jsonify({'error': 'No license plate detected by either method'}), 400
                    else:
                        logger.info(f"Plate detected using CV fallback: {cv_boxes}")
                        boxes = cv_boxes
                        scores = cv_scores
                else:
                    # Unpack the result if it's not None
                    boxes, scores = detection_result
                    if not boxes:
                        # TF detection ran but found nothing, try CV
                        logger.warning(f"TF detector found no boxes, trying OpenCV fallback...")
                        cv_boxes, cv_scores = detect_license_plate_cv(image)
                        if not cv_boxes:
                            logger.error(f"Both TF and CV detection failed (TF found no boxes): {unique_filename}")
                            try: os.remove(file_path)
                            except: pass
                            return jsonify({'error': 'No license plate detected'}), 400
                        else:
                            logger.info(f"Plate detected using CV fallback (after TF found none): {cv_boxes}")
                            boxes = cv_boxes
                            scores = cv_scores

                # --- Proceed with the detected boxes --- #
                # (Ensure boxes is not empty after the checks above)
                if not boxes:
                    # This case should ideally be covered by the checks above, but as a safeguard:
                    logger.error(f"Logic error: No boxes found after detection checks for {unique_filename}")
                    return jsonify({'error': 'Internal server error during detection'}), 500

                # For simplicity, process only the first detected box (highest score if TF worked)
                best_box = boxes[0]
                best_score = scores[0] if scores else 0.5 # Use 0.5 if only CV detection worked
                
                # Fix: Handle the case where best_box might contain arrays 
                try:
                    # Try the original approach first
                    y1, x1, y2, x2 = map(int, best_box) 
                except TypeError:
                    # If best_box contains arrays, extract the values properly
                    y1 = int(best_box[0]) if not isinstance(best_box[0], (list, np.ndarray)) else int(best_box[0].item())
                    x1 = int(best_box[1]) if not isinstance(best_box[1], (list, np.ndarray)) else int(best_box[1].item())
                    y2 = int(best_box[2]) if not isinstance(best_box[2], (list, np.ndarray)) else int(best_box[2].item())
                    x2 = int(best_box[3]) if not isinstance(best_box[3], (list, np.ndarray)) else int(best_box[3].item())
                
                logger.info(f"Processing detected plate with score {best_score:.2f} at box: [{y1},{x1},{y2},{x2}]")

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
                # recognize_text now returns (text, source)
                plate_text, confidence = recognize_text(plate_img)
                plate_text = plate_text if plate_text else "Error" # Ensure not None for display
                logger.info(f"Recognized text: '{plate_text}' (Confidence: {confidence})")

                # 3. Get confidence score 
                logger.info(f"Confidence score: {confidence:.2f}%")

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
                # Ensure each entry has required fields to avoid template errors
                for entry in history:
                    # Ensure all entries have processed_image field
                    if 'processed_image' not in entry:
                        # Try to construct it from other fields
                        if 'original_image' in entry:
                            # Strip url_for prefix if present
                            original = entry['original_image']
                            if original.startswith('/static/'):
                                original = original[8:]  # Remove '/static/' prefix
                            elif original.startswith('/'):
                                original = original[1:]  # Remove leading slash
                            
                            entry['processed_image'] = f"results/processed_{os.path.basename(original)}"
                        else:
                            entry['processed_image'] = 'placeholder.png'
                    
                    # Ensure numeric confidence values
                    if 'confidence' in entry:
                        try:
                            entry['confidence'] = float(entry['confidence'])
                        except (ValueError, TypeError):
                            entry['confidence'] = 0.0
                    else:
                        entry['confidence'] = 0.0
                    
                    # Ensure plate_text is a string
                    if 'plate_text' not in entry or entry['plate_text'] is None:
                        entry['plate_text'] = "Unknown"
                    
                    # Add timestamp if missing
                    if 'timestamp' not in entry:
                        entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            except json.JSONDecodeError:
                logger.error("Error parsing history JSON file")
                history = []
    
    return render_template('history.html', history=history)

@app.route('/correct', methods=['POST'])
def correct_text():
    """
    Endpoint for receiving text corrections from the user.
    Updates the recognition history and saves feedback data for future improvements.
    """
    try:
        data = request.json
        if not data:
            logger.error("Empty data in correction request")
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Extract correction data
        result_id = data.get('id', '')
        original = data.get('original', '')
        corrected = data.get('corrected', '')
        
        # Validate data
        if not original:
            logger.error("Missing original text in correction request")
            return jsonify({'success': False, 'error': 'Missing original text'}), 400
            
        # Check if this is a confirmation (original == corrected) or a correction
        is_confirmation = original == corrected
        
        # Log the feedback
        if is_confirmation:
            logger.info(f"User confirmed recognition: '{original}'")
        else:
            logger.info(f"Text correction: '{original}' -> '{corrected}'")
        
        # Find history entry if ID provided
        updated = False
        if result_id:
            history = get_processing_history()
            for entry in history:
                if entry.get('id') == result_id:
                    # Update history with correction
                    if not is_confirmation:
                        entry['plate_text'] = corrected
                        entry['corrected'] = True
                        entry['original_text'] = original
                    else:
                        entry['confirmed'] = True
                    
                    save_processing_history(history)
                    updated = True
                    break
        
        # Save feedback for recognition improvement
        try:
            # Call the proper feedback handler
            success = handle_edit_text_feedback(
                original_text=original,
                corrected_text=corrected,
                was_correct=is_confirmation
            )
            
            if not success:
                logger.warning(f"Failed to save feedback data: {original} -> {corrected}")
                return jsonify({'success': False, 'error': 'Failed to save feedback'}), 500
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Error saving feedback: {str(e)}'}), 500
                
        return jsonify({'success': True, 'updated': updated})
        
    except Exception as e:
        logger.error(f"Error processing correction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test')
def test():
    return "<h1>License Plate Detection API is running!</h1>"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def create_app():
    """Function to create and configure the Flask app for Waitress"""
    return app

# KEEP the Flask development server __main__ block
if __name__ == '__main__':
    # Ensure directories exist (redundant but safe)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    # Note: Use debug=True for local debugging
    logger.info("Starting Flask development server...")
    app.run(host='0.0.0.0', port=8080, debug=True) 