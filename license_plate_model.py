import cv2
import logging
import re
import os
import json

try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.warning("TensorFlow not found. Model loading/prediction will be disabled.")

logger = logging.getLogger(__name__)

_DETECTION_MODEL = None
_RECOGNITION_MODEL = None
_LABEL_MAP = {}

def enhance_plate_image(image):
    processed_images = []

    try:
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted image to grayscale.")

        # 1.5 Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)) # Adjust grid size if needed
        gray_enhanced = clahe.apply(gray)
        logger.debug("Applied CLAHE to grayscale image.")

        # 2. Resizing (on enhanced gray)
        scale_factor = 1.5 # Reduced scaling factor slightly
        width = int(gray_enhanced.shape[1] * scale_factor)
        height = int(gray_enhanced.shape[0] * scale_factor)
        # Use INTER_LINEAR for potentially smoother results than CUBIC
        resized = cv2.resize(gray_enhanced, (width, height), interpolation=cv2.INTER_LINEAR)

        # 3. Noise Reduction (on resized)
        denoised = cv2.medianBlur(resized, 3)

        # 4. Thresholding Variations (applied to denoised)
        #    a) Simple Otsu Thresholding
        _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(otsu_thresh)

        #    b) Adaptive Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
        processed_images.append(adaptive_thresh)

        # 5. Morphological Operations (Optional, applied to thresholds)
        # ... (morph ops as before) ...

        # 6. Add CLAHE enhanced grayscale (denoised version)
        processed_images.append(denoised) # Add denoised CLAHE version

        logger.debug(f"Generated {len(processed_images)} enhanced versions of the plate.")
        return processed_images
    except Exception as e:
        logger.error(f"Error enhancing plate image: {e}")
        return None

def get_confidence_score(text=None, source="unknown"):
    # ... (failure string handling, base confidence logic as before) ...

    # --- Adjustments Based on Text Characteristics --- #
    # ... (length_score, alpha_score logic as before) ...

    # --- Pattern Matching --- #
    pattern_score = -15.0 # Start with penalty
    is_known_format = False
    # Standard Indian format
    if re.match(r'^[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,2}\s?[0-9]{1,4}$', text):
        pattern_score = 15.0
        is_known_format = True
    # UK Format
    elif re.match(r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$', text):
         pattern_score = 15.0
         is_known_format = True
    # Basic alphanumeric check
    elif re.match(r'^[A-Z0-9\s]{6,12}$', text):
        pattern_score = 0.0 # Neutral if general format okay
        # is_known_format remains False

    # --- OCR Confusion Penalty --- #
    # Penalize if it doesn't match a known format AND contains common confusions
    confusion_penalty = 0.0
    if not is_known_format and source == "tesseract": # Only penalize Tesseract results
        confusions = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2'}
        text_no_space = re.sub(r'\s', '', text)
        has_confusion = False
        for char in text_no_space:
            # Check if a character has a common confusion counterpart also present
            if char in confusions and confusions[char] in text_no_space:
                has_confusion = True
                break
            # Check reverse confusion
            for k, v in confusions.items():
                 if char == v and k in text_no_space:
                      has_confusion = True
                      break
            if has_confusion: break
            
        if has_confusion:
            confusion_penalty = -20.0 # Apply significant penalty for likely OCR confusion
            logger.debug(f"Applying OCR confusion penalty ({confusion_penalty}) for text: '{text}'")

    # --- Combine Scores --- #
    final_confidence = base_confidence + length_score + alpha_score + pattern_score + confusion_penalty
    final_confidence = max(10.0, min(final_confidence, 96.0))

    logger.debug(f"Confidence - Base: {base_confidence:.1f}, Length: {length_score:.1f}, Alpha: {alpha_score:.1f}, Pattern: {pattern_score:.1f}, Confusion: {confusion_penalty:.1f} -> Final: {final_confidence:.2f}%")
    return final_confidence

def load_model(model_dir='model'):
    """
    Load detection and recognition models (if TensorFlow is available).
    Returns placeholder functions or loaded models.
    If placeholder files (empty sequential models) are found, returns None for the model component.
    """
    global _DETECTION_MODEL, _RECOGNITION_MODEL, _LABEL_MAP
    detection_model = None
    recognition_model = None
    label_map = {}

    # Define file paths
    detection_model_path = os.path.join(model_dir, 'plate_detection_model.h5')
    recognition_model_path = os.path.join(model_dir, 'plate_recognition_model.h5')
    label_map_path = os.path.join(model_dir, 'label_map.json')

    # Check if TensorFlow is available before attempting to load models
    if tf is None:
        logger.error("TensorFlow is not installed. Cannot load H5 models.")
        return {"detection": None, "recognition": None, "label_map": {}}

    # --- Load Detection Model ---
    if os.path.exists(detection_model_path):
        try:
            loaded_model = tf.keras.models.load_model(detection_model_path, compile=False)
            if isinstance(loaded_model, tf.keras.Sequential) and not loaded_model.layers:
                logger.warning("Detection model file found, but it's an empty placeholder. Detection model set to None.")
                detection_model = None
            else:
                detection_model = loaded_model
                logger.info(f"Detection model loaded from {detection_model_path}")
        except Exception as e:
            logger.error(f"Error loading detection model from {detection_model_path}: {e}")
            detection_model = None
    else:
        logger.warning(f"Detection model file not found at {detection_model_path}")
        detection_model = None

    # --- Load Recognition Model ---
    if os.path.exists(recognition_model_path):
        try:
            loaded_model = tf.keras.models.load_model(recognition_model_path, compile=False)
            if isinstance(loaded_model, tf.keras.Sequential) and not loaded_model.layers:
                logger.warning("Recognition model file found, but it's an empty placeholder. Recognition model set to None.")
                recognition_model = None
            else:
                recognition_model = loaded_model
                logger.info(f"Recognition model loaded from {recognition_model_path}")
        except Exception as e:
            logger.error(f"Error loading recognition model from {recognition_model_path}: {e}")
            recognition_model = None
    else:
        logger.warning(f"Recognition model file not found at {recognition_model_path}")
        recognition_model = None

    # --- Load Label Map ---
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
                if isinstance(label_map, dict) and label_map:
                     logger.info(f"Label map loaded from {label_map_path} ({len(label_map)} entries)")
                else:
                     logger.error(f"Label map file {label_map_path} is empty or not a valid JSON object. Using empty map.")
                     label_map = {}
        except Exception as e:
            logger.error(f"Error loading or parsing label map from {label_map_path}: {e}")
            label_map = {}
    else:
        logger.warning(f"Label map file not found at {label_map_path}. Using empty map.")
        label_map = {}

    # Update globals (optional, but kept for consistency with previous structure)
    _DETECTION_MODEL = detection_model
    _RECOGNITION_MODEL = recognition_model
    _LABEL_MAP = label_map

    return {
        "detection": detection_model,
        "recognition": recognition_model,
        "label_map": label_map
    }