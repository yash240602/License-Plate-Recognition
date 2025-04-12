# License Plate Recognition Web App

This project demonstrates a web application for detecting and recognizing license plates from uploaded images using computer vision and OCR techniques.

![License Plate Recognition Demo](docs/license_plate_demo.jpg)

## Key Features & Technologies

*   **Web Interface:** Simple, clean UI built with **Flask**, HTML, CSS, and basic JavaScript.
*   **Image Upload:** Easy drag-and-drop or file selection for image input.
*   **Plate Detection:** Utilizes **OpenCV** for image preprocessing (grayscale, blur, adaptive thresholding, CLAHE) and contour detection to locate potential license plate regions.
*   **OCR Engine:** Employs **Tesseract OCR** (via `pytesseract`) to extract text from the detected and enhanced plate images.
*   **Confidence Scoring:** Provides a heuristic-based confidence score evaluating the quality and format of the OCR result.
*   **User Feedback Loop:** Allows users to correct inaccurate OCR results, storing feedback (original vs. corrected text) for potential future improvements (Note: advanced retraining is not implemented in this version).
*   **Backend Server:** Runs on **Waitress**, a production-quality WSGI server.

## How It Works (High-Level)

1.  **Upload:** User uploads an image via the web interface.
2.  **Detection:** The Flask backend receives the image. OpenCV functions process the image to find the most likely rectangular license plate candidate.
3.  **Enhancement & OCR:** The detected plate region is cropped and enhanced using various OpenCV techniques (CLAHE, resizing, thresholding) to maximize OCR accuracy. Tesseract attempts to read the text from these enhanced versions.
4.  **Result Selection & Scoring:** The best Tesseract result (based on pattern matching) is selected, and a confidence score is calculated.
5.  **Display:** The original image, plate crop, recognized text, and confidence score are displayed to the user.
6.  **Feedback:** Users can submit corrections, which are logged.

## Running Locally (Step-by-Step)

Want to run this on your machine? Follow these steps:

**Prerequisites:**

1.  **Python:** Version **3.11.x** is recommended (as the included `tensorflow_env` was built with it). Download from [python.org](https://www.python.org/downloads/) if needed. *(Ensure Python is added to your system's PATH during installation)*.
2.  **Git (Optional but Recommended):** For easy cloning. Install from [git-scm.com](https://git-scm.com/).
3.  **Tesseract OCR Engine:** The core OCR component.
    *   **macOS:** Open Terminal and run: `brew install tesseract`
    *   **Ubuntu/Debian Linux:** Open Terminal and run: `sudo apt update && sudo apt install tesseract-ocr`
    *   **Windows:** Download an installer (e.g., from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)). **Crucially, ensure you add the Tesseract installation directory to your system's PATH environment variable** so the application can find `tesseract.exe`.

**Setup Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yash240602/License-Plate-Recognition.git
    cd License-Plate-Recognition
    ```
    *(Alternatively, download the ZIP from GitHub and extract it.)*

2.  **Create & Activate Virtual Environment:**
    ```bash
    # Create the environment (using Python 3.11)
    python3.11 -m venv tensorflow_env

    # Activate it
    # macOS/Linux:
    source tensorflow_env/bin/activate
    # Windows (CMD/PowerShell):
    # tensorflow_env\Scripts\activate
    ```
    *(Your terminal prompt should now start with `(tensorflow_env)`)*

3.  **Install Dependencies:**
    ```bash
    # Upgrade pip first
    pip install --upgrade pip

    # Install Python packages
    pip install -r requirements.txt

    # If on Apple Silicon (M1/M2/M3) Mac, install Metal plugin for TensorFlow:
    # pip install tensorflow-metal 
    # (Note: TF isn't used for prediction here, but metal helps avoid CPU fallback warnings if TF loads)
    ```

**Run the Application:**

1.  **Start the Server:**
    ```bash
    python run_server.py
    ```

2.  **Open in Browser:** Navigate to `http://localhost:8080`

Enjoy testing the license plate recognition!

## Project Structure

```
├── app.py                  # Main Flask application routes and logic
├── license_plate_model.py  # Core detection, OCR, enhancement functions
├── run_server.py           # Script to run the app using Waitress
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── LICENSE                 # Project license
├── model/                  # Directory for placeholder model files (.h5, .json)
├── static/
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript for frontend interaction
│   ├── uploads/            # User-uploaded images stored here
│   └── results/            # Processed images (plate crops, overlays) stored here
├── templates/
│   ├── index.html          # Main application page template
│   └── history.html        # (If implemented) Page to view past results
├── docs/                   # Project documentation/images
├── .gitignore              # Specifies intentionally untracked files (like venvs)
└── recognition_feedback.json # Stores user feedback (created on first feedback)
└── processing_history.json   # Stores results history (created on first upload)
```

## Limitations in this Version

*   **OCR Accuracy:** Relies heavily on Tesseract OCR. Accuracy varies with image quality, plate fonts, angles, and lighting.
*   **No Trained ML Model:** Uses placeholder `.h5` files. Detection is CV-based (OpenCV), and recognition is Tesseract-based. Real TensorFlow models would significantly improve robustness and accuracy.
*   **Basic Detection:** Contour-based detection can be unreliable.
*   **Simplified Feedback:** Feedback is logged but not used for automated retraining.

## Contributing

Contributions and improvements are welcome! Please feel free to fork and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE).
