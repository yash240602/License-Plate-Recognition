# License Plate Recognition

This project is a simple web application that uses computer vision techniques to detect and attempt to recognize text on license plates in uploaded images.

![License Plate Recognition Demo](docs/license_plate_demo.jpg)

## Features

*   **Image Upload:** Upload car images via drag-and-drop or file selection.
*   **Plate Detection:** Attempts to automatically locate the license plate region within the image using basic OpenCV techniques (contour finding).
*   **OCR Attempt:** Uses Tesseract OCR to try and read the text from the detected plate region.
*   **Feedback Mechanism:** Allows users to correct misread plates. This feedback is stored and can potentially be used to improve future results (though advanced learning isn't implemented in this simplified version).
*   **Result Display:** Shows the original image, the detected plate crop, the recognized text, and a confidence score.

## How It Works (Simplified)

1.  **Upload:** You provide an image.
2.  **Detection:** Basic image processing (grayscale, blur, thresholding, contour detection) is used to find rectangular areas that might be license plates.
3.  **Cropping:** The most likely candidate region is cropped.
4.  **Enhancement:** The cropped image undergoes processing (CLAHE, resizing, thresholding variations) to improve contrast and clarity for OCR.
5.  **OCR:** Tesseract OCR is run on the enhanced image(s).
6.  **Result Selection:** The best result from Tesseract (based on matching common plate patterns) is chosen.
7.  **Confidence Score:** A score is calculated based on the OCR result quality and pattern matching.
8.  **Feedback (Optional):** If the result is wrong, you can correct it, and this pair (original OCR vs. corrected) is stored.

## Running Locally (Easy Setup)

Getting this running on your own machine is straightforward!

**Prerequisites:**

1.  **Python:** You'll need Python installed (Version 3.9, 3.10, or 3.11 recommended). If you don't have it, grab it from [python.org](https://www.python.org/downloads/). Make sure to check "Add Python to PATH" during installation on Windows.
2.  **Git (Optional):** Useful for cloning the project, but you can also download the code as a ZIP file from GitHub.
3.  **Tesseract OCR:** This is essential for reading the text.
    *   **macOS:** `brew install tesseract`
    *   **Ubuntu/Debian Linux:** `sudo apt update && sudo apt install tesseract-ocr`
    *   **Windows:** Download the installer from the official [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. Make sure `tesseract.exe` is added to your system's PATH.

**Setup Steps:**

1.  **Get the Code:**
    ```bash
    # Clone the repository
    git clone https://github.com/yash240602/License-Plate-Recognition.git
    # Navigate into the project directory
    cd License-Plate-Recognition
    ```
    *(Or download and unzip the project)*

2.  **Create a Virtual Environment:** (Keeps dependencies tidy!)
    ```bash
    python3 -m venv tensorflow_env 
    ```
    *Note: We name it `tensorflow_env` because it needs specific Python versions compatible with TensorFlow, even though TF isn't actively used for prediction in this simplified version.*

3.  **Activate the Environment:**
    *   **macOS/Linux:** `source tensorflow_env/bin/activate`
    *   **Windows (Git Bash/WSL):** `source tensorflow_env/bin/activate`
    *   **Windows (CMD/PowerShell):** `tensorflow_env\Scripts\activate`
    *(You should see `(tensorflow_env)` at the start of your terminal prompt)*

4.  **Install Required Packages:**
    ```bash
    # Make sure pip is up-to-date
    pip install --upgrade pip 
    # Install from the requirements file
    pip install -r requirements.txt
    ```
    *If you are on an Apple Silicon Mac (M1/M2/M3), you also need TensorFlow acceleration:* 
    `pip install tensorflow-metal`

**Running the App:**

1.  **Start the Server:**
    ```bash
    python run_server.py
    ```

2.  **Access in Browser:** Open your web browser and go to:
    `http://localhost:8080`

You should see the License Plate Recognition interface! Upload an image to test it out.

## Tech Overview

*   **Backend:** Flask, Waitress
*   **Core Logic:** Python
*   **Image Processing:** OpenCV, Pillow
*   **OCR:** PyTesseract (wrapper for Tesseract)
*   **Environment:** Python 3.9-3.11 recommended

## Limitations in this Version

*   **No Deep Learning Model:** This simplified version relies primarily on OpenCV for detection and Tesseract for OCR. The placeholder model files (`.h5`) are included but contain no actual trained network. Accuracy will vary greatly depending on image quality and Tesseract's performance.
*   **Basic Detection:** The plate detection uses contour finding, which can be fooled by other rectangular shapes in the image.
*   **Limited Feedback Use:** Feedback is stored but not used for complex retraining in this version.

## Contributing

Feel free to fork the project, make improvements, and submit pull requests!

## License

MIT License - see [LICENSE](LICENSE).
