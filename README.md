# License Plate Recognition

Hey there! This is my license plate recognition project that I built for my computer vision class. It's basically a web app that can look at pictures of cars and read what's on their license plates!

![License Plate Recognition Demo](docs/license_plate_demo.jpg)

## What This Does

I spent way too many weekends building this, but it was pretty fun! Here's what it can do:

- **Upload Pictures:** Just drag and drop any car picture
- **Find the License Plate:** The app automatically finds where the license plate is in the image
- **Read the Text:** It figures out what letters and numbers are on the plate
- **Learn from Mistakes:** If it reads something wrong, you can correct it and it learns from that
- **Save Results:** You can download the results or check your history

## How It Works

When you use the app, it does these steps:

1. You upload a car picture
2. It finds the license plate (puts a green box around it)
3. It reads the text (like "DL 7 CN 5617" in the picture above)
4. You can tell it if it got it right or fix any mistakes

## Try It Yourself! (Kid-Friendly Steps)

Want to run this on your own computer? Here's how:

### Step 1: Get Python

If you don't already have Python:
- Go to [Python.org](https://www.python.org/downloads/)
- Download the latest version (3.8 or newer)
- During installation, make sure to check the box that says "Add Python to PATH"

### Step 2: Get This Project

```bash
# Copy my project to your computer
git clone https://github.com/yourusername/License-Plate-Recognition.git

# Go to the project folder
cd License-Plate-Recognition
```

Don't know what git is? No problem:
- Just click the green "Code" button at the top of this page
- Choose "Download ZIP"
- Unzip it somewhere on your computer

### Step 3: Set Everything Up

```bash
# Create a special environment (this keeps things organized)
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Mac or Linux:
source venv/bin/activate

# Install all the stuff this project needs
pip install -r requirements.txt

# Install Tesseract OCR (this helps read text from images)
# On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# On Mac:
brew install tesseract
# On Linux:
sudo apt install tesseract-ocr
```

### Step 4: Start It Up!

```bash
python run_server.py
```

Then open your web browser and go to: http://localhost:8080

## For CS Students & Recruiters

I built this as my final project after learning about computer vision and machine learning. I wanted to create something practical that solves a real problem. Here's some tech details:

### The Tech Stack

- **Frontend:** Just basic HTML/CSS/JavaScript with some Bootstrap to make it look decent
- **Backend:** Flask for the web server, running on Waitress for better performance
- **Computer Vision:** OpenCV for image processing and finding license plates
- **Text Recognition:** Tesseract OCR for reading the text
- **Machine Learning:** The system can use TensorFlow models for better detection (when available)
- **Continuous Learning:** The app remembers corrections and gets better over time

### Technical Parts I'm Proud Of

1. **Image Processing Pipeline**
   - Used adaptive thresholding to handle different lighting conditions
   - Applied contour detection to find rectangular plate shapes
   - Built a preprocessing pipeline that enhances text readability

2. **The Feedback System**
   - Implemented a correction mechanism that learns from user feedback
   - Created region-specific pattern recognition for different license plate formats
   - Built character-level correction for common OCR mistakes

3. **Error Handling**
   - Made the system gracefully fall back to simpler methods when needed
   - Implemented extensive logging for troubleshooting
   - Built detailed confidence scoring to estimate accuracy

## Known Limitations & Future Improvements

I'm still working on this! Some things I want to add:

- Support for multiple cars in one image
- Real-time video processing (if I can make it fast enough)
- Better support for international license plates
- A mobile app version

## Requirements

Check [requirements.txt](requirements.txt) for the full list of dependencies.

## License

This project is under the MIT License - see the [LICENSE](LICENSE) file for details.

## Want to Help?

Found a bug or have ideas to make this better? Feel free to contribute! Just fork, make your changes, and submit a pull request.

## Acknowledgements

Huge thanks to:
- OpenCV library (saved me so much time)
- Tesseract OCR (for the text recognition magic)
- My prof who didn't fail me despite all my deadline extensions 😅
- All the Stack Overflow posts that helped debug this thing!
