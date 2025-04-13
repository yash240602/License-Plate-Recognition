# 🚗 License Plate Recognition System

![License Plate Demo](docs/license_plate_demo.jpg)

## 👋 What is this project?

This is a web app that can **detect and read license plates** from car photos! Just upload a picture of a car, and the system will:

1. Find the license plate in the image
2. Read the text on the plate
3. Show you the results with a confidence score
4. Let you correct any mistakes to help the system learn

**Perfect for:** beginners learning AI, hobbyists, ML portfolio projects, or anyone wanting to see computer vision in action!

## 🔍 How it works (simplified)

1. You upload a picture of a car
2. The app finds rectangular shapes that look like license plates
3. It zooms in on the plate and makes it easier to read
4. Special software reads the text on the plate
5. The app shows you what it found and how confident it is
6. You can help improve the system by correcting any mistakes
## 🚀 Easy Setup (even for beginners!)
## 🧠 Technical skills showcased

This project demonstrates my skills in:

- **Computer Vision** - Using OpenCV for image processing and plate detection
- **Machine Learning** - Applying OCR with Tesseract and confidence scoring
- **Web Development** - Building a responsive Flask-based web application
- **User Experience Design** - Creating an intuitive interface for image processing
- **Error Handling** - Implementing robust fallback mechanisms
- **Software Engineering** - Designing modular, maintainable code
- **Full-stack Development** - Integrating frontend and backend components

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/License-Plate-Recognition.git
   cd License-Plate-Recognition
   ```

2. **Set up a virtual environment**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scriptsctivate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
   - **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Run the server**
   ```bash
   python run_server.py
   ```

6. **Access the web interface**
   Open your browser and go to: http://localhost:8080

## 📷 Example

Below is an example of successful license plate recognition on a Mercedes-Benz in India:

![License Plate Detection Example](docs/license_plate_demo.jpg)

The system correctly identified the license plate "MH 12 NE 8922" with a confidence of 94.64%.

## 📝 About Me

I'm a Machine Learning enthusiast passionate about computer vision and AI applications. This project showcases my ability to build end-to-end ML systems with practical applications.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Yash Shrivastava
