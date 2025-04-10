# Deploying License Plate Recognition to PythonAnywhere

## Step 1: Create a PythonAnywhere Account
1. Go to [PythonAnywhere](https://www.pythonanywhere.com/) and sign up for a free account

## Step 2: Set Up a Web App
1. Once logged in, click on the "Web" tab
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.11
5. Click "Next" and confirm the path

## Step 3: Clone Your Repository
1. Go to the "Consoles" tab and start a new Bash console
2. Clone your repository:
   ```
   git clone https://github.com/yash240602/License-Plate-Recognition.git
   ```
3. Navigate to the project directory:
   ```
   cd License-Plate-Recognition
   ```

## Step 4: Create a Virtual Environment
1. In the Bash console, create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Step 5: Configure the Web App
1. Go back to the "Web" tab
2. In the "Code" section, set:
   - Source code: `/home/yourusername/License-Plate-Recognition`
   - Working directory: `/home/yourusername/License-Plate-Recognition`
   - WSGI configuration file: Click on the WSGI file link and modify it as shown below

3. Replace the content of the WSGI file with:
   ```python
   import sys
   import os

   # Add your project directory to the path
   path = '/home/yourusername/License-Plate-Recognition'
   if path not in sys.path:
       sys.path.append(path)

   # Point to your virtual environment
   os.environ['VIRTUAL_ENV'] = '/home/yourusername/License-Plate-Recognition/venv'
   os.environ['PATH'] = '/home/yourusername/License-Plate-Recognition/venv/bin:' + os.environ['PATH']

   # Import your Flask app
   from app import app as application
   ```

4. Replace `yourusername` with your actual PythonAnywhere username

## Step 6: Set Up Static Files
1. Still in the "Web" tab, go to the "Static Files" section
2. Add these entries:
   - URL: `/static/` → Directory: `/home/yourusername/License-Plate-Recognition/static`

## Step 7: Start the Web App
1. Click the "Reload" button at the top of the web app page
2. Your app will be available at: `yourusername.pythonanywhere.com`

## Troubleshooting
If you encounter issues:
1. Check error logs in the "Web" tab under "Log files"
2. Ensure Tesseract is working by running this in a console:
   ```
   pytesseract --version
   ```
3. If needed, set environment variables in the WSGI file:
   ```python
   os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
   ```

## Advantages of PythonAnywhere
- Tesseract OCR is pre-installed
- No inactivity spin-down like Render
- Persistent storage for your feedback and history files
- Free tier includes 512MB storage and always-on service 