#!/bin/bash

# This script automates the PythonAnywhere setup process
# Run this in a Bash console on PythonAnywhere

# Clone the repository
git clone https://github.com/yash240602/License-Plate-Recognition.git
cd License-Plate-Recognition

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/uploads
mkdir -p static/results
touch static/uploads/.gitkeep
touch static/results/.gitkeep

# Create a basic wsgi file
cat > /var/www/yourusername_pythonanywhere_com_wsgi.py << EOL
import sys
import os

# Add project directory to path
path = '/home/yourusername/License-Plate-Recognition'
if path not in sys.path:
    sys.path.append(path)

# Configure environment variables
os.environ['VIRTUAL_ENV'] = '/home/yourusername/License-Plate-Recognition/venv'
os.environ['PATH'] = '/home/yourusername/License-Plate-Recognition/venv/bin:' + os.environ['PATH']
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'

# Import Flask app
from app import app as application
EOL

echo "Setup complete! Now you need to:"
echo "1. Go to the Web tab in PythonAnywhere"
echo "2. Set source code to: /home/yourusername/License-Plate-Recognition"
echo "3. Set working directory to: /home/yourusername/License-Plate-Recognition"
echo "4. Add static files URL: /static/ to directory: /home/yourusername/License-Plate-Recognition/static"
echo "5. Click the 'Reload' button to start your app" 