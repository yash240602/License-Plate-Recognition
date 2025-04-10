#!/bin/bash

# Create a Python virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install the required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo "Setup complete! You can now run the application with: python app.py" 