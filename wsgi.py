from app import app

# Needed for Gunicorn to find the Flask app
if __name__ == "__main__":
    app.run() 