from waitress import serve
from app import app, logger
import os

def main():
    # Create required directories
    upload_folder = 'static/uploads'
    result_folder = 'static/results'
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Configure and start the server
    PORT = 8080
    HOST = "0.0.0.0"
    
    logger.info(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    print(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    print(f"You can access the application at:")
    print(f"  - Local: http://127.0.0.1:{PORT}")
    print(f"  - Network: http://<your-ip-address>:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    
    # Start Waitress with 4 threads for better performance
    serve(app, host=HOST, port=PORT, threads=4)

if __name__ == "__main__":
    main() 