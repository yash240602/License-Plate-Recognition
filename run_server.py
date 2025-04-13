from waitress import serve
from app import app, logger
import os

# Ensure required directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

HOST = "0.0.0.0"
PORT = 8080

if __name__ == "__main__":
    logger.info(f"Starting Waitress server on {HOST}:{PORT}")
    print(f"Starting Waitress server - access at http://127.0.0.1:{PORT}")
    try:
        serve(app, host=HOST, port=PORT, threads=4)
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        print(f"Server crashed: {e}") 