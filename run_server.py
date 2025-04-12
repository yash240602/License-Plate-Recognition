from waitress import serve
from app import app, logger
import os
import signal
import sys
import subprocess
import time

def check_port_in_use(port):
    """Check if the port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def cleanup_existing_server(port):
    """Attempt to identify and clean up any server running on the specified port"""
    try:
        # Check platform-specific commands
        if sys.platform.startswith('win'):
            # Windows - use netstat and taskkill
            cmd = f'netstat -ano | findstr :{port}'
            output = subprocess.check_output(cmd, shell=True).decode()
            if output:
                for line in output.splitlines():
                    if 'LISTENING' in line:
                        pid = line.split()[-1]
                        print(f"Found process using port {port}: PID {pid}")
                        print(f"Attempting to terminate...")
                        os.system(f'taskkill /PID {pid} /F')
                        time.sleep(1)  # Give it time to terminate
                        return True
        else:
            # Unix-like systems (macOS, Linux) - use lsof
            cmd = f'lsof -i :{port} -t'
            try:
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                if output:
                    pids = output.split('\n')
                    for pid in pids:
                        if pid:
                            print(f"Found process using port {port}: PID {pid}")
                            print(f"Attempting to terminate...")
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(1)  # Give it time to terminate
                    return True
            except subprocess.CalledProcessError:
                # No process found
                pass
    except Exception as e:
        print(f"Error while trying to clean up existing server: {e}")
    
    return False

def main():
    # Create required directories
    upload_folder = 'static/uploads'
    result_folder = 'static/results'
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Configure and start the server
    PORT = 8080
    HOST = "0.0.0.0"
    
    # Check if port is already in use
    if check_port_in_use(PORT):
        print(f"Port {PORT} is already in use.")
        cleanup = cleanup_existing_server(PORT)
        if cleanup:
            print(f"Successfully cleaned up previous server instance.")
        else:
            print(f"Failed to clean up. Please manually stop any process using port {PORT}.")
            alternative_port = PORT + 1
            while check_port_in_use(alternative_port) and alternative_port < PORT + 10:
                alternative_port += 1
            
            if alternative_port < PORT + 10:
                print(f"Using alternative port {alternative_port}")
                PORT = alternative_port
            else:
                print(f"Could not find an available port. Please manually stop the server.")
                sys.exit(1)
    
    logger.info(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    print(f"Starting Waitress server on http://127.0.0.1:{PORT}")
    print(f"You can access the application at:")
    print(f"  - Local: http://127.0.0.1:{PORT}")
    print(f"  - Network: http://<your-ip-address>:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    
    try:
        # Start Waitress with 4 threads for better performance
        serve(app, host=HOST, port=PORT, threads=4)
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"Server error: {e}")
    finally:
        print("Server shutdown complete.")

if __name__ == "__main__":
    main() 