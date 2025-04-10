import os

# Gunicorn config variables
workers = 1
worker_class = "gthread"
threads = 4
timeout = 300
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
preload_app = True 