import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Create log directory
LOG_DIR = BASE_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Server settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
API_PORT = int(os.getenv('API_PORT', 8001))
VIEWER_PORT = int(os.getenv('VIEWER_PORT', 8080))

# Directories
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# Create necessary directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# URLs
BASE_URL = os.getenv('BASE_URL', f'http://localhost:{PORT}')
API_URL = os.getenv('API_URL', f'http://localhost:{API_PORT}')
VIEWER_URL = os.getenv('VIEWER_URL', f'http://localhost:{VIEWER_PORT}')

# API Keys
NGROK_AUTHTOKEN = os.getenv('NGROK_AUTHTOKEN', '')

# Processing settings
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 5)) 