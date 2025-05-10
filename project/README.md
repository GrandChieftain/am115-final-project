# Google Lens Image Processing System

This is a standalone application for processing images through Google Lens and managing the results. It consists of three main components:

1. **API Server**: FastAPI application that provides endpoints for image management and processing
2. **Image Processor**: Service that submits images to Google Lens and saves results
3. **Results Viewer**: Web interface to view images and their matches in a grid layout

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Configuration

Configuration settings are in `config.py`. You can override them by creating a `.env` file in the project directory with the following variables:

```
PORT=8000
API_PORT=8001
VIEWER_PORT=8080
BASE_URL=http://localhost:8000
NGROK_AUTHTOKEN=your_ngrok_auth_token  # Optional, for public access
MAX_RESULTS=5  # Number of results per image
```

## Directory Structure

- `images/`: Place your images here
- `results/`: Contains processing results
- `data/`: Additional data (e.g., matches.csv)
- `logs/`: Log files
- `api/`: API endpoints
- `services/`: Helper modules for Google Lens and ngrok

## Usage

### Using the consolidated system manager

The easiest way to use the system is with the consolidated system management script:

```bash
python system.py                  # Run the entire system interactively
python system.py --api            # Run just the API server
python system.py --viewer         # Run just the results viewer
python system.py --process        # Process images through Google Lens
python system.py --non-interactive # Run without interactive prompts
python system.py --help           # Show all available options
```

This script replaces the older `run_system.py`, `simple_server.py`, and `start_server.py` files.

### Running components separately

#### API Server

```bash
python main.py
```

This starts the FastAPI server on the port specified in config.py (default: 8000).
The API documentation is available at http://localhost:8000/docs

#### Results Viewer

```bash
python results_viewer.py
```

This starts the web interface for viewing processing results on port 8080 (default).

## API Endpoints

- `GET /api/images`: List all images
- `POST /api/images`: Upload a new image
- `DELETE /api/images/{filename}`: Delete an image and its results
- `GET /api/results`: List all processing results
- `POST /api/process`: Process images through Google Lens
- `POST /api/results/{image_id}/{match_index}`: Select a match for an image
- `GET /api/status`: Get system status

## How It Works

1. The API server serves the images via HTTP
2. When processing is initiated, the system makes images publicly accessible via ngrok
3. It then submits each image to Google Lens using Playwright browser automation
4. Results are saved as JSON files in the results directory
5. The Results Viewer displays the original images and their matches in a grid
6. Users can select the correct match for each image
7. Selected matches are saved in matches.csv 