import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from config import IMAGES_DIR, RESULTS_DIR, BASE_URL, PORT
from api import routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Processing API",
    description="API for processing images with Google Lens",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Include API routes
app.include_router(routes.router)

@app.get("/")
async def root():
    """Redirect to API docs"""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server at {BASE_URL}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True) 