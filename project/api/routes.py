import os
import json
import logging
import shutil
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

from config import IMAGES_DIR, RESULTS_DIR, MAX_RESULTS, BASE_URL
from services.lens import process_images_folder
from services.ngrok import ensure_public_images_root

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["images"])

# Models
class ImageInfo(BaseModel):
    filename: str
    path: str
    processed: bool

class ResultInfo(BaseModel):
    image_id: str
    selected_match: Optional[int] = None
    matches: List[Dict[str, Any]]

class ProcessRequest(BaseModel):
    image_ids: Optional[List[str]] = None  # If None, process all images

# Helper functions
def get_images() -> List[ImageInfo]:
    """Get all images in the images directory"""
    images = []
    for file in os.listdir(IMAGES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')) and not file.startswith('.'):
            # Check if this image has been processed
            base_name = os.path.splitext(file)[0]
            processed = os.path.exists(os.path.join(RESULTS_DIR, f"{base_name}.json"))
            
            images.append(ImageInfo(
                filename=file,
                path=f"/images/{file}",
                processed=processed
            ))
    
    return sorted(images, key=lambda x: x.filename)

def get_results() -> List[ResultInfo]:
    """Get all processing results"""
    results = []
    
    # Load matches data if it exists
    matches_data = {}
    matches_csv = os.path.join(RESULTS_DIR, "matches.csv")
    if os.path.exists(matches_csv):
        import csv
        with open(matches_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    image_id = row[0]
                    selected_match = int(row[1]) if row[1] else None
                    matches_data[image_id] = selected_match
    
    # Get results for each processed image
    for file in os.listdir(RESULTS_DIR):
        if file.endswith('.json') and not file.startswith('matches'):
            image_id = os.path.splitext(file)[0]
            
            try:
                with open(os.path.join(RESULTS_DIR, file), 'r') as f:
                    data = json.load(f)
                    
                selected_match = matches_data.get(image_id)
                
                results.append(ResultInfo(
                    image_id=image_id,
                    selected_match=selected_match,
                    matches=data.get('results', [])
                ))
            except Exception as e:
                logger.error(f"Error loading result {file}: {e}")
    
    return sorted(results, key=lambda x: x.image_id)

def process_images_background(image_ids: List[str] = None):
    """Process images in the background"""
    try:
        # Make sure images are publicly accessible via ngrok
        public_url = ensure_public_images_root()
        logger.info(f"Images accessible at: {public_url}")
        
        # Get the list of images to process
        all_images = [f for f in os.listdir(IMAGES_DIR) 
                      if f.lower().endswith('.jpg') and not f.startswith('.')]
        
        if not all_images:
            logger.warning("No images found to process")
            return
            
        # Filter by image_ids if provided
        if image_ids:
            images_to_process = [img for img in all_images if os.path.splitext(img)[0] in image_ids]
            logger.info(f"Processing {len(images_to_process)} of {len(all_images)} images")
        else:
            images_to_process = all_images
            logger.info(f"Processing all {len(all_images)} images")
            
        if not images_to_process:
            logger.warning("No matching images found to process")
            return
            
        # Process images
        process_images_folder(
            images_dir=str(IMAGES_DIR),
            results_dir=str(RESULTS_DIR),
            public_url_base=public_url,
            max_results=MAX_RESULTS
        )
        
        logger.info("Image processing completed")
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

# API Endpoints
@router.get("/images", response_model=List[ImageInfo])
async def list_images():
    """List all images in the images directory"""
    try:
        return get_images()
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/images", response_model=ImageInfo)
async def upload_image(file: UploadFile = File(...)):
    """Upload a new image"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Save the file
        file_path = os.path.join(IMAGES_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Uploaded image: {file.filename}")
        
        return ImageInfo(
            filename=file.filename,
            path=f"/images/{file.filename}",
            processed=False
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/images/{filename}")
async def delete_image(filename: str):
    """Delete an image and its associated results"""
    try:
        # Delete the image file
        image_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image {filename} not found")
        
        os.remove(image_path)
        
        # Delete the results file if it exists
        base_name = os.path.splitext(filename)[0]
        results_path = os.path.join(RESULTS_DIR, f"{base_name}.json")
        if os.path.exists(results_path):
            os.remove(results_path)
        
        # Update matches CSV if necessary
        matches_csv = os.path.join(RESULTS_DIR, "matches.csv")
        if os.path.exists(matches_csv):
            import csv
            # Load existing data
            matches_data = {}
            with open(matches_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] != base_name:
                        image_id = row[0]
                        selected_match = row[1]
                        matches_data[image_id] = selected_match
            
            # Write updated data
            with open(matches_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id', 'selected_match'])
                for image_id, selected_match in matches_data.items():
                    writer.writerow([image_id, selected_match])
        
        logger.info(f"Deleted image: {filename}")
        
        return {"message": f"Image {filename} and associated data deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results", response_model=List[ResultInfo])
async def list_results():
    """List all processing results"""
    try:
        return get_results()
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_images(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """Process images with Google Lens"""
    try:
        # Start processing in the background
        background_tasks.add_task(
            process_images_background, 
            request.image_ids
        )
        
        return {
            "message": "Image processing started in background",
            "images": request.image_ids if request.image_ids else "all"
        }
    except Exception as e:
        logger.error(f"Error starting image processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/results/{image_id}/{match_index}")
async def select_match(image_id: str, match_index: int):
    """Select a match for an image"""
    try:
        # Verify the image result exists
        results_path = os.path.join(RESULTS_DIR, f"{image_id}.json")
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail=f"Results for image {image_id} not found")
        
        # Load the results to verify match_index is valid
        with open(results_path, 'r') as f:
            data = json.load(f)
            if match_index >= len(data.get('results', [])):
                raise HTTPException(status_code=400, detail=f"Match index {match_index} out of range")
        
        # Update matches CSV
        matches_csv = os.path.join(RESULTS_DIR, "matches.csv")
        matches_data = {}
        
        # Load existing data if file exists
        if os.path.exists(matches_csv):
            import csv
            with open(matches_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        matches_data[row[0]] = row[1]
        
        # Update data
        matches_data[image_id] = str(match_index)
        
        # Write updated data
        import csv
        with open(matches_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'selected_match'])
            for img_id, sel_match in matches_data.items():
                writer.writerow([img_id, sel_match])
        
        logger.info(f"Selected match {match_index} for image {image_id}")
        
        return {"message": f"Match {match_index} selected for image {image_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get system status"""
    try:
        images = get_images()
        results = get_results()
        
        processed_count = sum(1 for img in images if img.processed)
        
        return {
            "status": "active",
            "total_images": len(images),
            "processed_images": processed_count,
            "results_count": len(results),
            "base_url": BASE_URL
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 