#!/usr/bin/env python
"""
rerank_results.py

• Reads JSON files in the results directory containing Google Lens results
• Processes each result image using YOLO to detect persons
• If a person is detected, uses the fashion detection model to extract clothing items
• Embeds the original query image and the processed results
• Reranks the results based on cosine similarity with the query image

Usage:
    python rerank_results.py --query_id <query_id> --label <clothing_item_label>

Example:
    python rerank_results.py --query_id 3f5d49b9bd1e4faeac8ba8223a37d14e --label shoes
"""

import argparse
import json
import os
import logging
import uuid
import sys
import csv
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any
import base64
import requests
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as v2
from transformers import AutoImageProcessor, AutoModelForObjectDetection, SwinModel, SwinConfig
from huggingface_hub import PyTorchModelHubMixin

from config import IMAGES_DIR, RESULTS_DIR, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rerank_results")

# Check for GPU availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
logger.info(f"Using device: {device}")

# CSV Loader for finding labels
class CropsCSVLoader:
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.crops_data = {}
        self._load_crops_data()
    
    def _load_crops_data(self):
        """Load all crops data from CSV files"""
        for csv_file in self.data_dir.glob("*.crops.csv"):
            logger.info(f"Loading crops data from {csv_file}")
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract filename from the crop_url
                    crop_url = row['crop_url']
                    if not crop_url:
                        continue
                    
                    # Extract just the filename part
                    filename = os.path.basename(crop_url)
                    self.crops_data[filename] = {
                        'origin': row['origin'],
                        'box': row['box'],
                        'label': row['label'],
                        'score': float(row['score'])
                    }
            
        logger.info(f"Loaded data for {len(self.crops_data)} crop images")
    
    def get_label_for_image(self, image_filename: str) -> Optional[str]:
        """Get label for a given image filename"""
        # If we get a complete URL or path, extract just the filename
        image_filename = os.path.basename(image_filename)
        
        if image_filename in self.crops_data:
            return self.crops_data[image_filename]['label']
        return None
    
    def get_labels_for_query(self, query_id: str) -> Dict[str, int]:
        """Analyze a query ID and return the most common clothing item labels"""
        # Load the query JSON file
        results_file = Path(RESULTS_DIR) / f"{query_id}.json"
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract query image filename
        query_url = results_data.get('query_url', '')
        query_filename = os.path.basename(query_url)
        
        # Check if query image is in our crops data
        if query_filename in self.crops_data:
            # Return just this label
            label = self.crops_data[query_filename]['label']
            return {label: 1}
        
        # If query image not found, look for any matches in the results
        labels_count = {}
        
        for result in results_data.get('results', []):
            image_url = result.get('image', result.get('thumbnail', ''))
            if not image_url:
                continue
                
            image_filename = os.path.basename(image_url)
            if image_filename in self.crops_data:
                label = self.crops_data[image_filename]['label']
                labels_count[label] = labels_count.get(label, 0) + 1
        
        return labels_count

# YOLO Person detection model
class PersonDetector:
    def __init__(self):
        logger.info("Loading YOLOv8 person detection model...")
        try:
            # Get the path to the local model file
            model_path = Path(os.path.dirname(os.path.abspath(__file__))) / "models" / "yolov8n-person.pt"
            if model_path.exists():
                logger.info(f"Using local model file: {model_path}")
                try:
                    # Try to import the YOLO class directly from ultralytics
                    from ultralytics import YOLO
                    self.model = YOLO(str(model_path))
                    logger.info("Loaded model using ultralytics.YOLO")
                except ImportError:
                    # Fall back to using torch.hub with trust_repo=True
                    logger.info("Ultralytics package not found, falling back to torch.hub")
                    self.model = torch.hub.load('ultralytics/yolov8', 'custom', 
                                               path=str(model_path), trust_repo=True)
            else:
                # Model file not found
                logger.error(f"Local model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model.to(device)
            logger.info("Person detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            sys.exit(1)
    
    def detect(self, image: Image.Image) -> bool:
        """Detect if there's a person in the image"""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                results = self.model(img_array)
            else:
                results = self.model(image)
            
            # Check if model.names exists and look for person class
            if hasattr(self.model, 'names') and 0 in self.model.names and self.model.names[0] == 'person':
                # For ultralytics YOLO format
                person_detected = False
                for r in results:
                    if hasattr(r, 'boxes'):
                        boxes = r.boxes
                        for box in boxes:
                            if hasattr(box, 'cls') and int(box.cls.item()) == 0:  # Person class is 0
                                person_detected = True
                                break
                return person_detected
            else:
                # For torch hub format
                for result in results.pred:
                    for det in result:
                        if det[-1] == 0:  # Person class is 0
                            return True
            return False
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return False


# Fashion object detector for clothing items
class FashionDetector:
    def __init__(self):
        logger.info("Loading fashion object detection model...")
        try:
            self.ckpt = 'yainage90/fashion-object-detection'
            self.image_processor = AutoImageProcessor.from_pretrained(self.ckpt)
            self.model = AutoModelForObjectDetection.from_pretrained(self.ckpt).to(device)
            logger.info("Fashion detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load fashion detection model: {e}")
            sys.exit(1)

    def detect(self, image: Image.Image, target_label: str) -> Optional[Dict[str, Any]]:
        """Detect clothing items and return the best match for target_label"""
        with torch.no_grad():
            inputs = self.image_processor(images=[image], return_tensors="pt")
            outputs = self.model(**inputs.to(device))
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=target_sizes)[0]

            # Create a list of detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score_val = score.item()
                label_val = self.model.config.id2label[label.item()]
                box_val = [i.item() for i in box]
                
                # If the label matches the target label, add to detections
                if label_val == target_label:
                    detections.append({
                        "score": score_val,
                        "label": label_val,
                        "box": box_val
                    })
            
            # Return the detection with the highest score, if any
            if detections:
                return max(detections, key=lambda x: x["score"])
            return None


# Image Encoder for feature extraction
class ImageEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config_path):
        super().__init__()
        # Initialize the backbone directly from pretrained
        self.backbone = SwinModel.from_pretrained(config_path)
        
        # Get hidden size from the model's config
        hidden_size = self.backbone.config.hidden_size
        
        # Create the projection head
        self.head = nn.Linear(hidden_size, 128)

    def forward(self, x):
        feat = self.backbone(x).pooler_output           # [B, hidden]
        emb = F.normalize(self.head(feat), p=2, dim=1)  # L2-norm 128-D
        return emb


class ImageEmbedder:
    def __init__(self):
        logger.info("Loading image embedding model...")
        try:
            # Use local model files
            local_model_path = Path(os.path.dirname(os.path.abspath(__file__))) / "models" / "fashion-image-feature-extractor"
            
            # Debug: Print the absolute path
            abs_path = os.path.abspath(local_model_path)
            logger.info(f"Absolute model path: {abs_path}")
            
            # Check if path and required files exist
            if not local_model_path.exists():
                logger.error(f"Local model directory not found: {local_model_path}")
                raise FileNotFoundError(f"Model directory not found: {local_model_path}")
            
            # Check individual files
            config_file = local_model_path / "config.json"
            model_file = local_model_path / "model.safetensors"
            processor_file = local_model_path / "preprocessor_config.json"
            
            logger.info(f"Config file exists: {config_file.exists()}")
            logger.info(f"Model file exists: {model_file.exists()}")
            logger.info(f"Processor file exists: {processor_file.exists()}")
                
            logger.info(f"Using local model from: {local_model_path}")
            
            # Load SwinModel directly from the local path
            logger.info("Attempting to load SwinModel...")
            try:
                self.backbone = SwinModel.from_pretrained(
                    str(local_model_path),
                    local_files_only=True  # never contact the Hub
                )
            except AttributeError as e:
                raise RuntimeError(
                    "Could not load local Swin weights – did you forget `git lfs pull` "
                    "or to upgrade `huggingface_hub`?"
                ) from e
            logger.info("SwinModel loaded successfully")
            
            self.backbone.to(device)
            self.backbone.eval()
            
            # Create projection head (128 dimensions for the embedding)
            hidden_size = self.backbone.config.hidden_size
            self.head = nn.Linear(hidden_size, 128).to(device)
            
            # Load image processor
            logger.info("Loading image processor...")
            self.image_processor = AutoImageProcessor.from_pretrained(str(local_model_path))
            logger.info("Image processor loaded successfully")
            
            # Setup transformer for preprocessing images
            self.transform = v2.Compose([
                v2.Resize((self.backbone.config.image_size, self.backbone.config.image_size)),
                v2.ToTensor(),
                v2.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
            ])
            
            logger.info("Image embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image embedding model: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Print full traceback
            sys.exit(1)
    
    def embed(self, image: Image.Image) -> np.ndarray:
        """Embed an image to get feature vector"""
        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Transform the image
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get features from backbone
                outputs = self.backbone(image_tensor)
                features = outputs.pooler_output
                
                # Project to embedding space
                embeddings = self.head(features)
                
                # Normalize embeddings
                normalized = F.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy array
                embedding = normalized.cpu().numpy()
                return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(128)  # Return zero vector as fallback


class ResultReranker:
    def __init__(self, query_id: str, target_label: Optional[str] = None):
        self.query_id = query_id
        self.results_dir = Path(RESULTS_DIR)
        self.images_dir = Path(IMAGES_DIR)
        
        # Load the query results
        self.results_file = self.results_dir / f"{query_id}.json"
        if not self.results_file.exists():
            logger.error(f"Results file not found: {self.results_file}")
            sys.exit(1)
        
        with open(self.results_file, 'r') as f:
            self.results_data = json.load(f)
        
        # Get the appropriate target label if not specified
        if target_label is None:
            # Load crops data
            crops_loader = CropsCSVLoader()
            labels_count = crops_loader.get_labels_for_query(query_id)
            
            if not labels_count:
                logger.warning(f"No label information found for query {query_id}. Using 'shoes' as default.")
                self.target_label = "shoes"
            else:
                # Get the most common label
                self.target_label = max(labels_count.items(), key=lambda x: x[1])[0]
                logger.info(f"Automatically determined target label: {self.target_label}")
        else:
            self.target_label = target_label
            logger.info(f"Using specified target label: {self.target_label}")
        
        # Initialize models
        self.person_detector = PersonDetector()
        self.fashion_detector = FashionDetector()
        self.image_embedder = ImageEmbedder()
        
        # Create images directory if it doesn't exist
        self.images_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Loaded results for query: {query_id}")
    
    def download_image(self, url_or_id: str) -> Optional[Image.Image]:
        """Get image from URL or local file"""
        try:
            # If this looks like a URL, try to download it
            if url_or_id.startswith('http'):
                logger.info(f"Downloading image from URL: {url_or_id}")
                response = requests.get(url_or_id, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to download image: HTTP status {response.status_code}")
                    return None
                return Image.open(BytesIO(response.content))
            
            # Otherwise, check if it's an image ID (filename without extension)
            image_id = os.path.basename(url_or_id).split('.')[0]
            
            # Look for the image in the images folder
            image_path = Path(IMAGES_DIR) / f"{image_id}.jpg"
            if image_path.exists():
                logger.info(f"Loading local image: {image_path}")
                return Image.open(image_path)
            
            # If not found, return None
            logger.error(f"Image not found locally: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single result, detecting person and embedding the image"""
        # Get image URL or ID, preferring the full image over thumbnail
        image_url_or_id = result.get('image', result.get('thumbnail', ''))
        if not image_url_or_id:
            return result
        
        # Download or load the image
        image = self.download_image(image_url_or_id)
        if image is None:
            return result
        
        # Check if there's a person in the image
        has_person = self.person_detector.detect(image)
        result['has_person'] = has_person
        
        # If there's a person and we have a target label, try to crop to clothing item
        cropped_image = None
        if has_person and self.target_label:
            detection = self.fashion_detector.detect(image, self.target_label)
            if detection:
                # Crop the image to the detected clothing item
                left, top, right, bottom = detection['box']
                cropped_image = image.crop((left, top, right, bottom))
                result['detection_box'] = detection['box']
                result['detection_score'] = detection['score']
            
        # Use the cropped image if available, otherwise the original
        embedding_image = cropped_image if cropped_image else image
        
        # Generate embedding for the image
        embedding = self.image_embedder.embed(embedding_image)
        
        # Store embedding (128-dimensional vector)
        result['embedding'] = embedding
        
        return result
    
    def rerank_results(self) -> Dict[str, Any]:
        """Process and rerank the results"""
        # Get the query image - could be a URL or an image ID
        query_url = self.results_data.get('query_url')
        if not query_url:
            # If no query_url, try using the query_id as the image ID
            query_url = self.query_id
            logger.info(f"No query URL found, using query ID as image ID: {query_url}")
        
        query_image = self.download_image(self.query_id)
        if query_image is None:
            logger.error(f"Failed to load query image from either URL or query ID")
            return self.results_data
        
        # Process the query image with the same pipeline as result images
        # Check if there's a person in the image and try to crop clothing item
        has_person = self.person_detector.detect(query_image)
        query_image_to_embed = query_image
        
        # If there's a person and we have a target label, try to crop to clothing item
        if has_person and self.target_label:
            detection = self.fashion_detector.detect(query_image, self.target_label)
            if detection:
                # Crop the image to the detected clothing item
                left, top, right, bottom = detection['box']
                cropped_query = query_image.crop((left, top, right, bottom))
                query_image_to_embed = cropped_query
                logger.info(f"Cropped query image to detected {self.target_label}")
        
        # Embed the query image
        query_embedding = self.image_embedder.embed(query_image_to_embed)
        
        # Process each result
        processed_results = []
        for result in self.results_data.get('results', []):
            processed_result = self.process_result(result)
            processed_results.append(processed_result)
        
        # Calculate cosine similarity for results with embeddings
        for result in processed_results:
            if 'embedding' in result:
                similarity = np.dot(query_embedding.flatten(), result['embedding'].flatten())
                result['similarity'] = float(similarity)
            else:
                result['similarity'] = 0.0
        
        # Rerank the results based on similarity
        reranked_results = sorted(processed_results, key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        # Remove the numpy arrays before returning
        for result in reranked_results:
            if 'embedding' in result:
                del result['embedding']
        
        # Create the output data
        output_data = {
            'query_id': self.query_id,
            'query_url': query_url,
            'target_label': self.target_label,
            'original_order': self.results_data.get('results', []),
            'reranked_results': reranked_results
        }
        
        # Save the reranked results
        output_file = self.results_dir / f"{self.query_id}_reranked_{self.target_label}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Reranked results saved to {output_file}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Rerank visual matches using YOLO and embeddings')
    parser.add_argument('--query_id', help='Optional: ID of a specific query image results to rerank')
    parser.add_argument('--all', action='store_true', help='Process all result files (this is the default behavior)')
    parser.add_argument('--label', choices=['bag', 'bottom', 'dress', 'hat', 'shoes', 'outer', 'top'],
                      help='Optional: Clothing item label to search for (if not specified, will be determined from CSV data)')
    args = parser.parse_args()
    
    # If query_id is provided, process only that file
    if args.query_id:
        logger.info(f"Processing single query ID: {args.query_id}")
        reranker = ResultReranker(args.query_id, args.label)
        results = reranker.rerank_results()
        
        # Print the top 3 reranked results
        print(f"\nTop 3 reranked results for {args.query_id} (target label: {reranker.target_label}):")
        for i, result in enumerate(results['reranked_results'][:3]):
            print(f"{i+1}. {result.get('title', 'No title')} - Similarity: {result.get('similarity', 0.0):.4f}")
        return
    
    # Default behavior (or if --all is specified): process all files
    # Process all JSON files in the results directory
    results_dir = Path(RESULTS_DIR)
    processed = 0
    skipped = 0
    
    # Get all JSON files that don't have '_reranked_' in their name
    result_files = [f for f in results_dir.glob('*.json') if '_reranked_' not in f.name]
    
    if not result_files:
        logger.warning("No result files found to process.")
        return
    
    logger.info(f"Found {len(result_files)} result files to process")
    
    for results_file in result_files:
        query_id = results_file.stem
        logger.info(f"Processing query ID: {query_id}")
        
        try:
            reranker = ResultReranker(query_id, args.label)
            results = reranker.rerank_results()
            processed += 1
            
            # Print top results for this query
            print(f"\nTop 3 reranked results for {query_id} (target label: {reranker.target_label}):")
            for i, result in enumerate(results['reranked_results'][:3]):
                print(f"{i+1}. {result.get('title', 'No title')} - Similarity: {result.get('similarity', 0.0):.4f}")
        except Exception as e:
            logger.error(f"Error processing {query_id}: {e}")
            skipped += 1
    
    logger.info(f"Processing complete. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
