#!/usr/bin/env python
"""
process_scraped_images.py

• Reads *.csv files containing Pinterest image URLs (`src` column)
• Downloads each image
• Runs a Fashion‑YOLOS model to detect clothing items
• Crops every detected bounding box and saves the sub‑image to IMAGES_ROOT
• Writes a second CSV next to each input file listing the new URLs

Usage
-----
    python -m app.scripts.process_scraped_images \
        --csv mens_outfits.csv women_outfits.csv

Dependencies
------------
pip install torch torchvision transformers pillow requests pandas tqdm
"""

from __future__ import annotations
import argparse
import csv
import uuid
import logging
from pathlib import Path
from io import BytesIO
from typing import List

import requests
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from tqdm import tqdm

from ..config import BASE_DIR, IMAGES_DIR, DATA_DIR

# Define the FULL_IMAGES_URL
FULL_IMAGES_URL = "/images/"

# ───────────────────────── log setup ────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s  %(message)s"
)
log = logging.getLogger("process_scraped_images")

# ───────────────────────── model load (once) ────────────────────
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

CKPT = "yainage90/fashion-object-detection"
log.info("Loading model %s …", CKPT)
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(CKPT)
MODEL = AutoModelForObjectDetection.from_pretrained(CKPT).to(DEVICE)
MODEL.eval()
log.info("Model ready on %s", DEVICE)

# score threshold for keeping boxes
THRESHOLD = 0.40

# ───────────────────────── helpers ──────────────────────────────
def download_image(url: str) -> Image.Image | None:
    """Download URL → PIL.Image (RGB) or None on failure."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("failed to download %s (%s)", url, exc)
        return None


def detect_boxes(img: Image.Image) -> List[dict]:
    """Run object detection → list of dicts with box coords and class."""
    # Define mapping for labels
    id2label = {
        0: 'bag',
        1: 'bottom',
        2: 'dress',
        3: 'hat', 
        4: 'shoes',
        5: 'outer',
        6: 'top'
    }
    
    with torch.no_grad():
        inputs = IMAGE_PROCESSOR(images=[img], return_tensors="pt").to(DEVICE)
        outputs = MODEL(**inputs)
        target_sizes = torch.tensor([[img.size[1], img.size[0]]]).to(DEVICE)
        results = IMAGE_PROCESSOR.post_process_object_detection(
            outputs, threshold=THRESHOLD, target_sizes=target_sizes
        )[0]
    
    boxes = []
    for i, (box, label, score) in enumerate(zip(results["boxes"], results["labels"], results["scores"])):
        boxes.append({
            "box": box.tolist(),
            "label": id2label[label.item()],
            "score": score.item()
        })
    
    return boxes


def save_crop(img: Image.Image, detection, dest_dir: Path) -> str:
    """Save a cropped image and return its public URL."""
    box = detection["box"]
    xmin, ymin, xmax, ymax = map(int, box)
    crop = img.crop((xmin, ymin, xmax, ymax))
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = dest_dir / filename
    crop.save(filepath, format="JPEG", quality=90)
    return f"{FULL_IMAGES_URL}{filename}"


def process_csv(csv_path: Path):
    log.info("Processing %s …", csv_path.name)
    output_rows = []
    
    # Use project/images directory for crops
    dest_dir = Path(IMAGES_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure data directory exists for CSV output
    data_dir = Path("data").resolve()
    data_dir.mkdir(exist_ok=True, parents=True)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        urls = [row["src"] for row in reader]

    for url in tqdm(urls, desc="images", unit="img"):
        img = download_image(url)
        if img is None:
            continue

        for detection in detect_boxes(img):
            public_url = save_crop(img, detection, dest_dir)
            output_rows.append(
                {
                    "origin": url,
                    "crop_url": public_url,
                    "box": ",".join(map(str, map(int, detection["box"]))),
                    "label": detection["label"],
                    "score": detection["score"]
                }
            )

    # write a "_crops.csv" in the data directory
    base_name = csv_path.stem
    out_csv = data_dir / f"{base_name}.crops.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["origin", "crop_url", "box", "label", "score"])
        writer.writeheader()
        writer.writerows(output_rows)

    log.info(" → %d crops saved, metadata → %s", len(output_rows), out_csv.name)


# ───────────────────────── CLI ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Detect clothing items and save crops."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Names of CSV files in the data directory (e.g., mens_outfits.csv)",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from ..config import DATA_DIR

    for csv_file in args.csv:
        csv_path = DATA_DIR / csv_file if not Path(csv_file).is_absolute() else Path(csv_file)
        if not csv_path.exists():
            log.error(f"CSV file not found: {csv_path}")
            continue
        process_csv(csv_path)