#!/usr/bin/env python3
import os
import sys
import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path

# Configuration
IMAGES_DIR = Path(__file__).resolve().parent / "images"
HOST = "10.250.131.177"  # Use your machine's IP for network access
PORT = 8080  # Different port from your main application

class ImageHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for image grid and deletion."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Serve the image grid page
        if parsed_path.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(self.generate_html().encode())
            return
            
        # Serve image files
        elif parsed_path.path.startswith("/images/"):
            image_path = Path(IMAGES_DIR / parsed_path.path.replace("/images/", ""))
            if not image_path.exists() or not image_path.is_file():
                self.send_error(404, "File not found")
                return
                
            # Determine content type
            content_type = "image/jpeg"  # Default
            if image_path.suffix.lower() == ".png":
                content_type = "image/png"
            elif image_path.suffix.lower() == ".gif":
                content_type = "image/gif"
            elif image_path.suffix.lower() == ".webp":
                content_type = "image/webp"
                
            # Serve the image
            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.end_headers()
            with open(image_path, "rb") as f:
                self.wfile.write(f.read())
            return
            
        # API to list images
        elif parsed_path.path == "/api/images":
            images = self.get_images()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(images).encode())
            return
            
        # If path not recognized, return 404
        else:
            self.send_error(404, "File not found")
            
    def do_DELETE(self):
        """Handle DELETE requests to delete images."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Delete an image
        if parsed_path.path.startswith("/api/images/"):
            filename = urllib.parse.unquote(parsed_path.path.replace("/api/images/", ""))
            image_path = IMAGES_DIR / filename
            
            if not image_path.exists() or not image_path.is_file():
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Image {filename} not found"}).encode())
                return
                
            try:
                os.remove(image_path)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"message": f"Image {filename} deleted successfully"}).encode())
                print(f"Deleted image: {filename}")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                print(f"Error deleting {filename}: {str(e)}")
        else:
            self.send_error(404, "Endpoint not found")
    
    def get_images(self):
        """Get list of images in the directory."""
        images = []
        for file in os.listdir(IMAGES_DIR):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) and not file.startswith('.'):
                images.append({
                    "filename": file,
                    "path": f"/images/{file}"
                })
        return images
    
    def generate_html(self):
        """Generate HTML for the image grid."""
        images = self.get_images()
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Management</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .image-container {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: black;
            transition: transform 0.2s;
            aspect-ratio: 1 / 1;
        }
        .image-container:hover {
            transform: translateY(-5px);
        }
        .image-container img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .delete-button {
            position: absolute;
            top: 0;
            right: 0;
            background: rgba(255, 0, 0, 0.7);
            color: white;
            border: none;
            width: 30px;
            height: 30px;
            border-radius: 0 0 0 8px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .image-container:hover .delete-button {
            opacity: 1;
        }
        .delete-button:hover {
            background: rgba(255, 0, 0, 0.9);
        }
        .counter {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            z-index: 1000;
        }
        .refresh-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .refresh-button:hover {
            background: #45a049;
        }
        .empty-message {
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <h1>Image Management</h1>
    
    <div class="counter">Images: <span id="image-count">""" + str(len(images)) + """</span></div>
    
    <div class="image-grid" id="image-grid">
"""
        
        if images:
            for image in images:
                html += f"""
        <div class="image-container" id="container-{image['filename']}">
            <img src="{image['path']}" alt="{image['filename']}">
            <button class="delete-button" onclick="deleteImage('{image['filename']}')">✕</button>
        </div>"""
        else:
            html += """
        <div class="empty-message">No images found</div>"""
            
        html += """
    </div>
    
    <button class="refresh-button" onclick="location.reload()">Refresh</button>

    <script>
        // Function to delete an image
        async function deleteImage(filename) {
            try {
                const response = await fetch(`/api/images/${encodeURIComponent(filename)}`, {
                    method: 'DELETE',
                });
                
                if (response.ok) {
                    // Remove the image container from the DOM
                    const container = document.getElementById(`container-${filename}`);
                    container.remove();
                    
                    // Update the counter
                    const counter = document.getElementById('image-count');
                    counter.textContent = parseInt(counter.textContent) - 1;
                    
                    // If no images left, show empty message
                    if (parseInt(counter.textContent) === 0) {
                        const grid = document.getElementById('image-grid');
                        grid.innerHTML = '<div class="empty-message">No images found</div>';
                    }
                } else {
                    const errorData = await response.json();
                    alert(`Error: ${errorData.error || 'Failed to delete image'}`);
                }
            } catch (error) {
                alert('An error occurred while deleting the image.');
                console.error(error);
            }
        }
    </script>
</body>
</html>
"""
        return html

def main():
    # Validate images directory
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        return
        
    # Set up and start the server
    handler = ImageHandler
    with socketserver.TCPServer((HOST, PORT), handler) as httpd:
        print(f"Server started at http://{HOST}:{PORT}")
        print(f"Image directory: {IMAGES_DIR}")
        print(f"Number of images: {len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) and not f.startswith('.')])}") 
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main() 