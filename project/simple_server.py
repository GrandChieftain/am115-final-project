#!/usr/bin/env python3
import http.server
import socketserver
import os
import urllib.parse
from pathlib import Path

# Set up the server to serve the images directory
PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

# Make sure the images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        """Log messages with paths to help debug"""
        print(f"{self.address_string()} - {format % args}")

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax."""
        # Parse the path
        parsed_path = urllib.parse.urlparse(path)
        path = parsed_path.path
        
        # If path is root, show directory listing
        if path == "/":
            return str(BASE_DIR)
            
        # If path starts with /images, serve from the images directory
        if path.startswith('/images/'):
            # Remove the /images/ prefix
            rel_path = path[8:]
            file_path = os.path.join(IMAGES_DIR, rel_path)
            return file_path
            
        # For any other path, try to serve it relative to BASE_DIR
        file_path = os.path.join(BASE_DIR, path.lstrip('/'))
        return file_path
        
    def do_GET(self):
        """Handle GET requests with extra debugging info."""
        # Add a special case for testing
        if self.path == '/test':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            # List all images in the directory
            images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            response = "Server is working!\n\n"
            response += f"Server root: {BASE_DIR}\n"
            response += f"Images directory: {IMAGES_DIR}\n"
            response += f"Images count: {len(images)}\n\n"
            
            if images:
                response += "Sample image URLs:\n"
                for img in images[:5]:  # Show first 5 images
                    response += f"http://localhost:{PORT}/images/{img}\n"
            
            self.wfile.write(response.encode())
            return
            
        return super().do_GET()

print(f"Starting server at http://localhost:{PORT}")
print(f"Serving images from: {IMAGES_DIR}")
print(f"Access images via: http://localhost:{PORT}/images/your_image.jpg")
print(f"Test server with: http://localhost:{PORT}/test")
print("Press Ctrl+C to stop the server")

httpd = socketserver.TCPServer(("", PORT), Handler)
httpd.serve_forever()
