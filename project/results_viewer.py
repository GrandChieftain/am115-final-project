#!/usr/bin/env python3
"""
Results Viewer for Google Lens Image Processing System
This script provides a web interface to view the results of Google Lens image processing.

Usage:
  python results_viewer.py             # Start the viewer
  python results_viewer.py --port 8080 # Start the viewer on a specific port
"""
import os
import json
import csv
import http.server
import socketserver
import urllib.parse
import argparse
from pathlib import Path
import shutil
from datetime import datetime
import socket
import threading
import time
import sys
import requests

# Import from config instead of redefining
from config import BASE_DIR, IMAGES_DIR, RESULTS_DIR, BASE_URL, VIEWER_PORT

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to matches CSV file
MATCHES_CSV = os.path.join(RESULTS_DIR, "matches.csv")

# Initialize matches_data from the file
matches_data = {}
if os.path.exists(MATCHES_CSV):
    with open(MATCHES_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                image_id = row[0]
                # Parse selected matches - might be a single value or comma-separated list
                if row[1]:
                    if ',' in row[1]:
                        # Multiple selections (comma-separated)
                        selected_matches = [int(idx) for idx in row[1].split(',') if idx.strip()]
                    else:
                        # Single selection
                        selected_matches = [int(row[1])]
                else:
                    selected_matches = []
                    
                matches_data[image_id] = selected_matches
                print(f"Loaded selections for {image_id}: {selected_matches}")

def is_port_in_use(port):
    """Check if a port is already in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def check_api_server():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=2)
        if response.status_code == 200:
            print(f"✓ API server is running at {BASE_URL}")
            return True
        else:
            print(f"⚠ Warning: API server returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Warning: Could not connect to API server: {e}")
        print(f"Some features will be limited. Make sure the API server is running at {BASE_URL}")
        return False

def get_results_data():
    """Get results data directly from files if API is not available."""
    results_data = {}
    images_with_results = []
    
    # Go through all JSON files in the results directory
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json') or filename == 'matches.csv':
            continue
            
        try:
            with open(os.path.join(RESULTS_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            image_id = os.path.splitext(filename)[0]
            results = data.get('results', [])
            selected_match = matches_data.get(image_id)
            
            results_data[image_id] = {
                "results": results,
                "selected_match": selected_match
            }
            
            # Keep track of the images that have results
            if data.get('query_image'):
                images_with_results.append(data.get('query_image'))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return results_data, images_with_results

def get_all_images():
    """Get all images from the images directory."""
    image_files = []
    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and not filename.startswith('.'):
            image_files.append(filename)
    return sorted(image_files)

class ResultsViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Serve images directly
        if parsed_path.path.startswith('/images/'):
            image_path = parsed_path.path[1:]  # Remove leading slash
            self.serve_file(os.path.join(BASE_DIR, image_path))
            return
        
        # Handle delete request
        if parsed_path.path.startswith('/delete/'):
            image_name = parsed_path.path[8:]  # Remove /delete/
            self.delete_image(image_name)
            self.send_response(302)  # Redirect
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        # Handle match selection
        if parsed_path.path.startswith('/select/'):
            parts = parsed_path.path[8:].split('/')
            if len(parts) == 2:
                image_id, match_index = parts
                success = False
                
                # Try API call first, then fall back to local update
                try:
                    response = requests.post(f"{BASE_URL}/api/results/{image_id}/{match_index}")
                    if response.status_code == 200:
                        print(f"Selected match {match_index} for {image_id} via API")
                        success = True
                except Exception:
                    # Update local matches data - handle multiple selections
                    if image_id not in matches_data:
                        matches_data[image_id] = []
                    
                    match_idx = int(match_index)
                    # Toggle selection: add if not there, remove if already there
                    if match_idx in matches_data[image_id]:
                        matches_data[image_id].remove(match_idx)
                    else:
                        matches_data[image_id].append(match_idx)
                    
                    # Sort selections for consistency
                    matches_data[image_id].sort()
                    
                    # Write to matches CSV
                    with open(MATCHES_CSV, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['image_id', 'selected_match'])
                        for img_id, matches in matches_data.items():
                            # Convert list of selections to comma-separated string
                            matches_str = ','.join(str(m) for m in matches) if matches else ''
                            writer.writerow([img_id, matches_str])
                    
                    print(f"Updated matches for {image_id} locally: {matches_data[image_id]}")
                    success = True
                
                # Return JSON response instead of redirecting
                self.send_response(200 if success else 500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response_data = {
                    "success": success,
                    "image_id": image_id,
                    "selected_match": matches_data[image_id] if success else None
                }
                self.wfile.write(json.dumps(response_data).encode())
                return
        
        # Serve the main page
        if parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.serve_main_page()
            return
        
        # Serve static files (CSS, JavaScript)
        if parsed_path.path.startswith('/static/'):
            # Handle static files like CSS, JS, etc.
            return super().do_GET()
        
        # Default: 404 Not Found
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'404 Not Found')

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Handle save selections
        if parsed_path.path == '/save-selections':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                # Parse the JSON data
                selections = json.loads(post_data)
                
                # Process each selection
                for image_id, match_indices in selections.items():
                    # Store the full list of selected matches
                    if match_indices and len(match_indices) > 0:
                        # Sort for consistency
                        sorted_indices = sorted(match_indices)
                        matches_data[image_id] = sorted_indices
                    else:
                        # Remove the match if none are selected
                        if image_id in matches_data:
                            del matches_data[image_id]
                
                # Write to matches CSV
                with open(MATCHES_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_id', 'selected_match'])
                    for img_id, matches in matches_data.items():
                        # Convert list of selections to comma-separated string
                        matches_str = ','.join(str(m) for m in matches) if matches else ''
                        writer.writerow([img_id, matches_str])
                
                print(f"Saved selections for {len(selections)} images")
                
                # Return success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response_data = {
                    "success": True,
                    "message": f"Saved selections for {len(selections)} images"
                }
                self.wfile.write(json.dumps(response_data).encode())
                return
            except Exception as e:
                print(f"Error saving selections: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response_data = {
                    "success": False,
                    "error": str(e)
                }
                self.wfile.write(json.dumps(response_data).encode())
                return
        
        # Default: 404 Not Found
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'404 Not Found')

    def serve_file(self, file_path):
        """Serve a file from the filesystem."""
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'404 Not Found')
            return
        
        # Determine content type based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        content_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.html': 'text/html',
        }.get(ext, 'application/octet-stream')
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())
    
    def delete_image(self, image_name):
        """Delete an image and its associated results using the API or directly."""
        try:
            # Try to use the API to delete the image
            api_success = False
            try:
                response = requests.delete(f"{BASE_URL}/api/images/{image_name}")
                if response.status_code == 200:
                    api_success = True
                    print(f"Deleted image {image_name} via API")
            except Exception:
                pass
                
            # If API failed, delete directly
            if not api_success:
                # Delete the image file
                image_path = os.path.join(IMAGES_DIR, image_name)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
                # Delete the results file
                base_name = os.path.splitext(image_name)[0]
                results_path = os.path.join(RESULTS_DIR, f"{base_name}.json")
                if os.path.exists(results_path):
                    os.remove(results_path)
                
                print(f"Deleted image {image_name} directly")
                
            # Remove from local matches data too
            base_name = os.path.splitext(image_name)[0]
            if base_name in matches_data:
                del matches_data[base_name]
                # Update matches CSV
                with open(MATCHES_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_id', 'selected_match'])
                    for img_id, match in matches_data.items():
                        writer.writerow([img_id, match])
        except Exception as e:
            print(f"Error deleting image: {e}")
    
    def serve_main_page(self):
        """Serve the main page with image grid."""
        try:
            api_working = False
            image_list = []
            results_list = []
            
            # Try to get data from API
            try:
                # Get images from API
                images_response = requests.get(f"{BASE_URL}/api/images")
                if images_response.status_code == 200:
                    image_list = images_response.json()
                    
                # Get results from API
                results_response = requests.get(f"{BASE_URL}/api/results")
                if results_response.status_code == 200:
                    results_list = results_response.json()
                    api_working = True
            except Exception:
                pass
                
            # If API failed, get data directly from files
            if not api_working:
                # Get results data and images with results
                result_data, images_with_results = get_results_data()
                
                # Convert result_data to results_list format
                results_list = []
                for image_id, data in result_data.items():
                    results_list.append({
                        "image_id": image_id,
                        "matches": data["results"],
                        "selected_match": data["selected_match"]
                    })
                
                # Get all images from directory
                all_image_files = get_all_images()
                
                # Convert to image_list format
                image_list = []
                for img_file in all_image_files:
                    image_id = os.path.splitext(img_file)[0]
                    processed = image_id in result_data
                    image_list.append({
                        "filename": img_file,
                        "path": f"/images/{img_file}",
                        "processed": processed
                    })
            
            # Convert to dictionary for easier access
            result_data = {}
            for result in results_list:
                image_id = result["image_id"]
                matches = result["matches"]
                selected_match = result["selected_match"]
                
                # Also update local matches data
                if selected_match and isinstance(selected_match, list):
                    matches_data[image_id] = selected_match
                
                result_data[image_id] = {
                    "results": matches,
                    "selected_match": selected_match
                }
            
            # Map images to their filenames for easier access
            image_files = []
            for img in image_list:
                image_files.append(img["filename"])
            image_files.sort()  # Sort alphabetically
            
            # Generate HTML
            html = self.generate_html(image_files, result_data, api_working)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())
    
    def generate_html(self, image_files, result_data, api_working=True):
        """Generate the HTML for the main page."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Results Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .info {{
            font-size: 14px;
            color: #666;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }}
        .row {{
            display: contents;
        }}
        .item {{
            position: relative;
            background-color: #fff;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .item.query {{
            border-right: 2px solid #333;
            margin-right: 10px;
            cursor: default;
        }}
        .item.selected {{
            border: 3px solid #4CAF50;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
        }}
        .image-container {{
            position: relative;
            width: 100%;
            height: 200px;
            background-color: #000;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .image-container img {{
            width: 100%;          /* always stretch to the full box width … */
            height: 100%;         /* … and height, letting object‑fit decide which wins */
            object-fit: contain;  /* preserves aspect ratio – bars only on one axis */
            display: block;
            margin: 0 auto;
            object-fit: contain;  /* contain will show the whole image with possible black bars */
        }}
        .delete-btn {{
            position: absolute;
            top: 5px;
            right: 5px;
            width: 20px;
            height: 20px;
            background-color: rgba(255, 0, 0, 0.7);
            color: white;
            font-weight: bold;
            border-radius: 50%;
            text-align: center;
            line-height: 20px;
            cursor: pointer;
            z-index: 10;
        }}
        .details {{
            padding: 10px;
            font-size: 12px;
        }}
        .source {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .title {{
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .count {{
            position: absolute;
            top: 5px;
            left: 5px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            z-index: 10;
        }}
        .vertical-separator {{
            grid-column: 1;
            border-right: 2px solid #333;
        }}
        .no-results {{
            grid-column: span 5;
            text-align: center;
            padding: 20px;
            color: #999;
        }}
        .api-info {{
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }}
        .api-warning {{
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        #save-button {{
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-left: 15px;
        }}
        #save-button.unsaved {{
            background-color: #dc3545;
            animation: pulse 1.5s infinite;
        }}
        #emergency-save {{
            padding: 5px 10px;
            background-color: #dc3545;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-left: 10px;
            font-size: 12px;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        .expand-btn {{
            position: absolute;
            bottom: 5px;
            right: 5px;
            width: 20px;
            height: 20px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            font-weight: bold;
            border-radius: 50%;
            text-align: center;
            line-height: 20px;
            cursor: pointer;
            z-index: 10;
            font-size: 14px;
        }}
    </style>
    <script>
        // Global state to track selections
        const selections = {{}};
        let hasUnsavedChanges = false;
        
        // Initialize with server-provided data
        const initialSelections = {json.dumps(matches_data)};
        
        // Initialize selections from existing data
        function initSelections() {{
            // First load any server-provided selections
            Object.assign(selections, initialSelections);
            
            // Then find any items with the 'selected' class in the DOM
            document.querySelectorAll('.row').forEach(row => {{
                const imageId = row.getAttribute('data-image-id');
                if (!imageId) return;
                
                if (!selections[imageId]) {{
                    selections[imageId] = [];
                }}
                
                // Add any pre-selected items from the DOM that aren't already in selections
                row.querySelectorAll('.item.selected').forEach(item => {{
                    const matchIndex = parseInt(item.getAttribute('data-match-index'));
                    if (matchIndex !== null && !isNaN(matchIndex) && !selections[imageId].includes(matchIndex)) {{
                        selections[imageId].push(matchIndex);
                    }}
                }});
                
                // Sort selections for consistency
                selections[imageId].sort();
            }});
        }}
        
        // Toggle selection of an item
        function toggleMatch(imageId, matchIndex) {{
            // Initialize array if it doesn't exist
            if (!selections[imageId]) {{
                selections[imageId] = [];
            }}
            
            // Convert to number for comparison
            matchIndex = parseInt(matchIndex);
            
            // Toggle selection state
            const index = selections[imageId].indexOf(matchIndex);
            if (index > -1) {{
                // Remove if already selected
                selections[imageId].splice(index, 1);
            }} else {{
                // Add if not selected
                selections[imageId].push(matchIndex);
            }}
            
            // Sort the array for consistency
            selections[imageId].sort();
            
            // Update UI
            updateUI(imageId);
            
            // Mark that there are unsaved changes
            hasUnsavedChanges = true;
            document.getElementById('save-button').classList.add('unsaved');
            
            return false;
        }}
        
        // Update UI based on current selections
        function updateUI(imageId) {{
            const row = document.querySelector('.row-' + imageId);
            if (!row) return;
            
            // Reset all items in this row
            row.querySelectorAll('.item:not(.query)').forEach(item => {{
                const itemIndex = parseInt(item.getAttribute('data-match-index'));
                if (selections[imageId] && selections[imageId].includes(itemIndex)) {{
                    item.classList.add('selected');
                }} else {{
                    item.classList.remove('selected');
                }}
            }});
        }}
        
        // Save all selections to server
        function saveSelections() {{
            const data = JSON.stringify(selections);
            
            fetch('/save-selections', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: data
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    hasUnsavedChanges = false;
                    document.getElementById('save-button').classList.remove('unsaved');
                    alert('Selections saved successfully!');
                }} else {{
                    alert('Error saving selections: ' + (data.error || 'Unknown error'));
                }}
            }})
            .catch(error => {{
                console.error('Error saving selections:', error);
                alert('Error saving selections. Please try again.');
            }});
        }}
        
        // Emergency save function - saves to both localStorage and server
        function emergencySave() {{
            // Save to localStorage first
            localStorage.setItem('savedSelections', JSON.stringify(selections));
            console.log("Emergency save to localStorage complete");
            
            // Then try to save to server
            saveSelections();
            
            return false;
        }}
        
        // Initialize when page loads
        window.onload = function() {{
            initSelections();
            
            // Try to restore selections from localStorage (if a reload happened)
            const savedSelectionsJSON = localStorage.getItem('savedSelections');
            if (savedSelectionsJSON) {{
                try {{
                    const savedSelections = JSON.parse(savedSelectionsJSON);
                    console.log("Found saved selections in localStorage:", savedSelections);
                    
                    // Merge with current selections, giving priority to saved ones
                    Object.assign(selections, savedSelections);
                    
                    // Update UI for all images with selections
                    for (const imageId in selections) {{
                        if (selections[imageId] && selections[imageId].length > 0) {{
                            updateUI(imageId);
                        }}
                    }}
                    
                    // Mark as having unsaved changes
                    hasUnsavedChanges = true;
                    document.getElementById('save-button').classList.add('unsaved');
                    
                    console.log("Successfully restored selections from localStorage");
                }} catch (error) {{
                    console.error("Error restoring selections:", error);
                }}
                
                // Clear localStorage to avoid restoring multiple times
                localStorage.removeItem('savedSelections');
            }}
            
            // Update UI for all images with selections from server
            for (const imageId in selections) {{
                if (selections[imageId] && selections[imageId].length > 0) {{
                    updateUI(imageId);
                }}
            }}
            
            // Log initial selections for debugging
            console.log("Final initialized selections:", JSON.stringify(selections));
            
            // Warn user about unsaved changes when leaving page
            window.addEventListener('beforeunload', function(e) {{
                if (hasUnsavedChanges) {{
                    e.preventDefault();
                    e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
                    return e.returnValue;
                }}
            }});
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>Image Results Viewer</h1>
        <div class="info">
            {len(image_files)} images | Click on any result to select it as the match
            <button id="save-button" onclick="saveSelections()">Save Selections</button>
            <button id="emergency-save" onclick="emergencySave()">Emergency Save</button>
        </div>
    </div>
"""

        # Show warning if API is not working
        if not api_working:
            html += """    <div class="api-warning">
        <strong>Warning:</strong> API server not detected. Running in limited mode with direct file access. 
        Some features may not work as expected.
    </div>
"""

        html += f"""    <div class="api-info">
        API: <a href="{BASE_URL}/docs" target="_blank">{BASE_URL}/docs</a>
    </div>
"""
        
        # Generate the grid
        html += '<div class="grid">\n'
        
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            html += f'    <div class="row row-{base_name}" data-image-id="{base_name}">\n'
            
            # Query image
            html += f"""        <div class="item query">
            <div class="image-container">
                <a href="/images/{image_file}" target="_blank">
                    <img src="/images/{image_file}" alt="{image_file}">
                </a>
                <div class="count">{image_files.index(image_file) + 1}/{len(image_files)}</div>
                <div class="delete-btn" onclick="if(confirm('Delete {image_file} and its results?')) window.location.href='/delete/{image_file}'">X</div>
                <a href="/images/{image_file}" target="_blank" class="expand-btn" title="View full image">+</a>
            </div>
            <div class="details">
                <div class="title">{base_name}</div>
            </div>
        </div>
"""
            
            # Results
            image_data = result_data.get(base_name, {})
            results = image_data.get("results", [])
            selected_match = image_data.get("selected_match") or matches_data.get(base_name)
            
            # Ensure selected_match is always a list for consistency
            if selected_match is None:
                selected_match = []
            elif not isinstance(selected_match, list):
                selected_match = [selected_match]  # Convert single value to list
            
            if not results:
                html += f'        <div class="no-results">No results available</div>\n'
            else:
                # Display available results
                for i, result in enumerate(results[:5]):
                    # Check if this result should be pre-selected
                    selected_class = " selected" if i in selected_match else ""
                    title = result.get("title", "No title")
                    source = result.get("source", "Unknown source")
                    
                    # Use image if available, otherwise use thumbnail
                    image_url = result.get("image", "")
                    if not image_url:
                        image_url = result.get("thumbnail", "")
                    
                    link_url = result.get("link", "#")
                    
                    html += f"""        <div class="item{selected_class} item-{i} result" data-match-index="{i}" onclick="return toggleMatch('{base_name}', {i})">
            <div class="image-container">
                <a href="{link_url}" target="_blank" onclick="event.stopPropagation();">
                    <img src="{image_url}" alt="{title}" onclick="event.stopPropagation();">
                </a>
                <div class="count">{i + 1}</div>
            </div>
            <div class="details">
                <div class="source">{source}</div>
                <div class="title">{title}</div>
            </div>
        </div>
"""
                
                # Add placeholders for missing results to keep grid layout consistent
                for i in range(len(results[:5]), 5):
                    html += f"""        <div class="item result" style="opacity: 0.3; cursor: default;">
            <div class="image-container">
                <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #999;">
                    Placeholder
                </div>
            </div>
            <div class="details">
                <div class="source">-</div>
                <div class="title">-</div>
            </div>
        </div>
"""
            
            html += '    </div>\n'
        
        html += '</div>\n'
        html += '</body>\n</html>'
        
        return html.encode()

def run_server(port=VIEWER_PORT):
    """Run the HTTP server on the specified port."""
    if is_port_in_use(port):
        print(f"⚠ Port {port} is already in use.")
        print(f"You can access the viewer at http://localhost:{port}/")
        return False
        
    handler = ResultsViewerHandler
    
    try:
        httpd = socketserver.TCPServer(("", port), handler)
        local_ip = get_local_ip()
        print(f"✓ Results Viewer running at:")
        print(f"  - http://localhost:{port}/")
        print(f"  - http://{local_ip}:{port}/ (local network)")
        if check_api_server():
            print(f"✓ API Documentation: {BASE_URL}/docs")
        
        print("\nPress Ctrl+C to stop the server")
        httpd.serve_forever()
        return True
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Google Lens Results Viewer")
    parser.add_argument("--port", type=int, default=VIEWER_PORT, help="Port to run the server on")
    args = parser.parse_args()
    
    # Create the results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create matches CSV if it doesn't exist
    if not os.path.exists(MATCHES_CSV):
        with open(MATCHES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'selected_match'])
    
    # Start the server
    print(f"Starting Results Viewer server on port {args.port}...")
    run_server(args.port)

if __name__ == "__main__":
    main() 