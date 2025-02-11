from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
import uuid
from flask_cors import CORS
import subprocess


app = Flask(__name__)
CORS(app)  # This will allow all domains to access your API
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CONVERTED_FOLDER'] = 'converted'
app.config['MAX_CONTENT_LENGTH'] = 5000000 * 1024 * 1024  # if it exceeds 50MB limit
app.config['FINAL_OUTPUT_FOLDER'] = 'finaloutput'
os.makedirs(app.config['FINAL_OUTPUT_FOLDER'], exist_ok=True)

# Add these additional directories to clear
CLEANUP_DIRS = ['output', 'input', 'images']

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERTED_FOLDER'], exist_ok=True)

app.config['SAVED_DATA_FOLDER'] = 'saved_data'
app.config['SAVED_ANNOTATIONS_FOLDER'] = 'saved_annotations'
os.makedirs(app.config['SAVED_DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVED_ANNOTATIONS_FOLDER'], exist_ok=True)

def clear_uploaded_images():
    """Delete all files in uploads folder"""
    upload_dir = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Function to clear files inside a folder, keeping the folder structure intact
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        clear_uploaded_images()
        original_name = file.filename
        base_name = os.path.splitext(original_name)[0]
        original_extension = os.path.splitext(original_name)[1][1:].lower()

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_name)
        file.save(upload_path)

        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}.png"
        output_path = os.path.join(app.config['CONVERTED_FOLDER'], output_filename)
        
        with Image.open(upload_path) as img:
            img.save(output_path, "PNG")

        return jsonify({
            'converted_url': f'/converted/{output_filename}',
            'original_name': original_name,
            'base_name': base_name,
            'original_extension': original_extension
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/detect-sgn', methods=['POST'])
def detect_sgn():
    try:
        # Find the uploaded image
        upload_dir = app.config['UPLOAD_FOLDER']
        uploaded_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        
        if not uploaded_files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400
            
        filepath = os.path.join(upload_dir, uploaded_files[0])
        final_output = app.config['FINAL_OUTPUT_FOLDER']

        # Run processing pipeline
        scripts = [
            ['python3', 'scripts/normalize.py', filepath, final_output],
            ['python3', 'scripts/splitimage.py', filepath, final_output],
            ['python3', 'scripts/detection_SGN.py', filepath, final_output],
            ['python3', 'scripts/mergeimage.py', filepath, final_output],
            ['python3', 'scripts/mergecsv.py', filepath, final_output]
        ]

        for script in scripts:
            result = subprocess.run(script, capture_output=True, text=True)
            if result.returncode != 0:
                return jsonify({
                    'error': f'Script {script[1]} failed',
                    'message': result.stderr
                }), 500

        # Cleanup directories
        for dir_name in CLEANUP_DIRS:
            dir_path = os.path.join(os.getcwd(), dir_name)
            if os.path.exists(dir_path):
                clear_folder(dir_path)

        # Return generated annotations
        csv_path = os.path.join(final_output, 'annotations.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': 'No annotations generated'}), 500

        with open(csv_path, 'r') as f:
            csv_data = f.read()

        return jsonify({'annotations': csv_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/converted/<filename>')
def serve_converted(filename):
    return send_from_directory(app.config['CONVERTED_FOLDER'], filename)

@app.route('/save-training-data', methods=['POST'])
def save_training_data():
    try:
        # Check if files are present
        if 'image' not in request.files or 'csv' not in request.files:
            return jsonify({'error': 'Missing image or CSV'}), 400

        image_file = request.files['image']
        csv_file = request.files['csv']

        # Validate filenames match
        image_name = os.path.splitext(image_file.filename)[0]
        csv_name = os.path.splitext(csv_file.filename)[0]
        
        if image_name != csv_name:
            return jsonify({'error': 'Filename mismatch'}), 400

        # Save files
        image_path = os.path.join(app.config['SAVED_DATA_FOLDER'], image_file.filename)
        csv_path = os.path.join(app.config['SAVED_ANNOTATIONS_FOLDER'], csv_file.filename)
        
        image_file.save(image_path)
        csv_file.save(csv_path)

        return jsonify({'message': 'Training data saved successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/clear-training-data', methods=['POST'])
def clear_training_data():
    try:
        # Clear saved data
        data_folder = app.config['SAVED_DATA_FOLDER']
        clear_folder(data_folder)
        
        # Clear saved annotations
        annotations_folder = app.config['SAVED_ANNOTATIONS_FOLDER']
        clear_folder(annotations_folder)
        
        return jsonify({'message': 'Training data cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)