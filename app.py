from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from PIL import Image
import uuid
from flask_cors import CORS
import subprocess
import shutil

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
    

@app.route('/detect-madm', methods=['POST'])
def detect_madm():
    try:
        # Find the uploaded image
        upload_dir = app.config['UPLOAD_FOLDER']
        uploaded_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        
        if not uploaded_files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400
            
        filepath = os.path.join(upload_dir, uploaded_files[0])
        final_output = app.config['FINAL_OUTPUT_FOLDER']

        # Run processing pipeline with detection.py instead
        scripts = [
            ['python3', 'scripts/normalize.py', filepath, final_output],
            ['python3', 'scripts/splitimage.py', filepath, final_output],
            ['python3', 'scripts/detection.py', filepath, final_output],  # Changed this line
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

@app.route('/train-saved', methods=['POST'])
def train_saved_data():
    try:
        # 1. Merge annotations with proper line breaks
        annotations_dir = app.config['SAVED_ANNOTATIONS_FOLDER']
        output_csv = os.path.join(app.config['SAVED_DATA_FOLDER'], 'merged_annotations.csv')
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(annotations_dir) if f.endswith('.csv')]
        if not csv_files:
            return jsonify({'error': 'No annotation files found in saved_annotations'}), 400

        # Merge files with clean line breaks
        with open(output_csv, 'w') as outfile:
            for i, fname in enumerate(csv_files):
                with open(os.path.join(annotations_dir, fname), 'r') as infile:
                    lines = infile.readlines()
                    
                    # Skip header for subsequent files after first
                    if i > 0:
                        lines = lines[1:]
                        
                    # Clean lines and ensure proper newlines
                    cleaned_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped:  # Skip empty lines
                            cleaned_lines.append(stripped + '\n')
                    
                    outfile.writelines(cleaned_lines)

        # 2. Verify merged file and images
        if not os.path.exists(output_csv):
            return jsonify({'error': 'Merged annotations failed to create'}), 500
            
        images_dir = app.config['SAVED_DATA_FOLDER']
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if not image_files:
            return jsonify({'error': 'No images found in saved_data folder'}), 400

        # 3. Get training parameters
        model_type = request.form.get('model_type', 'SGN')
        epochs = request.form.get('epochs', '10')
        
        # Validate epochs
        try:
            epochs = int(epochs)
            if epochs < 1:
                return jsonify({'error': 'Epochs must be at least 1'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid epochs value'}), 400

        # 4. Set class mapping
        classes_file = 'monochrome.csv' if model_type == 'SGN' else 'color.csv'
        if not os.path.exists(classes_file):
            return jsonify({'error': f'Class file {classes_file} not found'}), 400

        # 5. Verify weights exist
        weights_path = os.path.abspath('snapshots/combine.h5')
        if not os.path.exists(weights_path):
            return jsonify({'error': f'Weights file not found at {weights_path}'}), 400

        # 6. Build training command
        cmd = [
            'python3', 'keras_retinanet/bin/train.py',
            '--weights', weights_path,
            '--batch-size', '16',
            '--epochs', str(epochs),
            '--snapshot-path', 'snapshots/',
            'csv', 
            os.path.abspath(output_csv),
            os.path.abspath(classes_file)
        ]

        # 7. Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output to terminal
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)

        if process.returncode != 0:
            return jsonify({'error': 'Training process failed'}), 500

        # 8. Return latest snapshot
        snapshots = []
        for root, _, files in os.walk('snapshots'):
            for file in files:
                if file.endswith('.h5') and file != 'combine.h5':
                    snapshots.append(os.path.join(root, file))
        
        if not snapshots:
            return jsonify({'error': 'No training snapshots generated'}), 500
            
        latest_snapshot = max(snapshots, key=os.path.getctime)
        return send_file(latest_snapshot)

    except Exception as e:
        print(f"[ERROR] Saved data training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Clean and setup directories
        shutil.rmtree('ft_upload', ignore_errors=True)
        os.makedirs('ft_upload', exist_ok=True)

        # Save CSV
        csv_file = request.files['csv']
        csv_path = os.path.join('ft_upload', 'annotations.csv')  # Define csv_path
        csv_file.save(csv_path)

        # Save images
        for img in request.files.getlist('images'):
            img.save(os.path.join('ft_upload', img.filename))

        # Get training parameters
        model_type = request.form.get('model_type', 'SGN')
        epochs = request.form.get('epochs', '10')
        classes_file = 'monochrome.csv' if model_type == 'SGN' else 'color.csv'  # Define classes_file

        # Validate epochs
        try:
            epochs = int(epochs)
            if epochs < 1:
                return jsonify({'error': 'Epochs must be at least 1'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid epochs value'}), 400

        # Run training
        cmd = [
            'python3', 'keras_retinanet/bin/train.py',
            '--weights', 'snapshots/combine.h5',
            '--batch-size', '16',
            '--epochs', str(epochs),
            '--snapshot-path', 'snapshots/',
            'csv', 
            csv_path,  # Now defined
            classes_file  # Now defined
        ]

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output to terminal
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)

        if process.returncode != 0:
            return jsonify({'error': 'Training failed'}), 500

        # Return latest snapshot
        snapshots = [f for f in os.listdir('snapshots') 
                   if f.endswith('.h5') and f != 'combine.h5']
        if not snapshots:
            return jsonify({'error': 'No snapshots generated'}), 500
            
        latest = max(snapshots, key=lambda f: os.path.getctime(os.path.join('snapshots', f)))
        return send_file(os.path.join('snapshots', latest))

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)