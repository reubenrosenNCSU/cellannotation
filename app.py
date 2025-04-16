from werkzeug.utils import secure_filename  # ADD THIS AT TOP OF FILE
from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from PIL import Image
import uuid
from flask_cors import CORS
import subprocess
import shutil
import h5py
import time
from PIL import Image, ImageOps
import numpy as np
import zipfile
import io
import gc  # Garbage collector
import time  # For delays
from flask import session
from datetime import timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import atexit

app = Flask(__name__)
CORS(app, supports_credentials=True)  # This will allow all domains to access your API

app.secret_key = 'test'  # Replace with a real secret key


@app.before_request
def set_user_session():
    if 'user_id' not in session:
        session.permanent = True
        session['user_id'] = str(uuid.uuid4())
    
    # Always update directory modification time on any request
    user_id = session['user_id']
    user_dir = os.path.join('users', user_id)
    if os.path.exists(user_dir):
        os.utime(user_dir, None)  # Update mtime on every request
    # Create user directories if they don't exist
    user_id = session['user_id']
    user_dirs = [
        os.path.join('users', user_id, 'uploads'),
        os.path.join('users', user_id, 'converted'),
        os.path.join('users', user_id, 'saved_data'),
        os.path.join('users', user_id, 'saved_annotations'),
        os.path.join('users', user_id, 'finaloutput'),
        os.path.join('users', user_id, 'ft_upload'),
        os.path.join('users', user_id, 'images'),
        os.path.join('users', user_id, 'input'),
        os.path.join('users', user_id, 'output'),
        os.path.join('users', user_id, 'output/output_csv'),
        os.path.join('users', user_id, 'snapshots')
    ]
    for dir_path in user_dirs:
        os.makedirs(dir_path, exist_ok=True)

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Session expires after 1 hour

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        user_id = session.get('user_id')
        if user_id:
            user_dir = os.path.join('users', user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                print(f"Cleaned up directory for user: {user_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return jsonify({'status': 'error'}), 500



# Add these additional directories to clear
CLEANUP_DIRS = ['output', 'input', 'images']

# Create directories if they don't exist
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)



def clear_uploaded_images():
    """Delete all files in the current user's upload folder"""
    user_id = session.get('user_id', 'default')  # Handle unauthenticated edge case
    user_upload_dir = os.path.join('users', user_id, 'uploads')
    for filename in os.listdir(user_upload_dir):
        file_path = os.path.join(user_upload_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")



@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = session['user_id']
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
        user_upload_dir = os.path.join('users', user_id, 'uploads')
        user_converted_dir = os.path.join('users', user_id, 'converted')

        # Read image and ensure RGB (3 channels)
        with Image.open(file.stream) as img:
            # Convert RGBA/LA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
            
            # Save original to preservation folder
            original_preserve_path = os.path.join(user_upload_dir, original_name)
            img.save(original_preserve_path, format='TIFF', compression='tiff_deflate')
            
            # Save processed RGB copy to working uploads
            upload_path = os.path.join(user_upload_dir, original_name)
            img.save(upload_path, format='TIFF', compression='tiff_deflate')

        # Generate preview (must be inside the try block)
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}.png"
        output_path = os.path.join(user_converted_dir, output_filename)
        
        with Image.open(upload_path) as img:
            # Normalize 16-bit data for PNG preview
            if img.mode == 'I;16':
                # Convert to numpy array and normalize
                img_array = np.array(img).astype(np.uint16)
                min_val = np.min(img_array)
                max_val = np.max(img_array)
                if max_val > min_val:  # Avoid division by zero
                    normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    normalized = (img_array // 256).astype(np.uint8)  # Fallback scaling
                img = Image.fromarray(normalized)
            img.save(output_path, "PNG")

        return jsonify({
            'converted_url': f'/converted/{output_filename}',
            'original_name': original_name,
            'base_name': base_name,
            'original_extension': original_extension
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/export-annotations', methods=['POST'])
def export_annotations():
    user_id = session['user_id']
    try:
        # Get both CSV data and current image name from request
        data = request.json
        csv_data = data['csv_data']
        original_filename = data['original_filename']
        user_upload_dir = os.path.join('users', user_id, 'uploads')

        # Get current TIFF path
        tiff_path = os.path.join(user_upload_dir, original_filename)
        if not os.path.exists(tiff_path):
            return jsonify({'error': 'Current TIFF file not found'}), 404

        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV
            zipf.writestr('annotations.csv', csv_data)
            # Add TIFF
            zipf.write(tiff_path, os.path.basename(tiff_path))

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{os.path.splitext(original_filename)[0]}_export.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-cropped', methods=['POST'])
def upload_cropped_file():
    user_id = session['user_id']
    try:
        # Get crop coordinates and original filename
        original_name = request.form['original_filename']
        x = int(float(request.form['x']))
        y = int(float(request.form['y']))
        width = int(float(request.form['width']))
        height = int(float(request.form['height']))
        user_upload_dir = os.path.join('users', user_id, 'uploads')
        user_convert_dir = os.path.join('users', user_id, 'converted')

        # Path to original TIFF
        upload_path = os.path.join(user_upload_dir, original_name)
        
        # Open and crop original image
        with Image.open(upload_path) as img:
            # Perform crop on original TIFF
            cropped_img = img.crop((x, y, x + width, y + height))
            
            # Overwrite original file with cropped version
            cropped_img.save(upload_path, format='TIFF', compression='tiff_deflate')

        # Generate new PNG preview from updated TIFF
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}.png"
        output_path = os.path.join(user_convert_dir, output_filename)
        cropped_img.save(output_path, "PNG")

        return jsonify({
            'converted_url': f'/converted/{output_filename}',
            'original_name': original_name,  # Keep original filename
            'base_name': os.path.splitext(original_name)[0],
            'original_extension': 'tiff'
        })

    except Exception as e:
        print(f"Error in upload-cropped: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500
    
@app.route('/detect-sgn', methods=['POST'])
def detect_sgn():
    user_id = session['user_id']
    user_cleanup_dirs = [
        os.path.join('users', user_id, 'output'),
        os.path.join('users', user_id, 'input'),
        os.path.join('users', user_id, 'images')
    ]
    try:
        # Find the uploaded image
        upload_dir = os.path.join('users', user_id, 'uploads')
        finaloutput_dir = os.path.join('users', user_id, 'finaloutput')
        output_dir = os.path.join('users', user_id, 'output')
        output_csv_dir = os.path.join('users', user_id, 'output/output_csv')
        input_dir = os.path.join('users', user_id, 'input')
        images_dir = os.path.join('users', user_id, 'images')
        threshold = request.json.get('threshold', 0.5)
        uploaded_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        output_csv_file = os.path.join(finaloutput_dir, 'annotations.csv')
        
        if not uploaded_files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400
            
        filepath = os.path.join(upload_dir, uploaded_files[0])
        final_output = finaloutput_dir

        # Run processing pipeline
        scripts = [
            ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
            ['python3', 'scripts/splitimage.py', input_dir, images_dir],
            ['python3', 'scripts/detection_SGN.py', images_dir, output_dir, str(threshold)],
            ['python3', 'scripts/mergecsv.py', output_csv_dir, output_csv_file]
        ]

        for script in scripts:
            print(f"\n🚀 Running: {' '.join(script)}")
            process = subprocess.Popen(
                script,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Print output in real-time and capture it
            output = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())  # Show in terminal
                    output.append(line)

            if process.returncode != 0:
                return jsonify({
                    'error': f'Script {script[1]} failed',
                    'message': ''.join(output)
                }), 500

        # Cleanup directories
        for dir_name in user_cleanup_dirs:
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
    user_id = session['user_id']
    converted_dir = os.path.join('users', user_id, 'converted')
    return send_from_directory(converted_dir, filename)

@app.route('/save-training-data', methods=['POST'])
def save_training_data():
    user_id = session['user_id']
    upload_dir = os.path.join('users', user_id, 'uploads')
    saved_data_dir = os.path.join('users', user_id, 'saved_data')
    csv_data_dir = os.path.join('users', user_id, 'saved_annotations')
    try:
        # Get processing parameters
        original_filename = request.form['original_filename']
        
        # Load original image directly without any processing
        original_path = os.path.join(upload_dir, original_filename)
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original image not found'}), 400

        # Create output filename
        tiff_filename = original_filename
        image_path = os.path.join(saved_data_dir, tiff_filename)
        
        # Convert directly to TIFF without any adjustments
        with Image.open(original_path) as img:
            img.save(
                image_path,
                format='TIFF',
                compression='tiff_deflate'
            )

        # Save CSV (rest remains the same)
        csv_file = request.files['csv']
        csv_filename = f"{uuid.uuid4()}.csv"
        csv_path = os.path.join(csv_data_dir, csv_filename)
        csv_file.save(csv_path)

        return jsonify({'message': 'Training data saved successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/clear-training-data', methods=['POST'])
def clear_training_data():
    user_id = session['user_id']
    saved_data_dir = os.path.join('users', user_id, 'saved_data')
    saved_annotations_dir = os.path.join('users', user_id, 'saved_annotations')
    try:
        # Clear saved data
        data_folder = saved_data_dir
        clear_folder(data_folder)
        
        # Clear saved annotations
        annotations_folder = saved_annotations_dir
        clear_folder(annotations_folder)
        
        return jsonify({'message': 'Training data cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/detect-madm', methods=['POST'])
def detect_madm():
    user_id = session['user_id']
    try:
        # Find the uploaded image
        upload_dir = os.path.join('users', user_id, 'uploads')
        input_dir = os.path.join('users', user_id, 'input')
        images_dir = os.path.join('users', user_id, 'images')
        finaloutput_dir = os.path.join('users', user_id, 'finaloutput')
        output_dir = os.path.join('users', user_id, 'output')
        output_csv_dir = os.path.join('users', user_id, 'output/output_csv')
        uploaded_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        threshold = request.json.get('threshold', 0.5)

        user_cleanup_dirs = [
        os.path.join('users', user_id, 'output'),
        os.path.join('users', user_id, 'input'),
        os.path.join('users', user_id, 'images')
        ]
        
        if not uploaded_files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400
            
        filepath = os.path.join(upload_dir, uploaded_files[0])
        final_output = finaloutput_dir
        output_csv_file = os.path.join(finaloutput_dir, 'annotations.csv')

        # Run processing pipeline with detection.py
        scripts = [
            ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
            ['python3', 'scripts/splitimage.py', input_dir, images_dir],
            ['python3', 'scripts/detection.py', images_dir, output_dir, str(threshold)],
            ['python3', 'scripts/mergecsv.py', output_csv_dir, output_csv_file]
        ]

        for script in scripts:
            print(f"\n🚀 Running: {' '.join(script)}")
            process = subprocess.Popen(
                script,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )

            # Print output in real-time and capture it
            output = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())  # Show in terminal
                    output.append(line)

            if process.returncode != 0:
                return jsonify({
                    'error': f'Script {script[1]} failed',
                    'message': ''.join(output)
                }), 500

        # Cleanup directories
        for dir_name in user_cleanup_dirs:
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
    user_id = session['user_id']
    user_snapshot_dir = os.path.join('users', user_id, 'snapshots')
    try:
        # Get number of images from form
        num_images = int(request.form.get('num_images', 7))
        
        # 1. Copy pre-train images to saved_data
        pre_train_dir = 'pre_train_SGN'
        saved_data_dir = os.path.join('users', user_id, 'saved_data')

        # Get sorted list of image files
        all_images = sorted([f for f in os.listdir(pre_train_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        selected_images = []

        if num_images > 0:
            # Get sorted list of image files
            all_images = sorted([f for f in os.listdir(pre_train_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
            selected_images = all_images[:num_images]  # Take first N

            # Copy selected images
            for filename in selected_images:
                src_path = os.path.join(pre_train_dir, filename)
                dest_path = os.path.join(saved_data_dir, filename)
                shutil.copy2(src_path, dest_path)

        # 2. Merge annotations with pre_annot.csv
        annotations_dir = os.path.join('users', user_id, 'saved_annotations')
        output_csv = os.path.join(saved_data_dir, 'merged_annotations.csv')
        pre_annot_path = os.path.join(pre_train_dir, 'pre_annot.csv')

        with open(output_csv, 'w') as outfile:
            # Process saved annotations
            csv_files = [f for f in os.listdir(annotations_dir) if f.endswith('.csv')]
            for i, fname in enumerate(csv_files):
                with open(os.path.join(annotations_dir, fname), 'r') as infile:
                    lines = infile.readlines()
                    if i > 0:
                        lines = lines[1:]  # Skip header for subsequent files
                    cleaned_lines = [line.strip() + '\n' for line in lines if line.strip()]
                    outfile.writelines(cleaned_lines)

            # Add filtered pre_annot.csv
            if os.path.exists(pre_annot_path):
                with open(pre_annot_path, 'r') as pre_file:
                    lines = pre_file.readlines()
                    if lines:
                        # Filter lines based on selected images
                        header = lines[0]
                        for line in lines[1:]:  # Skip header
                            filename_in_csv = line.split(',')[0].strip()
                            if filename_in_csv in selected_images:
                                outfile.write(line.strip() + '\n')

        # 2. Verify merged file and images
        if not os.path.exists(output_csv):
            return jsonify({'error': 'Merged annotations failed to create'}), 500
            
        images_dir = saved_data_dir
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        if not image_files:
            return jsonify({'error': 'No images found in saved_data folder'}), 400

        # 3. Get training parameters
        model_type = request.form.get('model_type', 'SGN')
        epochs = request.form.get('epochs', '20')
        
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
        weights_path = 'snapshots/SGN_Rene.h5' if model_type == 'SGN' else 'snapshots/MADMweights.h5'
        if not os.path.exists(weights_path):
            return jsonify({'error': f'Weights file not found at {weights_path}'}), 400

        # 6. Build training command
        cmd = [
            'python3', 'keras_retinanet/keras_retinanet/bin/train.py',
            '--weights', weights_path,
            '--freeze-backbone',
            '--lr', '1e-4',
            '--batch-size', '8',
            '--epochs', str(epochs),
            '--snapshot-path', user_snapshot_dir,  # ✅ User-specific snapshots
            'csv', output_csv,
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
        epoch_str = f"{epochs:02d}"
        expected_filename = f'resnet50_csv_{epoch_str}.h5'
        snapshot_path = os.path.join(user_snapshot_dir, expected_filename)

        # ===== START CRITICAL FIX =====
        # Wait for file to finish writing
        time.sleep(1)  # Wait 1 second
        gc.collect()  # Clean up memory

        # Copy using safe binary method
        fixed_path = os.path.join(user_snapshot_dir, 'last_used.h5')  # ✅
        with open(snapshot_path, 'rb') as src_file, open(fixed_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)
        # ===== END CRITICAL FIX =====
        
        if not os.path.exists(snapshot_path):
            return jsonify({'error': f'Expected snapshot {expected_filename} not found'}), 500

        try:
            with h5py.File(snapshot_path, 'r') as f:
                if 'model_weights' not in f:
                    raise ValueError("Invalid model structure")
        except Exception as e:
            return jsonify({'error': f'Invalid model file: {str(e)}'}), 500

        return send_file(
            snapshot_path,
            as_attachment=True,
            download_name=expected_filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        print(f"[ERROR] Saved data training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    user_id = session['user_id']
    user_snapshot_dir = os.path.join('users', user_id, 'snapshots')

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
        weights_file = 'snapshots/SGN_Rene.h5' if model_type == 'SGN' else 'snapshots/MADMweights.h5'

        # Validate epochs
        try:
            epochs = int(epochs)
            if epochs < 1:
                return jsonify({'error': 'Epochs must be at least 1'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid epochs value'}), 400

        # Run training
        cmd = [
            'python3', 'keras_retinanet/keras_retinanet/bin/train.py',
            '--weights', weights_file,
            '--lr', '1e-4',
            '--batch-size', '8',
            '--epochs', str(epochs),
            '--snapshot-path', user_snapshot_dir,  # ✅ User-specific snapshots
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
        # Get specific snapshot based on epochs
        epoch_str = f"{epochs:02d}"
        expected_filename = f'resnet50_csv_{epoch_str}.h5'
        snapshot_path = os.path.join(user_snapshot_dir, expected_filename)
    
            # ===== START CRITICAL FIX =====
        # Wait for file to finish writing
        time.sleep(1)  # Wait 1 second
        gc.collect()  # Clean up memory

        # Copy using safe binary method
        fixed_path = os.path.join(user_snapshot_dir, 'last_used.h5')
        with open(snapshot_path, 'rb') as src_file, open(fixed_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)
        # ===== END CRITICAL FIX =====

        return send_file(
            snapshot_path,
            as_attachment=True,
            download_name=expected_filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/detect-custom', methods=['POST'])
def detect_custom():
    user_id = session['user_id']
    try:
        # Get uploaded model and image
        h5_file = request.files['h5_file']
        model_type = request.form.get('model_type', 'SGN')  # Get model type from form
        upload_dir = os.path.join('users', user_id, 'uploads')
        final_output = os.path.join('users', user_id, 'finaloutput')
        
        # Save model temporarily
        model_path = os.path.join(upload_dir, secure_filename(h5_file.filename))
        h5_file.save(model_path)

        # Find uploaded image
        image_files = [f for f in os.listdir(upload_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        if not image_files:
            return jsonify({'error': 'No image found'}), 400
            
        image_name = image_files[0]
        image_path = os.path.join(upload_dir, image_name)

        # Choose the correct script based on model type
        detection_script = 'scripts/custom_detection_color.py' if model_type == 'MADM' else 'scripts/custom_detection.py'

        # Run detection script
        subprocess.run([
            'python3', detection_script,
            image_path,
            model_path,
            final_output
        ], check=True)

        # CORRECTED: Use full image name with extension for CSV filename
        csv_filename = f"{os.path.basename(image_path)}_result.csv"
        csv_path = os.path.join(final_output, 'output_csv', csv_filename)
        
        if not os.path.exists(csv_path):
            return jsonify({
                'error': f'Annotations file not found at: {csv_path}',
                'searched_path': csv_path
            }), 500

        with open(csv_path, 'r') as f:
            csv_data = f.read()

        return jsonify({'annotations': csv_data})

    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': 'Detection failed',
            'details': str(e),
            'cmd': e.cmd,
            'output': e.output
        }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup model file
        try:
            os.remove(model_path)
        except:
            pass

@app.route('/scale-image', methods=['POST'])
def scale_image():
    user_id = session['user_id']
    upload_dir = os.path.join('users', user_id, 'uploads')
    converted_dir = os.path.join('users', user_id, 'converted')
    try:
        diameter = float(request.form['diameter'])
        original_filename = request.form['original_filename']
        
        # Check if scaling needed
        if abs(diameter - 34) / 34 <= 0.25:
            return jsonify({'message': 'No scaling required'}), 200
            
        scaling_factor = 34.0 / diameter

        # Get CURRENT image path (from UPLOAD_FOLDER, not original)
        current_path = os.path.join(upload_dir, original_filename)
        if not os.path.exists(current_path):
            return jsonify({'error': 'Current image not found'}), 400

        # Resize and save
        with Image.open(current_path) as img:
            new_width = int(img.width * scaling_factor)
            new_height = int(img.height * scaling_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save scaled version to uploads (overwrite current)
            upload_path = os.path.join(upload_dir, original_filename)
            resized_img.save(upload_path, format='TIFF', compression='tiff_deflate')
            
            # Generate new preview
            unique_id = str(uuid.uuid4())
            output_filename = f"{unique_id}.png"
            output_path = os.path.join(converted_dir, output_filename)
            resized_img.save(output_path, "PNG")

            return jsonify({
                'converted_url': f'/converted/{output_filename}',
                'scaling_factor': scaling_factor,
                'new_width': new_width,
                'new_height': new_height
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/detect-finetuned', methods=['POST'])
def detect_finetuned():
    user_id = session['user_id']
    user_cleanup_dirs = [
            os.path.join('users', user_id, 'output'),
            os.path.join('users', user_id, 'input'),
            os.path.join('users', user_id, 'images')
        ]
    try:
        # 1. Get the copied model
        user_snapshot_dir = os.path.join('users', user_id, 'snapshots')
        model_path = os.path.join(user_snapshot_dir, 'last_used.h5')  # ✅ User's model
        
        # 2. Basic validation
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found'}), 400

        # 3. Find uploaded image
        upload_dir = os.path.join('users', user_id, 'uploads')
        output_dir = os.path.join('users', user_id, 'output')
        output_csv_dir = os.path.join('users', user_id, 'output', 'output_csv')
        image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.tiff', '.tif'))]
        if not image_files:
            return jsonify({'error': 'No image uploaded'}), 400
        image_path = os.path.join(upload_dir, image_files[0])

        # 4. Use same CSV path as custom detection
        csv_filename = f"{os.path.basename(image_path)}_result.csv"
        csv_path = os.path.join(output_csv_dir, csv_filename)

        # 5. Run detection script
        subprocess.run([
            'python3',
            'scripts/custom_detection.py',
            image_path,
            model_path,
            output_dir
        ], check=True)



        # 6. Read and return results
        with open(csv_path, 'r') as f:
            csv_data = f.read()

        #clear directories to prevent buildup of old data:

        for dir_name in user_cleanup_dirs:
            dir_path = os.path.join(os.getcwd(), dir_name)
            if os.path.exists(dir_path):
                clear_folder(dir_path)

        return jsonify({'annotations': csv_data})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def batch_process_image(user_id, image_path, detection_type, threshold, model_path=None):
    """Process single image using existing pipeline"""
    try:
        # Get user directories
        upload_dir = os.path.join('users', user_id, 'uploads')
        input_dir = os.path.join('users', user_id, 'input')
        images_dir = os.path.join('users', user_id, 'images')
        output_dir = os.path.join('users', user_id, 'output')
        final_dir = os.path.join('users', user_id, 'finaloutput')

        # Clear directories before processing
        for dir_path in [upload_dir, input_dir, images_dir, output_dir, final_dir]:
            clear_folder(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # Copy image to uploads
        shutil.copy(image_path, os.path.join(upload_dir, os.path.basename(image_path)))


        if detection_type == 'SGN':
            scripts = [
                ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
                ['python3', 'scripts/splitimage.py', input_dir, images_dir],
                ['python3', 'scripts/detection_SGN.py', images_dir, output_dir, str(threshold)],
                ['python3', 'scripts/mergecsv.py', os.path.join(output_dir, 'output_csv'), 
                 os.path.join(final_dir, 'annotations.csv')]
            ]
        elif detection_type == 'MADM':
            scripts = [
                ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
                ['python3', 'scripts/splitimage.py', input_dir, images_dir],
                ['python3', 'scripts/detection.py', images_dir, output_dir, str(threshold)],
                ['python3', 'scripts/mergecsv.py', os.path.join(output_dir, 'output_csv'),
                 os.path.join(final_dir, 'annotations.csv')]
            ]
        else:  # Custom
            scripts = [
                ['python3', 'scripts/custom_detection.py',
                 os.path.join(upload_dir, os.path.basename(image_path)),
                 model_path,
                 final_dir]
            ]

        # Execute scripts
        for script in scripts:
            result = subprocess.run(script, capture_output=True, text=True)
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}

        # Read results
        csv_path = os.path.join(final_dir, 'annotations.csv')
        with open(csv_path, 'r') as f:
            return {'success': True, 'csv': f.read(), 'filename': os.path.basename(image_path)}

    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/batch-detect', methods=['POST'])
def batch_detect():
    user_id = session['user_id']
    try:
        # Create batch directory
        batch_dir = os.path.join('users', user_id, 'batch_temp')
        os.makedirs(batch_dir, exist_ok=True)
        clear_folder(batch_dir)  # Clear previous temp files

        # Save uploaded files
        for file in request.files.getlist('images'):
            file.save(os.path.join(batch_dir, secure_filename(file.filename)))

        # Get parameters
        detection_type = request.form['detection_type']
        threshold = float(request.form.get('threshold', 0.5))
        custom_model = request.files.get('custom_model')

        # Handle custom model
        model_path = None
        if detection_type == 'custom' and custom_model:
            model_path = os.path.join(batch_dir, 'custom_model.h5')
            custom_model.save(model_path)

        # Process each image
        results = []
        for filename in os.listdir(batch_dir):
            file_path = os.path.join(batch_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.tiff', '.tif')):
                result = batch_process_image(
                    user_id=user_id,
                    image_path=file_path,
                    detection_type=detection_type,
                    threshold=threshold,
                    model_path=model_path
                )
                if result['success']:
                    results.append({
                        'csv': result['csv'],
                        'image': filename,
                        'csv_name': f"{os.path.splitext(filename)[0]}_annotations.csv"
                    })

        # Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for result in results:
                # Add CSV
                zipf.writestr(result['csv_name'], result['csv'])
                # Add original image
                zipf.write(os.path.join(batch_dir, result['image']), result['image'])

        # Cleanup
        shutil.rmtree(batch_dir, ignore_errors=True)
        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='batch_results.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def delete_expired_sessions():
    now = datetime.datetime.utcnow()  # Use UTC time
    users_dir = 'users'
    for user_id in os.listdir(users_dir):
        user_path = os.path.join(users_dir, user_id)
        if os.path.isdir(user_path):
            try:
                mod_time = datetime.datetime.utcfromtimestamp(os.path.getmtime(user_path))
                if (now - mod_time).total_seconds() > 3600:
                    shutil.rmtree(user_path)
                    print(f"Cleaned expired session: {user_id}")
            except Exception as e:
                print(f"Error cleaning {user_id}: {str(e)}")
# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=delete_expired_sessions, trigger="interval", hours=1)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)