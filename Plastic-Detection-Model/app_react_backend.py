"""
Flask Backend for React Frontend - Plastic Detection System
Supports mobile capture, dashboard, and ensemble predictions
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import io
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
MODELS = []
CLASS_LABELS = []
ENSEMBLE_LOADED = False
DETECTION_HISTORY = []

def load_ensemble_models():
    """Load all ensemble models"""
    global MODELS, CLASS_LABELS, ENSEMBLE_LOADED
    
    if ENSEMBLE_LOADED:
        return True
    
    print("Loading ensemble models...")
    
    possible_paths = [
        'ensemble_models/tf_files_ensemble',
        'Plastic-Detection-Model/ensemble_models/tf_files_ensemble',
        '../ensemble_models/tf_files_ensemble'
    ]
    
    ensemble_dir = None
    for path in possible_paths:
        if Path(path).exists():
            ensemble_dir = Path(path)
            break
    
    if not ensemble_dir:
        print("❌ Ensemble models not found!")
        return False
    
    # Load class labels
    labels_file = ensemble_dir / 'class_indices.json'
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            class_indices = json.load(f)
        CLASS_LABELS = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    else:
        CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Load models
    model_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith('model_')])
    
    for model_dir in model_dirs:
        model_path = model_dir / 'final_model.h5'
        if model_path.exists():
            try:
                model = keras.models.load_model(str(model_path))
                MODELS.append(model)
                print(f"  ✅ Loaded {model_dir.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {model_dir.name}: {e}")
    
    if MODELS:
        print(f"\n✅ Loaded {len(MODELS)} models successfully!")
        ENSEMBLE_LOADED = True
        return True
    else:
        print("❌ No models loaded!")
        return False

def predict_with_ensemble(image):
    """Predict using ensemble of models"""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = []
    for model in MODELS:
        pred = model.predict(img_array, verbose=0)[0]
        predictions.append(pred)
    
    avg_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    
    top_idx = np.argmax(avg_prediction)
    top_label = CLASS_LABELS[top_idx]
    top_prob = avg_prediction[top_idx]
    top_std = std_prediction[top_idx]
    
    all_preds = {CLASS_LABELS[i]: float(avg_prediction[i]) for i in range(len(CLASS_LABELS))}
    
    return {
        'prediction': top_label,
        'confidence': round(top_prob * 100, 2),
        'uncertainty': round(top_std * 100, 1),
        'all_predictions': all_preds,
        'num_models': len(MODELS)
    }

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': ENSEMBLE_LOADED,
        'num_models': len(MODELS),
        'classes': CLASS_LABELS,
        'total_detections': len(DETECTION_HISTORY)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict waste type from image"""
    if not ENSEMBLE_LOADED:
        return jsonify({'error': 'Models not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        result = predict_with_ensemble(image)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-detection', methods=['POST'])
def save_detection():
    """Save detection with location data"""
    try:
        data = request.json
        
        detection = {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'prediction': data.get('prediction'),
            'confidence': data.get('confidence'),
            'location': data.get('location')
        }
        
        DETECTION_HISTORY.append(detection)
        
        return jsonify({'success': True, 'total': len(DETECTION_HISTORY)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    """Get detection history"""
    limit = request.args.get('limit', 50, type=int)
    
    # Return most recent detections
    recent = DETECTION_HISTORY[-limit:] if len(DETECTION_HISTORY) > limit else DETECTION_HISTORY
    recent.reverse()  # Most recent first
    
    return jsonify({
        'history': recent,
        'total': len(DETECTION_HISTORY)
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics for dashboard"""
    time_range = request.args.get('range', 'all')
    
    # Filter by time range
    now = datetime.now()
    filtered_history = []
    
    for detection in DETECTION_HISTORY:
        try:
            det_time = datetime.fromisoformat(detection['timestamp'])
            
            if time_range == 'today':
                if det_time.date() == now.date():
                    filtered_history.append(detection)
            elif time_range == 'week':
                if (now - det_time).days <= 7:
                    filtered_history.append(detection)
            elif time_range == 'month':
                if (now - det_time).days <= 30:
                    filtered_history.append(detection)
            else:  # all
                filtered_history.append(detection)
        except:
            filtered_history.append(detection)
    
    # Calculate statistics
    total = len(filtered_history)
    
    # Count by class
    by_class = defaultdict(int)
    total_confidence = 0
    
    for detection in filtered_history:
        pred = detection.get('prediction', 'unknown')
        by_class[pred] += 1
        total_confidence += detection.get('confidence', 0)
    
    avg_confidence = (total_confidence / total) if total > 0 else 0
    
    return jsonify({
        'total_detections': total,
        'by_class': dict(by_class),
        'avg_confidence': avg_confidence / 100,  # Convert to 0-1 range
        'time_range': time_range
    })

@app.route('/export-csv', methods=['GET'])
def export_csv():
    """Export detection history as CSV"""
    try:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Prediction', 'Confidence', 'Latitude', 'Longitude', 'Accuracy'])
        
        # Write data
        for detection in DETECTION_HISTORY:
            location = detection.get('location', {})
            writer.writerow([
                detection.get('timestamp', ''),
                detection.get('prediction', ''),
                f"{detection.get('confidence', 0)}%",
                location.get('latitude', '') if location else '',
                location.get('longitude', '') if location else '',
                f"{location.get('accuracy', '')}m" if location and location.get('accuracy') else ''
            ])
        
        # Create response
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear detection history"""
    global DETECTION_HISTORY
    DETECTION_HISTORY = []
    return jsonify({'success': True, 'message': 'History cleared'})

if __name__ == '__main__':
    print("=" * 80)
    print("PLASTIC DETECTION - REACT BACKEND")
    print("=" * 80)
    
    if load_ensemble_models():
        print("\n🚀 Starting Flask backend for React frontend...")
        print("📱 Backend: http://localhost:5000")
        print("🌐 Frontend: http://localhost:5173 (run 'npm run dev' in frontend folder)")
        print("\nEndpoints:")
        print("  GET  /health          - Health check")
        print("  POST /predict         - Predict waste type")
        print("  POST /save-detection  - Save detection with location")
        print("  GET  /history         - Get detection history")
        print("  GET  /stats           - Get dashboard statistics")
        print("  GET  /export-csv      - Export history as CSV")
        print("\nPress Ctrl+C to stop")
        print("=" * 80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models!")
        print("Make sure ensemble models are in the correct directory")
