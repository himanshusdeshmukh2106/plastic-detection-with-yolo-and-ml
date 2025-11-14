"""
Enhanced Flask Web Application for Plastic Detection
Combines ensemble models (96.5% accuracy) with advanced features
"""
from flask import Flask, request, jsonify, render_template_string, send_file
from tensorflow import keras
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import os
from pathlib import Path
import io
import base64
from datetime import datetime
import csv

app = Flask(__name__)

# Global variables
MODELS = []
CLASS_LABELS = []
ENSEMBLE_LOADED = False
PREDICTION_HISTORY = []

# Enhanced HTML Template with better UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plastic Detection System - 96.5% Accuracy</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .stats-bar {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        
        .upload-section:hover {
            background: #e9ecef;
            border-color: #764ba2;
            transform: translateY(-2px);
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: #495057;
            margin-bottom: 10px;
        }
        
        .upload-subtext {
            color: #6c757d;
        }
        
        #preview-container {
            margin: 30px 0;
            display: none;
        }
        
        #preview {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .image-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .info-label {
            font-weight: bold;
            color: #495057;
        }
        
        .info-value {
            color: #6c757d;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin: 30px 0;
        }
        
        button {
            flex: 1;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 18px 30px;
            font-size: 1.1em;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result-section {
            display: none;
            margin: 30px 0;
        }
        
        .result-card {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
        }
        
        .prediction-main {
            font-size: 2.5em;
            font-weight: bold;
            color: #2e7d32;
            margin: 15px 0;
            text-align: center;
        }
        
        .confidence-badge {
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .predictions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        
        .pred-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .pred-label {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #495057;
        }
        
        .pred-bar-container {
            background: #e9ecef;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .pred-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .history-section {
            margin-top: 40px;
            padding-top: 40px;
            border-top: 2px solid #dee2e6;
        }
        
        .history-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #495057;
        }
        
        .history-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .export-btn {
            background: #28a745;
            padding: 10px 20px;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .stats-bar {
                flex-direction: column;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .predictions-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌍 Plastic Detection System</h1>
        <p>Powered by 5-Model Ensemble | 96.5% Accuracy</p>
    </div>

    <div class="container">
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">96.5%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">5</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">6</div>
                <div class="stat-label">Classes</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="predCount">0</div>
                <div class="stat-label">Predictions</div>
            </div>
        </div>

        <div class="main-content">
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Click to Upload Image</div>
                <div class="upload-subtext">or drag and drop (JPG, PNG, JPEG)</div>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
            </div>

            <div id="preview-container">
                <img id="preview" alt="Preview">
            </div>

            <div class="image-info" id="imageInfo"></div>

            <div class="button-group" style="display:none" id="buttonGroup">
                <button onclick="predict()">🔮 Predict with Ensemble</button>
                <button onclick="reset()" style="background: #6c757d;">🔄 Reset</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing with 5-model ensemble...</p>
                <p style="color: #6c757d; margin-top: 10px;">This may take a few seconds</p>
            </div>

            <div class="result-section" id="resultSection">
                <div class="result-card">
                    <h2 style="text-align: center; color: #2e7d32;">📊 Prediction Results</h2>
                    <div class="prediction-main" id="predictionMain"></div>
                    <div style="text-align: center; margin: 20px 0;">
                        <span class="confidence-badge" id="confidenceBadge"></span>
                    </div>
                </div>

                <h3 style="margin: 30px 0 20px; color: #495057;">All Class Probabilities:</h3>
                <div class="predictions-grid" id="predictionsGrid"></div>
            </div>

            <div class="history-section" id="historySection" style="display:none">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 class="history-title">📜 Prediction History</h3>
                    <button class="export-btn" onclick="exportHistory()">📥 Export CSV</button>
                </div>
                <div id="historyList"></div>
            </div>
        </div>
    </div>

    <script>
        let currentImage = null;
        let predictionHistory = [];

        function handleFile(file) {
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = file;
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview-container').style.display = 'block';
                document.getElementById('buttonGroup').style.display = 'flex';
                document.getElementById('resultSection').style.display = 'none';
                
                // Show image info
                showImageInfo(file);
            };
            reader.readAsDataURL(file);
        }

        function showImageInfo(file) {
            const info = document.getElementById('imageInfo');
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            
            info.innerHTML = `
                <div class="info-row">
                    <span class="info-label">Filename:</span>
                    <span class="info-value">${file.name}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Size:</span>
                    <span class="info-value">${sizeMB} MB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Type:</span>
                    <span class="info-value">${file.type}</span>
                </div>
            `;
            info.style.display = 'block';
        }

        async function predict() {
            if (!currentImage) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';

            const formData = new FormData();
            formData.append('image', currentImage);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Display results
                document.getElementById('predictionMain').textContent = data.prediction.toUpperCase();
                document.getElementById('confidenceBadge').textContent = 
                    data.confidence + '% ±' + data.uncertainty + '%';

                // Display all predictions
                let html = '';
                for (const [label, prob] of Object.entries(data.all_predictions)) {
                    const percentage = (prob * 100).toFixed(2);
                    html += `
                        <div class="pred-card">
                            <div class="pred-label">${label.toUpperCase()}</div>
                            <div class="pred-bar-container">
                                <div class="pred-bar-fill" style="width: ${percentage}%">
                                    ${percentage}%
                                </div>
                            </div>
                        </div>
                    `;
                }
                document.getElementById('predictionsGrid').innerHTML = html;

                document.getElementById('resultSection').style.display = 'block';

                // Add to history
                addToHistory(data);

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function addToHistory(data) {
            const timestamp = new Date().toLocaleString();
            predictionHistory.push({
                timestamp,
                prediction: data.prediction,
                confidence: data.confidence,
                filename: currentImage.name
            });

            updateHistory();
            
            // Update prediction count
            document.getElementById('predCount').textContent = predictionHistory.length;
        }

        function updateHistory() {
            const historyList = document.getElementById('historyList');
            const historySection = document.getElementById('historySection');
            
            if (predictionHistory.length > 0) {
                historySection.style.display = 'block';
                
                let html = '';
                predictionHistory.slice().reverse().forEach((item, index) => {
                    html += `
                        <div class="history-item">
                            <div>
                                <strong>${item.prediction.toUpperCase()}</strong> (${item.confidence}%)
                                <br>
                                <small style="color: #6c757d;">${item.filename} - ${item.timestamp}</small>
                            </div>
                        </div>
                    `;
                });
                
                historyList.innerHTML = html;
            }
        }

        function exportHistory() {
            if (predictionHistory.length === 0) {
                alert('No predictions to export');
                return;
            }

            // Create CSV
            let csv = 'Timestamp,Filename,Prediction,Confidence\\n';
            predictionHistory.forEach(item => {
                csv += `"${item.timestamp}","${item.filename}","${item.prediction}","${item.confidence}%"\\n`;
            });

            // Download
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'prediction_history_' + new Date().getTime() + '.csv';
            a.click();
        }

        function reset() {
            currentImage = null;
            document.getElementById('preview-container').style.display = 'none';
            document.getElementById('buttonGroup').style.display = 'none';
            document.getElementById('resultSection').style.display = 'none';
            document.getElementById('imageInfo').style.display = 'none';
            document.getElementById('fileInput').value = '';
        }

        // Drag and drop
        const uploadArea = document.querySelector('.upload-section');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#e9ecef';
            uploadArea.style.borderColor = '#764ba2';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '#f8f9fa';
            uploadArea.style.borderColor = '#667eea';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#f8f9fa';
            uploadArea.style.borderColor = '#667eea';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });
    </script>
</body>
</html>
"""

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
        'confidence': f"{top_prob*100:.2f}",
        'uncertainty': f"{top_std*100:.1f}",
        'all_predictions': all_preds,
        'num_models': len(MODELS)
    }

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
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
        
        # Add to history
        PREDICTION_HISTORY.append({
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    return jsonify(PREDICTION_HISTORY)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': ENSEMBLE_LOADED,
        'num_models': len(MODELS),
        'classes': CLASS_LABELS,
        'total_predictions': len(PREDICTION_HISTORY)
    })

if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED PLASTIC DETECTION WEB APP")
    print("=" * 80)
    
    if load_ensemble_models():
        print("\n🚀 Starting enhanced web server...")
        print("📱 Open: http://localhost:5000")
        print("\nFeatures:")
        print("  ✅ 96.5% accuracy ensemble")
        print("  ✅ Beautiful responsive UI")
        print("  ✅ Prediction history")
        print("  ✅ CSV export")
        print("  ✅ Drag & drop support")
        print("\nPress Ctrl+C to stop")
        print("=" * 80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models!")
        print("Extract ensemble_models.zip first")
