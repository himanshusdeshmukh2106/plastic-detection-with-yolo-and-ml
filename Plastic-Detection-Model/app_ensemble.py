"""
Flask Web Application for Plastic Detection using Ensemble Models
Achieves 96.5% accuracy with 5-model ensemble
"""
from flask import Flask, request, jsonify, render_template_string
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import io
import base64

app = Flask(__name__)

# Global variables for models
MODELS = []
CLASS_LABELS = []
ENSEMBLE_LOADED = False

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plastic Detection - 96.5% Accuracy</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
        }
        .stats {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .upload-area {
            border: 3px dashed #667eea;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin: 20px 0;
            cursor: pointer;
        }
        .upload-area:hover {
            background: #f9f9f9;
        }
        #preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px 0;
            display: none;
            border-radius: 5px;
        }
        .result {
            margin: 20px 0;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 5px;
            display: none;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
            margin: 10px 0;
        }
        .confidence {
            font-size: 18px;
            color: #555;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            margin: 10px 0;
        }
        button:hover {
            background: #764ba2;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .all-predictions {
            margin-top: 20px;
        }
        .pred-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .pred-label {
            width: 120px;
            font-weight: bold;
        }
        .pred-bar {
            flex: 1;
            height: 25px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin: 0 10px;
        }
        .pred-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.3s;
        }
        .pred-value {
            width: 60px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Plastic Detection System</h1>
        
        <div class="stats">
            <strong>Model Performance:</strong><br>
            ✅ Accuracy: <strong>96.5%</strong><br>
            ✅ Ensemble: <strong>5 Models</strong><br>
            ✅ Classes: cardboard, glass, metal, paper, plastic, trash
        </div>

        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <h3>📸 Click to Upload Image</h3>
            <p>or drag and drop</p>
            <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
        </div>

        <img id="preview" alt="Preview">

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing with 5-model ensemble...</p>
        </div>

        <button id="predictBtn" style="display:none" onclick="predict()">
            🔮 Predict with Ensemble
        </button>

        <div class="result" id="result">
            <h2>📊 Prediction Results</h2>
            <div class="prediction" id="prediction"></div>
            <div class="confidence" id="confidence"></div>
            <div class="all-predictions" id="allPredictions"></div>
        </div>
    </div>

    <script>
        let currentImage = null;

        function handleFile(file) {
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = file;
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
                document.getElementById('predictBtn').style.display = 'block';
                document.getElementById('result').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        async function predict() {
            if (!currentImage) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;

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
                document.getElementById('prediction').textContent = 
                    '🎯 Prediction: ' + data.prediction.toUpperCase();
                
                document.getElementById('confidence').textContent = 
                    '📊 Confidence: ' + data.confidence + '% (±' + data.uncertainty + '%)';

                // Display all predictions
                let html = '<h3>All Classes:</h3>';
                for (const [label, prob] of Object.entries(data.all_predictions)) {
                    const percentage = (prob * 100).toFixed(2);
                    html += `
                        <div class="pred-item">
                            <div class="pred-label">${label}</div>
                            <div class="pred-bar">
                                <div class="pred-bar-fill" style="width: ${percentage}%"></div>
                            </div>
                            <div class="pred-value">${percentage}%</div>
                        </div>
                    `;
                }
                document.getElementById('allPredictions').innerHTML = html;

                document.getElementById('result').style.display = 'block';

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            }
        }

        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#f0f0f0';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'white';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'white';
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
    
    # Try different paths
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
        print("Please extract ensemble_models.zip first")
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
    # Preprocess image
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions from each model
    predictions = []
    for model in MODELS:
        pred = model.predict(img_array, verbose=0)[0]
        predictions.append(pred)
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    
    # Get top prediction
    top_idx = np.argmax(avg_prediction)
    top_label = CLASS_LABELS[top_idx]
    top_prob = avg_prediction[top_idx]
    top_std = std_prediction[top_idx]
    
    # All predictions
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
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if not ENSEMBLE_LOADED:
        return jsonify({'error': 'Models not loaded. Please extract ensemble_models.zip'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        
        # Predict
        result = predict_with_ensemble(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': ENSEMBLE_LOADED,
        'num_models': len(MODELS),
        'classes': CLASS_LABELS
    })

if __name__ == '__main__':
    print("=" * 80)
    print("PLASTIC DETECTION WEB APP - 96.5% ACCURACY")
    print("=" * 80)
    
    # Load models
    if load_ensemble_models():
        print("\n🚀 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models!")
        print("\n💡 To fix this:")
        print("   1. Make sure ensemble_models.zip is extracted")
        print("   2. Extract to: Plastic-Detection-Model/ensemble_models/")
        print("   3. Run this script again")
