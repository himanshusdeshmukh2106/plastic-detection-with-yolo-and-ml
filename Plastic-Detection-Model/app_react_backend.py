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
import random

app = Flask(__name__)
# Enable CORS with explicit configuration
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:5173", "http://localhost:3000"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

# Global variables
MODELS = []
CLASS_LABELS = []
ENSEMBLE_LOADED = False
DETECTION_HISTORY = []
CLEANUP_TEAMS = []

# Mock authentication - accepts any password for demo purposes
MOCK_USERS = {
    'admin@rewater.com': {
        'password': 'admin123',  # Default password, but we'll accept any password
        'user': {
            'id': '1',
            'name': 'Admin User',
            'email': 'admin@rewater.com',
            'role': 'admin'
        }
    }
}

def generate_demo_data():
    """Generate comprehensive demo detection data across Indian coastline"""
    # Coastal locations with real coordinates across India
    locations = [
        # Mumbai & Maharashtra Coast
        {'name': 'Juhu Beach, Mumbai', 'lat': 19.0968, 'lng': 72.8265, 'district': 'Mumbai', 'type': 'tourist'},
        {'name': 'Versova Beach, Mumbai', 'lat': 19.1347, 'lng': 72.8114, 'district': 'Mumbai', 'type': 'fishing'},
        {'name': 'Alibaug Beach', 'lat': 18.6400, 'lng': 72.8756, 'district': 'Raigad', 'type': 'tourist'},
        {'name': 'Kashid Beach', 'lat': 18.4376, 'lng': 72.8607, 'district': 'Raigad', 'type': 'tourist'},
        {'name': 'Murud Beach', 'lat': 18.3237, 'lng': 72.9647, 'district': 'Raigad', 'type': 'tourist'},
        
        # Ratnagiri District
        {'name': 'Ganpatipule Beach', 'lat': 17.1495, 'lng': 73.2649, 'district': 'Ratnagiri', 'type': 'tourist'},
        {'name': 'Ratnagiri Fishing Harbor', 'lat': 16.9915, 'lng': 73.3102, 'district': 'Ratnagiri', 'type': 'fishing'},
        {'name': 'Bhatye Beach, Ratnagiri', 'lat': 17.0050, 'lng': 73.3167, 'district': 'Ratnagiri', 'type': 'tourist'},
        
        # Sindhudurg District
        {'name': 'Tarkarli Beach', 'lat': 16.0154, 'lng': 73.4893, 'district': 'Sindhudurg', 'type': 'tourist'},
        {'name': 'Vengurla Beach', 'lat': 15.8594, 'lng': 73.6284, 'district': 'Sindhudurg', 'type': 'coastal'},
        {'name': 'Malvan Fishing Port', 'lat': 16.0594, 'lng': 73.4644, 'district': 'Sindhudurg', 'type': 'fishing'},
        
        # Goa Beaches
        {'name': 'Baga Beach, Goa', 'lat': 15.5523, 'lng': 73.7518, 'district': 'North Goa', 'type': 'tourist'},
        {'name': 'Candolim Beach, Goa', 'lat': 15.5154, 'lng': 73.7684, 'district': 'North Goa', 'type': 'tourist'},
        {'name': 'Arambol Beach, Goa', 'lat': 15.6871, 'lng': 73.7049, 'district': 'North Goa', 'type': 'tourist'},
        {'name': 'Vagator Beach, Goa', 'lat': 15.5925, 'lng': 73.7395, 'district': 'North Goa', 'type': 'tourist'},
        {'name': 'Palolem Beach, Goa', 'lat': 15.0072, 'lng': 73.9985, 'district': 'South Goa', 'type': 'tourist'},
        {'name': 'Agonda Beach, Goa', 'lat': 15.0472, 'lng': 73.9925, 'district': 'South Goa', 'type': 'tourist'},
        {'name': 'Vasco Fishing Harbor, Goa', 'lat': 15.3989, 'lng': 73.8155, 'district': 'South Goa', 'type': 'fishing'},
        
        # Karnataka Coast
        {'name': 'Gokarna Beach', 'lat': 14.5479, 'lng': 74.3188, 'district': 'Karnataka', 'type': 'tourist'},
        {'name': 'Karwar Beach', 'lat': 14.8137, 'lng': 74.1294, 'district': 'Karnataka', 'type': 'coastal'},
        {'name': 'Malpe Beach, Udupi', 'lat': 13.3499, 'lng': 74.7040, 'district': 'Karnataka', 'type': 'fishing'},
        
        # Kerala Coast
        {'name': 'Kovalam Beach', 'lat': 8.4004, 'lng': 76.9790, 'district': 'Kerala', 'type': 'tourist'},
        {'name': 'Varkala Beach', 'lat': 8.7355, 'lng': 76.7132, 'district': 'Kerala', 'type': 'tourist'},
        {'name': 'Kochi Harbor', 'lat': 9.9312, 'lng': 76.2673, 'district': 'Kerala', 'type': 'port'},
        {'name': 'Cherai Beach', 'lat': 10.1412, 'lng': 76.1787, 'district': 'Kerala', 'type': 'tourist'},
        {'name': 'Alappuzha Beach', 'lat': 9.4981, 'lng': 76.3388, 'district': 'Kerala', 'type': 'coastal'},
        
        # Tamil Nadu Coast
        {'name': 'Marina Beach, Chennai', 'lat': 13.0474, 'lng': 80.2785, 'district': 'Tamil Nadu', 'type': 'tourist'},
        {'name': 'Mahabalipuram Beach', 'lat': 12.6269, 'lng': 80.1977, 'district': 'Tamil Nadu', 'type': 'tourist'},
        {'name': 'Kanyakumari Beach', 'lat': 8.0883, 'lng': 77.5385, 'district': 'Tamil Nadu', 'type': 'tourist'},
        {'name': 'Rameswaram Fishing Harbor', 'lat': 9.2876, 'lng': 79.3129, 'district': 'Tamil Nadu', 'type': 'fishing'},
        {'name': 'Pondicherry Beach', 'lat': 11.9139, 'lng': 79.8145, 'district': 'Puducherry', 'type': 'tourist'},
        
        # Andhra Pradesh
        {'name': 'Visakhapatnam Beach', 'lat': 17.7231, 'lng': 83.3014, 'district': 'Andhra Pradesh', 'type': 'tourist'},
        {'name': 'Kakinada Port', 'lat': 16.9891, 'lng': 82.2475, 'district': 'Andhra Pradesh', 'type': 'port'},
        
        # Odisha
        {'name': 'Puri Beach', 'lat': 19.8135, 'lng': 85.8312, 'district': 'Odisha', 'type': 'tourist'},
        {'name': 'Chandrabhaga Beach', 'lat': 19.8875, 'lng': 86.0954, 'district': 'Odisha', 'type': 'coastal'},
        
        # West Bengal
        {'name': 'Digha Beach', 'lat': 21.6765, 'lng': 87.5077, 'district': 'West Bengal', 'type': 'tourist'},
        {'name': 'Mandarmani Beach', 'lat': 21.6598, 'lng': 87.7729, 'district': 'West Bengal', 'type': 'tourist'},
    ]
    
    plastic_types = ['plastic', 'bottles', 'bags', 'fishing_nets', 'microplastics', 'packaging', 'straws', 'containers']
    sources = ['manual', 'camera', 'drone', 'satellite']
    
    demo_data = []
    detection_id = 1
    
    # Generate data over the past 90 days
    for days_ago in range(90):
        timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # More detections on weekends and tourist areas
        num_detections = random.randint(3, 12) if days_ago < 7 else random.randint(2, 6)
        
        for _ in range(num_detections):
            location = random.choice(locations)
            
            # Add some random variance to coordinates (within ~1km)
            lat_variance = random.uniform(-0.008, 0.008)
            lng_variance = random.uniform(-0.008, 0.008)
            
            detection = {
                'id': detection_id,
                'timestamp': timestamp.isoformat(),
                'waste_type': random.choice(plastic_types),
                'confidence': round(random.uniform(75, 98), 1),
                'latitude': location['lat'] + lat_variance,
                'longitude': location['lng'] + lng_variance,
                'source': random.choice(sources),
                'nearest_landmark': location['name'],
                'district': location['district'],
                'location_type': location['type'],
                'status': random.choice(['pending'] * 6 + ['in-progress'] * 3 + ['completed']),
                'severity': random.choice(['low', 'low', 'medium', 'medium', 'medium', 'high']),
                'estimated_quantity': random.randint(5, 500),
                'weather_condition': random.choice(['clear', 'cloudy', 'partly_cloudy', 'rainy']),
            }
            
            demo_data.append(detection)
            detection_id += 1
    
    return demo_data

def generate_cleanup_teams():
    """Generate cleanup team data positioned across Indian coastline"""
    teams = [
        {
            'id': 'team-01',
            'name': 'Mumbai Coastal Squad',
            'location': {'lat': 19.0968, 'lng': 72.8265},
            'region': 'Mumbai',
            'status': 'available',
            'capacity': 5,
            'current_load': 0,
            'members': 4,
            'equipment': ['pickup_truck', 'bags', 'gloves', 'gps'],
            'assigned_detections': [],
            'total_cleaned': random.randint(150, 300),
            'efficiency_rating': round(random.uniform(4.2, 5.0), 1),
        },
        {
            'id': 'team-02',
            'name': 'Raigad Beach Cleaners',
            'location': {'lat': 18.6400, 'lng': 72.8756},
            'region': 'Raigad',
            'status': 'busy',
            'capacity': 4,
            'current_load': 3,
            'members': 3,
            'equipment': ['boat', 'nets', 'bags', 'gps'],
            'assigned_detections': [],
            'total_cleaned': random.randint(100, 250),
            'efficiency_rating': round(random.uniform(4.0, 4.8), 1),
        },
        {
            'id': 'team-03',
            'name': 'Konkan Cleanup Force',
            'location': {'lat': 17.1495, 'lng': 73.2649},
            'region': 'Ratnagiri',
            'status': 'available',
            'capacity': 6,
            'current_load': 1,
            'members': 5,
            'equipment': ['pickup_truck', 'trailer', 'bags', 'sorting_bins'],
            'assigned_detections': [],
            'total_cleaned': random.randint(200, 400),
            'efficiency_rating': round(random.uniform(4.3, 5.0), 1),
        },
        {
            'id': 'team-04',
            'name': 'Sindhudurg Sea Warriors',
            'location': {'lat': 16.0154, 'lng': 73.4893},
            'region': 'Sindhudurg',
            'status': 'available',
            'capacity': 5,
            'current_load': 2,
            'members': 4,
            'equipment': ['boat', 'diving_gear', 'underwater_nets'],
            'assigned_detections': [],
            'total_cleaned': random.randint(120, 280),
            'efficiency_rating': round(random.uniform(4.1, 4.9), 1),
        },
        {
            'id': 'team-05',
            'name': 'Goa Beach Patrol',
            'location': {'lat': 15.5523, 'lng': 73.7518},
            'region': 'North Goa',
            'status': 'available',
            'capacity': 7,
            'current_load': 4,
            'members': 6,
            'equipment': ['atv', 'bags', 'gps', 'first_aid'],
            'assigned_detections': [],
            'total_cleaned': random.randint(300, 500),
            'efficiency_rating': round(random.uniform(4.5, 5.0), 1),
        },
        {
            'id': 'team-06',
            'name': 'Karnataka Coastal Care',
            'location': {'lat': 14.5479, 'lng': 74.3188},
            'region': 'Karnataka',
            'status': 'busy',
            'capacity': 4,
            'current_load': 4,
            'members': 3,
            'equipment': ['pickup_truck', 'bags', 'rakes'],
            'assigned_detections': [],
            'total_cleaned': random.randint(90, 200),
            'efficiency_rating': round(random.uniform(3.9, 4.6), 1),
        },
        {
            'id': 'team-07',
            'name': 'Kerala Backwater Guardians',
            'location': {'lat': 9.9312, 'lng': 76.2673},
            'region': 'Kerala',
            'status': 'available',
            'capacity': 5,
            'current_load': 1,
            'members': 4,
            'equipment': ['boat', 'nets', 'bags', 'sorting_bins'],
            'assigned_detections': [],
            'total_cleaned': random.randint(180, 350),
            'efficiency_rating': round(random.uniform(4.4, 4.9), 1),
        },
        {
            'id': 'team-08',
            'name': 'Chennai Marina Cleanup',
            'location': {'lat': 13.0474, 'lng': 80.2785},
            'region': 'Tamil Nadu',
            'status': 'available',
            'capacity': 6,
            'current_load': 3,
            'members': 5,
            'equipment': ['pickup_truck', 'bags', 'gps', 'compactor'],
            'assigned_detections': [],
            'total_cleaned': random.randint(250, 450),
            'efficiency_rating': round(random.uniform(4.2, 4.8), 1),
        },
        {
            'id': 'team-09',
            'name': 'Visakhapatnam Beach Brigade',
            'location': {'lat': 17.7231, 'lng': 83.3014},
            'region': 'Andhra Pradesh',
            'status': 'available',
            'capacity': 4,
            'current_load': 0,
            'members': 3,
            'equipment': ['bags', 'gps', 'shovels'],
            'assigned_detections': [],
            'total_cleaned': random.randint(100, 220),
            'efficiency_rating': round(random.uniform(4.0, 4.7), 1),
        },
        {
            'id': 'team-10',
            'name': 'Puri Coastline Protectors',
            'location': {'lat': 19.8135, 'lng': 85.8312},
            'region': 'Odisha',
            'status': 'available',
            'capacity': 5,
            'current_load': 2,
            'members': 4,
            'equipment': ['pickup_truck', 'bags', 'gps'],
            'assigned_detections': [],
            'total_cleaned': random.randint(130, 280),
            'efficiency_rating': round(random.uniform(4.1, 4.8), 1),
        },
    ]
    
    return teams

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two coordinates in kilometers using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r

def assign_detection_to_team(detection_id):
    """Automatically assign a detection to the nearest available team"""
    global CLEANUP_TEAMS, DETECTION_HISTORY
    
    # Find the detection
    detection = None
    for d in DETECTION_HISTORY:
        if d['id'] == detection_id:
            detection = d
            break
    
    if not detection:
        return None
    
    # Find nearest available team with capacity
    best_team = None
    min_distance = float('inf')
    
    for team in CLEANUP_TEAMS:
        if team['current_load'] < team['capacity']:
            distance = calculate_distance(
                detection['latitude'], detection['longitude'],
                team['location']['lat'], team['location']['lng']
            )
            
            if distance < min_distance:
                min_distance = distance
                best_team = team
    
    if best_team:
        # Assign detection to team
        best_team['assigned_detections'].append(detection_id)
        best_team['current_load'] = len(best_team['assigned_detections'])
        
        if best_team['current_load'] >= best_team['capacity']:
            best_team['status'] = 'busy'
        
        # Update detection status
        detection['status'] = 'assigned'
        detection['assigned_team'] = best_team['id']
        detection['distance_to_team'] = round(min_distance, 2)
        
        return {
            'team': best_team,
            'detection': detection,
            'distance': round(min_distance, 2)
        }
    
    return None

def find_hotspots(detections, radius_km=5):
    """Identify pollution hotspots based on detection density"""
    hotspots = []
    processed_indices = set()
    
    for i, detection in enumerate(detections):
        if i in processed_indices:
            continue
        
        # Find all detections within radius
        cluster = [detection]
        cluster_indices = {i}
        
        for j, other in enumerate(detections):
            if j <= i or j in processed_indices:
                continue
            
            distance = calculate_distance(
                detection['latitude'], detection['longitude'],
                other['latitude'], other['longitude']
            )
            
            if distance <= radius_km:
                cluster.append(other)
                cluster_indices.add(j)
        
        # If cluster has enough detections, mark as hotspot
        if len(cluster) >= 5:
            processed_indices.update(cluster_indices)
            
            # Calculate center point
            avg_lat = sum(d['latitude'] for d in cluster) / len(cluster)
            avg_lng = sum(d['longitude'] for d in cluster) / len(cluster)
            
            # Determine severity
            if len(cluster) > 20:
                severity = 'critical'
            elif len(cluster) > 10:
                severity = 'high'
            else:
                severity = 'medium'
            
            hotspot = {
                'id': f'hotspot-{len(hotspots) + 1}',
                'latitude': avg_lat,
                'longitude': avg_lng,
                'detection_count': len(cluster),
                'radius': radius_km,
                'severity': severity,
                'detections': [d['id'] for d in cluster],
                'nearest_landmark': cluster[0].get('nearest_landmark', 'Unknown'),
            }
            
            hotspots.append(hotspot)
    
    return hotspots

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
        print("[ERROR] Ensemble models not found!")
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
                print(f"  [OK] Loaded {model_dir.name}")
            except Exception as e:
                print(f"  [WARN] Error loading {model_dir.name}: {e}")
    
    if MODELS:
        print(f"\n[SUCCESS] Loaded {len(MODELS)} models successfully!")
        ENSEMBLE_LOADED = True
        return True
    else:
        print("[ERROR] No models loaded!")
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

@app.route('/auth/login', methods=['POST'])
def login():
    """Mock login endpoint - accepts any password for demo"""
    try:
        data = request.json
        email = data.get('email', '')
        password = data.get('password', '')
        
        # For demo purposes, accept any password for admin@rewater.com
        if email == 'admin@rewater.com' and password:
            user_data = MOCK_USERS.get(email)
            return jsonify({
                'token': 'mock-jwt-token-' + email,
                'refreshToken': 'mock-refresh-token',
                'user': user_data['user']
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/refresh', methods=['POST'])
def refresh():
    """Mock refresh token endpoint"""
    try:
        data = request.json
        refresh_token = data.get('refreshToken', '')
        
        if refresh_token:
            # Return new tokens
            return jsonify({
                'token': 'mock-jwt-token-refreshed',
                'refreshToken': 'mock-refresh-token-new',
                'user': MOCK_USERS['admin@rewater.com']['user']
            })
        
        return jsonify({'error': 'Invalid refresh token'}), 401
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            'id': len(DETECTION_HISTORY) + 1,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'waste_type': data.get('waste_type', data.get('prediction')),
            'confidence': data.get('confidence', 0),
            'latitude': data.get('latitude', 0),
            'longitude': data.get('longitude', 0),
            'source': data.get('source', 'manual'),
            'status': 'pending'
        }
        
        DETECTION_HISTORY.append(detection)
        
        return jsonify({'success': True, 'total': len(DETECTION_HISTORY), 'id': detection['id']})
    
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
    today_detections = []
    week_detections = []
    
    for detection in DETECTION_HISTORY:
        try:
            det_time = datetime.fromisoformat(detection['timestamp'])
            days_diff = (now - det_time).days
            
            if det_time.date() == now.date():
                today_detections.append(detection)
            
            if days_diff <= 7:
                week_detections.append(detection)
            
            if time_range == 'today':
                if det_time.date() == now.date():
                    filtered_history.append(detection)
            elif time_range == 'week':
                if days_diff <= 7:
                    filtered_history.append(detection)
            elif time_range == 'month':
                if days_diff <= 30:
                    filtered_history.append(detection)
            else:  # all
                filtered_history.append(detection)
        except:
            filtered_history.append(detection)
    
    # Calculate statistics
    total = len(DETECTION_HISTORY)
    
    # Count by class
    by_class = defaultdict(int)
    total_confidence = 0
    
    for detection in filtered_history:
        pred = detection.get('waste_type', detection.get('prediction', 'unknown'))
        by_class[pred] += 1
        total_confidence += detection.get('confidence', 0)
    
    avg_confidence = (total_confidence / len(filtered_history)) if filtered_history else 0
    
    return jsonify({
        'total_detections': total,
        'today_detections': len(today_detections),
        'week_detections': len(week_detections),
        'active_hotspots': max(3, int(total / 100)),  # Simulated
        'cleanup_ratio': min(85.5, 50 + (total / 10)),  # Simulated
        'coverage_area': 450,  # km² - Konkan coastline
        'last_detection': DETECTION_HISTORY[-1]['timestamp'] if DETECTION_HISTORY else None,
        'system_uptime': 99.2,  # Simulated
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

@app.route('/hotspots', methods=['GET'])
def hotspots():
    """Get plastic pollution hotspots"""
    # Analyze detection history to find hotspots
    hotspot_data = defaultdict(lambda: {'count': 0, 'lat': 0, 'lng': 0, 'detections': []})
    
    for detection in DETECTION_HISTORY:
        lat = detection.get('latitude', 0)
        lng = detection.get('longitude', 0)
        
        if lat != 0 or lng != 0:
            # Round to 2 decimal places to group nearby detections
            key = f"{round(lat, 2)},{round(lng, 2)}"
            hotspot_data[key]['count'] += 1
            hotspot_data[key]['lat'] += lat
            hotspot_data[key]['lng'] += lng
            hotspot_data[key]['detections'].append(detection)
    
    # Convert to hotspot list
    hotspots_list = []
    for key, data in hotspot_data.items():
        if data['count'] >= 2:  # Only consider areas with 2+ detections
            hotspots_list.append({
                'id': key,
                'latitude': data['lat'] / data['count'],
                'longitude': data['lng'] / data['count'],
                'count': data['count'],
                'severity': 'high' if data['count'] > 10 else 'medium' if data['count'] > 5 else 'low'
            })
    
    # Sort by count
    hotspots_list.sort(key=lambda x: x['count'], reverse=True)
    
    return jsonify({'hotspots': hotspots_list[:20]})  # Return top 20

@app.route('/system-status', methods=['GET'])
def system_status():
    """Get system status information"""
    now = datetime.now()
    uptime_hours = 24  # Simulated
    
    return jsonify({
        'status': 'operational',
        'uptime': uptime_hours * 3600,  # in seconds
        'models_loaded': ENSEMBLE_LOADED,
        'num_models': len(MODELS),
        'total_detections': len(DETECTION_HISTORY),
        'memory_usage': 45.2,  # Simulated percentage
        'cpu_usage': 12.8,  # Simulated percentage
        'last_detection': DETECTION_HISTORY[-1]['timestamp'] if DETECTION_HISTORY else None,
        'version': '1.0.0'
    })

@app.route('/collection/status', methods=['GET'])
def collection_status():
    """Get data collection channel status"""
    now = datetime.now()
    
    # Analyze detection sources
    source_counts = defaultdict(int)
    for detection in DETECTION_HISTORY:
        source = detection.get('source', 'manual')
        source_counts[source] += 1
    
    # Build channel status
    channels = []
    
    # Manual uploads
    manual_count = source_counts.get('manual', 0)
    channels.append({
        'id': 'manual',
        'name': 'Manual Uploads',
        'type': 'manual',
        'status': 'operational' if manual_count > 0 else 'degraded',
        'lastIngestion': DETECTION_HISTORY[-1]['timestamp'] if DETECTION_HISTORY and DETECTION_HISTORY[-1].get('source') == 'manual' else None,
        'successRate': 98.5,
        'queueDepth': 0,
        'totalProcessed': manual_count
    })
    
    # Camera feeds
    camera_count = source_counts.get('camera', 0)
    channels.append({
        'id': 'camera',
        'name': 'Camera Feeds',
        'type': 'camera',
        'status': 'operational' if camera_count > 0 else 'offline',
        'lastIngestion': (now - timedelta(minutes=5)).isoformat() if camera_count > 0 else None,
        'successRate': 95.2,
        'queueDepth': 2,
        'totalProcessed': camera_count
    })
    
    # Drone footage
    drone_count = source_counts.get('drone', 0)
    channels.append({
        'id': 'drone',
        'name': 'Drone Footage',
        'type': 'drone',
        'status': 'operational' if drone_count > 0 else 'offline',
        'lastIngestion': (now - timedelta(hours=2)).isoformat() if drone_count > 0 else None,
        'successRate': 92.0,
        'queueDepth': 0,
        'totalProcessed': drone_count
    })
    
    # Satellite imagery
    satellite_count = source_counts.get('satellite', 0)
    channels.append({
        'id': 'satellite',
        'name': 'Satellite Imagery',
        'type': 'satellite',
        'status': 'degraded' if satellite_count > 0 else 'offline',
        'lastIngestion': (now - timedelta(days=1)).isoformat() if satellite_count > 0 else None,
        'successRate': 88.5,
        'queueDepth': 5,
        'totalProcessed': satellite_count
    })
    
    return jsonify({
        'channels': channels,
        'overall_status': 'operational',
        'total_channels': len(channels),
        'operational_channels': sum(1 for c in channels if c['status'] == 'operational')
    })

@app.route('/collection/ingest-jobs', methods=['GET'])
def ingest_jobs():
    """Get scheduled auto-ingest jobs"""
    return jsonify({
        'jobs': [
            {
                'id': 'drone-auto-1',
                'source': 'drone',
                'frequency': 30,
                'nextRun': (datetime.now() + timedelta(minutes=15)).isoformat(),
                'status': 'active'
            },
            {
                'id': 'satellite-auto-1',
                'source': 'satellite',
                'frequency': 120,
                'nextRun': (datetime.now() + timedelta(hours=1)).isoformat(),
                'status': 'active'
            }
        ]
    })

@app.route('/collection/schedule-ingest', methods=['POST'])
def schedule_ingest():
    """Schedule automated data ingestion"""
    try:
        data = request.json
        source = data.get('source')
        frequency = data.get('frequency', 30)
        
        # In a real system, this would create a scheduled job
        job_id = f"{source}-auto-{datetime.now().timestamp()}"
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Scheduled {source} ingestion every {frequency} minutes'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detections', methods=['GET'])
def detections():
    """Get all detections with filtering"""
    status_filter = request.args.get('status', None)
    limit = request.args.get('limit', 100, type=int)
    
    filtered = DETECTION_HISTORY
    
    if status_filter:
        filtered = [d for d in filtered if d.get('status') == status_filter]
    
    # Return most recent
    recent = filtered[-limit:] if len(filtered) > limit else filtered
    recent.reverse()
    
    return jsonify({
        'detections': recent,
        'total': len(filtered),
        'filtered': status_filter is not None
    })

@app.route('/detections/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get a specific detection by ID"""
    for detection in DETECTION_HISTORY:
        if detection.get('id') == detection_id:
            return jsonify(detection)
    
    return jsonify({'error': 'Detection not found'}), 404

@app.route('/detections/<int:detection_id>', methods=['PUT'])
def update_detection(detection_id):
    """Update a detection"""
    try:
        data = request.json
        
        for detection in DETECTION_HISTORY:
            if detection.get('id') == detection_id:
                # Update fields
                if 'status' in data:
                    detection['status'] = data['status']
                if 'notes' in data:
                    detection['notes'] = data['notes']
                
                return jsonify(detection)
        
        return jsonify({'error': 'Detection not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/teams', methods=['GET'])
def get_teams():
    """Get all cleanup teams"""
    return jsonify({'teams': CLEANUP_TEAMS})

@app.route('/teams/<string:team_id>', methods=['GET'])
def get_team(team_id):
    """Get a specific team by ID"""
    for team in CLEANUP_TEAMS:
        if team['id'] == team_id:
            return jsonify(team)
    
    return jsonify({'error': 'Team not found'}), 404

@app.route('/teams/<string:team_id>', methods=['PUT'])
def update_team(team_id):
    """Update team information"""
    try:
        data = request.json
        
        for team in CLEANUP_TEAMS:
            if team['id'] == team_id:
                # Update allowed fields
                if 'status' in data:
                    team['status'] = data['status']
                if 'location' in data:
                    team['location'] = data['location']
                
                return jsonify(team)
        
        return jsonify({'error': 'Team not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/teams/<string:team_id>/complete', methods=['POST'])
def complete_team_assignment(team_id):
    """Mark a team's current assignments as completed"""
    global CLEANUP_TEAMS, DETECTION_HISTORY
    
    try:
        team = None
        for t in CLEANUP_TEAMS:
            if t['id'] == team_id:
                team = t
                break
        
        if not team:
            return jsonify({'error': 'Team not found'}), 404
        
        # Mark all assigned detections as completed
        completed_count = 0
        for detection_id in team['assigned_detections']:
            for detection in DETECTION_HISTORY:
                if detection['id'] == detection_id:
                    detection['status'] = 'completed'
                    completed_count += 1
                    break
        
        # Update team stats
        team['total_cleaned'] += completed_count
        team['assigned_detections'] = []
        team['current_load'] = 0
        team['status'] = 'available'
        
        return jsonify({
            'success': True,
            'team': team,
            'completed_count': completed_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assign', methods=['POST'])
def assign_detection():
    """Assign a detection to nearest available team"""
    try:
        data = request.json
        detection_id = data.get('detection_id')
        
        if not detection_id:
            return jsonify({'error': 'detection_id required'}), 400
        
        result = assign_detection_to_team(detection_id)
        
        if result:
            return jsonify({
                'success': True,
                'assignment': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No available teams or detection not found'
            }), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assign/bulk', methods=['POST'])
def bulk_assign():
    """Assign multiple detections to teams automatically"""
    try:
        data = request.json
        detection_ids = data.get('detection_ids', [])
        
        if not detection_ids:
            # Auto-assign all pending detections
            detection_ids = [d['id'] for d in DETECTION_HISTORY if d.get('status') == 'pending']
        
        assignments = []
        failed = []
        
        for detection_id in detection_ids:
            result = assign_detection_to_team(detection_id)
            if result:
                assignments.append(result)
            else:
                failed.append(detection_id)
        
        return jsonify({
            'success': True,
            'assigned_count': len(assignments),
            'failed_count': len(failed),
            'assignments': assignments
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hotspots/calculate', methods=['GET'])
def calculate_hotspots():
    """Calculate and return pollution hotspots"""
    try:
        radius = request.args.get('radius', 5, type=float)
        
        # Only consider recent pending/in-progress detections
        recent_detections = [
            d for d in DETECTION_HISTORY 
            if d.get('status') in ['pending', 'in-progress', 'assigned']
        ]
        
        hotspots = find_hotspots(recent_detections, radius_km=radius)
        
        return jsonify({
            'hotspots': hotspots,
            'total_hotspots': len(hotspots),
            'radius_km': radius
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("PLASTIC DETECTION - REACT BACKEND")
    print("=" * 80)
    
    # Load demo data
    print("\nLoading demo data...")
    demo_data = generate_demo_data()
    DETECTION_HISTORY.extend(demo_data)
    
    # Show detection stats
    sources = defaultdict(int)
    districts = defaultdict(int)
    for d in DETECTION_HISTORY:
        sources[d.get('source', 'unknown')] += 1
        districts[d.get('district', 'unknown')] += 1
    
    print(f"[SUCCESS] Loaded {len(DETECTION_HISTORY)} demo detection records")
    print(f"  Sources: Manual={sources['manual']}, Camera={sources['camera']}, Drone={sources['drone']}, Satellite={sources['satellite']}")
    print(f"  Top Regions: {', '.join([f'{k}={v}' for k, v in sorted(districts.items(), key=lambda x: x[1], reverse=True)[:5]])}")
    
    # Load cleanup teams
    print("\nLoading cleanup teams...")
    teams = generate_cleanup_teams()
    CLEANUP_TEAMS.extend(teams)
    available_teams = sum(1 for t in CLEANUP_TEAMS if t['status'] == 'available')
    print(f"[SUCCESS] Loaded {len(CLEANUP_TEAMS)} cleanup teams ({available_teams} available, {len(CLEANUP_TEAMS) - available_teams} busy)")
    
    # Calculate initial hotspots
    print("\nCalculating pollution hotspots...")
    pending = [d for d in DETECTION_HISTORY if d.get('status') == 'pending']
    hotspots = find_hotspots(pending, radius_km=5)
    print(f"[SUCCESS] Identified {len(hotspots)} pollution hotspots")
    
    if load_ensemble_models():
        print("\n[START] Starting Flask backend for React frontend...")
        print("[BACKEND] http://localhost:5000")
        print("[FRONTEND] http://localhost:5173 (run 'npm run dev' in frontend folder)")
        print("\nEndpoints:")
        print("  Authentication:")
        print("    POST /auth/login      - Login (email: admin@rewater.com, password: any)")
        print("    POST /auth/refresh    - Refresh token")
        print("\n  System:")
        print("    GET  /health          - Health check")
        print("    GET  /system-status   - System status and metrics")
        print("\n  Predictions:")
        print("    POST /predict         - Predict waste type from image")
        print("    POST /save-detection  - Save detection with location")
        print("\n  Detections:")
        print("    GET  /detections           - Get all detections (with filtering)")
        print("    GET  /detections/<id>      - Get specific detection")
        print("    PUT  /detections/<id>      - Update detection")
        print("    GET  /history              - Get detection history")
        print("\n  Analytics:")
        print("    GET  /stats           - Dashboard statistics")
        print("    GET  /hotspots        - Pollution hotspots")
        print("\n  Data Collection:")
        print("    GET  /collection/status         - Collection channel status")
        print("    GET  /collection/ingest-jobs    - Scheduled auto-ingest jobs")
        print("    POST /collection/schedule-ingest - Schedule automated ingestion")
        print("\n  Cleanup Teams:")
        print("    GET  /teams                 - Get all cleanup teams")
        print("    GET  /teams/<id>            - Get specific team")
        print("    PUT  /teams/<id>            - Update team info")
        print("    POST /teams/<id>/complete   - Mark team assignments completed")
        print("\n  Assignment:")
        print("    POST /assign                - Assign detection to nearest team")
        print("    POST /assign/bulk           - Bulk assign detections to teams")
        print("\n  Hotspots:")
        print("    GET  /hotspots/calculate    - Calculate pollution hotspots")
        print("\n  Export:")
        print("    GET  /export-csv      - Export history as CSV")
        print("    POST /clear-history   - Clear detection history")
        print("\nPress Ctrl+C to stop")
        print("=" * 80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n[ERROR] Failed to load models!")
        print("Make sure ensemble models are in the correct directory")
