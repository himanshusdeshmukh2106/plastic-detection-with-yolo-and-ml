"""
Predict using ensemble of models
Averages predictions from multiple models for better accuracy
"""
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import sys
from pathlib import Path

def load_ensemble_models(ensemble_dir='tf_files_ensemble', num_models=3):
    """
    Load all models in the ensemble
    """
    models = []
    
    print(f"Loading {num_models} models from ensemble...")
    
    for i in range(1, num_models + 1):
        model_path = os.path.join(ensemble_dir, f'model_{i}', 'final_model.h5')
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model {i} not found at: {model_path}")
            continue
        
        model = keras.models.load_model(model_path)
        models.append(model)
        print(f"  ✅ Loaded model {i}")
    
    if not models:
        raise FileNotFoundError("No models found in ensemble directory!")
    
    print(f"\n✅ Loaded {len(models)} models successfully")
    return models


def load_class_labels(ensemble_dir='tf_files_ensemble'):
    """Load class labels"""
    labels_file = os.path.join(ensemble_dir, 'class_indices.json')
    
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            class_indices = json.load(f)
        # Reverse mapping: index -> label
        labels = {v: k for k, v in class_indices.items()}
        labels = [labels[i] for i in range(len(labels))]
    else:
        # Fallback to default labels
        labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    return labels


def predict_ensemble(models, image_path, labels):
    """
    Predict using ensemble of models
    
    Args:
        models: List of Keras models
        image_path: Path to image
        labels: List of class labels
    
    Returns:
        dict: Averaged predictions
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions from each model
    predictions = []
    for i, model in enumerate(models, 1):
        pred = model.predict(img_array, verbose=0)[0]
        predictions.append(pred)
        print(f"  Model {i}: {labels[np.argmax(pred)]} ({np.max(pred)*100:.2f}%)")
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    
    # Calculate variance (uncertainty)
    std_prediction = np.std(predictions, axis=0)
    
    # Create results dictionary
    results = {}
    for i, label in enumerate(labels):
        results[label] = {
            'probability': float(avg_prediction[i]),
            'std': float(std_prediction[i])
        }
    
    return results, avg_prediction, std_prediction


def predict_ensemble_with_tta(models, image_path, labels, num_augmentations=5):
    """
    Predict using ensemble + Test-Time Augmentation
    Combines both techniques for maximum accuracy
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    all_predictions = []
    
    # For each augmentation
    for aug_idx in range(num_augmentations):
        if aug_idx == 0:
            # Original image
            aug_img = img_array.copy()
        else:
            # Augmented image
            aug_img = img_array.copy()
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                aug_img = np.fliplr(aug_img)
            
            # Random brightness
            brightness = np.random.uniform(0.9, 1.1)
            aug_img = np.clip(aug_img * brightness, 0, 1)
        
        # Predict with all models
        for model in models:
            pred = model.predict(np.expand_dims(aug_img, 0), verbose=0)[0]
            all_predictions.append(pred)
    
    # Average all predictions
    avg_prediction = np.mean(all_predictions, axis=0)
    std_prediction = np.std(all_predictions, axis=0)
    
    results = {}
    for i, label in enumerate(labels):
        results[label] = {
            'probability': float(avg_prediction[i]),
            'std': float(std_prediction[i])
        }
    
    return results, avg_prediction, std_prediction


if __name__ == '__main__':
    print("=" * 80)
    print("ENSEMBLE PREDICTION")
    print("=" * 80)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python predict_ensemble.py <image_path> [--tta]")
        print("\nExamples:")
        print("  python predict_ensemble.py test_image.jpg")
        print("  python predict_ensemble.py test_image.jpg --tta  # With TTA")
        sys.exit(1)
    
    image_path = sys.argv[1]
    use_tta = '--tta' in sys.argv
    
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Load models and labels
    try:
        models = load_ensemble_models()
        labels = load_class_labels()
    except Exception as e:
        print(f"\n❌ Error loading ensemble: {e}")
        print("\nMake sure you've trained the ensemble first:")
        print("  python train_ensemble.py 3")
        sys.exit(1)
    
    print(f"\n📸 Image: {image_path}")
    print(f"🔮 Predicting with {len(models)}-model ensemble...")
    
    if use_tta:
        print(f"🔄 Using Test-Time Augmentation (5 augmentations per model)")
        print(f"   Total predictions: {len(models) * 5}")
    
    print()
    
    # Predict
    if use_tta:
        results, avg_pred, std_pred = predict_ensemble_with_tta(models, image_path, labels)
    else:
        results, avg_pred, std_pred = predict_ensemble(models, image_path, labels)
    
    # Sort by probability
    sorted_results = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("ENSEMBLE PREDICTIONS")
    print("=" * 80)
    
    for i, (label, data) in enumerate(sorted_results, 1):
        prob = data['probability']
        std = data['std']
        bar = '█' * int(prob * 50)
        uncertainty = "±" + f"{std*100:.1f}%"
        print(f"{i}. {label:<15} {prob*100:>6.2f}% {uncertainty:<8} {bar}")
    
    # Top prediction
    top_label = sorted_results[0][0]
    top_prob = sorted_results[0][1]['probability']
    top_std = sorted_results[0][1]['std']
    
    print("\n" + "=" * 80)
    print(f"🎯 PREDICTION: {top_label}")
    print(f"📊 CONFIDENCE: {top_prob*100:.2f}% (±{top_std*100:.1f}%)")
    
    # Confidence assessment
    if top_std < 0.05:
        print(f"✅ Very consistent across models (low uncertainty)")
    elif top_std < 0.10:
        print(f"✅ Consistent across models")
    else:
        print(f"⚠️  Models disagree (high uncertainty)")
    
    if top_prob > 0.9:
        print("✅ Very confident!")
    elif top_prob > 0.7:
        print("✅ Confident")
    elif top_prob > 0.5:
        print("⚠️  Moderate confidence")
    else:
        print("❌ Low confidence - uncertain")
    
    print("=" * 80)
    
    # Compare with single model (if available)
    single_model_path = 'tf_files_optimized/final_model.h5'
    if os.path.exists(single_model_path):
        print("\n💡 Comparing with single model...")
        single_model = keras.models.load_model(single_model_path)
        
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        single_pred = single_model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        
        single_top_idx = np.argmax(single_pred)
        single_top_conf = single_pred[single_top_idx]
        
        print(f"  Single model: {labels[single_top_idx]} ({single_top_conf*100:.2f}%)")
        print(f"  Ensemble:     {top_label} ({top_prob*100:.2f}%)")
        
        if top_prob > single_top_conf:
            improvement = (top_prob - single_top_conf) * 100
            print(f"  ✅ Ensemble improved confidence by +{improvement:.2f}%")
        
        # Check if predictions agree
        if labels[single_top_idx] == top_label:
            print(f"  ✅ Both models agree on the prediction")
        else:
            print(f"  ⚠️  Models disagree! Ensemble is more reliable.")
