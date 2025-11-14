"""
Quick test of ensemble models
"""
from tensorflow import keras
import numpy as np
from PIL import Image
from pathlib import Path
import json

def test_ensemble():
    print("=" * 80)
    print("TESTING ENSEMBLE MODELS")
    print("=" * 80)
    
    # Find ensemble directory
    possible_paths = [
        'ensemble_models/tf_files_ensemble',
        'Plastic-Detection-Model/ensemble_models/tf_files_ensemble',
    ]
    
    ensemble_dir = None
    for path in possible_paths:
        if Path(path).exists():
            ensemble_dir = Path(path)
            break
    
    if not ensemble_dir:
        print("\n❌ Ensemble models not found!")
        print("\n💡 Please extract ensemble_models.zip first:")
        print("   unzip ensemble_models.zip")
        return False
    
    print(f"\n✅ Found ensemble directory: {ensemble_dir}")
    
    # Load models
    models = []
    model_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith('model_')])
    
    print(f"\n📦 Loading {len(model_dirs)} models...")
    for model_dir in model_dirs:
        model_path = model_dir / 'final_model.h5'
        if model_path.exists():
            model = keras.models.load_model(str(model_path))
            models.append(model)
            print(f"  ✅ {model_dir.name}")
    
    if not models:
        print("\n❌ No models loaded!")
        return False
    
    print(f"\n✅ Successfully loaded {len(models)} models!")
    
    # Load class labels
    labels_file = ensemble_dir / 'class_indices.json'
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            class_indices = json.load(f)
        labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    else:
        labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    print(f"\n📋 Classes: {labels}")
    
    # Test with a dummy image
    print("\n🧪 Testing with dummy image...")
    dummy_img = np.random.rand(224, 224, 3)
    dummy_img = np.expand_dims(dummy_img, axis=0)
    
    predictions = []
    for i, model in enumerate(models, 1):
        pred = model.predict(dummy_img, verbose=0)[0]
        predictions.append(pred)
        top_class = labels[np.argmax(pred)]
        print(f"  Model {i}: {top_class} ({np.max(pred)*100:.2f}%)")
    
    # Ensemble prediction
    avg_pred = np.mean(predictions, axis=0)
    ensemble_class = labels[np.argmax(avg_pred)]
    ensemble_conf = np.max(avg_pred)
    
    print(f"\n🎯 Ensemble: {ensemble_class} ({ensemble_conf*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\n🚀 Ready to use! Run:")
    print("   python app_ensemble.py")
    
    return True

if __name__ == '__main__':
    test_ensemble()
