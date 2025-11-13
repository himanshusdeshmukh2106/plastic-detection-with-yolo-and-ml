"""
Test the improved classification model
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

def load_model(model_path='tf_files_improved/final_model.h5'):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    return model

def load_labels(labels_path='tf_files_improved/labels.txt'):
    """Load class labels"""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def predict_image(model, labels, image_path, img_size=(224, 224)):
    """
    Predict class for a single image
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top predictions
    top_indices = predictions.argsort()[-3:][::-1]
    
    results = {}
    print(f"\n📸 Predictions for: {os.path.basename(image_path)}")
    print("-" * 50)
    for i, idx in enumerate(top_indices, 1):
        label = labels[idx]
        confidence = predictions[idx] * 100
        results[label] = confidence
        print(f"  {i}. {label:<15} {confidence:>6.2f}%")
    
    return results

def test_on_samples(model, labels, dataset_path='training_dataset/training_dataset', samples_per_class=3):
    """
    Test model on sample images from each class
    """
    print("\n" + "=" * 80)
    print("TESTING ON SAMPLE IMAGES")
    print("=" * 80)
    
    from pathlib import Path
    
    correct = 0
    total = 0
    
    for class_name in labels:
        class_dir = Path(dataset_path) / class_name
        if not class_dir.exists():
            continue
        
        # Get sample images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if not images:
            continue
        
        samples = images[:samples_per_class]
        
        print(f"\n📁 Testing class: {class_name}")
        print("-" * 50)
        
        for img_path in samples:
            results = predict_image(model, labels, str(img_path))
            
            # Check if correct
            top_prediction = max(results, key=results.get)
            if top_prediction == class_name:
                correct += 1
                print("  ✅ Correct!")
            else:
                print(f"  ❌ Wrong! Expected: {class_name}")
            total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print("\n" + "=" * 80)
    print(f"📊 Sample Test Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    import sys
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    print(f"\n📋 Classes: {labels}")
    
    # Test on specific image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            predict_image(model, labels, image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        # Test on sample images from dataset
        test_on_samples(model, labels)
        
        print("\n💡 To test on a specific image:")
        print("   python test_improved_model.py path/to/image.jpg")
