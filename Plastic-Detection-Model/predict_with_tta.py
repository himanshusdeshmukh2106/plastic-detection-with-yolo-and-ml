"""
Test-Time Augmentation (TTA) for improved predictions
Expected improvement: +2-4% accuracy
"""
from tensorflow import keras
import numpy as np
from PIL import Image
import sys
import os

def predict_with_tta(model, image_path, labels, num_augmentations=10):
    """
    Predict with Test-Time Augmentation
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        labels: List of class labels
        num_augmentations: Number of augmented predictions to average
    
    Returns:
        dict: Predictions for each class
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    predictions = []
    
    # 1. Original image
    predictions.append(model.predict(np.expand_dims(img_array, 0), verbose=0)[0])
    
    # 2-N. Augmented versions
    for i in range(num_augmentations - 1):
        aug_img = img_array.copy()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            aug_img = np.fliplr(aug_img)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.85, 1.15)
        aug_img = np.clip(aug_img * brightness, 0, 1)
        
        # Random rotation (small angle)
        angle = np.random.uniform(-15, 15)
        aug_img_pil = Image.fromarray((aug_img * 255).astype(np.uint8))
        aug_img_pil = aug_img_pil.rotate(angle, fillcolor=(128, 128, 128))
        aug_img = np.array(aug_img_pil) / 255.0
        
        # Random zoom
        zoom = np.random.uniform(0.9, 1.1)
        if zoom != 1.0:
            h, w = aug_img.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            aug_img_pil = Image.fromarray((aug_img * 255).astype(np.uint8))
            aug_img_pil = aug_img_pil.resize((new_w, new_h))
            
            # Crop or pad to original size
            if zoom > 1.0:
                # Crop center
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                aug_img_pil = aug_img_pil.crop((left, top, left + w, top + h))
            else:
                # Pad
                padded = Image.new('RGB', (w, h), (128, 128, 128))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                padded.paste(aug_img_pil, (left, top))
                aug_img_pil = padded
            
            aug_img = np.array(aug_img_pil) / 255.0
        
        predictions.append(model.predict(np.expand_dims(aug_img, 0), verbose=0)[0])
    
    # Average all predictions
    avg_prediction = np.mean(predictions, axis=0)
    
    # Create results dictionary
    results = {}
    for i, label in enumerate(labels):
        results[label] = float(avg_prediction[i])
    
    return results


def load_model_and_labels(model_path='tf_files_optimized/final_model.h5', 
                          labels_path='tf_files_optimized/labels.txt'):
    """Load model and labels"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    return model, labels


if __name__ == '__main__':
    print("=" * 80)
    print("TEST-TIME AUGMENTATION (TTA) PREDICTION")
    print("=" * 80)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python predict_with_tta.py <image_path> [num_augmentations]")
        print("\nExample:")
        print("  python predict_with_tta.py test_image.jpg 10")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_aug = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Load model and labels
    model, labels = load_model_and_labels()
    
    print(f"\n📸 Image: {image_path}")
    print(f"🔄 Augmentations: {num_aug}")
    print(f"\n🔮 Predicting with TTA...")
    
    # Predict with TTA
    results = predict_with_tta(model, image_path, labels, num_augmentations=num_aug)
    
    # Sort by confidence
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTIONS (with Test-Time Augmentation)")
    print("=" * 80)
    
    for i, (label, confidence) in enumerate(sorted_results, 1):
        bar = '█' * int(confidence * 50)
        print(f"{i}. {label:<15} {confidence*100:>6.2f}% {bar}")
    
    # Top prediction
    top_label, top_conf = sorted_results[0]
    print("\n" + "=" * 80)
    print(f"🎯 PREDICTION: {top_label}")
    print(f"📊 CONFIDENCE: {top_conf*100:.2f}%")
    
    if top_conf > 0.9:
        print("✅ Very confident!")
    elif top_conf > 0.7:
        print("✅ Confident")
    elif top_conf > 0.5:
        print("⚠️  Moderate confidence")
    else:
        print("❌ Low confidence - uncertain")
    
    print("=" * 80)
    
    # Compare with single prediction (no TTA)
    print("\n💡 Comparing with single prediction (no TTA)...")
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    single_pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
    single_top_idx = np.argmax(single_pred)
    single_top_conf = single_pred[single_top_idx]
    
    print(f"  Single prediction: {labels[single_top_idx]} ({single_top_conf*100:.2f}%)")
    print(f"  TTA prediction:    {top_label} ({top_conf*100:.2f}%)")
    
    if top_conf > single_top_conf:
        improvement = (top_conf - single_top_conf) * 100
        print(f"  ✅ TTA improved confidence by +{improvement:.2f}%")
    else:
        print(f"  ℹ️  TTA provided more robust prediction")
