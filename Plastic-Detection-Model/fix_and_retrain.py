"""
Fixed training script with correct dataset path
This ensures we use ALL 5,054 images, not just 2,527
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the training module
from train_improved_classification import *

# Override the CONFIG with correct path
CONFIG['dataset_path'] = 'training_dataset/training_dataset'

# Verify the path exists and has all images
from pathlib import Path
dataset_path = Path(CONFIG['dataset_path'])

if not dataset_path.exists():
    print(f"❌ Dataset not found at: {dataset_path}")
    print("\n💡 Checking alternative paths...")
    
    # Try to find the correct path
    alternatives = [
        'training_dataset',
        '../training_dataset/training_dataset',
    ]
    
    for alt in alternatives:
        alt_path = Path(alt)
        if alt_path.exists():
            # Count total images
            total = 0
            for class_dir in alt_path.iterdir():
                if class_dir.is_dir():
                    total += len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            
            print(f"  Found: {alt} with {total} images")
            
            if total > 4000:  # Should be ~5054
                CONFIG['dataset_path'] = str(alt_path)
                print(f"  ✅ Using this path!")
                break
    else:
        print("\n❌ Could not find dataset with 5000+ images")
        print("Please check your dataset extraction")
        sys.exit(1)

print("\n" + "=" * 80)
print("FIXED TRAINING CONFIGURATION")
print("=" * 80)
print(f"Dataset path: {CONFIG['dataset_path']}")
print(f"Expected images: ~5,054")
print("=" * 80)

# Now run the training with correct configuration
if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    print(f"\n✅ Verification:")
    print(f"   Training samples: {train_gen.samples} (should be ~4,000)")
    print(f"   Validation samples: {val_gen.samples} (should be ~1,000)")
    
    if train_gen.samples < 3000:
        print("\n⚠️  WARNING: Training samples are too few!")
        print("   This will result in poor accuracy.")
        print("   Please check your dataset extraction.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Calculate class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_gen)
    
    # Create model
    model, base_model = create_model(len(train_gen.class_indices))
    
    # Train model
    history1, history2 = train_model(model, base_model, train_gen, val_gen, class_weights)
    
    # Evaluate model
    results = evaluate_model(model, val_gen)
    
    # Save model
    save_model(model, train_gen)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nFinal Validation Accuracy: {results[1]*100:.2f}%")
    
    if results[1] < 0.50:
        print("\n⚠️  Accuracy is below 50% - this indicates a problem!")
        print("   Possible issues:")
        print("   1. Dataset is too small or incorrectly loaded")
        print("   2. Images are corrupted or mislabeled")
        print("   3. Model architecture is not suitable")
    elif results[1] < 0.70:
        print("\n⚠️  Accuracy is below 70% - room for improvement")
        print("   Consider:")
        print("   1. Training for more epochs")
        print("   2. Using a different model architecture")
        print("   3. Adjusting hyperparameters")
    else:
        print("\n✅ Good accuracy achieved!")
    
    print(f"\nModel saved to: {CONFIG['output_dir']}/")
