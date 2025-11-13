"""
Analyze the training dataset for the Plastic Detection Model
"""
import os
from pathlib import Path
from collections import Counter
import random

def analyze_dataset(dataset_path='training_dataset/training_dataset'):
    """
    Analyze the image classification dataset
    """
    print("=" * 80)
    print("DATASET ANALYSIS - Plastic Detection Model")
    print("=" * 80)
    
    dataset_dir = Path(dataset_path)
    
    # Try alternative paths if default doesn't exist
    if not dataset_dir.exists():
        alternative_paths = [
            'training_dataset',
            '../training_dataset/training_dataset',
            './training_dataset/training_dataset'
        ]
        for alt_path in alternative_paths:
            alt_dir = Path(alt_path)
            if alt_dir.exists() and any(alt_dir.iterdir()):
                dataset_dir = alt_dir
                print(f"📁 Found dataset at: {dataset_dir.absolute()}")
                break
        else:
            print(f"❌ Dataset not found at: {dataset_path}")
            print(f"❌ Also tried: {alternative_paths}")
            print(f"\n💡 Please extract training_dataset.zip first:")
            print(f"   !unzip -q training_dataset.zip")
            return None, 0
    
    # Get all class directories
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in: {dataset_dir}")
        return None, 0
    
    print(f"\n📁 Dataset Location: {dataset_dir.absolute()}")
    print(f"📊 Number of Classes: {len(class_dirs)}")
    
    # Analyze each class
    class_stats = {}
    total_images = 0
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Count images (jpg, jpeg, png)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        num_images = len(image_files)
        total_images += num_images
        
        # Get sample file sizes
        if image_files:
            sample_files = random.sample(image_files, min(10, len(image_files)))
            file_sizes = [f.stat().st_size for f in sample_files]
            avg_size = sum(file_sizes) / len(file_sizes) / 1024  # KB
        else:
            avg_size = 0
        
        class_stats[class_name] = {
            'count': num_images,
            'avg_size_kb': avg_size
        }
    
    # Print class distribution
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)
    print(f"\n{'Class':<20} {'Images':<10} {'Percentage':<12} {'Avg Size':<12} {'Bar'}")
    print("-" * 80)
    
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        count = stats['count']
        percentage = (count / total_images * 100) if total_images > 0 else 0
        avg_size = stats['avg_size_kb']
        bar = '█' * int(percentage / 2)
        
        print(f"{class_name:<20} {count:<10} {percentage:>10.2f}% {avg_size:>9.1f} KB  {bar}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_images:<10} {100.0:>10.2f}%")
    
    # Calculate class imbalance
    if class_stats:
        counts = [s['count'] for s in class_stats.values()]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n⚖️  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 5:
            print("  ⚠️  WARNING: Significant class imbalance detected!")
            print("  💡 Recommendations:")
            print("     - Use class weights during training")
            print("     - Oversample minority classes (trash)")
            print("     - Use data augmentation more aggressively on minority classes")
        elif imbalance_ratio > 3:
            print("  ⚠️  Moderate class imbalance")
            print("  💡 Consider using class weights during training")
        else:
            print("  ✅ Class distribution is relatively balanced")
    
    # Dataset quality assessment
    print("\n" + "=" * 80)
    print("DATASET QUALITY ASSESSMENT")
    print("=" * 80)
    
    print(f"\n📊 Total Images: {total_images}")
    
    if total_images < 1000:
        print("  ⚠️  Small dataset - may lead to overfitting")
        print("  💡 Recommendation: Use aggressive data augmentation")
    elif total_images < 3000:
        print("  ✅ Moderate dataset size")
        print("  💡 Recommendation: Use data augmentation")
    else:
        print("  ✅ Good dataset size")
    
    # Images per class assessment
    avg_per_class = total_images / len(class_stats) if class_stats else 0
    print(f"\n📈 Average Images per Class: {avg_per_class:.0f}")
    
    if avg_per_class < 200:
        print("  ⚠️  Low images per class")
        print("  💡 Recommendation: Collect more data or use transfer learning")
    elif avg_per_class < 500:
        print("  ✅ Acceptable images per class")
    else:
        print("  ✅ Good images per class")
    
    # Check for problematic classes
    print("\n🔍 Class-Specific Issues:")
    for class_name, stats in sorted(class_stats.items()):
        if stats['count'] < 200:
            print(f"  ⚠️  '{class_name}': Only {stats['count']} images (consider collecting more)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS FOR TRAINING")
    print("=" * 80)
    
    print("\n1. Data Split:")
    print(f"   - Training: {int(total_images * 0.8)} images (80%)")
    print(f"   - Validation: {int(total_images * 0.1)} images (10%)")
    print(f"   - Testing: {int(total_images * 0.1)} images (10%)")
    
    print("\n2. Data Augmentation (Recommended):")
    print("   - Horizontal flip: 50%")
    print("   - Rotation: ±15°")
    print("   - Zoom: 0.8-1.2x")
    print("   - Brightness: ±20%")
    print("   - Contrast: ±20%")
    
    if imbalance_ratio > 3:
        print("\n3. Handle Class Imbalance:")
        print("   - Use class weights in loss function")
        print("   - Oversample minority classes")
        print("   - Use focal loss instead of cross-entropy")
    
    print("\n4. Model Selection:")
    if total_images < 2000:
        print("   - Use smaller models (MobileNet, EfficientNet-B0)")
        print("   - Strong regularization (dropout=0.5)")
        print("   - Transfer learning is ESSENTIAL")
    else:
        print("   - Can use medium models (ResNet50, EfficientNet-B3)")
        print("   - Moderate regularization (dropout=0.3)")
        print("   - Transfer learning recommended")
    
    print("\n5. Training Strategy:")
    print("   - Start with frozen base model (transfer learning)")
    print("   - Train for 20-30 epochs")
    print("   - Fine-tune last layers for 10-20 more epochs")
    print("   - Use early stopping (patience=10)")
    
    print("\n" + "=" * 80)
    
    return class_stats, total_images


def check_image_quality(dataset_path='training_dataset/training_dataset', sample_size=20):
    """
    Check image dimensions and quality
    """
    try:
        from PIL import Image
        
        print("\n" + "=" * 80)
        print("IMAGE QUALITY ANALYSIS")
        print("=" * 80)
        
        dataset_dir = Path(dataset_path)
        
        # Collect sample images from all classes
        all_images = []
        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    all_images.extend(list(class_dir.glob(ext)))
        
        if not all_images:
            print("No images found!")
            return
        
        # Sample random images
        sample_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        widths = []
        heights = []
        aspect_ratios = []
        modes = []
        
        for img_path in sample_images:
            try:
                img = Image.open(img_path)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w/h)
                modes.append(img.mode)
            except Exception as e:
                print(f"  ⚠️  Error reading {img_path.name}: {e}")
        
        if widths:
            print(f"\n📐 Image Dimensions (sampled {len(widths)} images):")
            print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.0f}")
            print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.0f}")
            print(f"  Aspect Ratio: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, avg={sum(aspect_ratios)/len(aspect_ratios):.2f}")
            
            # Check color modes
            mode_counts = Counter(modes)
            print(f"\n🎨 Color Modes:")
            for mode, count in mode_counts.items():
                print(f"  {mode}: {count} images ({count/len(modes)*100:.1f}%)")
            
            # Recommendations
            print(f"\n💡 Recommended Input Size for Training:")
            avg_w = sum(widths)/len(widths)
            avg_h = sum(heights)/len(heights)
            
            if avg_w > 500 or avg_h > 500:
                print(f"  224x224 or 299x299 (standard for transfer learning)")
            else:
                print(f"  Keep original size or resize to 224x224")
            
            # Check for consistency
            if max(widths) / min(widths) > 3 or max(heights) / min(heights) > 3:
                print("\n  ⚠️  Large variation in image sizes detected")
                print("  💡 Recommendation: Resize all images to consistent size during preprocessing")
            else:
                print("\n  ✅ Image sizes are relatively consistent")
        
        print("=" * 80)
        
    except ImportError:
        print("\n⚠️  PIL (Pillow) not installed. Skipping image quality analysis.")
        print("Install with: pip install Pillow")


if __name__ == '__main__':
    print("\n🔍 Starting Dataset Analysis...\n")
    
    # Analyze dataset
    result = analyze_dataset()
    
    if result is not None:
        class_stats, total_images = result
        
        # Check image quality
        check_image_quality()
        
        print("\n✅ Analysis Complete!")
        print("\nNext Steps:")
        print("  1. Review the class distribution and imbalance")
        print("  2. Check if you need to collect more data")
        print("  3. Prepare data augmentation strategy")
        print("  4. Run training with appropriate hyperparameters")
    else:
        print("\n❌ Analysis failed. Please check the dataset path.")
