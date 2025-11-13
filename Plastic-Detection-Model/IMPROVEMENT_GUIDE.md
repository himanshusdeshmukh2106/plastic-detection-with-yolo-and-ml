# Model Improvement Guide

Current Performance: **82.11%** → Target: **85-90%+**

---

## 🎯 Quick Wins (Easy Implementation)

### 1. Test-Time Augmentation (TTA)
**Expected Gain:** +2-4%  
**Effort:** Low  
**Time:** 5 minutes

Predict on multiple augmented versions of the same image and average results.

```python
# Create this file: predict_with_tta.py
from tensorflow import keras
import numpy as np
from PIL import Image
import tensorflow as tf

def predict_with_tta(model, image_path, num_augmentations=5):
    """
    Predict with Test-Time Augmentation
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    predictions = []
    
    # Original image
    predictions.append(model.predict(np.expand_dims(img_array, 0), verbose=0)[0])
    
    # Augmented versions
    for _ in range(num_augmentations - 1):
        # Random augmentations
        aug_img = img_array.copy()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            aug_img = np.fliplr(aug_img)
        
        # Random brightness
        brightness = np.random.uniform(0.9, 1.1)
        aug_img = np.clip(aug_img * brightness, 0, 1)
        
        # Random rotation (small)
        angle = np.random.uniform(-10, 10)
        aug_img_pil = Image.fromarray((aug_img * 255).astype(np.uint8))
        aug_img_pil = aug_img_pil.rotate(angle)
        aug_img = np.array(aug_img_pil) / 255.0
        
        predictions.append(model.predict(np.expand_dims(aug_img, 0), verbose=0)[0])
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

# Usage
model = keras.models.load_model('tf_files_optimized/final_model.h5')
prediction = predict_with_tta(model, 'test_image.jpg', num_augmentations=10)
```

---

### 2. Ensemble Multiple Models
**Expected Gain:** +3-5%  
**Effort:** Medium  
**Time:** 2-3 hours (training time)

Train 3-5 models with different random seeds and average predictions.

```python
# train_ensemble.py
import numpy as np
from train_optimized_small_dataset import *

# Train 5 models with different seeds
models = []
for seed in [42, 123, 456, 789, 1011]:
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL {len(models)+1}/5 (seed={seed})")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Update output directory
    CONFIG['output_dir'] = f'tf_files_ensemble/model_{seed}'
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Train model
    train_gen, val_gen = create_data_generators()
    class_weights = calculate_class_weights(train_gen)
    model, base_model = create_model(len(train_gen.class_indices))
    train_model(model, base_model, train_gen, val_gen, class_weights)
    
    models.append(model)
    print(f"✅ Model {len(models)} trained!")

# Save ensemble
print("\n💾 Saving ensemble models...")
for i, model in enumerate(models):
    model.save(f'tf_files_ensemble/model_{i+1}.h5')

print("✅ Ensemble training complete!")
```

**Predict with ensemble:**
```python
# predict_ensemble.py
from tensorflow import keras
import numpy as np

# Load all models
models = []
for i in range(1, 6):
    model = keras.models.load_model(f'tf_files_ensemble/model_{i}.h5')
    models.append(model)

# Predict
def predict_ensemble(image_array):
    predictions = []
    for model in models:
        pred = model.predict(np.expand_dims(image_array, 0), verbose=0)[0]
        predictions.append(pred)
    
    # Average predictions
    return np.mean(predictions, axis=0)
```

---

### 3. Adjust Class Weights
**Expected Gain:** +1-2%  
**Effort:** Low  
**Time:** 5 minutes

Fine-tune class weights to focus more on underperforming classes.

```python
# In train_optimized_small_dataset.py, modify calculate_class_weights():

def calculate_class_weights_v2(train_generator):
    """Enhanced class weights focusing on plastic"""
    class_counts = {}
    for class_name, class_idx in train_generator.class_indices.items():
        class_dir = Path(CONFIG['dataset_path']) / class_name
        count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
        class_counts[class_idx] = count
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        
        # Boost weight for plastic and cardboard (underperforming)
        class_name = list(train_generator.class_indices.keys())[class_idx]
        if class_name in ['plastic', 'cardboard']:
            weight *= 1.5  # 50% boost
        
        class_weights[class_idx] = weight
    
    return class_weights
```

---

### 4. Lower Confidence Threshold
**Expected Gain:** Better practical performance  
**Effort:** None  
**Time:** Instant

Instead of improving accuracy, improve usability by setting confidence thresholds.

```python
def predict_with_confidence(model, image, threshold=0.7):
    """Only return predictions above threshold"""
    prediction = model.predict(image, verbose=0)[0]
    max_conf = np.max(prediction)
    
    if max_conf < threshold:
        return "uncertain", max_conf
    
    class_idx = np.argmax(prediction)
    return labels[class_idx], max_conf

# Usage
result, confidence = predict_with_confidence(model, img_array, threshold=0.7)
if result == "uncertain":
    print(f"Model is uncertain (confidence: {confidence:.2%})")
else:
    print(f"Prediction: {result} (confidence: {confidence:.2%})")
```

---

## 🚀 Medium Impact Improvements

### 5. Focal Loss (Handle Class Imbalance Better)
**Expected Gain:** +2-3%  
**Effort:** Medium  
**Time:** 30 minutes

Replace categorical crossentropy with focal loss.

```python
# Add to train_optimized_small_dataset.py

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss_fixed

# In model.compile(), replace:
# loss='categorical_crossentropy'
# with:
# loss=focal_loss(gamma=2.0, alpha=0.25)
```

---

### 6. Mixup Data Augmentation
**Expected Gain:** +2-4%  
**Effort:** Medium  
**Time:** 1 hour

Mix two images and their labels during training.

```python
# Add to train_optimized_small_dataset.py

def mixup_generator(generator, alpha=0.2):
    """
    Mixup data augmentation
    """
    while True:
        # Get two batches
        batch_x1, batch_y1 = next(generator)
        batch_x2, batch_y2 = next(generator)
        
        # Generate lambda from beta distribution
        lam = np.random.beta(alpha, alpha, size=batch_x1.shape[0])
        lam = lam.reshape(-1, 1, 1, 1)
        
        # Mix images
        mixed_x = lam * batch_x1 + (1 - lam) * batch_x2
        
        # Mix labels
        lam_y = lam.reshape(-1, 1)
        mixed_y = lam_y * batch_y1 + (1 - lam_y) * batch_y2
        
        yield mixed_x, mixed_y

# Usage in model.fit():
# train_gen_mixup = mixup_generator(train_gen, alpha=0.2)
# model.fit(train_gen_mixup, ...)
```

---

### 7. Progressive Resizing
**Expected Gain:** +1-3%  
**Effort:** Medium  
**Time:** 2 hours

Train on smaller images first, then larger images.

```python
# train_progressive.py

# Phase 1: Train on 128x128
CONFIG['img_size'] = (128, 128)
train_gen, val_gen = create_data_generators()
model, base_model = create_model(6)
train_model(model, base_model, train_gen, val_gen, class_weights, epochs=20)

# Phase 2: Fine-tune on 224x224
CONFIG['img_size'] = (224, 224)
train_gen, val_gen = create_data_generators()
# Rebuild model with new input size
model, base_model = create_model(6)
# Load weights from phase 1 (where possible)
train_model(model, base_model, train_gen, val_gen, class_weights, epochs=30)
```

---

### 8. Use EfficientNetB0 (Middle Ground)
**Expected Gain:** +2-4%  
**Effort:** Low  
**Time:** 1 hour

Try EfficientNetB0 - lighter than B3, heavier than MobileNetV2.

```python
# In train_optimized_small_dataset.py, replace MobileNetV2 with:

from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(*CONFIG['img_size'], 3)
)
```

---

## 🔥 Advanced Improvements (High Impact)

### 9. Collect More Data (HIGHEST IMPACT)
**Expected Gain:** +5-10%  
**Effort:** High  
**Time:** Days/Weeks

**Target:**
- Plastic: 482 → 800+ images
- Cardboard: 403 → 600+ images
- Trash: 137 → 400+ images

**Sources:**
1. **Web scraping** - Google Images, Flickr
2. **Data augmentation** - Generate synthetic images
3. **External datasets** - TrashNet, TACO dataset
4. **Manual collection** - Take photos yourself

```python
# download_additional_data.py
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

# Download plastic images
arguments = {
    "keywords": "plastic waste, plastic bottle, plastic bag",
    "limit": 500,
    "print_urls": True,
    "format": "jpg",
    "output_directory": "training_dataset/training_dataset/plastic"
}
response.download(arguments)
```

---

### 10. Advanced Augmentation (Albumentations)
**Expected Gain:** +2-3%  
**Effort:** Medium  
**Time:** 1 hour

Use Albumentations library for better augmentation.

```python
# Install: !pip install albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(p=1),
        A.GaussianBlur(p=1),
        A.MotionBlur(p=1),
    ], p=0.3),
    A.OneOf([
        A.OpticalDistortion(p=1),
        A.GridDistortion(p=1),
    ], p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=2, p=1),
        A.Sharpen(p=1),
        A.Emboss(p=1),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom data generator with Albumentations
class AlbumentationsGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            img = Image.open(self.image_paths[i])
            img = np.array(img)
            
            # Apply augmentation
            augmented = self.transform(image=img)
            img = augmented['image']
            
            batch_x.append(img)
            batch_y.append(self.labels[i])
        
        return np.array(batch_x), np.array(batch_y)
```

---

### 11. Knowledge Distillation
**Expected Gain:** +2-4%  
**Effort:** High  
**Time:** 3-4 hours

Train a larger "teacher" model, then distill knowledge to smaller "student" model.

```python
# train_with_distillation.py

# Step 1: Train teacher model (EfficientNetB3)
teacher = create_large_model()  # EfficientNetB3
train_model(teacher, ...)

# Step 2: Train student model with distillation
student = create_small_model()  # MobileNetV2

def distillation_loss(y_true, y_pred, teacher_pred, temperature=3.0, alpha=0.5):
    """
    Combine student loss with teacher knowledge
    """
    # Student loss
    student_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Distillation loss (soft targets from teacher)
    teacher_soft = tf.nn.softmax(teacher_pred / temperature)
    student_soft = tf.nn.softmax(y_pred / temperature)
    distill_loss = keras.losses.categorical_crossentropy(teacher_soft, student_soft)
    
    # Combined loss
    return alpha * student_loss + (1 - alpha) * distill_loss * (temperature ** 2)

# Train student with teacher guidance
for batch_x, batch_y in train_gen:
    teacher_pred = teacher.predict(batch_x, verbose=0)
    student.train_on_batch(batch_x, batch_y, 
                          sample_weight=distillation_loss(...))
```

---

### 12. Self-Training (Pseudo-Labeling)
**Expected Gain:** +3-5%  
**Effort:** High  
**Time:** 4-6 hours

Use model to label unlabeled data, then retrain.

```python
# self_training.py

# Step 1: Train initial model
model = train_initial_model()

# Step 2: Collect unlabeled images (e.g., from web)
unlabeled_images = collect_unlabeled_images()

# Step 3: Predict on unlabeled images
pseudo_labels = []
confident_images = []

for img in unlabeled_images:
    pred = model.predict(img)
    confidence = np.max(pred)
    
    # Only use high-confidence predictions
    if confidence > 0.9:
        pseudo_labels.append(np.argmax(pred))
        confident_images.append(img)

# Step 4: Combine with original dataset
combined_images = original_images + confident_images
combined_labels = original_labels + pseudo_labels

# Step 5: Retrain on combined dataset
model = train_model(combined_images, combined_labels)
```

---

## 📊 Improvement Priority Matrix

| Technique | Expected Gain | Effort | Time | Priority |
|-----------|--------------|--------|------|----------|
| **Test-Time Augmentation** | +2-4% | Low | 5 min | 🔥 HIGH |
| **Ensemble (3-5 models)** | +3-5% | Medium | 2-3 hrs | 🔥 HIGH |
| **Adjust Class Weights** | +1-2% | Low | 5 min | 🔥 HIGH |
| **Focal Loss** | +2-3% | Medium | 30 min | ⭐ MEDIUM |
| **Mixup Augmentation** | +2-4% | Medium | 1 hr | ⭐ MEDIUM |
| **EfficientNetB0** | +2-4% | Low | 1 hr | ⭐ MEDIUM |
| **Collect More Data** | +5-10% | High | Days | 🚀 HIGHEST |
| **Advanced Augmentation** | +2-3% | Medium | 1 hr | ⭐ MEDIUM |
| **Knowledge Distillation** | +2-4% | High | 3-4 hrs | ⚠️ LOW |
| **Self-Training** | +3-5% | High | 4-6 hrs | ⚠️ LOW |

---

## 🎯 Recommended Action Plan

### Week 1: Quick Wins
1. ✅ Implement Test-Time Augmentation
2. ✅ Train ensemble of 3 models
3. ✅ Adjust class weights for plastic/cardboard

**Expected Result:** 82% → 87-89%

### Week 2: Data Collection
1. Collect 300+ more plastic images
2. Collect 200+ more cardboard images
3. Collect 200+ more trash images

**Expected Result:** 87-89% → 90-92%

### Week 3: Advanced Techniques
1. Implement Focal Loss
2. Try EfficientNetB0
3. Add Mixup augmentation

**Expected Result:** 90-92% → 92-94%

---

## 🧪 Testing Improvements

After each improvement, test systematically:

```python
# test_improvement.py

def evaluate_improvement(model_old, model_new, test_data):
    """
    Compare two models
    """
    results_old = model_old.evaluate(test_data)
    results_new = model_new.evaluate(test_data)
    
    improvement = (results_new[1] - results_old[1]) * 100
    
    print(f"Old Model: {results_old[1]*100:.2f}%")
    print(f"New Model: {results_new[1]*100:.2f}%")
    print(f"Improvement: +{improvement:.2f}%")
    
    return improvement
```

---

## 📈 Expected Final Results

With all improvements:
- **Current:** 82.11%
- **After Quick Wins:** 87-89%
- **After Data Collection:** 90-92%
- **After Advanced Techniques:** 92-94%

**Realistic Target:** **90-92% accuracy** with 2-3 weeks of work

---

## 💡 Pro Tips

1. **Start with easy wins** - TTA and ensemble give quick results
2. **Data > Algorithms** - More data beats better algorithms
3. **Test incrementally** - Don't combine multiple changes at once
4. **Monitor overfitting** - Keep train-val gap < 10%
5. **Use version control** - Track which changes help
6. **Document everything** - Note what works and what doesn't

---

**Ready to improve? Start with Test-Time Augmentation - it's the easiest +2-4% boost!** 🚀
