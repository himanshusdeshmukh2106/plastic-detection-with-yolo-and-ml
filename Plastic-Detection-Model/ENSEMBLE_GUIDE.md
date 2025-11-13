# Ensemble Training Guide

Train multiple models and average their predictions for +3-5% accuracy boost.

---

## 🎯 What is Ensemble Learning?

Instead of relying on one model, train 3-5 models with different random seeds and average their predictions. This reduces variance and improves accuracy.

**Benefits:**
- ✅ +3-5% accuracy improvement
- ✅ More robust predictions
- ✅ Uncertainty estimation
- ✅ Reduces overfitting

---

## 🚀 Quick Start

### Step 1: Train Ensemble (2-3 hours)

```bash
# In Colab or local machine
python train_ensemble.py 3
```

This will train 3 models with seeds: 42, 123, 456

**What happens:**
- Creates `tf_files_ensemble/` directory
- Trains model_1, model_2, model_3
- Each model takes ~40-50 minutes
- Total time: ~2-3 hours

### Step 2: Predict with Ensemble

```bash
# Basic ensemble prediction
python predict_ensemble.py test_image.jpg

# Ensemble + Test-Time Augmentation (best accuracy)
python predict_ensemble.py test_image.jpg --tta
```

---

## 📊 Expected Results

### Single Model (Current)
- Accuracy: 82.11%
- Confidence: Variable

### Ensemble (3 models)
- Accuracy: **85-87%** (+3-5%)
- Confidence: More stable
- Uncertainty: Quantified

### Ensemble + TTA
- Accuracy: **87-89%** (+5-7%)
- Confidence: Very stable
- Best overall performance

---

## 🔧 Training Options

### Train 3 Models (Recommended)
```bash
python train_ensemble.py 3
```
- Time: 2-3 hours
- Disk: ~45 MB
- Accuracy gain: +3-4%

### Train 5 Models (Maximum Accuracy)
```bash
python train_ensemble.py 5
```
- Time: 3-4 hours
- Disk: ~75 MB
- Accuracy gain: +4-5%

### Custom Seeds
Edit `train_ensemble.py` and modify:
```python
seeds = [42, 123, 456, 789, 1011]
```

---

## 📁 Output Structure

After training, you'll have:

```
tf_files_ensemble/
├── class_indices.json          # Class label mapping
├── model_1/
│   ├── final_model.h5         # Model 1 (seed=42)
│   ├── best_phase1.h5
│   └── best_phase2.h5
├── model_2/
│   ├── final_model.h5         # Model 2 (seed=123)
│   ├── best_phase1.h5
│   └── best_phase2.h5
└── model_3/
    ├── final_model.h5         # Model 3 (seed=456)
    ├── best_phase1.h5
    └── best_phase2.h5
```

---

## 🧪 Testing the Ensemble

### Test on Single Image
```python
from predict_ensemble import *

# Load ensemble
models = load_ensemble_models()
labels = load_class_labels()

# Predict
results, avg_pred, std_pred = predict_ensemble(models, 'test.jpg', labels)

# Top prediction
top_class = max(results, key=lambda x: results[x]['probability'])
confidence = results[top_class]['probability']
uncertainty = results[top_class]['std']

print(f"Prediction: {top_class}")
print(f"Confidence: {confidence*100:.2f}% ±{uncertainty*100:.1f}%")
```

### Test on Validation Set
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load validation data
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    'training_dataset/training_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict with ensemble
all_predictions = []
for batch_x, batch_y in val_gen:
    batch_preds = []
    for model in models:
        pred = model.predict(batch_x, verbose=0)
        batch_preds.append(pred)
    
    # Average predictions
    avg_pred = np.mean(batch_preds, axis=0)
    all_predictions.append(avg_pred)
    
    if len(all_predictions) * 32 >= val_gen.samples:
        break

# Calculate accuracy
all_predictions = np.vstack(all_predictions)[:val_gen.samples]
y_pred = np.argmax(all_predictions, axis=1)
y_true = val_gen.classes

accuracy = (y_pred == y_true).mean()
print(f"Ensemble Accuracy: {accuracy*100:.2f}%")
```

---

## 💡 Tips for Best Results

### 1. Use Different Seeds
Different random seeds create diverse models that complement each other.

### 2. Train Fully
Don't stop training early - let each model reach its best performance.

### 3. Check Individual Models
```bash
# Test each model individually
python test_improved_model.py tf_files_ensemble/model_1/final_model.h5
python test_improved_model.py tf_files_ensemble/model_2/final_model.h5
python test_improved_model.py tf_files_ensemble/model_3/final_model.h5
```

All models should have similar accuracy (±2%). If one model is much worse, retrain it.

### 4. Combine with TTA
For maximum accuracy, use both ensemble and TTA:
```bash
python predict_ensemble.py image.jpg --tta
```

### 5. Monitor Uncertainty
High uncertainty (std > 0.1) means models disagree - the prediction might be unreliable.

---

## 🔬 Advanced: Weighted Ensemble

Instead of simple averaging, weight models by their validation accuracy:

```python
# In predict_ensemble.py, modify:

def predict_weighted_ensemble(models, model_weights, image_path, labels):
    """
    Weighted ensemble prediction
    """
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    weighted_predictions = []
    for model, weight in zip(models, model_weights):
        pred = model.predict(img_array, verbose=0)[0]
        weighted_predictions.append(pred * weight)
    
    # Weighted average
    avg_prediction = np.sum(weighted_predictions, axis=0)
    avg_prediction = avg_prediction / np.sum(model_weights)
    
    return avg_prediction

# Usage:
# If model accuracies are: 82%, 83%, 81%
model_weights = [0.82, 0.83, 0.81]
prediction = predict_weighted_ensemble(models, model_weights, 'test.jpg', labels)
```

---

## 📊 Performance Comparison

| Method | Accuracy | Inference Time | Disk Space |
|--------|----------|----------------|------------|
| Single Model | 82.11% | 20ms | 15 MB |
| Ensemble (3) | 85-87% | 60ms | 45 MB |
| Ensemble (5) | 86-88% | 100ms | 75 MB |
| Ensemble + TTA | 87-89% | 300ms | 45 MB |

---

## 🚀 Deployment Options

### Option 1: Deploy All Models
- Best accuracy
- Higher latency (3x slower)
- More disk space

### Option 2: Deploy Best Model Only
- Good accuracy (82%)
- Fast inference
- Less disk space

### Option 3: Deploy 2-Model Ensemble
- Balance between accuracy and speed
- 2x slower than single model
- +2-3% accuracy boost

---

## 🎯 When to Use Ensemble

**Use Ensemble When:**
- ✅ Accuracy is critical
- ✅ You have GPU for inference
- ✅ Latency < 100ms is acceptable
- ✅ You have disk space

**Use Single Model When:**
- ✅ Speed is critical
- ✅ Running on CPU/edge devices
- ✅ Disk space is limited
- ✅ 82% accuracy is sufficient

---

## 📝 Troubleshooting

### Issue: Out of Memory During Training
**Solution:**
```python
# In train_ensemble.py, reduce batch size
'batch_size': 8,  # Instead of 16
```

### Issue: Models Have Very Different Accuracies
**Solution:**
- Check if dataset was shuffled differently
- Retrain the outlier model
- Use weighted ensemble

### Issue: Ensemble Not Improving Accuracy
**Solution:**
- Models might be too similar
- Try different architectures (MobileNetV2 + EfficientNetB0)
- Increase diversity with different augmentation

---

## ✅ Success Checklist

- [ ] Trained 3+ models successfully
- [ ] Each model has 80-85% accuracy
- [ ] Models saved in `tf_files_ensemble/`
- [ ] Tested ensemble prediction
- [ ] Ensemble accuracy > single model accuracy
- [ ] Uncertainty is reasonable (std < 0.15)

---

## 🎉 Expected Timeline

| Task | Time |
|------|------|
| Train Model 1 | 40-50 min |
| Train Model 2 | 40-50 min |
| Train Model 3 | 40-50 min |
| **Total Training** | **2-3 hours** |
| Test Ensemble | 5 min |
| Deploy | 10 min |

---

**Ready to train? Run:**
```bash
python train_ensemble.py 3
```

Then wait 2-3 hours and enjoy your +3-5% accuracy boost! 🚀
