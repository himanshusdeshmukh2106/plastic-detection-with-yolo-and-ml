# Google Colab Setup Guide

## 🚀 Quick Setup for Google Colab

### Step 1: Clone Repository
```python
!git clone https://github.com/himanshusdeshmukh2106/plastic-detection-with-yolo-and-ml.git
%cd plastic-detection-with-yolo-and-ml/Plastic-Detection-Model
```

### Step 2: Extract Dataset
```python
# Extract the dataset
!unzip -q training_dataset.zip

# Verify extraction
!ls -la training_dataset/

# Check dataset structure
!python check_dataset_path.py
```

This will create: `training_dataset/training_dataset/` with 6 class folders

**Note:** The zip file contains a nested structure, so the path will be `training_dataset/training_dataset/`

### Step 3: Install Dependencies
```python
!pip install -q tensorflow pillow numpy
```

### Step 4: Verify Dataset
```python
!python analyze_dataset.py
```

### Step 5: Train Model
```python
!python train_improved_classification.py
```

⏱️ Takes ~30-45 minutes on Colab GPU (T4)

### Step 6: Test Model
```python
!python test_improved_model.py
```

---

## 📓 Complete Colab Notebook

Copy and paste these cells into a new Colab notebook:

### Cell 1: Setup
```python
# Clone repository
!git clone https://github.com/himanshusdeshmukh2106/plastic-detection-with-yolo-and-ml.git
%cd plastic-detection-with-yolo-and-ml/Plastic-Detection-Model

# Extract dataset
!unzip -q training_dataset.zip

# Verify dataset
!python check_dataset_path.py

# Install dependencies
!pip install -q tensorflow pillow numpy

print("✅ Setup complete!")
```

### Cell 2: Analyze Dataset
```python
!python analyze_dataset.py
```

### Cell 3: Train Model
```python
# Enable GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Train model
!python train_improved_classification.py
```

### Cell 4: Test Model
```python
# Test on sample images
!python test_improved_model.py
```

### Cell 5: Test on Custom Image
```python
# Upload your own image
from google.colab import files
uploaded = files.upload()

# Get filename
import os
filename = list(uploaded.keys())[0]

# Test
!python test_improved_model.py {filename}
```

### Cell 6: Download Trained Model
```python
# Download the trained model
from google.colab import files

# Download model
files.download('tf_files_improved/final_model.h5')
files.download('tf_files_improved/labels.txt')

print("✅ Model downloaded!")
```

---

## 🎯 Expected Output

### After Training:
```
================================================================================
FINAL EVALUATION
================================================================================

📊 Validation Results:
  Loss: 0.3245
  Accuracy: 89.26%
  Top-2 Accuracy: 96.82%

📋 Per-Class Performance:
  cardboard       Accuracy:  87.65%
  glass           Accuracy:  91.23%
  metal           Accuracy:  88.45%
  paper           Accuracy:  92.15%
  plastic         Accuracy:  90.34%
  trash           Accuracy:  85.12%

✅ Training complete!
```

---

## 🔧 Troubleshooting

### Issue: "No GPU available"
**Solution:**
```python
# Enable GPU in Colab
# Runtime → Change runtime type → Hardware accelerator → GPU
```

### Issue: "Out of memory"
**Solution:**
```python
# Edit train_improved_classification.py
# Change batch_size from 32 to 16
CONFIG = {
    'batch_size': 16,  # Reduced from 32
    ...
}
```

### Issue: "Dataset not found"
**Solution:**
```python
# Make sure you're in the correct directory
%cd /content/plastic-detection-with-yolo-and-ml/Plastic-Detection-Model

# Check current location
!pwd

# Check if zip exists
!ls -lh training_dataset.zip

# Extract the dataset
!unzip -q training_dataset.zip

# Verify the structure
!ls -la training_dataset/
!ls -la training_dataset/training_dataset/

# Use the helper script
!python check_dataset_path.py
```

**Expected structure after extraction:**
```
Plastic-Detection-Model/
├── training_dataset.zip
└── training_dataset/
    └── training_dataset/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

### Issue: "Training too slow"
**Solution:**
```python
# Reduce epochs for faster training
# Edit train_improved_classification.py
CONFIG = {
    'epochs': 30,  # Reduced from 50
    ...
}
```

---

## 📊 Colab-Specific Tips

### 1. Keep Session Alive
```python
# Run this in a cell to prevent timeout
import time
from IPython.display import clear_output

while True:
    time.sleep(60)
    clear_output()
    print("Session active...")
```

### 2. Monitor GPU Usage
```python
!nvidia-smi
```

### 3. Check Training Progress
```python
# View training logs
!tail -f tf_files_improved/training.log
```

### 4. Save to Google Drive
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy trained model to Drive
!cp -r tf_files_improved /content/drive/MyDrive/plastic_detection_model
```

---

## 🎓 Full Colab Notebook Template

```python
# ============================================================================
# PLASTIC DETECTION MODEL - GOOGLE COLAB TRAINING
# ============================================================================

# 1. SETUP
print("📦 Setting up environment...")
!git clone https://github.com/himanshusdeshmukh2106/plastic-detection-with-yolo-and-ml.git
%cd plastic-detection-with-yolo-and-ml/Plastic-Detection-Model
!unzip -q training_dataset.zip
!pip install -q tensorflow pillow numpy

# 2. VERIFY GPU
import tensorflow as tf
print(f"\n🎮 GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"TensorFlow Version: {tf.__version__}")

# 3. ANALYZE DATASET
print("\n📊 Analyzing dataset...")
!python analyze_dataset.py

# 4. TRAIN MODEL
print("\n🏋️ Starting training...")
!python train_improved_classification.py

# 5. TEST MODEL
print("\n🧪 Testing model...")
!python test_improved_model.py

# 6. SAVE TO GOOGLE DRIVE (OPTIONAL)
from google.colab import drive
drive.mount('/content/drive')
!cp -r tf_files_improved /content/drive/MyDrive/plastic_detection_model
print("\n✅ Model saved to Google Drive!")

# 7. DOWNLOAD MODEL (OPTIONAL)
from google.colab import files
files.download('tf_files_improved/final_model.h5')
files.download('tf_files_improved/labels.txt')
print("\n✅ All done!")
```

---

## ⏱️ Expected Timeline on Colab

| Task | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Setup & Extract | 2 min | 2 min |
| Dataset Analysis | 1 min | 1 min |
| Phase 1 Training | 20-25 min | 2-3 hours |
| Phase 2 Training | 15-20 min | 1-2 hours |
| Testing | 2 min | 2 min |
| **Total** | **~40-50 min** | **~3-5 hours** |

---

## 💡 Pro Tips

1. **Use GPU:** Always enable GPU in Colab (Runtime → Change runtime type)
2. **Save frequently:** Copy models to Google Drive to avoid losing progress
3. **Monitor training:** Check the output regularly to catch errors early
4. **Reduce batch size:** If you get OOM errors, reduce batch_size to 16 or 8
5. **Use Colab Pro:** For longer training sessions without interruption

---

## 📝 Quick Commands Reference

```bash
# Navigate to project
%cd /content/plastic-detection-with-yolo-and-ml/Plastic-Detection-Model

# Extract dataset
!unzip -q training_dataset.zip

# Analyze dataset
!python analyze_dataset.py

# Train model
!python train_improved_classification.py

# Test model
!python test_improved_model.py

# Check GPU
!nvidia-smi

# List files
!ls -lh

# Check training output
!ls -lh tf_files_improved/
```

---

## 🎉 Success Checklist

- [ ] Repository cloned
- [ ] Dataset extracted (should see `training_dataset/training_dataset/` folder)
- [ ] GPU enabled in Colab
- [ ] Dependencies installed
- [ ] Dataset analyzed (5,054 images, 6 classes)
- [ ] Training completed (Phase 1 + Phase 2)
- [ ] Model saved (tf_files_improved/final_model.h5)
- [ ] Model tested (85-92% accuracy)
- [ ] Model downloaded or saved to Drive

---

**Ready to train? Copy the commands above into Google Colab and run!** 🚀
