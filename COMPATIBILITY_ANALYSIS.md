# Model Compatibility Analysis

## Problem Summary

The ensemble models **cannot be loaded** on the different device due to a **Keras version mismatch**.

---

## Current Setup (Working Device)

Based on the error log, the models were created with:
- **Keras Version**: 3.10.0
- **TensorFlow**: 2.16+ (required for Keras 3.x)
- **Python**: Unknown (but likely 3.9-3.11)
- **Model Format**: HDF5 (.h5)

---

## Different Device Setup (Failing)

The device where models fail to load has:
- **Python**: 3.12.6
- **TensorFlow**: 2.20.0
- **Keras**: 3.12.0

---

## Root Cause

The models were saved with **Keras 3.10.0** but the different device has **Keras 3.12.0**. Even though both are Keras 3.x, there are breaking changes between minor versions:

1. **Deprecated Parameters**: Models contain `batch_shape` and `synchronized` parameters that were removed/changed in newer Keras versions
2. **Serialization Format Changes**: Keras 3.10 → 3.12 changed how models are serialized
3. **Internal Architecture Changes**: Layer configuration format changed between versions

---

## Why This Happens

Keras 3.x is still evolving rapidly, and each minor version (3.10, 3.11, 3.12) can introduce breaking changes. Models saved in one version may not load in another, even within the same major version.

---

## Solutions

### Option 1: Match the Original Environment (Quick Fix)

Install the exact same versions that were used to create the models:

```bash
pip install tensorflow==2.16.2 keras==3.10.0 "numpy<2"
```

**Pros**: Models will load immediately
**Cons**: Using older versions, may have security/bug issues

---

### Option 2: Retrain Models (Recommended)

Retrain the ensemble models with the current environment (TF 2.20.0 + Keras 3.12.0):

```bash
# In Plastic-Detection-Model directory
python train_ensemble.py 5
```

This will create 5 new models compatible with your current setup.

**Pros**: 
- Models will work on current and future setups
- Can use latest features and optimizations
- No version conflicts

**Cons**: 
- Takes 3-4 hours to train
- Requires training dataset

---

### Option 3: Use SavedModel Format (Best Long-term)

Instead of HDF5 (.h5), use TensorFlow's SavedModel format which is more stable across versions:

```python
# When saving models, use:
model.save('model_dir')  # SavedModel format (directory)
# Instead of:
model.save('model.h5')   # HDF5 format (file)
```

Then update loading code:
```python
model = keras.models.load_model('model_dir')
```

**Pros**: Better cross-version compatibility
**Cons**: Requires retraining

---

## Immediate Action Plan

### For Quick Testing (Option 1):

1. Create a virtual environment:
```bash
python -m venv tfenv_310
tfenv_310\Scripts\activate
```

2. Install matching versions:
```bash
pip install tensorflow==2.16.2 keras==3.10.0 numpy==1.26.4 flask flask-cors pillow
```

3. Run the backend:
```bash
python app_react_backend.py
```

---

### For Production (Option 2):

1. Verify you have the training dataset:
```bash
dir training_dataset\training_dataset
```

2. Train new ensemble models:
```bash
python train_ensemble.py 5
```

3. Models will be saved to `tf_files_ensemble/` with current Keras version

4. Test loading:
```bash
python predict_ensemble.py test_image.jpg
```

---

## Prevention for Future

1. **Document Environment**: Always save `requirements.txt`:
```bash
pip freeze > requirements.txt
```

2. **Use SavedModel Format**: More stable than HDF5

3. **Version Lock**: Pin exact versions in requirements:
```
tensorflow==2.20.0
keras==3.12.0
numpy==1.26.4
```

4. **Test Immediately**: After training, test loading on a fresh environment

---

## Key Differences Between Setups

| Aspect | Original (Working) | Different Device (Failing) |
|--------|-------------------|---------------------------|
| Python | Unknown | 3.12.6 |
| TensorFlow | 2.16.x | 2.20.0 |
| Keras | 3.10.0 | 3.12.0 |
| Model Format | HDF5 (.h5) | Same |
| Issue | - | Version mismatch |

---

## Technical Details

The error occurs because:

1. **Keras 3.10.0** saved models with certain parameter formats
2. **Keras 3.12.0** changed how these parameters are handled
3. When loading, Keras 3.12.0 doesn't recognize the old format
4. Results in errors like:
   - `ValueError: Unrecognized keyword arguments: ['batch_shape']`
   - `AttributeError: 'str' object has no attribute 'as_list'`

---

## Recommendation

**Use Option 1 for immediate testing**, then **switch to Option 2 for production**.

This ensures:
- ✅ Quick verification that everything else works
- ✅ Long-term stability with current versions
- ✅ No future compatibility issues
