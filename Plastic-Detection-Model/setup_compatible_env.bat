@echo off
echo ================================================================================
echo SETTING UP COMPATIBLE ENVIRONMENT FOR ENSEMBLE MODELS
echo ================================================================================
echo.
echo This will create a virtual environment with Keras 3.10.0 to match the models
echo.

REM Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv tfenv_310
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo   Done!

REM Activate and install packages
echo.
echo [2/4] Installing TensorFlow 2.16.2 and Keras 3.10.0...
call tfenv_310\Scripts\activate.bat
pip install --upgrade pip
pip install tensorflow==2.16.2 keras==3.10.0 "numpy<2" flask flask-cors pillow werkzeug
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo   Done!

REM Verify installation
echo.
echo [3/4] Verifying installation...
python -c "import tensorflow as tf; import keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__)"
if errorlevel 1 (
    echo ERROR: Verification failed
    pause
    exit /b 1
)
echo   Done!

REM Test model loading
echo.
echo [4/4] Testing model loading...
python -c "from tensorflow import keras; import os; model_path = 'ensemble_models/tf_files_ensemble/model_1/final_model.h5'; print('Testing:', model_path); model = keras.models.load_model(model_path) if os.path.exists(model_path) else None; print('SUCCESS: Model loaded!' if model else 'Model file not found')"

echo.
echo ================================================================================
echo SETUP COMPLETE!
echo ================================================================================
echo.
echo To use this environment:
echo   1. Run: tfenv_310\Scripts\activate.bat
echo   2. Run: python app_react_backend.py
echo.
echo To deactivate: deactivate
echo.
pause
