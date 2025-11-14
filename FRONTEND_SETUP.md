# Plastic Detection Frontend - Complete Setup Guide

## 🎯 Overview

You now have a complete React + Vite frontend with:
- Mobile camera capture interface
- Real-time waste classification
- Geolocation tracking
- Interactive dashboard with statistics
- Detection history and CSV export

## 📁 Project Structure

```
plastic-detection-frontend/
├── src/
│   ├── components/
│   │   ├── MobileCapture.jsx    # Camera & image upload
│   │   ├── MobileCapture.css
│   │   ├── Dashboard.jsx        # Stats & history
│   │   └── Dashboard.css
│   ├── App.jsx                  # Main app with routing
│   ├── App.css                  # Global styles
│   └── main.tsx
├── package.json
└── vite.config.ts

Plastic-Detection-Model/
└── app_react_backend.py         # Flask backend with CORS
```

## 🚀 Quick Start

### Step 1: Start Backend Server

```bash
cd Plastic-Detection-Model
python app_react_backend.py
```

Backend runs on: **http://localhost:5000**

### Step 2: Start Frontend Server

```bash
cd plastic-detection-frontend
npm run dev
```

Frontend runs on: **http://localhost:5173**

### Step 3: Open in Browser

Navigate to: **http://localhost:5173**

## 📱 Features

### Mobile Capture Page (/)
- **Camera Access**: Click "Open Camera" to use device camera
- **Upload**: Or upload existing images
- **Real-time Prediction**: Get instant classification results
- **Geolocation**: Automatic GPS tagging
- **Confidence Scores**: See prediction confidence for all classes

### Dashboard Page (/dashboard)
- **Statistics Cards**: Total detections by waste type
- **Distribution Chart**: Visual breakdown of detections
- **Confidence Meter**: Average model confidence
- **History Table**: Recent detections with timestamps
- **CSV Export**: Download all data
- **Time Filters**: Today, Week, Month, All Time

## 🔌 API Endpoints

Backend provides these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check & model status |
| POST | `/predict` | Predict waste type from image |
| POST | `/save-detection` | Save detection with location |
| GET | `/history?limit=50` | Get detection history |
| GET | `/stats?range=today` | Get dashboard statistics |
| GET | `/export-csv` | Export history as CSV |

## 🎨 UI Features

- **Gradient Design**: Modern purple gradient theme
- **Responsive**: Works on mobile and desktop
- **Animations**: Smooth transitions and hover effects
- **Icons**: Emoji-based icons for visual appeal
- **Loading States**: Spinners and progress indicators
- **Error Handling**: User-friendly error messages

## 🔧 Configuration

### Backend Configuration

Edit `app_react_backend.py`:
- Port: Default 5000
- CORS: Enabled for all origins
- Model path: Auto-detects ensemble models

### Frontend Configuration

Edit `vite.config.ts` if needed:
- Port: Default 5173
- Proxy settings (if required)

## 📊 Data Flow

1. User captures/uploads image
2. Frontend sends image to `/predict`
3. Backend runs ensemble prediction
4. Results returned with confidence scores
5. Frontend saves detection to `/save-detection`
6. Dashboard fetches stats from `/stats` and `/history`

## 🌐 Browser Requirements

- **Chrome/Edge**: Recommended (best camera support)
- **Firefox**: Full support
- **Safari**: iOS 11+ required
- **Mobile**: Camera API support needed

## 🔒 Permissions

The app requires:
- **Camera**: For mobile capture feature
- **Location**: For GPS tagging (optional)

## 🐛 Troubleshooting

### Frontend won't start
```bash
cd plastic-detection-frontend
npm install
npm run dev
```

### Backend won't start
```bash
cd Plastic-Detection-Model
pip install flask flask-cors tensorflow pillow numpy
python app_react_backend.py
```

### Camera not working
- Use HTTPS or localhost
- Check browser permissions
- Try different browser

### CORS errors
- Ensure backend has `flask-cors` installed
- Check backend is running on port 5000

### Predictions failing
- Verify ensemble models are in `ensemble_models/tf_files_ensemble/`
- Check backend console for model loading errors

## 📦 Build for Production

```bash
cd plastic-detection-frontend
npm run build
```

Output in `dist/` folder. Serve with:
```bash
npm run preview
```

## 🎯 Next Steps

1. **Test Mobile**: Open on phone to test camera
2. **Add Detections**: Capture some waste images
3. **View Dashboard**: Check statistics and history
4. **Export Data**: Download CSV of detections
5. **Customize**: Modify colors, add features

## 📝 Notes

- Backend must be running for predictions
- Ensemble models must be loaded (96.5% accuracy)
- Location is optional but enhances tracking
- History is stored in memory (resets on restart)

## 🎉 You're All Set!

Your complete plastic detection system is ready:
- ✅ React frontend with mobile support
- ✅ Flask backend with ensemble models
- ✅ Dashboard with statistics
- ✅ Location tracking
- ✅ CSV export

Enjoy building a cleaner world! 🌍♻️
