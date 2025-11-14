# Plastic Detection Frontend

Modern React + Vite frontend for the Plastic Detection System with mobile camera support and real-time dashboard.

## Features

- 📱 **Mobile Capture**: Use device camera to capture waste images
- 🔮 **Real-time Prediction**: Instant classification with 96.5% accuracy ensemble
- 📍 **Geolocation**: Automatic location tagging for detections
- 📊 **Dashboard**: Statistics, charts, and detection history
- 📥 **Export**: Download detection history as CSV
- 🎨 **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- React 19
- Vite
- React Router DOM
- CSS3 with modern gradients and animations

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Backend Server

In the `Plastic-Detection-Model` folder:

```bash
python app_react_backend.py
```

Backend will run on: http://localhost:5000

### 3. Start Frontend Development Server

```bash
npm run dev
```

Frontend will run on: http://localhost:5173

## Usage

### Mobile Capture Page

1. Click "Open Camera" to use device camera
2. Or click "Upload Image" to select from gallery
3. Capture/select an image of waste
4. Click "Analyze Waste" to get prediction
5. View results with confidence scores and location data

### Dashboard Page

- View total detections and breakdown by waste type
- See average model confidence
- Browse recent detection history
- Export all data to CSV

## API Endpoints

The frontend connects to these backend endpoints:

- `POST /predict` - Predict waste type from image
- `POST /save-detection` - Save detection with location
- `GET /history` - Get detection history
- `GET /stats` - Get dashboard statistics
- `GET /export-csv` - Export history as CSV
- `GET /health` - Health check

## Project Structure

```
plastic-detection-frontend/
├── src/
│   ├── components/
│   │   ├── MobileCapture.jsx    # Camera & upload interface
│   │   ├── MobileCapture.css
│   │   ├── Dashboard.jsx        # Statistics & history
│   │   └── Dashboard.css
│   ├── App.jsx                  # Main app with routing
│   ├── App.css                  # Global styles
│   └── main.tsx                 # Entry point
├── public/
├── package.json
└── vite.config.ts
```

## Build for Production

```bash
npm run build
```

Built files will be in the `dist/` folder.

## Browser Support

- Chrome/Edge (recommended for camera access)
- Firefox
- Safari (iOS 11+)
- Mobile browsers with camera API support

## Notes

- Camera access requires HTTPS in production (or localhost for development)
- Geolocation requires user permission
- Backend must be running for predictions to work
- Ensemble models must be loaded in backend

## Troubleshooting

### Camera not working
- Check browser permissions
- Use HTTPS or localhost
- Try different browser

### Predictions failing
- Ensure backend is running on port 5000
- Check that ensemble models are loaded
- Verify CORS is enabled

### Location not available
- Grant location permissions in browser
- Check device GPS is enabled
- Some browsers block location on HTTP

## License

Part of the Plastic Detection System project.
