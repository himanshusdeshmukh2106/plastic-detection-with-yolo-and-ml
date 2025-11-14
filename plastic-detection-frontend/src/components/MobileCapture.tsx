import { type ChangeEvent, type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './MobileCapture.css'

type GeoLocation = {
  latitude: number
  longitude: number
  accuracy?: number
}

type PredictionResponse = {
  prediction: string
  confidence: number
  all_predictions: Record<string, number>
  [key: string]: unknown
}

type LocationStatus = 'idle' | 'loading' | 'ready' | 'denied' | 'unsupported' | 'error'

const PREDICTION_ICONS: Record<string, string> = {
  plastic: '🥤',
  paper: '📄',
  glass: '🍾',
  metal: '🥫',
  cardboard: '📦'
}

const MobileCapture = (): ReactElement => {
  const [imageBlob, setImageBlob] = useState<Blob | File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [location, setLocation] = useState<GeoLocation | null>(null)
  const [locationStatus, setLocationStatus] = useState<LocationStatus>('idle')
  const [cameraActive, setCameraActive] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (!('geolocation' in navigator)) {
      setLocationStatus('unsupported')
      return
    }

    setLocationStatus('loading')

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy
        })
        setLocationStatus('ready')
      },
      (error) => {
        console.error('Location error:', error)
        if (error.code === error.PERMISSION_DENIED) {
          setLocationStatus('denied')
        } else {
          setLocationStatus('error')
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 8000,
        maximumAge: 300000
      }
    )
  }, [])

  const stopCamera = useCallback(() => {
    const tracks = (videoRef.current?.srcObject as MediaStream | null)?.getTracks() ?? []
    tracks.forEach((track) => track.stop())
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }, [])

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [stopCamera])

  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const startCamera = useCallback(async () => {
    try {
      setErrorMessage(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setCameraActive(true)
      }
    } catch (error) {
      console.error('Camera error:', error)
      setErrorMessage('Camera access denied or unavailable. Please allow camera permission or upload a photo instead.')
    }
  }, [])

  const capturePhoto = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas) {
      setErrorMessage('Camera unavailable. Please retry or upload an image.')
      return
    }

    const { videoWidth, videoHeight } = video
    canvas.width = videoWidth
    canvas.height = videoHeight

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      setErrorMessage('Unable to capture from camera.')
      return
    }

    ctx.drawImage(video, 0, 0, videoWidth, videoHeight)

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setErrorMessage('Could not capture image. Please try again.')
          return
        }

        stopCamera()
        setImageBlob(blob)
        setPrediction(null)
        const url = canvas.toDataURL('image/jpeg', 0.95)
        setPreviewUrl(url)
      },
      'image/jpeg',
      0.95
    )
  }, [stopCamera])

  const handleFileUpload = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setImageBlob(file)
    setPrediction(null)
    if (previewUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(URL.createObjectURL(file))
    event.target.value = ''
  }, [previewUrl])

  const saveDetection = useCallback(async (predictionData: PredictionResponse) => {
    try {
      await fetch('http://localhost:5000/save-detection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction: predictionData.prediction,
          confidence: predictionData.confidence,
          location,
          timestamp: new Date().toISOString()
        })
      })
    } catch (error) {
      console.error('Error saving detection:', error)
    }
  }, [location])

  const analyzeImage = useCallback(async () => {
    if (!imageBlob) {
      setErrorMessage('Please capture or upload an image before analyzing.')
      return
    }

    setIsAnalyzing(true)
    setErrorMessage(null)
    setPrediction(null)

    const formData = new FormData()
    formData.append('image', imageBlob)

    if (location) {
      formData.append('latitude', `${location.latitude}`)
      formData.append('longitude', `${location.longitude}`)
      if (location.accuracy) {
        formData.append('accuracy', `${location.accuracy}`)
      }
    }

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      })

      const data: PredictionResponse = await response.json()

      if ('error' in data) {
        const message = typeof data.error === 'string' ? data.error : 'Model failed to analyze the image.'
        setErrorMessage(message)
        return
      }

      setPrediction(data)
      await saveDetection(data)
    } catch (error) {
      console.error('Prediction error:', error)
      setErrorMessage('Unable to reach the AI service. Please ensure the backend is running.')
    } finally {
      setIsAnalyzing(false)
    }
  }, [imageBlob, location, saveDetection])

  const resetCapture = useCallback(() => {
    stopCamera()
    setImageBlob(null)
    setPrediction(null)
    if (previewUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(null)
    setErrorMessage(null)
  }, [previewUrl, stopCamera])

  const predictionBreakdown = useMemo(() => {
    if (!prediction) return []
    return Object.entries(prediction.all_predictions)
      .map(([label, probability]) => ({
        label,
        probability
      }))
      .sort((a, b) => b.probability - a.probability)
  }, [prediction])

  return (
    <section className="mobile-capture">
      <div className="capture-shell">
        <header className="capture-header">
          <div className="capture-title-group">
            <span className="capture-badge">Field Scout</span>
            <h2>Capture litter in real time</h2>
            <p>Snap a photo or upload from your gallery. The model will classify the material and log its location automatically.</p>
          </div>

          <div className="location-status" data-state={locationStatus}>
            {locationStatus === 'loading' && 'Locating you…'}
            {locationStatus === 'ready' && location && (
              <>
                <span className="status-dot" />
                {location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}
              </>
            )}
            {locationStatus === 'denied' && 'Enable location to map detections'}
            {locationStatus === 'unsupported' && 'Location not supported on this device'}
            {locationStatus === 'error' && 'We could not read your location'}
          </div>
        </header>

        <div className="capture-body">
          {!cameraActive && !previewUrl && (
            <div className="capture-options">
              <button type="button" className="accent-button" onClick={startCamera}>
                Start camera
              </button>

              <div className="divider" role="presentation">
                <span>or</span>
              </div>

              <label className="secondary-button">
                Upload from gallery
                <input
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleFileUpload}
                />
              </label>
            </div>
          )}

          {cameraActive && (
            <div className="camera-view">
              <video ref={videoRef} autoPlay playsInline muted />
              <canvas ref={canvasRef} hidden />

              <div className="camera-controls">
                <button type="button" className="capture-button" onClick={capturePhoto}>
                  Capture
                </button>
                <button type="button" className="ghost-button" onClick={resetCapture}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          {previewUrl && !cameraActive && (
            <div className="preview-card">
              <img src={previewUrl} alt="Captured waste" />

              <div className="preview-actions">
                <button
                  type="button"
                  className="accent-button"
                  disabled={isAnalyzing}
                  onClick={analyzeImage}
                >
                  {isAnalyzing ? 'Analyzing…' : 'Analyze material'}
                </button>
                <button type="button" className="ghost-button" onClick={resetCapture}>
                  Retake
                </button>
              </div>
            </div>
          )}
        </div>

        {errorMessage && (
          <div className="capture-alert" role="alert">
            {errorMessage}
          </div>
        )}

        {prediction && (
          <aside className="prediction-panel">
            <div className="prediction-header">
              <span className="prediction-icon">
                {PREDICTION_ICONS[prediction.prediction] ?? '🗑️'}
              </span>
              <div>
                <h3>{prediction.prediction.toUpperCase()}</h3>
                <p>{prediction.confidence}% model confidence</p>
              </div>
            </div>

            <div className="prediction-breakdown">
              {predictionBreakdown.map(({ label, probability }) => (
                <div key={label} className="prediction-row">
                  <div className="prediction-label">
                    {PREDICTION_ICONS[label] ?? '♻️'}
                    <span>{label}</span>
                  </div>
                  <div className="progress-bar">
                    <span
                      className="progress-fill"
                      style={{ width: `${Math.max(probability * 100, 6)}%` }}
                    />
                    <span className="progress-value">{(probability * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>

            {location && (
              <div className="prediction-location">
                <h4>Mapped location</h4>
                <p>
                  {location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}
                  {location.accuracy ? ` · ±${Math.round(location.accuracy)}m` : ''}
                </p>
                <a
                  className="inline-link"
                  target="_blank"
                  rel="noopener noreferrer"
                  href={`https://www.google.com/maps?q=${location.latitude},${location.longitude}`}
                >
                  View in Google Maps
                </a>
              </div>
            )}
          </aside>
        )}
      </div>
    </section>
  )
}

export default MobileCapture
