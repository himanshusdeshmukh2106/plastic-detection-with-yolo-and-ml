import { useState, useRef, useEffect, useMemo } from 'react';
import { 
  Upload, Camera, MapPin, Clock, Image as ImageIcon,
  CheckCircle, AlertCircle, Loader, X, Video, PlayCircle,
  Globe, RefreshCw, Timer
} from 'lucide-react';
import { Api } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './UploadInterface.css';

interface UploadResult {
  success: boolean;
  prediction?: string;
  confidence?: number;
  location?: { lat: number; lng: number };
  timestamp?: string;
}

type PreprocessingKey = 'resolution' | 'brightness' | 'sharpness' | 'geo';

interface PreprocessingStep {
  key: PreprocessingKey;
  label: string;
  complete: boolean;
}

const INITIAL_PREPROCESSING_STEPS: PreprocessingStep[] = [
  { key: 'resolution', label: 'Resolution baseline', complete: false },
  { key: 'brightness', label: 'Brightness & exposure normalization', complete: false },
  { key: 'sharpness', label: 'Sharpness baseline', complete: false },
  { key: 'geo', label: 'Geo-tag synchronised', complete: false },
];

const buildPreprocessingSteps = (hasLocation: boolean): PreprocessingStep[] =>
  INITIAL_PREPROCESSING_STEPS.map((step) => ({
    ...step,
    complete: step.key === 'geo' ? hasLocation : false,
  }));

const BRIGHTNESS_RANGE = { min: 35, max: 85 };
const SHARPNESS_THRESHOLD = 25;
const ASPECT_RATIO_RANGE = { min: 0.5, max: 2.2 };

const UploadInterface = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [locationError, setLocationError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [inputSource, setInputSource] = useState<'manual' | 'camera' | 'drone' | 'satellite'>('manual');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [autoIngestEnabled, setAutoIngestEnabled] = useState(false);
  const [ingestFrequency, setIngestFrequency] = useState(30);
  const [ingestSubmitting, setIngestSubmitting] = useState(false);
  const [ingestFeedback, setIngestFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [preprocessingSteps, setPreprocessingSteps] = useState<PreprocessingStep[]>(() => buildPreprocessingSteps(false));
  const [qualityMetrics, setQualityMetrics] = useState<{
    brightness: number | null;
    sharpness: number | null;
    aspectRatio: number | null;
    resolution: { width: number; height: number } | null;
  }>({ brightness: null, sharpness: null, aspectRatio: null, resolution: null });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const { token } = useAuth();

  useEffect(() => {
    // Get GPS location on component mount
    getLocation();
  }, []);

  useEffect(() => {
    if (fileInputRef.current) {
      if (inputSource === 'drone') {
        fileInputRef.current.accept = 'image/*,video/*,.zip';
      } else if (inputSource === 'satellite') {
        fileInputRef.current.accept = 'image/*,.tif,.tiff,.zip';
      } else {
        fileInputRef.current.accept = 'image/*';
      }
    }
  }, [inputSource]);

  useEffect(() => {
    if (!['drone', 'satellite'].includes(inputSource)) {
      setAutoIngestEnabled(false);
      setIngestFeedback(null);
    }
  }, [inputSource]);

  useEffect(() => {
    setPreprocessingSteps((steps) =>
      steps.map((step) => (step.key === 'geo' ? { ...step, complete: !!location } : step)),
    );
  }, [location]);

  const getLocation = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          setLocationError('');
        },
        (_error) => {
          setLocationError('Unable to get location. Using default coordinates.');
          // Default to Konkan coastline
          setLocation({ lat: 16.7, lng: 73.3 });
        }
      );
    } else {
      setLocationError('Geolocation not supported');
      setLocation({ lat: 16.7, lng: 73.3 });
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      validateAndPreview(file);
    }
  };

  const validateAndPreview = (file: File) => {
    const errors: string[] = [];

    const isImage = file.type.startsWith('image/');

    if (inputSource !== 'drone' && inputSource !== 'satellite' && !isImage) {
      errors.push('Only image formats are allowed for manual or camera uploads.');
    }

    if (file.size > 12 * 1024 * 1024) {
      errors.push('File size exceeds 12MB limit.');
    }

    setValidationErrors(errors);

    if (errors.length > 0) {
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setQualityMetrics({ brightness: null, sharpness: null, aspectRatio: null, resolution: null });
    setPreprocessingSteps(buildPreprocessingSteps(!!location));

    if (isImage) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result as string;
        setPreview(dataUrl);
        runImageQualityChecks(dataUrl);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview('');
      setPreprocessingSteps((steps) =>
        steps.map((step) => (step.key === 'geo' ? { ...step, complete: !!location } : { ...step, complete: true })),
      );
    }
  };

  const runImageQualityChecks = (dataUrl: string) => {
    const image = new Image();
    image.onload = () => {
      const { width, height } = image;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(image, 0, 0, width, height);
      const { data } = ctx.getImageData(0, 0, width, height);

      const totalPixels = width * height;
      const grayscale = new Float32Array(totalPixels);
      let brightnessSum = 0;

      for (let i = 0; i < totalPixels; i += 1) {
        const idx = i * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        grayscale[i] = gray;
        brightnessSum += gray;
      }

      const brightness = brightnessSum / totalPixels;
      const brightnessNormalized = (brightness / 255) * 100;

      const step = Math.max(1, Math.floor(Math.sqrt(totalPixels) / 200));
      let laplacianSum = 0;
      let laplacianSqSum = 0;
      let sampleCount = 0;

      for (let y = 1; y < height - 1; y += step) {
        for (let x = 1; x < width - 1; x += step) {
          const idx = y * width + x;
          const center = grayscale[idx];
          const top = grayscale[idx - width];
          const bottom = grayscale[idx + width];
          const left = grayscale[idx - 1];
          const right = grayscale[idx + 1];
          const laplacian = top + bottom + left + right - 4 * center;
          laplacianSum += laplacian;
          laplacianSqSum += laplacian * laplacian;
          sampleCount += 1;
        }
      }

      const laplacianMean = sampleCount ? laplacianSum / sampleCount : 0;
      const laplacianVariance = sampleCount ? laplacianSqSum / sampleCount - laplacianMean * laplacianMean : 0;
      const sharpnessScore = Math.max(laplacianVariance, 0);
      const sharpnessNormalized = Math.min((sharpnessScore / (SHARPNESS_THRESHOLD * 3)) * 100, 100);

      const aspectRatio = height === 0 ? 1 : width / height;
      const resolutionOk = width >= 640 && height >= 640;
      const brightnessOk = brightnessNormalized >= BRIGHTNESS_RANGE.min && brightnessNormalized <= BRIGHTNESS_RANGE.max;
      const sharpnessOk = sharpnessScore >= SHARPNESS_THRESHOLD;
      const aspectRatioOk = aspectRatio >= ASPECT_RATIO_RANGE.min && aspectRatio <= ASPECT_RATIO_RANGE.max;

      setQualityMetrics({
        brightness: Number(brightnessNormalized.toFixed(1)),
        sharpness: Number(sharpnessNormalized.toFixed(1)),
        aspectRatio: Number(aspectRatio.toFixed(2)),
        resolution: { width, height },
      });

      setPreprocessingSteps((steps) =>
        steps.map((step) => {
          if (step.key === 'resolution') return { ...step, complete: resolutionOk };
          if (step.key === 'brightness') return { ...step, complete: brightnessOk };
          if (step.key === 'sharpness') return { ...step, complete: sharpnessOk };
          if (step.key === 'geo') return { ...step, complete: !!location };
          return step;
        }),
      );

      const warnings: string[] = [];
      if (!resolutionOk) warnings.push('Minimum resolution of 640x640 required.');
      if (!brightnessOk) warnings.push('Brightness outside optimal range (35-85%).');
      if (!sharpnessOk) warnings.push('Image appears blurred – adjust focus or stabilize the capture.');
      if (!aspectRatioOk) warnings.push('Extreme aspect ratio detected – ensure proper framing.');

      if (warnings.length > 0) {
        setValidationErrors((prev) => {
          const existing = new Set(prev);
          warnings.forEach((warning) => existing.add(warning));
          return Array.from(existing);
        });
      }
    };
    image.src = dataUrl;
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera');
    }
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            setSelectedFile(file);
            setPreview(canvas.toDataURL());
            stopCamera();
          }
        });
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      setCameraActive(false);
    }
  };

  const handleScheduleAutoIngest = async () => {
    if (!autoIngestEnabled) return;

    if (!['drone', 'satellite'].includes(inputSource)) {
      setIngestFeedback({ type: 'error', message: 'Automated ingestion is available for drone or satellite sources.' });
      return;
    }

    setIngestSubmitting(true);
    setIngestFeedback(null);

    try {
      await Api.scheduleAutoIngest({
        source: inputSource,
        frequencyMinutes: ingestFrequency,
        startImmediately: true,
      }, token || undefined);

      setIngestFeedback({ type: 'success', message: 'Ingestion scheduled successfully.' });
    } catch (error) {
      console.error('Ingestion schedule failed', error);
      setIngestFeedback({ type: 'error', message: 'Unable to schedule ingestion. Please try again shortly.' });
    } finally {
      setIngestSubmitting(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !location) {
      alert('Please select an image and ensure location is available');
      return;
    }

    if (validationErrors.length > 0) {
      alert('Resolve validation errors before uploading.');
      return;
    }

    setUploading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('latitude', location.lat.toString());
    formData.append('longitude', location.lng.toString());
    formData.append('source', inputSource);

    try {
      const prediction = await Api.uploadDetection(formData, token || undefined);

      if (prediction?.prediction) {
        const timestamp = new Date().toISOString();
        setResult({
          success: true,
          prediction: prediction.prediction,
          confidence: prediction.confidence,
          location,
          timestamp,
        });

        await Api.saveDetection({
          waste_type: prediction.prediction,
          confidence: prediction.confidence,
          latitude: location.lat,
          longitude: location.lng,
          timestamp,
          source: inputSource,
        }, token || undefined);
      } else {
        setResult({ success: false });
      }
    } catch (error) {
      console.error('Upload error:', error);
      setResult({ success: false });
    } finally {
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreview('');
    setResult(null);
    setValidationErrors([]);
    setPreprocessingSteps(buildPreprocessingSteps(!!location));
    setQualityMetrics({ brightness: null, sharpness: null, aspectRatio: null, resolution: null });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const sourceDescription = useMemo(() => {
    switch (inputSource) {
      case 'camera':
        return 'Capture live footage from field devices and auto-stream to the processing queue.';
      case 'drone':
        return 'Upload drone video batches or enable scheduled ingestion for autonomous flights.';
      case 'satellite':
        return 'Ingest orthorectified satellite passes or zipped rasters for broad coastline surveillance.';
      default:
        return 'Manually upload still images from any device for immediate analysis.';
    }
  }, [inputSource]);

  return (
    <div className="upload-interface">
      <div className="upload-header">
        <div>
          <h1>Upload Detection Data</h1>
          <p className="subtitle">Upload images for plastic detection analysis</p>
        </div>
      </div>

      <div className="upload-content">
        {/* Input Source Selection */}
        <div className="source-selection card">
          <h3>Input Source</h3>
          <div className="source-buttons">
            <button
              className={`source-btn ${inputSource === 'manual' ? 'active' : ''}`}
              onClick={() => setInputSource('manual')}
            >
              <Upload size={20} />
              <span>Manual Upload</span>
            </button>
            <button
              className={`source-btn ${inputSource === 'camera' ? 'active' : ''}`}
              onClick={() => {
                setInputSource('camera');
                if (!cameraActive) startCamera();
              }}
            >
              <Camera size={20} />
              <span>Camera Capture</span>
            </button>
            <button
              className={`source-btn ${inputSource === 'drone' ? 'active' : ''}`}
              onClick={() => {
                setInputSource('drone');
                if (cameraActive) {
                  stopCamera();
                }
              }}
            >
              <Video size={20} />
              <span>Drone Footage</span>
            </button>
            <button
              className={`source-btn ${inputSource === 'satellite' ? 'active' : ''}`}
              onClick={() => {
                setInputSource('satellite');
                if (cameraActive) {
                  stopCamera();
                }
              }}
            >
              <Globe size={20} />
              <span>Satellite Imagery</span>
            </button>
          </div>
          <p className="source-description">{sourceDescription}</p>
        </div>

        <div className="automation-panel card">
          <h3>Automated Ingestion</h3>
          <p className="subtitle">Schedule unattended pipelines for drone sorties or satellite pass ingestion.</p>
          <div className="automation-controls">
            <label className="auto-ingest">
              <input
                type="checkbox"
                checked={autoIngestEnabled}
                onChange={(e) => {
                  setAutoIngestEnabled(e.target.checked);
                  setIngestFeedback(null);
                }}
              />
              <span>Enable scheduled ingestion</span>
            </label>

            {autoIngestEnabled && (
              <div className="automation-form">
                <label>
                  <span>Source</span>
                  <div className="automation-input">
                    <RefreshCw size={16} />
                    <span>{inputSource === 'drone' ? 'Drone Fleet' : inputSource === 'satellite' ? 'Satellite Imagery' : 'Manual'}</span>
                  </div>
                </label>
                <label>
                  <span>Frequency</span>
                  <div className="automation-select">
                    <Timer size={16} />
                    <select
                      value={ingestFrequency}
                      onChange={(e) => setIngestFrequency(Number(e.target.value))}
                    >
                      <option value={15}>Every 15 minutes</option>
                      <option value={30}>Every 30 minutes</option>
                      <option value={60}>Every 60 minutes</option>
                      <option value={120}>Every 2 hours</option>
                    </select>
                  </div>
                </label>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={async () => {
                    if (ingestSubmitting) return;
                    await handleScheduleAutoIngest();
                  }}
                  disabled={ingestSubmitting}
                >
                  {ingestSubmitting ? 'Scheduling...' : 'Schedule Job'}
                </button>
              </div>
            )}

            {ingestFeedback && (
              <div className={`automation-feedback ${ingestFeedback.type}`}>
                {ingestFeedback.message}
              </div>
            )}
          </div>
        </div>

        <div className="upload-grid">
          {/* Upload Area */}
          <div className="upload-area card">
            {!cameraActive ? (
              <>
                {!preview ? (
                  <div 
                    className="dropzone"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload size={48} />
                    <h3>Drop image here or click to browse</h3>
                    <p>
                      Supports: {inputSource === 'drone'
                        ? 'JPG, PNG, MP4, MOV, ZIP'
                        : inputSource === 'satellite'
                          ? 'JPG, PNG, TIFF, ZIP'
                          : 'JPG, PNG, JPEG'} (Max 12MB)
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      onChange={handleFileSelect}
                      style={{ display: 'none' }}
                    />
                  </div>
                ) : (
                  <div className="preview-container">
                    <button className="remove-btn" onClick={resetUpload}>
                      <X size={18} />
                    </button>
                    <img src={preview} alt="Preview" className="preview-image" />
                    <div className="preview-info">
                      <p className="file-name">{selectedFile?.name}</p>
                      <p className="file-size">
                        {((selectedFile?.size || 0) / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="camera-container">
                <video ref={videoRef} autoPlay playsInline className="camera-video" />
                <div className="camera-controls">
                  <button className="btn btn-primary" onClick={capturePhoto}>
                    <Camera size={18} />
                    Capture Photo
                  </button>
                  <button className="btn btn-secondary" onClick={stopCamera}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Metadata Panel */}
          <div className="metadata-panel card">
            <h3>Detection Metadata</h3>
            
            <div className="metadata-section">
              <div className="metadata-item">
                <MapPin size={18} />
                <div>
                  <label>GPS Coordinates</label>
                  {location ? (
                    <p className="metadata-value">
                      {location.lat.toFixed(6)}, {location.lng.toFixed(6)}
                    </p>
                  ) : (
                    <p className="metadata-error">{locationError}</p>
                  )}
                  <button className="btn-link" onClick={getLocation}>
                    Refresh Location
                  </button>
                </div>
              </div>

              <div className="metadata-item">
                <Clock size={18} />
                <div>
                  <label>Timestamp</label>
                  <p className="metadata-value">
                    {new Date().toLocaleString()}
                  </p>
                </div>
              </div>

              <div className="metadata-item">
                <ImageIcon size={18} />
                <div>
                  <label>Source Type</label>
                  <p className="metadata-value capitalize">{inputSource}</p>
                </div>
              </div>
            </div>

            {/* Upload Button */}
            <button
              className="btn btn-primary btn-large"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? (
                <>
                  <Loader size={18} className="spinner" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload size={18} />
                  Analyze & Upload
                </>
              )}
            </button>

            {/* Result */}
            {result && (
              <div className={`result-box ${result.success ? 'success' : 'error'}`}>
                {result.success ? (
                  <>
                    <CheckCircle size={24} />
                    <div>
                      <h4>Detection Successful!</h4>
                      <p><strong>Type:</strong> {result.prediction}</p>
                      <p><strong>Confidence:</strong> {((result.confidence || 0) * 100).toFixed(1)}%</p>
                      <p className="result-note">Detection saved to database</p>
                    </div>
                  </>
                ) : (
                  <>
                    <AlertCircle size={24} />
                    <div>
                      <h4>Detection Failed</h4>
                      <p>Unable to process image. Please try again.</p>
                    </div>
                  </>
                )}
              </div>
            )}

            {validationErrors.length > 0 && (
              <div className="validation-errors">
                {validationErrors.map((message) => (
                  <p key={message}>{message}</p>
                ))}
              </div>
            )}

            <div className="preprocessing card-light">
              <h4>Preprocessing Pipeline</h4>
              <ul>
                {preprocessingSteps.map((step) => (
                  <li key={step.key} className={step.complete ? 'complete' : ''}>
                    {step.complete ? <CheckCircle size={16} /> : <PlayCircle size={16} />}
                    <span>{step.label}</span>
                  </li>
                ))}
              </ul>
            </div>

            {qualityMetrics.brightness !== null && (
              <div className="quality-metrics card-light">
                <h4>Quality Metrics</h4>
                <div className="metric-row">
                  <span>Brightness</span>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(Math.max(qualityMetrics.brightness ?? 0, 0), 100)}%` }}
                    ></div>
                  </div>
                  <span>{qualityMetrics.brightness?.toFixed?.(0)}%</span>
                </div>
                <div className="metric-row">
                  <span>Sharpness</span>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(Math.max(qualityMetrics.sharpness ?? 0, 0), 100)}%` }}
                    ></div>
                  </div>
                  <span>{qualityMetrics.sharpness?.toFixed?.(0)}%</span>
                </div>
                <div className="metric-row metric-inline">
                  <span>Aspect Ratio</span>
                  <strong>{qualityMetrics.aspectRatio}</strong>
                </div>
                {qualityMetrics.resolution && (
                  <div className="metric-row metric-inline">
                    <span>Resolution</span>
                    <strong>
                      {qualityMetrics.resolution.width} × {qualityMetrics.resolution.height}
                    </strong>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Quality Guidelines */}
        <div className="guidelines card">
          <h3>Image Quality Guidelines</h3>
          <div className="guidelines-grid">
            <div className="guideline-item">
              <CheckCircle size={18} className="check-icon" />
              <span>Clear, well-lit images</span>
            </div>
            <div className="guideline-item">
              <CheckCircle size={18} className="check-icon" />
              <span>Plastic waste clearly visible</span>
            </div>
            <div className="guideline-item">
              <CheckCircle size={18} className="check-icon" />
              <span>Minimum resolution: 640x640</span>
            </div>
            <div className="guideline-item">
              <CheckCircle size={18} className="check-icon" />
              <span>GPS location enabled</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadInterface;
