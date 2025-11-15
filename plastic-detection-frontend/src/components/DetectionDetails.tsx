import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  MapPin,
  Clock,
  Target,
  AlertTriangle,
  CheckCircle,
  Download,
  Share2,
  Flag,
  ArrowLeft,
  Edit,
  Send,
  Weight,
  Compass,
} from 'lucide-react';
import { Api, type DetectionRecord } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './DetectionDetails.css';

interface BoundingBox {
  id?: string;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

type ExtendedDetection = DetectionRecord & { bounding_boxes?: BoundingBox[] };

const DetectionDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [detection, setDetection] = useState<ExtendedDetection | null>(null);
  const [loading, setLoading] = useState(true);
  const [notes, setNotes] = useState('');
  const [savingNotes, setSavingNotes] = useState(false);
  const [statusUpdating, setStatusUpdating] = useState(false);
  const [alertSending, setAlertSending] = useState(false);
  const { token } = useAuth();

  useEffect(() => {
    fetchDetection();
  }, [id]);

  const fetchDetection = async () => {
    try {
      if (!id) return;
      try {
        const detailed = await Api.fetchDetection(id, token || undefined);
        setDetection(detailed as ExtendedDetection);
        setNotes(detailed.notes || '');
      } catch (error) {
        const history = await Api.fetchHistory(1000, token || undefined);
        const found = history.history?.find((d) => d.id === parseInt(id, 10));
        if (found) {
          setDetection(found as ExtendedDetection);
          setNotes(found.notes || '');
        }
      }
    } catch (error) {
      console.error('Error fetching detection:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkForCleanup = async () => {
    if (!detection) return;
    setStatusUpdating(true);
    try {
      const updated = await Api.updateDetection(detection.id, { status: 'priority' }, token || undefined);
      setDetection((prev) => (prev ? { ...prev, ...updated } : prev));
    } catch (error) {
      console.error('Failed to update status', error);
      alert('Unable to mark for cleanup.');
    } finally {
      setStatusUpdating(false);
    }
  };

  const handleFlagFalsePositive = async () => {
    if (!detection) return;
    setStatusUpdating(true);
    try {
      const updated = await Api.updateDetection(detection.id, { status: 'false-positive' }, token || undefined);
      setDetection((prev) => (prev ? { ...prev, ...updated } : prev));
    } catch (error) {
      console.error('Failed to flag detection', error);
      alert('Unable to flag detection.');
    } finally {
      setStatusUpdating(false);
    }
  };

  const handleShare = () => {
    const url = window.location.href;
    navigator.clipboard.writeText(url);
    alert('Link copied to clipboard!');
  };

  const handleDownloadReport = () => {
    if (!id) return;
    window.open(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'}/detections/${id}/report`, '_blank');
  };

  const handleSendAlert = async () => {
    if (!detection) return;
    const team = prompt('Send alert to which team? (e.g., Team Konkan)');
    if (!team) return;
    setAlertSending(true);
    try {
      await Api.triggerAlert({ detectionId: detection.id, message: `High priority cleanup required at detection ${detection.id}`, teamId: team }, token || undefined);
      alert('Alert dispatched successfully.');
    } catch (error) {
      console.error('Alert dispatch failed', error);
      alert('Unable to send alert.');
    } finally {
      setAlertSending(false);
    }
  };

  const handleSaveNotes = async () => {
    if (!detection) return;
    setSavingNotes(true);
    try {
      const updated = await Api.updateDetection(detection.id, { notes }, token || undefined);
      setDetection((prev) => (prev ? { ...prev, ...updated } : prev));
    } catch (error) {
      console.error('Failed to save notes', error);
      alert('Unable to save notes.');
    } finally {
      setSavingNotes(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading detection details...</div>;
  }

  if (!detection) {
    return (
      <div className="not-found">
        <h2>Detection Not Found</h2>
        <button className="btn btn-primary" onClick={() => navigate('/map')}>
          Back to Map
        </button>
      </div>
    );
  }

  const plasticBreakdown = detection.plastic_breakdown ?? [];
  const statusLabel = detection.status ?? 'pending';
  const densityLabel = detection.density ? `${detection.density.toFixed(1)} items/m²` : 'Not available';
  const massLabel = detection.estimated_mass_kg ? `${detection.estimated_mass_kg.toFixed(2)} kg` : 'Not estimated';
  const boundingBoxes = detection.bounding_boxes ?? [];

  return (
    <div className="detection-details">
      <button className="back-btn" onClick={() => navigate(-1)}>
        <ArrowLeft size={18} />
        Back
      </button>

      <div className="details-header">
        <div>
          <h1>Detection #{detection.id}</h1>
          <p className="subtitle">Detailed information and actions</p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={handleShare}>
            <Share2 size={18} />
            Share
          </button>
          <button className="btn btn-secondary" onClick={handleDownloadReport}>
            <Download size={18} />
            Download Report
          </button>
        </div>
      </div>

      <div className="details-grid">
        {/* Main Info */}
        <div className="details-main card">
          <div className="detection-image">
            {detection.image_path ? (
              <div className="image-wrapper">
                <img src={detection.image_path} alt="Detection" />
                {boundingBoxes.map((box, index) => (
                  <div
                    key={box.id ?? index}
                    className="bbox"
                    style={{
                      left: `${box.x * 100}%`,
                      top: `${box.y * 100}%`,
                      width: `${box.width * 100}%`,
                      height: `${box.height * 100}%`,
                    }}
                  >
                    <span>{box.label} {(box.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-image">
                <AlertTriangle size={48} />
                <p>No image available</p>
              </div>
            )}
          </div>

          <div className="detection-info">
            <div className="info-row">
              <div className="info-item">
                <label>Waste Type</label>
                <div className="info-value large">{detection.waste_type}</div>
              </div>
              <div className="info-item">
                <label>Confidence Score</label>
                <div className="info-value large">
                  {(detection.confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${detection.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="info-row">
              <div className="info-item">
                <MapPin size={18} />
                <div>
                  <label>GPS Coordinates</label>
                  <div className="info-value">
                    {detection.latitude.toFixed(6)}, {detection.longitude.toFixed(6)}
                  </div>
                </div>
              </div>
              <div className="info-item">
                <Clock size={18} />
                <div>
                  <label>Timestamp</label>
                  <div className="info-value">
                    {new Date(detection.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>

            <div className="info-row">
              <div className="info-item">
                <Target size={18} />
                <div>
                  <label>Plastic Density</label>
                  <div className="info-value">
                    {detection.density || 24.5} items/m²
                  </div>
                </div>
              </div>
              <div className="info-item">
                <CheckCircle size={18} />
                <div>
                  <label>Status</label>
                  <span className={`badge badge-${statusLabel === 'completed' ? 'success' : statusLabel === 'priority' ? 'warning' : 'info'}`}>
                    {statusLabel.replace('-', ' ')}
                  </span>
                </div>
              </div>
            </div>

            <div className="info-row">
              <div className="info-item">
                <Weight size={18} />
                <div>
                  <label>Estimated Mass</label>
                  <div className="info-value">{massLabel}</div>
                </div>
              </div>
              <div className="info-item">
                <Compass size={18} />
                <div>
                  <label>Nearest Landmark</label>
                  <div className="info-value">{detection.nearest_landmark ?? 'Konkan Coast'}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Actions Panel */}
        <div className="details-sidebar">
          <div className="card actions-card">
            <h3>Actions</h3>
            <div className="action-buttons">
              <button className="action-btn priority" onClick={handleMarkForCleanup} disabled={statusUpdating}>
                <AlertTriangle size={18} />
                <span>{statusUpdating ? 'Updating...' : 'Mark for Cleanup Priority'}</span>
              </button>
              <button className="action-btn" onClick={handleSendAlert} disabled={alertSending}>
                <Send size={18} />
                <span>{alertSending ? 'Sending...' : 'Send Alert to Team'}</span>
              </button>
              <button className="action-btn" onClick={handleFlagFalsePositive} disabled={statusUpdating}>
                <Flag size={18} />
                <span>{statusUpdating ? 'Updating...' : 'Flag False Positive'}</span>
              </button>
            </div>
          </div>

          <div className="card notes-card">
            <h3>Notes & Observations</h3>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add notes or observations..."
              rows={6}
            />
            <button className="btn btn-primary" onClick={handleSaveNotes} disabled={savingNotes}>
              <Edit size={18} />
              {savingNotes ? 'Saving...' : 'Save Notes'}
            </button>
          </div>

          <div className="card location-card">
            <h3>Location Details</h3>
            <div className="location-info">
              <p><strong>Nearest Landmark:</strong> {detection.nearest_landmark ?? 'Konkan Beach'}</p>
              <p><strong>Source:</strong> {detection.source ?? 'Unknown'}</p>
              <p><strong>Density:</strong> {densityLabel}</p>
              <p><strong>Report Link:</strong> <a href={`https://www.google.com/maps/search/?api=1&query=${detection.latitude},${detection.longitude}`} target="_blank" rel="noreferrer">Open in Maps</a></p>
            </div>
          </div>
        </div>
      </div>

      <div className="analytics-row">
        <div className="card breakdown-card">
          <h3>Plastic Type Breakdown</h3>
          <ul>
            {plasticBreakdown.length > 0 ? (
              plasticBreakdown.map((entry) => (
                <li key={entry.type}>
                  <span className="breakdown-type">{entry.type}</span>
                  <span className="breakdown-value">{entry.percentage}%</span>
                </li>
              ))
            ) : (
              <li>No breakdown data</li>
            )}
          </ul>
        </div>

        <div className="card info-card">
          <h3>Detection Metadata</h3>
          <div className="info-grid">
            <div>
              <label>Latitude</label>
              <p>{detection.latitude.toFixed(6)}</p>
            </div>
            <div>
              <label>Longitude</label>
              <p>{detection.longitude.toFixed(6)}</p>
            </div>
            <div>
              <label>Timestamp</label>
              <p>{new Date(detection.timestamp).toLocaleString()}</p>
            </div>
            <div>
              <label>Confidence</label>
              <p>{((detection.confidence ?? 0) * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectionDetails;
