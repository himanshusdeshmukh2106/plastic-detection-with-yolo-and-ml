import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CalendarCheck, MapPin, Users, Leaf, Activity, AlertTriangle } from 'lucide-react';
import { Api, type DetectionRecord } from '../services/api';
import './PublicPortal.css';

interface PublicStats {
  totalDetections: number;
  cleanedRatio: number;
  activeHotspots: number;
  teamsDeployed: number;
  lastUpdated?: string;
}

const PublicPortal = () => {
  const [stats, setStats] = useState<PublicStats | null>(null);
  const [recentDetections, setRecentDetections] = useState<DetectionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [statsResponse, historyResponse] = await Promise.all([
          Api.fetchStats('month'),
          Api.fetchHistory(20),
        ]);

        setStats({
          totalDetections: statsResponse.total_detections ?? 0,
          cleanedRatio: statsResponse.cleanup_ratio ?? 0,
          activeHotspots: statsResponse.active_hotspots ?? 0,
          teamsDeployed: statsResponse.teams_deployed ?? 0,
          lastUpdated: statsResponse.last_detection,
        });

        setRecentDetections(historyResponse.history.slice(0, 6));
      } catch (error) {
        console.error('Public portal load failed', error);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  if (loading) {
    return <div className="loading">Loading public insights...</div>;
  }

  return (
    <div className="public-portal">
      <section className="hero card">
        <div className="hero-text">
          <h1>Keeping Konkan Coastlines Clean</h1>
          <p className="subtitle">
            ReWater crowdsources detection data across Maharashtra's Konkan belt to track and remediate plastic pollution. Explore live metrics and the impact of our cleanup missions.
          </p>
          <div className="hero-actions">
            <Link to="/login" className="btn btn-primary">Partner Login</Link>
            <Link to="/map" className="btn btn-secondary">View Monitoring Map</Link>
          </div>
        </div>
        <div className="hero-stats">
          <div className="stat-pod">
            <span>Total Detections</span>
            <strong>{stats?.totalDetections.toLocaleString() ?? '—'}</strong>
          </div>
          <div className="stat-pod">
            <span>Cleanup Success</span>
            <strong>{stats?.cleanedRatio.toFixed?.(1) ?? 0}%</strong>
          </div>
          <div className="stat-pod">
            <span>Active Hotspots</span>
            <strong>{stats?.activeHotspots ?? 0}</strong>
          </div>
          <div className="stat-pod">
            <span>Teams Deployed</span>
            <strong>{stats?.teamsDeployed ?? 0}</strong>
          </div>
        </div>
      </section>

      <section className="impact-grid">
        <div className="impact-card card">
          <div className="icon-bubble" style={{ background: '#dbeafe', color: '#1d4ed8' }}>
            <MapPin size={22} />
          </div>
          <h3>Hotspot Monitoring</h3>
          <p>Our AI-enhanced drones continuously scan shores from Palghar to Sindhudurg, flagging plastic hotspots before they escalate.</p>
        </div>
        <div className="impact-card card">
          <div className="icon-bubble" style={{ background: '#dcfce7', color: '#047857' }}>
            <Leaf size={22} />
          </div>
          <h3>Cleanup Collaborations</h3>
          <p>Local communities and NGOs receive real-time alerts, enabling rapid cleanups and measurable impact tracking.</p>
        </div>
        <div className="impact-card card">
          <div className="icon-bubble" style={{ background: '#fef3c7', color: '#b45309' }}>
            <Users size={22} />
          </div>
          <h3>Citizen Science</h3>
          <p>Beachgoers and fishermen share sightings via the ReWater app, enriching our data collection with on-ground intelligence.</p>
        </div>
      </section>

      <section className="recent-detections card">
        <div className="section-header">
          <h2>Recent Detections</h2>
          {stats?.lastUpdated && (
            <span className="timestamp">
              <CalendarCheck size={16} /> Updated {new Date(stats.lastUpdated).toLocaleString()}
            </span>
          )}
        </div>

        <div className="detections-list">
          {recentDetections.map((detection) => (
            <div key={detection.id} className="detection-item">
              <div className="meta">
                <span className="badge badge-info">{detection.waste_type}</span>
                <strong className="location">
                  {detection.latitude.toFixed(2)}, {detection.longitude.toFixed(2)}
                </strong>
              </div>
              <div className="details">
                <span>
                  Confidence {(detection.confidence * 100).toFixed(0)}%
                </span>
                <span>
                  {new Date(detection.timestamp).toLocaleString()}
                </span>
              </div>
              <Link to={`/detection/${detection.id}`} className="btn-link">
                View details
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section className="callout card">
        <div className="callout-header">
          <Activity size={20} />
          <h2>How to Help</h2>
        </div>
        <ul>
          <li>
            <AlertTriangle size={16} /> Report plastic hotspots with pictures and GPS coordinates via the ReWater mobile app.
          </li>
          <li>
            <Users size={16} /> Volunteer for weekend cleanups organised with district authorities.
          </li>
          <li>
            <Leaf size={16} /> Adopt a stretch of coastline and receive monthly pollution analytics.
          </li>
        </ul>
      </section>
    </div>
  );
};

export default PublicPortal;
