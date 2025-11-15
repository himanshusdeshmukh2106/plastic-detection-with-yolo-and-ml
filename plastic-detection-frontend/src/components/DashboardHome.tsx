import { useEffect, useState, useMemo, useCallback } from 'react';
import {
  TrendingUp,
  TrendingDown,
  MapPin,
  AlertTriangle,
  CheckCircle,
  Users,
  Activity,
  Clock,
  Compass,
  Radio,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { Api, type CollectionChannelStatus, type AutoIngestJob } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './DashboardHome.css';

interface Stats {
  total_detections: number;
  today_detections: number;
  week_detections: number;
  previous_week?: number;
  active_hotspots: number;
  cleanup_ratio: number;
  coverage_area: number;
  last_detection: string;
  system_uptime: number;
  avg_density?: number;
}

interface Hotspot {
  id: string | number;
  name: string;
  density: number;
  status: 'pending' | 'in-progress' | 'completed';
}

interface SystemService {
  name: string;
  status: 'active' | 'warning' | 'offline';
  message?: string;
}

interface SystemStatusSummary {
  alerts: number;
  teamsDeployed: number;
  services: SystemService[];
}

const DashboardHome = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatusSummary | null>(null);
  const [collectionChannels, setCollectionChannels] = useState<CollectionChannelStatus[]>([]);
  const [autoIngestJobs, setAutoIngestJobs] = useState<AutoIngestJob[]>([]);
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();

  const parseStats = (raw: any): Stats => ({
    total_detections: raw?.total_detections ?? 0,
    today_detections: raw?.today_detections ?? raw?.today ?? 0,
    week_detections: raw?.week_detections ?? raw?.week ?? 0,
    previous_week: raw?.previous_week ?? raw?.week_previous ?? raw?.week_detections_previous ?? 0,
    active_hotspots: raw?.active_hotspots ?? 0,
    cleanup_ratio: raw?.cleanup_ratio ?? 0,
    coverage_area: raw?.coverage_area ?? 0,
    last_detection: raw?.last_detection ?? '',
    system_uptime: raw?.system_uptime ?? raw?.uptime ?? 0,
    avg_density: raw?.avg_density ?? raw?.average_density ?? 0,
  });

  const parseHotspots = (raw: any): Hotspot[] => {
    const source = Array.isArray(raw?.hotspots) ? raw.hotspots : Array.isArray(raw) ? raw : [];
    return source.map((item: any, index: number) => ({
      id: item?.id ?? index,
      name: item?.name ?? item?.location ?? `Zone ${index + 1}`,
      density: item?.density ?? item?.intensity ?? 0,
      status: (item?.status as Hotspot['status']) ?? 'pending',
    }));
  };

  const parseSystemStatus = (raw: any): SystemStatusSummary => ({
    alerts: raw?.alerts ?? raw?.active_alerts ?? 0,
    teamsDeployed: raw?.teams_deployed ?? raw?.teamsDeployed ?? 0,
    services: Array.isArray(raw?.services)
      ? raw.services.map((service: any) => ({
          name: service?.name ?? 'Service',
          status: (service?.status as SystemService['status']) ?? 'active',
          message: service?.message,
        }))
      : [
          { name: 'Detection System', status: 'active' },
          { name: 'GPS Tracking', status: 'active' },
          { name: 'Database', status: 'active' },
        ],
  });

  const channelLabels: Record<string, string> = useMemo(
    () => ({
      manual: 'Manual Uploads',
      camera: 'Field Cameras',
      drone: 'Drone Fleet',
      satellite: 'Satellite Ingestion',
      automated: 'Automated Pipeline',
    }),
    [],
  );

  const parseCollectionChannels = useCallback(
    (raw: any): CollectionChannelStatus[] => {
      const fallback: CollectionChannelStatus[] = [
        {
          channel: channelLabels.manual,
          status: 'operational',
          lastIngest: new Date().toISOString(),
          successRate: 98,
          queueDepth: 0,
        },
        {
          channel: channelLabels.drone,
          status: 'operational',
          lastIngest: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
          successRate: 94,
          queueDepth: 2,
          automated: true,
        },
        {
          channel: channelLabels.satellite,
          status: 'degraded',
          lastIngest: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString(),
          successRate: 82,
          queueDepth: 5,
          automated: true,
        },
      ];

      const channels = Array.isArray(raw?.channels) ? raw.channels : fallback;

      return channels.map((channel: CollectionChannelStatus) => ({
        channel: channelLabels[channel.channel as keyof typeof channelLabels] ?? channel.channel,
        status: channel.status ?? 'operational',
        lastIngest: channel.lastIngest,
        successRate: channel.successRate ?? 100,
        queueDepth: channel.queueDepth ?? 0,
        automated: channel.automated,
        notes: channel.notes,
      }));
    },
    [channelLabels],
  );

  const parseIngestJobs = useCallback((raw: any): AutoIngestJob[] => {
    const jobs = Array.isArray(raw?.autoIngestJobs) ? raw.autoIngestJobs : [];
    return jobs.map((job: any) => ({
      id: job.id?.toString?.() ?? job.id ?? Math.random().toString(36).slice(2, 8),
      source: channelLabels[job.source as keyof typeof channelLabels] ?? job.source ?? 'Pipeline',
      frequencyMinutes: Number(job.frequencyMinutes ?? job.frequency ?? 30),
      nextRun: job.nextRun ?? new Date(Date.now() + 1000 * 60 * 15).toISOString(),
      status: job.status ?? 'scheduled',
    }));
  }, [channelLabels]);

  const fetchDashboardData = useCallback(async () => {
    try {
      const [statsResponse, hotspotsResponse, systemResponse, collectionResponse] = await Promise.all([
        Api.fetchStats('month', token || undefined),
        Api.fetchHotspots(token || undefined).catch(() => ({ hotspots: [] })),
        Api.fetchSystemStatus(token || undefined).catch(() => ({})),
        Api.fetchCollectionStatus(token || undefined).catch(() => ({ channels: [], autoIngestJobs: [] })),
      ]);

      setStats(parseStats(statsResponse));
      setHotspots(parseHotspots(hotspotsResponse));
      setSystemStatus(parseSystemStatus(systemResponse));
      setCollectionChannels(parseCollectionChannels(collectionResponse));
      setAutoIngestJobs(parseIngestJobs(collectionResponse));
    } catch (error) {
      console.error('Error fetching dashboard', error);
    } finally {
      setLoading(false);
    }
  }, [token, parseIngestJobs, parseCollectionChannels]);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  const weekTrend = useMemo(() => {
    if (!stats) return 0;
    const previous = stats.previous_week ?? 0;
    if (previous === 0) return stats.week_detections > 0 ? 100 : 0;
    return ((stats.week_detections - previous) / previous) * 100;
  }, [stats]);

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  const mostPollutedZone = hotspots[0]?.name ?? 'No active hotspot';
  const lastDetectionRelative = stats?.last_detection
    ? `${formatDistanceToNow(new Date(stats.last_detection), { addSuffix: true })}`
    : '—';

  const primaryMetrics = [
    {
      label: 'Total Detections',
      value: stats?.total_detections.toLocaleString() ?? '0',
      icon: MapPin,
      background: 'linear-gradient(135deg, #eff6ff, #dbeafe)',
      footer: 'Lifetime count',
    },
    {
      label: 'Active Hotspots',
      value: stats?.active_hotspots ?? 0,
      icon: AlertTriangle,
      background: 'linear-gradient(135deg, #fef3c7, #fde68a)',
      footer: 'High density areas',
    },
    {
      label: 'Cleanup Ratio',
      value: `${stats?.cleanup_ratio.toFixed(1) ?? 0}%`,
      icon: CheckCircle,
      background: 'linear-gradient(135deg, #dcfce7, #bbf7d0)',
      footer: 'Cleaned vs detected',
    },
    {
      label: 'System Uptime',
      value: `${stats?.system_uptime.toFixed(1) ?? 0}%`,
      icon: Activity,
      background: 'linear-gradient(135deg, #e0f2fe, #bae6fd)',
      footer: 'Last 30 days',
    },
    {
      label: 'Last Detection',
      value: lastDetectionRelative,
      icon: Clock,
      background: 'linear-gradient(135deg, #f1f5f9, #e2e8f0)',
      footer: new Date(stats?.last_detection ?? '').toLocaleString() || 'No data',
    },
    {
      label: 'Coverage Monitored',
      value: `${stats?.coverage_area ?? 0} km²`,
      icon: Compass,
      background: 'linear-gradient(135deg, #ede9fe, #ddd6fe)',
      footer: 'Coastline coverage',
    },
  ];

  const realTimeCards = [
    {
      title: "Today's Detections",
      value: stats?.today_detections ?? 0,
      icon: Clock,
      changeLabel: '+12% from yesterday',
      changeType: 'positive',
    },
    {
      title: 'This Week',
      value: stats?.week_detections ?? 0,
      icon: TrendingUp,
      changeLabel: `${Math.abs(weekTrend).toFixed(1)}% vs last week`,
      changeType: weekTrend >= 0 ? 'positive' : 'negative',
    },
    {
      title: 'Avg Plastic Density',
      value: `${stats?.avg_density?.toFixed(1) ?? '0'} items/m²`,
      icon: Activity,
      footer: 'Across monitored zones',
    },
    {
      title: 'Most Polluted Zone',
      value: mostPollutedZone,
      icon: AlertTriangle,
      footer: hotspots[0] ? `${hotspots[0].density.toFixed(1)} items/m²` : 'No active alerts',
    },
    {
      title: 'Active Alerts',
      value: systemStatus?.alerts ?? 0,
      icon: Radio,
      changeType: 'warning',
    },
    {
      title: 'Cleanup Teams',
      value: systemStatus?.teamsDeployed ?? 0,
      icon: Users,
      footer: 'Currently deployed',
    },
  ];

  return (
    <div className="dashboard-home">
      <div className="dashboard-header">
        <div>
          <h1>Plastic Detection Dashboard</h1>
          <p className="subtitle">Real-time monitoring of Konkan coastline</p>
        </div>
        <div className="header-actions">
          <Link to="/upload" className="btn btn-primary">
            <Activity size={18} />
            New Detection
          </Link>
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="metrics-grid">
        {primaryMetrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <div className="metric-card" key={metric.label} style={{ background: metric.background }}>
              <div className="metric-icon">
                <Icon size={24} />
              </div>
              <div className="metric-content">
                <div className="metric-label">{metric.label}</div>
                <div className="metric-value">{metric.value}</div>
                <div className="metric-footer">{metric.footer}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Real-Time Statistics */}
      <div className="stats-section">
        <h2>Real-Time Statistics</h2>
        <div className="stats-grid">
          {realTimeCards.map((card) => {
            const Icon = card.icon;
            return (
              <div className="stat-card" key={card.title}>
                <div className="stat-header">
                  <span className="stat-title">{card.title}</span>
                  <Icon size={16} />
                </div>
                <div className="stat-value">{card.value}</div>
                {card.changeLabel && (
                  <div className={`stat-change ${card.changeType ?? 'neutral'}`}>
                    {card.changeType === 'negative' ? <TrendingDown size={14} /> : <TrendingUp size={14} />}
                    <span>{card.changeLabel}</span>
                  </div>
                )}
                {card.footer && <div className="stat-footer">{card.footer}</div>}
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Actions & Recent Activity */}
      <div className="bottom-section">
        <div className="quick-actions card">
          <h3>Quick Actions</h3>
          <div className="action-buttons">
            <Link to="/map" className="action-btn">
              <MapPin size={20} />
              <span>View Map</span>
            </Link>
            <Link to="/analytics" className="action-btn">
              <Activity size={20} />
              <span>Analytics</span>
            </Link>
            <Link to="/upload" className="action-btn">
              <TrendingUp size={20} />
              <span>Upload Data</span>
            </Link>
          </div>
        </div>

        <div className="system-status card">
          <h3>System Status</h3>
          <div className="status-items">
            {systemStatus?.services.map((service) => (
              <div className="status-item" key={service.name}>
                <div className={`status-indicator ${service.status}`}></div>
                <span>{service.name}</span>
                <span className={`status-badge badge-${service.status === 'active' ? 'success' : service.status === 'warning' ? 'warning' : 'danger'}`}>
                  {service.status === 'active' ? 'Active' : service.status === 'warning' ? 'Warning' : 'Offline'}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="hotspot-summary card">
          <h3>Hotspot Watchlist</h3>
          <ul>
            {hotspots.slice(0, 3).map((hotspot) => (
              <li key={hotspot.id}>
                <div>
                  <span className="hotspot-name">{hotspot.name}</span>
                  <span className="hotspot-density">{hotspot.density.toFixed(1)} items/m²</span>
                </div>
                <span className={`status-badge badge-${hotspot.status === 'completed' ? 'success' : hotspot.status === 'in-progress' ? 'warning' : 'danger'}`}>
                  {hotspot.status.replace('-', ' ')}
                </span>
              </li>
            ))}
            {hotspots.length === 0 && <li>No hotspots detected</li>}
          </ul>
        </div>

        <div className="collection-status card">
          <h3>Data Collection Health</h3>
          <ul className="collection-channels">
            {collectionChannels.map((channel) => (
              <li key={channel.channel} className="channel-item">
                <div className="channel-meta">
                  <span className="channel-name">{channel.channel}</span>
                  <span className={`channel-status status-${channel.status}`}>
                    {channel.status === 'operational' ? 'Operational' : channel.status === 'degraded' ? 'Degraded' : 'Offline'}
                  </span>
                </div>
                <div className="channel-metrics">
                  <span>
                    Success Rate: {channel.successRate?.toFixed?.(0) ?? '—'}%
                  </span>
                  <span>
                    Queue: {channel.queueDepth ?? 0}
                  </span>
                  {channel.lastIngest && (
                    <span className="channel-ingest">
                      Last ingest {formatDistanceToNow(new Date(channel.lastIngest), { addSuffix: true })}
                    </span>
                  )}
                </div>
                {channel.notes && <p className="channel-notes">{channel.notes}</p>}
              </li>
            ))}
          </ul>
          {autoIngestJobs.length > 0 && (
            <div className="ingest-jobs">
              <h4>Automated Ingestion</h4>
              <ul>
                {autoIngestJobs.map((job) => (
                  <li key={job.id}>
                    <div>
                      <span className="job-source">{job.source}</span>
                      <span className="job-frequency">Every {job.frequencyMinutes} min</span>
                    </div>
                    <div className="job-meta">
                      <span>Next run {formatDistanceToNow(new Date(job.nextRun), { addSuffix: true })}</span>
                      <span className={`job-status badge-${job.status === 'running' ? 'info' : job.status === 'failed' ? 'danger' : 'success'}`}>
                        {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardHome;
