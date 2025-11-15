import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  GoogleMap,
  useLoadScript,
  Marker,
  InfoWindow,
  MarkerClusterer,
  // HeatmapLayer, // Deprecated - removed
  // DrawingManager, // Deprecated - removed
  Polyline,
} from '@react-google-maps/api';
import type { Libraries } from '@react-google-maps/api';
import {
  Layers,
  Maximize2,
  Filter,
  Calendar,
  Sliders,
  Download,
  Navigation,
  Ruler,
  PauseCircle,
  PlayCircle,
  History,
} from 'lucide-react';
import { format, formatDistanceToNow, parseISO } from 'date-fns';
import { Api, type DetectionRecord } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './MapInterface.css';

const mapContainerStyle = {
  width: '100%',
  height: '100%',
};

// Note: 'visualization' and 'drawing' are deprecated but still functional
// They will be removed in May 2026. Consider migrating to alternative solutions.
const libraries: Libraries = ['geometry', 'places'];

const defaultCenter: google.maps.LatLngLiteral = {
  lat: 16.7,
  lng: 73.3,
};

const konkanBoundary: google.maps.LatLngLiteral[] = [
  { lat: 19.6967, lng: 72.7653 },
  { lat: 18.8915, lng: 72.8131 },
  { lat: 18.5627, lng: 72.8893 },
  { lat: 17.7886, lng: 73.1139 },
  { lat: 17.0947, lng: 73.2608 },
  { lat: 16.4334, lng: 73.3533 },
  { lat: 15.9246, lng: 73.6788 },
];

const touristBeaches = [
  { name: 'Ganpatipule Beach', lat: 17.1495, lng: 73.2649 },
  { name: 'Alibaug Beach', lat: 18.6400, lng: 72.8756 },
  { name: 'Tarkarli Beach', lat: 16.0154, lng: 73.4893 },
  { name: 'Kashid Beach', lat: 18.4376, lng: 72.8607 },
  { name: 'Murud Beach', lat: 18.3237, lng: 72.9647 },
];

type DrawingShape = google.maps.Polygon | google.maps.Rectangle | google.maps.Polyline;

const RECENT_WINDOW_DAYS = 7;

const MapInterface = () => {
  const { token } = useAuth();
  const [detections, setDetections] = useState<DetectionRecord[]>([]);
  const [mapType, setMapType] = useState<'roadmap' | 'satellite' | 'hybrid' | 'terrain'>('satellite');
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);
  const [dateRange] = useState<'all' | 'today' | 'week' | 'month'>('month');
  const [selectedType, setSelectedType] = useState('all');
  const [selectedDetection, setSelectedDetection] = useState<DetectionRecord | null>(null);
  const [mapRef, setMapRef] = useState<google.maps.Map | null>(null);
  const [zoomLevel, setZoomLevel] = useState(10);
  const [userLocation, setUserLocation] = useState<google.maps.LatLngLiteral | null>(null);
  const [timelineDates, setTimelineDates] = useState<string[]>([]);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationHandle = useRef<number | null>(null);
  const [drawingMode, setDrawingMode] = useState<google.maps.drawing.OverlayType | null>(null);
  const [activeShape, setActiveShape] = useState<DrawingShape | null>(null);
  const [measurement, setMeasurement] = useState<{ area?: number; distance?: number; detections: DetectionRecord[] }>(
    { detections: [] },
  );
  const [detectionAge, setDetectionAge] = useState<'all' | 'recent' | 'historical'>('all');
  const [teams, setTeams] = useState<any[]>([]);
  const [hotspots, setHotspots] = useState<any[]>([]);
  const [showTeams, setShowTeams] = useState(true);
  const [showHotspots, setShowHotspots] = useState(true);
  const [showSidebar, setShowSidebar] = useState(true);
  const [selectedTeam, setSelectedTeam] = useState<any | null>(null);

  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '',
    libraries,
  });

  useEffect(() => {
    const loadDetections = async () => {
      try {
        const response = await Api.fetchHistory(500, token || undefined);
        setDetections(response.history || []);
      } catch (error) {
        console.error('Error fetching detections:', error);
      }
    };

    const loadTeams = async () => {
      try {
        const response = await Api.fetchTeams(token || undefined);
        setTeams(response.teams || []);
      } catch (error) {
        console.error('Error fetching teams:', error);
      }
    };

    const loadHotspots = async () => {
      try {
        const response = await Api.calculateHotspots(5, token || undefined);
        setHotspots(response.hotspots || []);
      } catch (error) {
        console.error('Error fetching hotspots:', error);
      }
    };

    loadDetections();
    loadTeams();
    loadHotspots();
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      loadDetections();
      loadTeams();
      loadHotspots();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [token]);

  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setUserLocation({ lat: position.coords.latitude, lng: position.coords.longitude });
      },
      () => {
        setUserLocation(null);
      },
    );
  }, []);

  useEffect(() => {
    if (detections.length === 0) return;
    const uniqueDays = Array.from(
      new Set(
        detections
          .map((d) => format(parseISO(d.timestamp), 'yyyy-MM-dd'))
          .sort((a, b) => new Date(a).getTime() - new Date(b).getTime()),
      ),
    );
    setTimelineDates(uniqueDays);
    setTimelineIndex(uniqueDays.length - 1);
  }, [detections]);

  useEffect(() => {
    if (!isAnimating || timelineDates.length < 2) {
      if (animationHandle.current) {
        window.clearInterval(animationHandle.current);
      }
      return;
    }

    animationHandle.current = window.setInterval(() => {
      setTimelineIndex((prev) => (prev + 1) % timelineDates.length);
    }, 1800);

    return () => {
      if (animationHandle.current) {
        window.clearInterval(animationHandle.current);
      }
    };
  }, [isAnimating, timelineDates]);

  const filteredDetections = useMemo(() => {
    const timelineCutoff = timelineDates[timelineIndex];
    const now = new Date();
    return detections.filter((detection) => {
      const detectionDate = new Date(detection.timestamp);

      if (dateRange !== 'all') {
        const now = new Date();
        const cutoff = new Date();
        if (dateRange === 'today') {
          cutoff.setHours(0, 0, 0, 0);
        } else if (dateRange === 'week') {
          cutoff.setDate(now.getDate() - 7);
        } else if (dateRange === 'month') {
          cutoff.setMonth(now.getMonth() - 1);
        }
        if (detectionDate < cutoff) return false;
      }

      if (selectedType !== 'all' && detection.waste_type !== selectedType) {
        return false;
      }

      const ageInDays = (now.getTime() - detectionDate.getTime()) / (1000 * 60 * 60 * 24);
      if (detectionAge === 'recent' && ageInDays > RECENT_WINDOW_DAYS) {
        return false;
      }
      if (detectionAge === 'historical' && ageInDays <= RECENT_WINDOW_DAYS) {
        return false;
      }

      if (timelineCutoff) {
        return detectionDate <= new Date(`${timelineCutoff}T23:59:59`);
      }

      return true;
    });
  }, [detections, dateRange, selectedType, timelineDates, timelineIndex, detectionAge]);

  const heatmapData = useMemo(() => {
    if (!isLoaded || !window.google?.maps) return [];
    return filteredDetections.map((detection) => ({
      location: new google.maps.LatLng(detection.latitude, detection.longitude),
      weight: detection.density ?? detection.confidence ?? 1,
    }));
  }, [filteredDetections, isLoaded]);

  const getDetectionAgeCategory = (timestamp: string) => {
    const ageInDays = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60 * 24);
    return ageInDays <= RECENT_WINDOW_DAYS ? 'recent' : 'historical';
  };

  const getMarkerIcon = (detection: DetectionRecord) => {
    const status = detection.status ?? 'pending';
    const ageCategory = getDetectionAgeCategory(detection.timestamp);

    // Color by status: pending=red, assigned=orange, in-progress=blue, completed=green
    let color = '#ef4444';
    if (status === 'assigned') color = '#f59e0b';
    else if (status === 'in-progress') color = '#3b82f6';
    else if (status === 'completed') color = '#10b981';

    const isRecent = ageCategory === 'recent';
    const fillOpacity = isRecent ? 0.95 : 0.55;
    const scale = zoomLevel >= 12 ? (isRecent ? 10 : 8) : isRecent ? 8 : 6.5;

    return {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity,
      strokeColor: '#ffffff',
      strokeWeight: 2,
      scale,
    };
  };

  const getTeamIcon = (team: any) => {
    // Color by status: available=green, busy=orange, offline=gray
    let color = '#10b981';
    if (team.status === 'busy') color = '#f59e0b';
    else if (team.status === 'offline') color = '#6b7280';

    return {
      path: 'M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z',
      fillColor: color,
      fillOpacity: 1,
      strokeColor: '#ffffff',
      strokeWeight: 2,
      scale: 1.5,
      anchor: new google.maps.Point(12, 22),
    };
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  const exportData = () => {
    const params = new URLSearchParams({
      dateRange,
      type: selectedType,
    });
    Api.exportDetections(params, token || undefined)
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'detection-export.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
      })
      .catch((error) => console.error('Export failed', error));
  };

  const onLoad = useCallback((map: google.maps.Map) => {
    setMapRef(map);
  }, []);

  const onUnmount = useCallback(() => {
    setMapRef(null);
  }, []);

  const handleZoomChanged = () => {
    if (mapRef) {
      setZoomLevel(mapRef.getZoom() || 10);
    }
  };

  const clearActiveShape = () => {
    activeShape?.setMap(null);
    setActiveShape(null);
    setMeasurement({ detections: [] });
  };

  const detectionsInsideShape = (shape: DrawingShape) => {
    if (!window.google?.maps || filteredDetections.length === 0) {
      return [];
    }

    if (shape instanceof google.maps.Polygon) {
      return filteredDetections.filter((detection) => {
        const point = new google.maps.LatLng(detection.latitude, detection.longitude);
        return google.maps.geometry.poly.containsLocation(point, shape);
      });
    }

    if (shape instanceof google.maps.Rectangle) {
      const bounds = shape.getBounds();
      if (!bounds) return [];
      return filteredDetections.filter((detection) => {
        const point = new google.maps.LatLng(detection.latitude, detection.longitude);
        return bounds.contains(point);
      });
    }

    return [];
  };

  const handleShapeComplete = (shape: DrawingShape) => {
    clearActiveShape();
    setActiveShape(shape);
    setDrawingMode(null);

    if ('setEditable' in shape && typeof shape.setEditable === 'function') {
      shape.setEditable(true);
    }

    const selectedDetections = detectionsInsideShape(shape);
    let area: number | undefined;
    let distance: number | undefined;

    if (shape instanceof google.maps.Polygon) {
      area = google.maps.geometry.spherical.computeArea(shape.getPath());
    }
    if (shape instanceof google.maps.Rectangle) {
      const bounds = shape.getBounds();
      if (bounds) {
        const ne = bounds.getNorthEast();
        const sw = bounds.getSouthWest();
        const width = google.maps.geometry.spherical.computeDistanceBetween(
          new google.maps.LatLng(ne.lat(), sw.lng()),
          new google.maps.LatLng(ne.lat(), ne.lng()),
        );
        const height = google.maps.geometry.spherical.computeDistanceBetween(
          new google.maps.LatLng(ne.lat(), sw.lng()),
          new google.maps.LatLng(sw.lat(), sw.lng()),
        );
        area = width * height;
      }
    }
    if (shape instanceof google.maps.Polyline) {
      distance = google.maps.geometry.spherical.computeLength(shape.getPath());
    }

    setMeasurement({
      area,
      distance,
      detections: selectedDetections,
    });
  };

  const exportSelectedArea = () => {
    if (!measurement.detections.length) return;
    const rows = measurement.detections.map((d) => [
      d.id,
      d.waste_type,
      (d.confidence ?? 0).toFixed(2),
      d.latitude,
      d.longitude,
      d.timestamp,
      d.status ?? 'pending',
    ]);
    const header = ['ID', 'Type', 'Confidence', 'Latitude', 'Longitude', 'Timestamp', 'Status'];
    const csv = [header, ...rows]
      .map((line) => line.join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'selected-area-detections.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  if (loadError) {
    return <div className="map-error">Error loading Google Maps</div>;
  }

  if (!isLoaded) {
    return <div className="map-loading">Loading Google Maps...</div>;
  }

  return (
    <div className="map-interface">
      <div className="map-header">
        <div>
          <h1>Interactive Map</h1>
          <p className="subtitle">{filteredDetections.length} detections visualised</p>
        </div>
        <div className="map-header-actions">
          <button className="btn btn-secondary" onClick={exportData}>
            <Download size={18} />
            Export Data
          </button>
        </div>
      </div>

      <div className="map-container-wrapper">
        <div className="map-controls">
          <div className="control-section">
            <h3><Layers size={16} /> Map Type</h3>
            <div className="control-buttons">
              {(['satellite', 'roadmap', 'terrain', 'hybrid'] as const).map((type) => (
                <button
                  key={type}
                  className={`control-btn ${mapType === type ? 'active' : ''}`}
                  onClick={() => setMapType(type)}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Heatmap controls - Deprecated API, temporarily disabled */}
          {/* 
          <div className="control-section">
            <h3><Sliders size={16} /> Heatmap</h3>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
              <span>Show Heatmap</span>
            </label>
            {showHeatmap && (
              <div className="slider-control">
                <label>Opacity</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                />
                <span>{(heatmapOpacity * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
          */}

          <div className="control-section">
            <h3><Calendar size={16} /> Timeline</h3>
            <div className="timeline-control">
              <input
                type="range"
                min={0}
                max={Math.max(timelineDates.length - 1, 0)}
                value={timelineIndex}
                onChange={(e) => setTimelineIndex(Number(e.target.value))}
              />
              <div className="timeline-meta">
                <span>{timelineDates[timelineIndex] ?? '—'}</span>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setIsAnimating((prev) => !prev)}
                >
                  {isAnimating ? <PauseCircle size={16} /> : <PlayCircle size={16} />}
                  {isAnimating ? 'Pause' : 'Animate'}
                </button>
              </div>
            </div>
          </div>

          <div className="control-section">
            <h3><Filter size={16} /> Plastic Type</h3>
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="control-select"
            >
              <option value="all">All Types</option>
              <option value="Plastic">Plastic</option>
              <option value="Metal">Metal</option>
              <option value="Glass">Glass</option>
              <option value="Paper">Paper</option>
              <option value="Organic">Organic</option>
            </select>
          </div>

          <div className="control-section">
            <h3><History size={16} /> Detection Age</h3>
            <div className="pill-group">
              {(['all', 'recent', 'historical'] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`pill ${detectionAge === option ? 'active' : ''}`}
                  onClick={() => setDetectionAge(option)}
                >
                  {option === 'all' && 'All'}
                  {option === 'recent' && 'Current (< 7d)'}
                  {option === 'historical' && 'Historical'}
                </button>
              ))}
            </div>
          </div>

          {/* Measurement Tools - Depends on deprecated DrawingManager API, temporarily disabled */}
          {/*
          <div className="control-section">
            <h3><Ruler size={16} /> Measurement Tools</h3>
            <div className="control-buttons">
              <button
                className={`control-btn ${drawingMode === 'polygon' ? 'active' : ''}`}
                onClick={() => setDrawingMode('polygon' as google.maps.drawing.OverlayType)}
              >
                Area
              </button>
              <button
                className={`control-btn ${drawingMode === 'polyline' ? 'active' : ''}`}
                onClick={() => setDrawingMode('polyline' as google.maps.drawing.OverlayType)}
              >
                Distance
              </button>
              <button
                className={`control-btn ${drawingMode === 'rectangle' ? 'active' : ''}`}
                onClick={() => setDrawingMode('rectangle' as google.maps.drawing.OverlayType)}
              >
                Boundary
              </button>
              <button className="control-btn" onClick={() => {
                setDrawingMode(null);
                clearActiveShape();
              }}>
                Clear
              </button>
            </div>
            {measurement.area && (
              <p className="measurement-stat">Area: {(measurement.area / 1_000_000).toFixed(2)} km²</p>
            )}
            {measurement.distance && (
              <p className="measurement-stat">Distance: {(measurement.distance / 1000).toFixed(2)} km</p>
            )}
            {!!measurement.detections.length && (
              <button className="btn btn-secondary btn-sm" onClick={exportSelectedArea}>
                <Download size={14} /> Export Selected Area ({measurement.detections.length})
              </button>
            )}
          </div>
          */}

          <div className="control-section">
            <h3>Legend</h3>
            <div className="legend">
              <div className="legend-item">
                <div className="legend-marker" style={{ background: '#0ea5e9' }}></div>
                <span>High confidence (&gt;80%)</span>
              </div>
              <div className="legend-item">
                <div className="legend-marker" style={{ background: '#f59e0b' }}></div>
                <span>Medium confidence</span>
              </div>
              <div className="legend-item">
                <div className="legend-marker" style={{ background: '#ef4444' }}></div>
                <span>Low confidence</span>
              </div>
              <div className="legend-item">
                <div className="legend-marker" style={{ background: '#22c55e' }}></div>
                <span>Cleanup completed</span>
              </div>
            </div>
          </div>
        </div>

        <div className="map-wrapper">
          <button className="fullscreen-btn" onClick={toggleFullscreen}>
            <Maximize2 size={18} />
          </button>

          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={userLocation || defaultCenter}
            zoom={10}
            mapTypeId={mapType}
            onLoad={onLoad}
            onUnmount={onUnmount}
            onZoomChanged={handleZoomChanged}
            options={{
              zoomControl: true,
              streetViewControl: false,
              mapTypeControl: false,
              fullscreenControl: false,
              gestureHandling: 'greedy',
            }}
          >
            {/* DrawingManager - Deprecated API, temporarily disabled */}
            {/* 
            <DrawingManager
              drawingMode={drawingMode}
              onPolygonComplete={handleShapeComplete}
              onRectangleComplete={handleShapeComplete}
              onPolylineComplete={handleShapeComplete}
              options={{
                drawingControl: false,
                polygonOptions: {
                  fillColor: '#2563eb',
                  fillOpacity: 0.1,
                  strokeColor: '#2563eb',
                  strokeWeight: 2,
                },
                rectangleOptions: {
                  fillColor: '#2563eb',
                  fillOpacity: 0.08,
                  strokeColor: '#2563eb',
                  strokeWeight: 2,
                },
                polylineOptions: {
                  strokeColor: '#2563eb',
                  strokeWeight: 3,
                },
              }}
            />
            */}

            {userLocation && (
              <Marker
                position={userLocation}
                icon={{
                  path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                  fillColor: '#22c55e',
                  fillOpacity: 0.9,
                  strokeWeight: 2,
                  strokeColor: '#14532d',
                  scale: 6,
                }}
              />
            )}

            <MarkerClusterer>
              {(clusterer) => (
                <>
                  {filteredDetections.map((detection) => (
                    <Marker
                      key={detection.id}
                      clusterer={clusterer}
                      position={{ lat: detection.latitude, lng: detection.longitude }}
                      icon={getMarkerIcon(detection)}
                      onClick={() => setSelectedDetection(detection)}
                    />
                  ))}
                </>
              )}
            </MarkerClusterer>

            {/* Team Markers */}
            {showTeams && teams.map((team) => (
              <Marker
                key={team.id}
                position={{ lat: team.location.lat, lng: team.location.lng }}
                icon={getTeamIcon(team)}
                onClick={() => setSelectedTeam(team)}
                title={`${team.name} (${team.status})`}
              />
            ))}

            {/* Hotspot Circles */}
            {showHotspots && hotspots.map((hotspot) => {
              const severityColor = hotspot.severity === 'critical' ? '#dc2626' : hotspot.severity === 'high' ? '#ea580c' : '#f59e0b';
              return (
                <div key={hotspot.id}>
                  <Marker
                    position={{ lat: hotspot.latitude, lng: hotspot.longitude }}
                    icon={{
                      path: google.maps.SymbolPath.CIRCLE,
                      fillColor: severityColor,
                      fillOpacity: 0.3,
                      strokeColor: severityColor,
                      strokeWeight: 2,
                      scale: 15,
                    }}
                  />
                </div>
              );
            })}

            {/* Assignment Lines - Connect teams to their assigned detections */}
            {teams.map((team) => 
              team.assigned_detections?.map((detectionId: number) => {
                const detection = detections.find(d => d.id === detectionId);
                if (!detection) return null;
                
                return (
                  <Polyline
                    key={`${team.id}-${detectionId}`}
                    path={[
                      { lat: team.location.lat, lng: team.location.lng },
                      { lat: detection.latitude, lng: detection.longitude }
                    ]}
                    options={{
                      strokeColor: '#3b82f6',
                      strokeOpacity: 0.6,
                      strokeWeight: 2,
                      geodesic: true,
                    }}
                  />
                );
              })
            )}

            {/* HeatmapLayer - Deprecated API, temporarily disabled */}
            {/* 
            {showHeatmap && heatmapData.length > 0 && (
              <HeatmapLayer
                data={heatmapData}
                options={{ opacity: heatmapOpacity, radius: zoomLevel >= 12 ? 30 : 45 }}
              />
            )}
            */}

            <Polyline
              path={konkanBoundary}
              options={{
                strokeColor: '#0f172a',
                strokeOpacity: 0.7,
                strokeWeight: 3,
                geodesic: true,
              }}
            />

            <MarkerClusterer>
              {(clusterer) => (
                <>
                  {touristBeaches.map((beach) => (
                    <Marker
                      key={beach.name}
                      clusterer={clusterer}
                      position={{ lat: beach.lat, lng: beach.lng }}
                      title={beach.name}
                      icon={{
                        path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
                        fillColor: '#facc15',
                        fillOpacity: 0.9,
                        strokeColor: '#b45309',
                        strokeWeight: 2,
                        scale: 6,
                      }}
                    />
                  ))}
                </>
              )}
            </MarkerClusterer>

            {selectedDetection && (
              <InfoWindow
                position={{ lat: selectedDetection.latitude, lng: selectedDetection.longitude }}
                onCloseClick={() => setSelectedDetection(null)}
              >
                <div className="marker-popup">
                  <h4>{selectedDetection.waste_type}</h4>
                  <p><Navigation size={14} /> {selectedDetection.latitude.toFixed(4)}, {selectedDetection.longitude.toFixed(4)}</p>
                  <p><strong>Confidence:</strong> {(selectedDetection.confidence ?? 0) * 100}%</p>
                  <p><strong>Detected:</strong> {new Date(selectedDetection.timestamp).toLocaleString()}</p>
                  <p>
                    <strong>Age:</strong> {formatDistanceToNow(new Date(selectedDetection.timestamp), { addSuffix: true })}
                    {' '}({getDetectionAgeCategory(selectedDetection.timestamp) === 'recent' ? 'Current' : 'Historical'})
                  </p>
                  {selectedDetection.status && (
                    <p><strong>Status:</strong> {selectedDetection.status}</p>
                  )}
                  {selectedDetection.assigned_team && (
                    <p><strong>Assigned Team:</strong> {teams.find(t => t.id === selectedDetection.assigned_team)?.name || selectedDetection.assigned_team}</p>
                  )}
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => window.location.href = `/detection/${selectedDetection.id}`}
                  >
                    View Details
                  </button>
                </div>
              </InfoWindow>
            )}

            {selectedTeam && (
              <InfoWindow
                position={{ lat: selectedTeam.location.lat, lng: selectedTeam.location.lng }}
                onCloseClick={() => setSelectedTeam(null)}
              >
                <div className="marker-popup">
                  <h4>{selectedTeam.name}</h4>
                  <p><strong>Region:</strong> {selectedTeam.region}</p>
                  <p><strong>Status:</strong> <span className={`status-badge badge-${selectedTeam.status === 'available' ? 'success' : selectedTeam.status === 'busy' ? 'warning' : 'danger'}`}>{selectedTeam.status}</span></p>
                  <p><strong>Capacity:</strong> {selectedTeam.current_load} / {selectedTeam.capacity}</p>
                  <p><strong>Members:</strong> {selectedTeam.members}</p>
                  <p><strong>Total Cleaned:</strong> {selectedTeam.total_cleaned}</p>
                  <p><strong>Rating:</strong> ⭐ {selectedTeam.efficiency_rating}</p>
                  {selectedTeam.assigned_detections && selectedTeam.assigned_detections.length > 0 && (
                    <p><strong>Assigned:</strong> {selectedTeam.assigned_detections.length} locations</p>
                  )}
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        </div>

        {/* Sidebar with Teams, Assignments, and Hotspots */}
        {showSidebar && (
          <div className="map-sidebar">
            <div className="sidebar-header">
              <h3>Cleanup Operations</h3>
              <button className="btn-icon" onClick={() => setShowSidebar(false)}>×</button>
            </div>

            {/* Teams Section */}
            <div className="sidebar-section">
              <h4>Cleanup Teams ({teams.length})</h4>
              <div className="team-list">
                {teams.map(team => (
                  <div key={team.id} className={`team-card ${team.status}`} onClick={() => {
                    setSelectedTeam(team);
                    mapRef?.panTo({ lat: team.location.lat, lng: team.location.lng });
                  }}>
                    <div className="team-header">
                      <span className="team-name">{team.name}</span>
                      <span className={`status-dot ${team.status}`}></span>
                    </div>
                    <div className="team-info">
                      <span>{team.region}</span>
                      <span>{team.current_load}/{team.capacity} assigned</span>
                    </div>
                    {team.assigned_detections && team.assigned_detections.length > 0 && (
                      <div className="team-assignments">
                        <strong>Assigned Locations:</strong>
                        {team.assigned_detections.slice(0, 3).map((detId: number) => {
                          const det = detections.find(d => d.id === detId);
                          return det ? (
                            <div key={detId} className="assigned-location" onClick={(e) => {
                              e.stopPropagation();
                              setSelectedDetection(det);
                              mapRef?.panTo({ lat: det.latitude, lng: det.longitude });
                            }}>
                              📍 {det.nearest_landmark || `Detection #${detId}`}
                            </div>
                          ) : null;
                        })}
                        {team.assigned_detections.length > 3 && (
                          <span className="more-count">+{team.assigned_detections.length - 3} more</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Hotspots Section */}
            <div className="sidebar-section">
              <h4>Pollution Hotspots ({hotspots.length})</h4>
              <div className="hotspot-list">
                {hotspots.map(hotspot => (
                  <div key={hotspot.id} className={`hotspot-card severity-${hotspot.severity}`} onClick={() => {
                    mapRef?.panTo({ lat: hotspot.latitude, lng: hotspot.longitude });
                    mapRef?.setZoom(13);
                  }}>
                    <div className="hotspot-header">
                      <span className="hotspot-name">{hotspot.nearest_landmark}</span>
                      <span className={`severity-badge ${hotspot.severity}`}>{hotspot.severity}</span>
                    </div>
                    <div className="hotspot-info">
                      <span>{hotspot.detection_count} detections</span>
                      <span>{hotspot.radius}km radius</span>
                    </div>
                  </div>
                ))}
                {hotspots.length === 0 && (
                  <p className="empty-state">No active hotspots</p>
                )}
              </div>
            </div>

            {/* Status Legend */}
            <div className="sidebar-section">
              <h4>Legend</h4>
              <div className="legend">
                <div className="legend-item">
                  <span className="legend-dot" style={{backgroundColor: '#ef4444'}}></span>
                  <span>Pending</span>
                </div>
                <div className="legend-item">
                  <span className="legend-dot" style={{backgroundColor: '#f59e0b'}}></span>
                  <span>Assigned</span>
                </div>
                <div className="legend-item">
                  <span className="legend-dot" style={{backgroundColor: '#3b82f6'}}></span>
                  <span>In Progress</span>
                </div>
                <div className="legend-item">
                  <span className="legend-dot" style={{backgroundColor: '#10b981'}}></span>
                  <span>Completed</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {!showSidebar && (
          <button className="sidebar-toggle" onClick={() => setShowSidebar(true)}>
            ☰ Teams & Hotspots
          </button>
        )}
      </div>
    </div>
  );
};

export default MapInterface;
