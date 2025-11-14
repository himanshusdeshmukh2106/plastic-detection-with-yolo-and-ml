import { type ReactElement, useEffect, useMemo, useState } from 'react'
import { GoogleMap, HeatmapLayerF, MarkerF, useJsApiLoader } from '@react-google-maps/api'
import './Dashboard.css'

type TimeRange = 'today' | 'week' | 'month' | 'all'

type StatsResponse = {
  total_detections: number
  by_class: Record<string, number>
  avg_confidence: number
}

type DetectionLocation = {
  latitude: number
  longitude: number
  accuracy?: number
}

type DetectionHistoryItem = {
  prediction: string
  confidence: number
  timestamp: string
  location?: DetectionLocation | null
}

type DashboardPayload = {
  history: DetectionHistoryItem[]
}

const API_BASE_URL = 'http://localhost:5000'

const MATERIAL_COLORS: Record<string, string> = {
  plastic: '#ff5722',
  paper: '#1e88e5',
  glass: '#43a047',
  metal: '#9c27b0',
  cardboard: '#ffb300'
}

type LatLngLiteral = google.maps.LatLngLiteral

const FALLBACK_CENTER: LatLngLiteral = { lat: 18.5204, lng: 73.8567 }

const Dashboard = (): ReactElement => {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [history, setHistory] = useState<DetectionHistoryItem[]>([])
  const [timeRange, setTimeRange] = useState<TimeRange>('today')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined

  const { isLoaded: isMapLoaded } = useJsApiLoader({
    id: 'plastic-watch-map-loader',
    googleMapsApiKey: googleMapsApiKey ?? '',
    libraries: ['visualization']
  })

  useEffect(() => {
    const fetchDashboardData = async () => {
      setIsLoading(true)
      setError(null)

      try {
        const [statsResponse, historyResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/stats?range=${timeRange}`),
          fetch(`${API_BASE_URL}/history?limit=75`)
        ])

        if (!statsResponse.ok || !historyResponse.ok) {
          throw new Error('The dashboard service returned an error.')
        }

        const statsData: StatsResponse = await statsResponse.json()
        const historyData: DashboardPayload = await historyResponse.json()

        setStats(statsData)
        setHistory(historyData.history ?? [])
      } catch (err) {
        console.error('Dashboard fetch failed:', err)
        setError('We could not load the dashboard data. Please verify the backend service is running.')
      } finally {
        setIsLoading(false)
      }
    }

    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000)
    return () => clearInterval(interval)
  }, [timeRange])

  const detectionsWithLocation = useMemo(
    () => history.filter((item) => item.location && item.location.latitude && item.location.longitude),
    [history]
  )

  const mapCenter = useMemo(() => {
    if (detectionsWithLocation.length === 0) {
      return FALLBACK_CENTER
    }

    const latSum = detectionsWithLocation.reduce((sum, item) => sum + (item.location?.latitude ?? 0), 0)
    const lngSum = detectionsWithLocation.reduce((sum, item) => sum + (item.location?.longitude ?? 0), 0)
    return {
      lat: latSum / detectionsWithLocation.length,
      lng: lngSum / detectionsWithLocation.length
    }
  }, [detectionsWithLocation])

  const heatmapPoints = useMemo(() => {
    if (!isMapLoaded) return []

    return detectionsWithLocation.map((item) => ({
      location: new google.maps.LatLng(item.location!.latitude, item.location!.longitude),
      weight: Math.max(item.confidence / 100, 0.5)
    }))
  }, [detectionsWithLocation, isMapLoaded])

  const materialTotals = useMemo(() => {
    if (!stats) return []
    return Object.entries(stats.by_class ?? {})
      .map(([material, value]) => ({
        material,
        value
      }))
      .sort((a, b) => b.value - a.value)
  }, [stats])

  return (
    <section className="dashboard">
      <div className="dashboard-shell">
        <header className="dashboard-header">
          <div>
            <span className="dashboard-badge">Impact Hub</span>
            <h2>Monitor every detection</h2>
            <p>Track material trends and visualize the hotspots your team has mapped on the field.</p>
          </div>

          <div className="header-controls">
            <select
              className="range-select"
              value={timeRange}
              onChange={(event) => setTimeRange(event.target.value as TimeRange)}
            >
              <option value="today">Today</option>
              <option value="week">Past 7 days</option>
              <option value="month">Past 30 days</option>
              <option value="all">All time</option>
            </select>

            <button
              type="button"
              className="outline-button"
              onClick={async () => {
                try {
                  const response = await fetch(`${API_BASE_URL}/export-csv`)
                  const blob = await response.blob()
                  const url = window.URL.createObjectURL(blob)
                  const anchor = document.createElement('a')
                  anchor.href = url
                  anchor.download = `detections_${new Date().toISOString().split('T')[0]}.csv`
                  anchor.click()
                  window.URL.revokeObjectURL(url)
                } catch (err) {
                  console.error('Export failed:', err)
                  setError('We could not export the CSV file. Please retry.')
                }
              }}
            >
              Export CSV
            </button>
          </div>
        </header>

        {error && (
          <div className="dashboard-alert" role="alert">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="loading-block">
            <span className="spinner" />
            <p>Loading fresh insights…</p>
          </div>
        ) : (
          <>
            <div className="stats-grid">
              <article className="stat-card">
                <h3>Total detections</h3>
                <p className="stat-value">{stats?.total_detections ?? 0}</p>
                <span className="stat-footnote">Logged in selected range</span>
              </article>

              <article className="stat-card">
                <h3>Average confidence</h3>
                <p className="stat-value">
                  {((stats?.avg_confidence ?? 0) * 100).toFixed(1)}%
                </p>
                <span className="stat-footnote">Ensemble certainty</span>
              </article>

              <article className="stat-card">
                <h3>Active hotspots</h3>
                <p className="stat-value">{detectionsWithLocation.length}</p>
                <span className="stat-footnote">Records mapped</span>
              </article>
            </div>

            <div className="dashboard-grid">
              <section className="map-card">
                <div className="map-card-header">
                  <div>
                    <h3>Heatmap of detections</h3>
                    <p>Each hotspot is weighted by the model confidence for that capture.</p>
                  </div>
                  {!googleMapsApiKey && (
                    <span className="map-warning">Add VITE_GOOGLE_MAPS_API_KEY to enable the map.</span>
                  )}
                </div>

                <div className="map-container">
                  {googleMapsApiKey && isMapLoaded ? (
                    <GoogleMap
                      mapContainerClassName="map-instance"
                      center={mapCenter}
                      zoom={detectionsWithLocation.length > 1 ? 11 : 12}
                      options={{
                        disableDefaultUI: true,
                        styles: [
                          { elementType: 'geometry', stylers: [{ color: '#1f1f1f' }] },
                          { elementType: 'labels.text.fill', stylers: [{ color: '#d4d4d4' }] },
                          { elementType: 'labels.text.stroke', stylers: [{ color: '#1f1f1f' }] },
                          { featureType: 'water', stylers: [{ color: '#253858' }] }
                        ]
                      }}
                      onTilesLoaded={() => undefined}
                    >
                      {heatmapPoints.length > 0 && (
                        <HeatmapLayerF
                          data={heatmapPoints.map((point) => point.location)}
                          options={{ radius: 40, opacity: 0.6 }}
                        />
                      )}

                      {detectionsWithLocation.map((item, index) => (
                        <MarkerF
                          key={`${item.timestamp}-${index}`}
                          position={{
                            lat: item.location!.latitude,
                            lng: item.location!.longitude
                          }}
                          icon={{
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: 7,
                            fillOpacity: 1,
                            fillColor: MATERIAL_COLORS[item.prediction] ?? '#ffffff',
                            strokeColor: '#1f1f1f',
                            strokeWeight: 2
                          }}
                        />
                      ))}
                    </GoogleMap>
                  ) : (
                    <div className="map-placeholder">
                      <p>
                        {googleMapsApiKey
                          ? 'Loading map…'
                          : 'Provide a Google Maps API key to see the live heatmap of detections.'}
                      </p>
                    </div>
                  )}
                </div>
              </section>

              <section className="materials-card">
                <h3>Material breakdown</h3>
                <ul>
                  {materialTotals.length === 0 && <li>No detections captured yet.</li>}
                  {materialTotals.map(({ material, value }) => (
                    <li key={material}>
                      <span className="material-chip" style={{ background: MATERIAL_COLORS[material] ?? '#607d8b' }} />
                      <span className="material-label">{material}</span>
                      <span className="material-value">{value}</span>
                    </li>
                  ))}
                </ul>
              </section>
            </div>

            <section className="timeline-card">
              <h3>Latest field captures</h3>
              {history.length === 0 ? (
                <div className="empty-block">
                  <p>No detections logged yet. Capture waste in the field to populate the dashboard.</p>
                </div>
              ) : (
                <ol>
                  {history.slice(0, 10).map((item, index) => (
                    <li key={`${item.timestamp}-${index}`}>
                      <div className="timeline-icon" style={{ background: MATERIAL_COLORS[item.prediction] ?? '#455a64' }}>
                        {item.prediction.slice(0, 1).toUpperCase()}
                      </div>
                      <div className="timeline-content">
                        <header>
                          <span className="timeline-title">{item.prediction}</span>
                          <span className={`confidence-tag ${item.confidence > 90 ? 'high' : item.confidence > 70 ? 'medium' : 'low'}`}>
                            {item.confidence}% confidence
                          </span>
                        </header>
                        <p>{new Date(item.timestamp).toLocaleString()}</p>
                        {item.location ? (
                          <a
                            className="inline-link"
                            href={`https://maps.google.com/?q=${item.location.latitude},${item.location.longitude}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            View on map
                          </a>
                        ) : (
                          <span className="no-location">No location saved</span>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </>
        )}
      </div>
    </section>
  )
}

export default Dashboard
