import { useEffect, useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
} from 'recharts';
import { Calendar, TrendingUp, MapPin, Package, Thermometer } from 'lucide-react';
import { format, startOfWeek, formatISO, getHours, getMonth, getYear } from 'date-fns';
import { Api, type DetectionRecord } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './Analytics.css';

interface AnalyticsData {
  temporal: Array<{ date: string; count: number }>;
  weekly: Array<{ week: string; detections: number }>;
  monthly: Array<{ month: string; detections: number }>;
  seasonal: Array<{ season: string; count: number }>;
  hourlyHeatmap: Array<{ hour: number; count: number }>;
  dayOfWeek: Array<{ day: string; count: number }>;
  yoy: Array<{ year: string; count: number }>;
  spatial: Array<{ location: string; count: number }>;
  districtBreakdown: Array<{ district: string; count: number }>;
  touristComparison: Array<{ category: string; count: number }>;
  coastalVsInland: Array<{ category: string; count: number }>;
  typeDistribution: Array<{ type: string; count: number; percentage: string }>;
  typeTrendSeries: Array<Record<string, number | string>>;
  sourceBreakdown: Array<{ category: string; count: number; percentage: string }>;
}

const Analytics = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [timeRange, setTimeRange] = useState('month');
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const limit = timeRange === 'week' ? 200 : timeRange === 'year' ? 2000 : 1000;
      const historyResponse = await Api.fetchHistory(limit, token || undefined);
      const history = historyResponse.history || [];
      const processed = processAnalyticsData(history);
      setData(processed);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeason = (month: number) => {
    if ([5, 6, 7, 8].includes(month)) return 'Monsoon';
    if ([9, 10, 11].includes(month)) return 'Post-Monsoon';
    if ([0, 1, 2].includes(month)) return 'Winter';
    return 'Pre-Monsoon';
  };

  const classifyDistrict = (lat: number): string => {
    if (lat >= 18.8) return 'Palghar/Raigad';
    if (lat >= 17.5) return 'Ratnagiri';
    return 'Sindhudurg';
  };

  const processAnalyticsData = (history: DetectionRecord[]): AnalyticsData => {
    const dailyMap = new Map<string, number>();
    const weeklyMap = new Map<string, number>();
    const monthlyMap = new Map<string, number>();
    const seasonalMap = new Map<string, number>();
    const hourlyMap = new Map<number, number>();
    const dayOfWeekMap = new Map<string, number>();
    const yoyMap = new Map<string, number>();
    const locationMap = new Map<string, number>();
    const districtMap = new Map<string, number>();
    const touristMap = new Map<string, number>([['Tourist Areas', 0], ['Non-Tourist', 0]]);
    const coastalMap = new Map<string, number>([['Coastal', 0], ['Inland', 0]]);
    const typeMap = new Map<string, number>();
    const typeTrend = new Map<string, Map<string, number>>();
    const sourceOriginMap = new Map<string, number>();

    const inferSourceOrigin = (item: DetectionRecord): string => {
      const landmark = item.nearest_landmark?.toLowerCase?.() ?? '';
      const source = item.source?.toLowerCase?.() ?? '';

      const fishingKeywords = ['port', 'harbor', 'jetty', 'dock', 'wharf', 'trawler', 'fishing'];
      const touristKeywords = ['beach', 'resort', 'tourist', 'promenade', 'hotel', 'market'];

      if (fishingKeywords.some((keyword) => landmark.includes(keyword))) {
        return 'Fishing Waste';
      }

      if (touristKeywords.some((keyword) => landmark.includes(keyword))) {
        return 'Tourist Waste';
      }

      if (source === 'satellite') {
        return 'Industrial/Other';
      }

      if (source === 'drone' && landmark.includes('offshore')) {
        return 'Industrial/Other';
      }

      return source === 'drone' ? 'Fishing Waste' : 'Industrial/Other';
    };

    history.forEach((item) => {
      const date = new Date(item.timestamp);
      const dayKey = formatISO(date, { representation: 'date' });
      dailyMap.set(dayKey, (dailyMap.get(dayKey) || 0) + 1);

      const weekKey = format(startOfWeek(date, { weekStartsOn: 1 }), 'dd MMM');
      weeklyMap.set(weekKey, (weeklyMap.get(weekKey) || 0) + 1);

      const monthKey = format(date, 'MMM yyyy');
      monthlyMap.set(monthKey, (monthlyMap.get(monthKey) || 0) + 1);

      const season = getSeason(getMonth(date));
      seasonalMap.set(season, (seasonalMap.get(season) || 0) + 1);

      const hour = getHours(date);
      hourlyMap.set(hour, (hourlyMap.get(hour) || 0) + 1);

      const dayName = format(date, 'EEE');
      dayOfWeekMap.set(dayName, (dayOfWeekMap.get(dayName) || 0) + 1);

      const year = getYear(date).toString();
      yoyMap.set(year, (yoyMap.get(year) || 0) + 1);

      const locationKey = `${item.latitude.toFixed(2)}, ${item.longitude.toFixed(2)}`;
      locationMap.set(locationKey, (locationMap.get(locationKey) || 0) + 1);

      const district = item.nearest_landmark?.split(',')[1]?.trim() || classifyDistrict(item.latitude);
      districtMap.set(district, (districtMap.get(district) || 0) + 1);

      const isTourist = item.nearest_landmark?.toLowerCase().includes('beach') || item.nearest_landmark?.toLowerCase().includes('resort');
      touristMap.set(isTourist ? 'Tourist Areas' : 'Non-Tourist', (touristMap.get(isTourist ? 'Tourist Areas' : 'Non-Tourist') || 0) + 1);

      const isCoastal = Math.abs(item.longitude - 73.0) < 0.25;
      coastalMap.set(isCoastal ? 'Coastal' : 'Inland', (coastalMap.get(isCoastal ? 'Coastal' : 'Inland') || 0) + 1);

      const wasteType = item.waste_type ?? 'Unknown';
      typeMap.set(wasteType, (typeMap.get(wasteType) || 0) + 1);

      if (!typeTrend.has(wasteType)) {
        typeTrend.set(wasteType, new Map<string, number>());
      }
      const typeSeries = typeTrend.get(wasteType)!;
      typeSeries.set(dayKey, (typeSeries.get(dayKey) || 0) + 1);

      const sourceOrigin = inferSourceOrigin(item);
      sourceOriginMap.set(sourceOrigin, (sourceOriginMap.get(sourceOrigin) || 0) + 1);
    });

    const temporal = Array.from(dailyMap.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
      .slice(-30)
      .map(([date, count]) => ({ date, count }));

    const weekly = Array.from(weeklyMap.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
      .map(([week, detections]) => ({ week, detections }))
      .slice(-8);

    const monthly = Array.from(monthlyMap.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
      .map(([month, detections]) => ({ month, detections }))
      .slice(-12);

    const seasonal = Array.from(seasonalMap.entries()).map(([season, count]) => ({ season, count }));

    const hourlyHeatmap = Array.from({ length: 24 }, (_, hour) => ({
      hour,
      count: hourlyMap.get(hour) || 0,
    }));

    const dayOrder = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const dayOfWeek = dayOrder.map((day) => ({ day, count: dayOfWeekMap.get(day) || 0 }));

    const yoy = Array.from(yoyMap.entries())
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([year, count]) => ({ year, count }))
      .slice(-5);

    const spatial = Array.from(locationMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([location, count]) => ({ location, count }));

    const districtBreakdown = Array.from(districtMap.entries())
      .map(([district, count]) => ({ district, count }));

    const touristComparison = Array.from(touristMap.entries()).map(([category, count]) => ({ category, count }));

    const coastalVsInland = Array.from(coastalMap.entries()).map(([category, count]) => ({ category, count }));

    const typeDistribution = Array.from(typeMap.entries()).map(([type, count]) => ({ type, count, percentage: '0' }));
    const totalTypes = typeDistribution.reduce((sum, item) => sum + item.count, 0);
    typeDistribution.forEach((item) => {
      item.percentage = totalTypes ? ((item.count / totalTypes) * 100).toFixed(1) : '0';
    });

    const sourceCategories = ['Fishing Waste', 'Tourist Waste', 'Industrial/Other'];
    const sourceBreakdown = sourceCategories.map((category) => ({
      category,
      count: sourceOriginMap.get(category) ?? 0,
      percentage: '0',
    }));
    const totalSource = sourceBreakdown.reduce((sum, entry) => sum + entry.count, 0);
    sourceBreakdown.forEach((entry) => {
      entry.percentage = totalSource ? ((entry.count / totalSource) * 100).toFixed(1) : '0';
    });

    const typeTrendSeries: Array<Record<string, number | string>> = [];
    temporal.forEach(({ date }) => {
      const row: Record<string, number | string> = { date };
      typeTrend.forEach((series, type) => {
        row[type] = series.get(date) || 0;
      });
      typeTrendSeries.push(row);
    });

    return {
      temporal,
      weekly,
      monthly,
      seasonal,
      hourlyHeatmap,
      dayOfWeek,
      yoy,
      spatial,
      districtBreakdown,
      touristComparison,
      coastalVsInland,
      typeDistribution,
      typeTrendSeries,
      sourceBreakdown,
    };
  };

  const heatmapMax = useMemo(() => Math.max(...(data?.hourlyHeatmap.map((item) => item.count) ?? [1])), [data]);

  if (loading) {
    return <div className="loading">Loading analytics...</div>;
  }

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  return (
    <div className="analytics">
      <div className="analytics-header">
        <div>
          <h1>Plastic Analytics Dashboard</h1>
          <p className="subtitle">Comprehensive data analysis and insights</p>
        </div>
        <div className="header-actions">
          <select 
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="time-range-select"
          >
            <option value="week">Last 7 Days</option>
            <option value="month">Last 30 Days</option>
            <option value="year">Last Year</option>
            <option value="all">All Time</option>
          </select>
        </div>
      </div>

      {/* Temporal Analysis */}
      <div className="analytics-section">
        <div className="section-header">
          <Calendar size={20} />
          <h2>Temporal Analysis</h2>
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <h3>Daily Detection Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={data?.temporal}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="date" 
                  stroke="#94a3b8"
                  style={{ fontSize: '0.75rem' }}
                />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip 
                  contentStyle={{ 
                    background: '#fff', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px'
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#3b82f6" 
                  fillOpacity={1}
                  fill="url(#colorCount)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Day of Week Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data?.dayOfWeek}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="day"
                  stroke="#94a3b8"
                  style={{ fontSize: '0.75rem' }}
                />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip 
                  contentStyle={{ 
                    background: '#fff', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="count" fill="#10b981" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <h3>Weekly Detections</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data?.weekly}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="week" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Bar dataKey="detections" fill="#60a5fa" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Monthly Detections</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data?.monthly}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="month" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Line type="monotone" dataKey="detections" stroke="#f97316" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Year-over-Year Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data?.yoy}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="year" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#22c55e" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Spatial Analysis */}
      <div className="analytics-section">
        <div className="section-header">
          <MapPin size={20} />
          <h2>Spatial Analysis</h2>
        </div>
        
        <div className="chart-card full-width">
          <h3>Top 10 Most Polluted Locations</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={data?.spatial} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
              <YAxis 
                type="category" 
                dataKey="location" 
                stroke="#94a3b8"
                style={{ fontSize: '0.75rem' }}
                width={120}
              />
              <Tooltip 
                contentStyle={{ 
                  background: '#fff', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="count" fill="#ef4444" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="location-stats">
          <div className="stat-box">
            <div className="stat-icon" style={{ background: '#fee2e2', color: '#dc2626' }}>
              <MapPin size={24} />
            </div>
            <div>
              <div className="stat-value">
                {data?.spatial[0]?.location || 'N/A'}
              </div>
              <div className="stat-label">Most Polluted Zone</div>
            </div>
          </div>
          <div className="stat-box">
            <div className="stat-icon" style={{ background: '#dbeafe', color: '#2563eb' }}>
              <TrendingUp size={24} />
            </div>
            <div>
              <div className="stat-value">
                {data?.spatial.length || 0}
              </div>
              <div className="stat-label">Active Monitoring Zones</div>
            </div>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <h3>District-wise Breakdown</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data?.districtBreakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="district" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Tourist vs Non-Tourist Areas</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data?.touristComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="category" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#f97316" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Plastic Type Distribution */}
      <div className="analytics-section">
        <div className="section-header">
          <Package size={20} />
          <h2>Plastic Type Distribution</h2>
        </div>
        
        <div className="charts-grid">
          <div className="chart-card">
            <h3>Type Breakdown</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={data?.typeDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry: any) => `${entry.type}: ${entry.percentage}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {data?.typeDistribution.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Most Common Items</h3>
            <div className="items-list">
              {data?.typeDistribution.map((item: any, index) => (
                <div key={index} className="item-row">
                  <div className="item-info">
                    <div 
                      className="item-color" 
                      style={{ background: COLORS[index % COLORS.length] }}
                    ></div>
                    <span className="item-name">{item.type}</span>
                  </div>
                  <div className="item-stats">
                    <span className="item-count">{item.count}</span>
                    <span className="item-percentage">{item.percentage}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="chart-card">
            <h3>Source Analysis</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={data?.sourceBreakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="category" stroke="#94a3b8" style={{ fontSize: '0.7rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.7rem' }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#14b8a6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="source-summary">
              {data?.sourceBreakdown.map((entry) => (
                <div key={entry.category} className="source-row">
                  <span>{entry.category}</span>
                  <strong>{entry.count}</strong>
                  <span>{entry.percentage}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="chart-card full-width">
          <h3>Type-specific Trends</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data?.typeTrendSeries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
              <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
              <Tooltip />
              {data?.typeDistribution.map((type, index) => (
                <Line
                  key={type.type}
                  type="monotone"
                  dataKey={type.type}
                  stroke={COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Seasonal & Hourly Analysis */}
      <div className="analytics-section">
        <div className="section-header">
          <Thermometer size={20} />
          <h2>Seasonal & Hourly Insights</h2>
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <h3>Seasonal Pattern</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data?.seasonal}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="season" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Peak Pollution Hours</h3>
            <div className="heatmap-grid">
              {data?.hourlyHeatmap.map((cell) => {
                const intensity = heatmapMax ? cell.count / heatmapMax : 0;
                const background = `rgba(59, 130, 246, ${Math.max(intensity * 0.85, 0.15)})`;
                return (
                  <div key={cell.hour} className="heatmap-cell" style={{ background }}>
                    <span>{cell.hour}</span>
                    <strong>{cell.count}</strong>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="chart-card full-width">
          <h3>Coastal vs Inland Pollution</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data?.coastalVsInland}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="category" stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
              <YAxis stroke="#94a3b8" style={{ fontSize: '0.75rem' }} />
              <Tooltip />
              <Bar dataKey="count" fill="#34d399" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
