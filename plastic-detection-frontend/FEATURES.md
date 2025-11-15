# ReWater - Plastic Detection Frontend Features

## Overview
A comprehensive light-themed dashboard for plastic detection and monitoring along the Konkan coastline.

## Implemented Features

### 1. Dashboard Home (`/dashboard`)
- **Primary Metrics Display**
  - Total plastic detections (lifetime count)
  - Active hotspots count
  - Plastic cleaned vs detected ratio
  - System uptime and health status
  - Last detection timestamp
  - Coverage area statistics (km²)

- **Real-Time Statistics Cards**
  - Today's detection count
  - This week's detection trend with percentage change
  - Average plastic density across monitored areas
  - Most polluted zone indicator
  - Active alerts count
  - Cleanup teams deployed count

- **Quick Actions & System Status**
  - Quick navigation to key features
  - Real-time system health monitoring
  - Service status indicators

### 2. Interactive Map Interface (`/map`)
- **Map Visualization**
  - Satellite/Street/Terrain view toggle
  - Full-screen map mode
  - Zoom controls and navigation
  - Konkan coastline focus

- **Heatmap Overlay**
  - Color-coded density visualization (green → yellow → red)
  - Adjustable opacity slider
  - Confidence-based coloring

- **Marker System**
  - Individual detection pins with color-coded confidence
  - Click to view detection details popup
  - GPS coordinates display
  - Timestamp information

- **Filters**
  - Date range filtering (Today, Last 7 Days, Last 30 Days, All Time)
  - Plastic type filtering
  - Real-time data updates

- **Export**
  - Export filtered data as CSV

### 3. Analytics Dashboard (`/analytics`)
- **Temporal Analysis**
  - Daily detection time-series graph
  - Weekly trend comparison
  - Day-of-week distribution chart

- **Spatial Analysis**
  - Top 10 most polluted locations (ranked list)
  - Horizontal bar chart visualization
  - Active monitoring zones count
  - Most polluted zone highlighting

- **Plastic Type Distribution**
  - Pie chart showing plastic categories detected
  - Most common plastic items list
  - Percentage breakdown
  - Color-coded categories

### 4. Upload Interface (`/upload`)
- **Multiple Input Sources**
  - Manual file upload (drag & drop)
  - Camera capture (live photo)
  - Drone footage support

- **GPS Coordinate Tagging**
  - Automatic location detection
  - Manual location refresh
  - Coordinates display

- **Metadata Recording**
  - Timestamp recording
  - Source type tracking
  - Image quality validation

- **Real-time Processing**
  - Instant prediction results
  - Confidence score display
  - Automatic database saving

- **Image Quality Guidelines**
  - Clear quality requirements
  - Resolution specifications
  - Best practices display

### 5. Detection Details (`/detection/:id`)
- **Individual Detection Information**
  - High-resolution detection image
  - GPS coordinates
  - Exact timestamp and date
  - Plastic type with confidence score
  - Confidence visualization bar
  - Plastic density metric
  - Status indicator

- **Detection Actions**
  - Mark for cleanup priority
  - Send alert to specific team
  - Flag false positives
  - Share detection (copy link)
  - Download detection report

- **Notes & Observations**
  - Add custom notes
  - Save observations
  - Edit detection information

- **Location Details**
  - Nearest landmark
  - District information
  - Zone type (tourist/non-tourist)
  - Distance from shore

### 6. Admin Panel (`/admin`)
- **System Health Monitoring**
  - API server status
  - ML models status
  - Database connection
  - Storage usage

- **Server Statistics**
  - CPU usage with progress bar
  - Memory usage
  - Disk space monitoring

- **System Actions**
  - Restart services
  - Export logs
  - Backup database
  - Clear cache

- **User Management**
  - User list with roles
  - Status indicators
  - Edit capabilities

- **Database Management**
  - Total records count
  - Database size
  - Last backup time
  - Connection pool status
  - Export/optimize/backup actions

- **Settings Configuration**
  - Detection confidence threshold
  - Auto-cleanup threshold
  - Alert email configuration
  - Email notifications toggle
  - Auto-backup settings

## Design Features

### Light Theme
- Clean, modern light color scheme
- High contrast for readability
- Professional blue primary color (#2563eb)
- Success green (#10b981)
- Warning orange (#f59e0b)
- Danger red (#ef4444)

### Responsive Design
- Mobile-friendly layouts
- Tablet optimization
- Desktop-first approach
- Collapsible sidebar
- Adaptive grids

### User Experience
- Smooth transitions and animations
- Hover effects on interactive elements
- Loading states
- Error handling
- Toast notifications
- Intuitive navigation

### Components
- Reusable card components
- Consistent button styles
- Badge system for status
- Progress bars
- Charts and graphs (Recharts)
- Interactive maps (Leaflet)

## Technology Stack
- **Frontend**: React 19 + TypeScript
- **Routing**: React Router DOM v7
- **Charts**: Recharts
- **Maps**: React Leaflet + Leaflet
- **Icons**: Lucide React
- **Styling**: Custom CSS with CSS Variables
- **Build Tool**: Vite
- **Date Handling**: date-fns

## API Integration
All components integrate with the Flask backend at `http://localhost:5000`:
- `/health` - Health check
- `/predict` - Image prediction
- `/save-detection` - Save detection data
- `/history` - Get detection history
- `/stats` - Dashboard statistics
- `/export-csv` - Export data

## Future Enhancements
- User authentication system
- Real-time WebSocket updates
- Advanced filtering options
- Custom report generation
- Team management features
- Notification system
- Mobile app version
- Offline mode support
