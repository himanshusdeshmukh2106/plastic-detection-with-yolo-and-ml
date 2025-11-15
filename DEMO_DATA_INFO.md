# Demo Data Information

## Overview
The backend now includes comprehensive placeholder data with **500+ detection records** across the entire Indian coastline for demonstration purposes.

## Coverage Area

### West Coast (Arabian Sea)
- **Mumbai Metropolitan Region**: Juhu Beach, Versova Beach
- **Raigad District**: Alibaug, Kashid, Murud
- **Ratnagiri District**: Ganpatipule, Ratnagiri Harbor, Bhatye Beach
- **Sindhudurg District**: Tarkarli, Vengurla, Malvan Port
- **Goa**: Baga, Candolim, Arambol, Vagator, Palolem, Agonda, Vasco Harbor
- **Karnataka**: Gokarna, Karwar, Malpe (Udupi)
- **Kerala**: Kovalam, Varkala, Kochi Harbor, Cherai, Alappuzha

### South Coast
- **Tamil Nadu**: Marina Beach (Chennai), Mahabalipuram, Kanyakumari, Rameswaram
- **Puducherry**: Pondicherry Beach

### East Coast (Bay of Bengal)
- **Andhra Pradesh**: Visakhapatnam, Kakinada Port
- **Odisha**: Puri, Chandrabhaga
- **West Bengal**: Digha, Mandarmani

## Data Characteristics

### Time Range
- **90 days of historical data** (rolling window)
- Recent data weighted higher (more detections in last 7 days)
- Distributed across all hours of the day

### Plastic Types
- Plastic (general)
- Bottles
- Bags
- Fishing nets
- Microplastics
- Packaging materials
- Straws
- Containers

### Detection Sources
- **Manual uploads**: Field workers and volunteers
- **Camera feeds**: Fixed coastal monitoring cameras
- **Drone footage**: Aerial surveys
- **Satellite imagery**: Remote sensing data

### Location Types
- **Tourist beaches**: High-traffic areas (Goa, Mumbai, Kerala)
- **Fishing harbors**: Commercial fishing ports
- **Coastal areas**: General coastline
- **Major ports**: Commercial shipping areas

### Detection Attributes
- **Confidence levels**: 75-98%
- **Status**: Pending (60%), In-Progress (30%), Completed (10%)
- **Severity**: Low, Medium, High
- **Estimated quantity**: 5-500 items per detection
- **Weather conditions**: Clear, Cloudy, Partly Cloudy, Rainy

## Geographic Distribution

Data is spread across:
- **37 distinct locations**
- **15 states/territories**
- **Real GPS coordinates** from actual Indian coastal locations
- **Coordinate variance** of ±1km to simulate detection spread

## Usage

This demo data automatically loads when the backend starts, providing:
- Realistic map visualizations with clustered markers
- Analytics charts showing temporal and spatial trends
- Hotspot identification based on detection density
- Source analysis (fishing vs tourist waste patterns)
- Collection channel status monitoring

## To Start Backend with Demo Data

```bash
cd Plastic-Detection-Model
python app_react_backend.py
```

You should see:
```
Loading demo data...
[SUCCESS] Loaded 500+ demo detection records
  Sources: Manual=XXX, Camera=XXX, Drone=XXX, Satellite=XXX
  Top Regions: North Goa=XX, Mumbai=XX, Kerala=XX...
```

## Note

All coordinates are real locations along the Indian coastline. Detection data is randomly generated but follows realistic patterns (higher concentrations near tourist areas and fishing ports, temporal variations, etc.).
