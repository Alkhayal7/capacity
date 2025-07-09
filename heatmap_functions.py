import pandas as pd
import folium
from folium.plugins import HeatMap, MeasureControl
import numpy as np
from math import radians, cos, sin, asin, sqrt, exp
from scipy.spatial import cKDTree
import os

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance calculation
    Can handle scalar to array or array to array comparisons
    """
    # Convert to radians - handle both scalar and array inputs
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)  
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in meters
    return 6371000 * c

def calculate_capacity_with_decay_vectorized(distances, capacity_min, capacity_max, max_range_meters, decay_strength=1.0):
    """
    Vectorized capacity calculation with exponential decay
    decay_strength: Controls how quickly capacity drops with distance 
                   (1.0 = normal, >1.0 = faster decay, <1.0 = slower decay)
    """
    # Set distances beyond max range to infinity to get 0 capacity
    distances = np.where(distances > max_range_meters, np.inf, distances)
    
    # Calculate base decay constant
    k_base = -np.log(capacity_min / capacity_max) / max_range_meters
    
    # Apply decay strength multiplier
    k = k_base * decay_strength
    
    # Calculate capacity with exponential decay
    capacities = capacity_max * np.exp(-k * distances)
    
    # Apply minimum capacity and zero out beyond range
    capacities = np.where(distances <= max_range_meters, 
                         np.maximum(capacities, capacity_min), 
                         0)
    
    return capacities

def apply_noise_vectorized(capacities, distances, noise_strength, noise_distance_factor):
    """
    Vectorized noise application
    """
    # Calculate noise factor based on distance
    noise_factors = noise_strength * (1 + distances * noise_distance_factor)
    noise_factors = np.minimum(noise_factors, 1.0)  # Cap at 1.0
    
    # Generate random noise reductions
    noise_reductions = np.random.random(len(capacities)) * noise_factors
    
    # Apply noise (always reduces capacity)
    return capacities * (1 - noise_reductions)

def degrees_to_meters_approx(lat):
    """
    Approximate conversion from degrees to meters at given latitude
    """
    # At equator: 1 degree ≈ 111,320 meters
    # Longitude varies with latitude: 1 degree longitude ≈ 111,320 * cos(latitude)
    lat_to_m = 111320
    lon_to_m = 111320 * np.cos(np.radians(lat))
    return lat_to_m, lon_to_m

def load_tower_data_from_upload(uploaded_data):
    """
    Load tower data from uploaded DataFrame
    """
    df = uploaded_data.copy()
    if 'power' in df.columns:
        df = df.drop('power', axis=1)
    return df

def generate_capacity_grid(df, capacity_min, capacity_max, max_range_meters, 
                          noise_strength, noise_distance_factor, grid_resolution,
                          decay_strength=1.0, progress_callback=None):
    """
    Generate the capacity grid for heatmap
    """
    # Calculate map boundaries
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    # Add padding
    lat_padding = (lat_max - lat_min) * 0.1
    lon_padding = (lon_max - lon_min) * 0.1
    
    lat_min -= lat_padding
    lat_max += lat_padding
    lon_min -= lon_padding
    lon_max += lon_padding
    
    # Create coordinate grids
    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
    
    # Create meshgrid for all combinations
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    grid_lats = lat_mesh.flatten()
    grid_lons = lon_mesh.flatten()
    
    if progress_callback:
        progress_callback(f"Processing {len(grid_lats):,} grid points...")
    
    # Convert tower coordinates to projected coordinates for spatial indexing
    center_lat = (lat_min + lat_max) / 2
    lat_to_m, lon_to_m = degrees_to_meters_approx(center_lat)
    
    # Convert towers to meter coordinates (approximate)
    tower_lats = df['latitude'].values
    tower_lons = df['longitude'].values
    tower_y = (tower_lats - center_lat) * lat_to_m
    tower_x = (tower_lons - (lon_min + lon_max) / 2) * lon_to_m
    
    # Convert grid to meter coordinates
    grid_y = (grid_lats - center_lat) * lat_to_m
    grid_x = (grid_lons - (lon_min + lon_max) / 2) * lon_to_m
    
    # Build spatial index for fast nearest neighbor queries
    if progress_callback:
        progress_callback("Building spatial index...")
    tower_coords = np.column_stack([tower_x, tower_y])
    tree = cKDTree(tower_coords)
    
    # Find all towers within range for each grid point
    if progress_callback:
        progress_callback("Finding nearby towers...")
    grid_coords = np.column_stack([grid_x, grid_y])
    
    # Query for all towers within MAX_RANGE_METERS
    nearby_indices = tree.query_ball_point(grid_coords, max_range_meters)
    
    # Calculate capacities efficiently
    if progress_callback:
        progress_callback("Calculating capacities...")
    max_capacities = np.zeros(len(grid_lats))
    
    for i, indices in enumerate(nearby_indices):
        if len(indices) > 0:
            # Get tower positions for this grid point
            nearby_tower_lats = tower_lats[indices]
            nearby_tower_lons = tower_lons[indices]
            
            # Calculate precise distances using haversine
            distances = haversine_distance_vectorized(
                grid_lats[i], grid_lons[i], 
                nearby_tower_lats, nearby_tower_lons
            )
            
            # Filter to only towers actually within range (spatial index is approximate)
            in_range = distances <= max_range_meters
            if np.any(in_range):
                distances = distances[in_range]
                
                # Calculate capacities with decay
                capacities = calculate_capacity_with_decay_vectorized(
                    distances, capacity_min, capacity_max, max_range_meters, decay_strength
                )
                
                # Apply noise
                noisy_capacities = apply_noise_vectorized(
                    capacities, distances, noise_strength, noise_distance_factor
                )
                
                # Take maximum capacity from all towers
                max_capacities[i] = np.max(noisy_capacities)
        
        # Progress update
        if progress_callback and (i + 1) % 50000 == 0:
            progress_callback(f"Processed {i+1:,}/{len(grid_lats):,} points...")
    
    # Filter out zero capacity points and create heatmap data
    if progress_callback:
        progress_callback("Filtering and preparing heatmap data...")
    non_zero_mask = max_capacities > 0
    heat_data = [[grid_lats[i], grid_lons[i], max_capacities[i]] 
                 for i in range(len(grid_lats)) if non_zero_mask[i]]
    
    return heat_data

def create_folium_map(df, heat_data, capacity_min, capacity_max, max_range_meters, 
                     noise_strength, heatmap_radius, heatmap_blur, decay_strength=1.0, 
                     heatmap_opacity=0.7, map_style='CartoDB voyager'):
    """
    Create the Folium map with heatmap and markers
    """
    # Create base map
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles=map_style)
    
    # Add heatmap layer
    if heat_data:
        capacities = [point[2] for point in heat_data]
        max_capacity_found = max(capacities)
        min_capacity_found = min(capacities)
        
        HeatMap(
            heat_data, 
            radius=heatmap_radius, 
            blur=heatmap_blur, 
            max_zoom=18,
            zoom=1,
            min_opacity=0.1,
            max_opacity=heatmap_opacity,
            gradient={0.0: 'blue', 0.3: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        # Add colorbar (moved to bottom left, under scale)
        colorbar_html = f'''
        <div style="position: fixed; 
                    bottom: 120px; left: 20px; width: 220px; height: 90px; 
                    background-color: rgba(255, 255, 255, 0.95);
                    border: 2px solid #333; z-index: 10000; 
                    font-size: 11px; border-radius: 8px;
                    padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px; color: #333;">Capacity (Mbps)</p>
        <div style="background: linear-gradient(to right, #0000ff, #00ffff, #ffff00, #ff8000, #ff0000); 
                    height: 20px; width: 196px; margin: 5px 0; border: 1px solid #333; border-radius: 3px;"></div>
        <div style="display: flex; justify-content: space-between; width: 196px; font-size: 10px; color: #333; font-weight: 500;">
            <span>{min_capacity_found:.0f}</span>
            <span>{(min_capacity_found + max_capacity_found) / 2:.0f}</span>
            <span>{max_capacity_found:.0f}</span>
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(colorbar_html))
    
    # Add tower markers with labels
    for index, row in df.iterrows():
        is_kitchen = row["original_name"].startswith("Kitchen")
        marker_color = 'white' if is_kitchen else 'blue'
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8,
            weight=2,
            popup=f"Tower: {row['unique_name']}<br>Base Capacity: {capacity_max}<br>Max Range: {max_range_meters}m",
            tooltip=row['unique_name']
        ).add_to(m)
        
        # Add permanent text label above the marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 8pt; font-weight: bold; color: {marker_color}; text-shadow: 1px 1px 1px rgba(0,0,0,0.8); text-align: center; margin-top: -25px;">{row["unique_name"]}</div>',
                icon_size=(50, 20),
                icon_anchor=(25, 10)
            )
        ).add_to(m)
    
    # Add legend (moved to bottom right)
    legend_html = '''
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 160px; height: 100px; 
                background-color: rgba(255, 255, 255, 0.95);
                border: 2px solid #333; z-index: 10000; 
                font-size: 12px; border-radius: 8px;
                padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 8px 0; font-weight: bold; color: #333;">Tower Legend</p>
    <p style="margin: 6px 0; color: #333;"><span style="color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">●</span> Kitchen Towers</p>
    <p style="margin: 6px 0; color: #333;"><span style="color: blue; font-weight: bold;">●</span> Operation Towers</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add scale control/ruler for distance measurement
    measure_control = MeasureControl(
        position='topright',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares'
    )
    measure_control.add_to(m)
    
    # Add dynamic scale bar that updates with zoom
    scale_html = '''
    <style>
    .dynamic-scale-bar {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #333;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 11px;
        font-weight: bold;
        color: #333;
        z-index: 10000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        min-width: 120px;
    }
    .dynamic-scale-line {
        height: 3px;
        background: #333;
        margin: 4px 0;
        position: relative;
        transition: width 0.3s ease;
    }
    .dynamic-scale-line::before, .dynamic-scale-line::after {
        content: '';
        position: absolute;
        width: 2px;
        height: 8px;
        background: #333;
        top: -2px;
    }
    .dynamic-scale-line::before { left: 0; }
    .dynamic-scale-line::after { right: 0; }
    </style>
    <div class="dynamic-scale-bar" id="scaleBar">
        <div id="scaleTitle">Scale</div>
        <div class="dynamic-scale-line" id="scaleLine"></div>
        <div id="scaleText" style="text-align: center; margin-top: 2px;">Calculating...</div>
    </div>
    
    <script>
    var globalMap = null;
    var scaleUpdateInterval = null;
    
    function findMap() {
        // Try multiple ways to find the map
        if (globalMap) return globalMap;
        
        // Method 1: Look for map in window object
        for (var key in window) {
            if (key.includes('map') && window[key] && typeof window[key].getZoom === 'function') {
                globalMap = window[key];
                return globalMap;
            }
        }
        
        // Method 2: Look in document for leaflet map
        var mapDivs = document.querySelectorAll('.folium-map');
        if (mapDivs.length > 0) {
            for (var i = 0; i < mapDivs.length; i++) {
                if (mapDivs[i]._leaflet_id) {
                    var mapId = mapDivs[i]._leaflet_id;
                    if (window.L && window.L._map && window.L._map[mapId]) {
                        globalMap = window.L._map[mapId];
                        return globalMap;
                    }
                }
            }
        }
        
        return null;
    }
    
    function updateScale() {
        var map = findMap();
        if (!map) {
            document.getElementById('scaleText').textContent = 'Map loading...';
            return;
        }
        
        try {
            var zoom = map.getZoom();
            var center = map.getCenter();
            
            // Calculate meters per pixel at current zoom and latitude
            var metersPerPixel = 40075016.686 * Math.abs(Math.cos(center.lat * Math.PI / 180)) / Math.pow(2, zoom + 8);
            
            // Define scale bar length in pixels (base length)
            var basePixels = 80;
            var scaleMeters = metersPerPixel * basePixels;
            
            // Round to nice numbers
            var niceScales = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
            var niceScale = niceScales.find(scale => scale >= scaleMeters) || niceScales[niceScales.length - 1];
            
            // Calculate actual pixel width for the nice scale
            var actualPixels = Math.round(niceScale / metersPerPixel);
            
            // Update the scale bar
            var scaleLine = document.getElementById('scaleLine');
            var scaleText = document.getElementById('scaleText');
            
            if (scaleLine && scaleText) {
                scaleLine.style.width = actualPixels + 'px';
                
                if (niceScale >= 1000) {
                    var km = niceScale / 1000;
                    scaleText.textContent = (km % 1 === 0 ? km.toFixed(0) : km.toFixed(1)) + ' km';
                } else {
                    scaleText.textContent = niceScale + ' m';
                }
            }
        } catch (e) {
            document.getElementById('scaleText').textContent = 'Scale error';
        }
    }
    
    function initializeScale() {
        var attempts = 0;
        var maxAttempts = 20;
        
        function tryUpdate() {
            attempts++;
            var map = findMap();
            
            if (map) {
                updateScale();
                map.on('zoomend', updateScale);
                map.on('moveend', updateScale);
                
                // Clear any existing interval
                if (scaleUpdateInterval) {
                    clearInterval(scaleUpdateInterval);
                }
            } else if (attempts < maxAttempts) {
                setTimeout(tryUpdate, 500);
            } else {
                document.getElementById('scaleText').textContent = 'Map unavailable';
            }
        }
        
        tryUpdate();
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeScale);
    } else {
        initializeScale();
    }
    
    // Also try after a delay for Streamlit
    setTimeout(initializeScale, 2000);
    </script>
    '''
    m.get_root().html.add_child(folium.Element(scale_html))
    
    return m

def generate_heatmap(capacity_min=20, capacity_max=190, max_range_meters=400, 
                    noise_strength=0.1, noise_distance_factor=0.001, 
                    grid_resolution=1000, heatmap_radius=8, heatmap_blur=1,
                    decay_strength=1.0, heatmap_opacity=0.7, progress_callback=None, 
                    uploaded_data=None, map_style='CartoDB voyager'):
    """
    Main function to generate the complete heatmap
    Can work with either a CSV file (csv_file parameter) or uploaded data (uploaded_data parameter)
    """
    # Load data - prioritize uploaded_data over csv_file
    if uploaded_data is not None:
        df = load_tower_data_from_upload(uploaded_data)
    else:
        raise ValueError("Either uploaded_data or csv_file must be provided")
    
    df['base_capacity'] = capacity_max
    
    # Generate heatmap data
    heat_data = generate_capacity_grid(
        df, capacity_min, capacity_max, max_range_meters,
        noise_strength, noise_distance_factor, grid_resolution,
        decay_strength, progress_callback
    )
    
    # Create map
    m = create_folium_map(
        df, heat_data, capacity_min, capacity_max, max_range_meters,
        noise_strength, heatmap_radius, heatmap_blur, decay_strength, 
        heatmap_opacity, map_style
    )
    
    return m, heat_data, df 