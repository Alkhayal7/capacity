import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np
from math import radians, cos, sin, asin, sqrt, exp
from scipy.spatial import cKDTree

# --- CONFIGURABLE PARAMETERS ---
# Capacity parameters
CAPACITY_MIN = 20           # Minimum capacity at maximum range
CAPACITY_MAX = 190          # Maximum capacity at tower location
MAX_RANGE_METERS = 400       # Maximum effective range in meters

# Noise parameters
NOISE_STRENGTH = 0.1        # Base noise strength (0-1, where 1 = can reduce capacity to 0)
NOISE_DISTANCE_FACTOR = 0.001 # How much noise increases with distance (per meter)

# Decay parameters
DECAY_STRENGTH = 1.0        # Controls how quickly capacity drops with distance (1.0 = normal)

# Heatmap grid parameters
GRID_RESOLUTION = 1000       # Number of points per dimension for heatmap grid
HEATMAP_RADIUS = 8          # Visual radius for heatmap points
HEATMAP_BLUR = 5            # Blur amount for heatmap
HEATMAP_OPACITY = 0.7       # Heatmap transparency (0.1-1.0, lower = more transparent)

# --- FAST UTILITY FUNCTIONS ---
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

def calculate_capacity_with_decay_vectorized(distances):
    """
    Vectorized capacity calculation with exponential decay
    """
    # Set distances beyond max range to infinity to get 0 capacity
    distances = np.where(distances > MAX_RANGE_METERS, np.inf, distances)
    
    # Calculate decay constant
    k = -np.log(CAPACITY_MIN / CAPACITY_MAX) / MAX_RANGE_METERS
    
    # Calculate capacity with exponential decay
    capacities = CAPACITY_MAX * np.exp(-k * distances)
    
    # Apply minimum capacity and zero out beyond range
    capacities = np.where(distances <= MAX_RANGE_METERS, 
                         np.maximum(capacities, CAPACITY_MIN), 
                         0)
    
    return capacities

def apply_noise_vectorized(capacities, distances):
    """
    Vectorized noise application
    """
    # Calculate noise factor based on distance
    noise_factors = NOISE_STRENGTH * (1 + distances * NOISE_DISTANCE_FACTOR)
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

# --- 1. Load Your Data ---
# try:
#     df = pd.read_csv('coordinates_detailed.csv')
# except FileNotFoundError:
#     print("Error: 'coordinates_detailed.csv' not found. Make sure the file is in the same directory as the script.")
#     exit()

# print(f"Loaded {len(df)} cell towers")

# --- 2. Prepare the Capacity Data ---
if 'power' in df.columns:
    df = df.drop('power', axis=1)

df['base_capacity'] = CAPACITY_MAX

# --- 3. Create Grid for Heatmap (OPTIMIZED) ---
print("Generating optimized capacity grid...")

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
lat_grid = np.linspace(lat_min, lat_max, GRID_RESOLUTION)
lon_grid = np.linspace(lon_min, lon_max, GRID_RESOLUTION)

# Create meshgrid for all combinations
lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
grid_lats = lat_mesh.flatten()
grid_lons = lon_mesh.flatten()

print(f"Processing {len(grid_lats):,} grid points...")

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
print("Building spatial index...")
tower_coords = np.column_stack([tower_x, tower_y])
tree = cKDTree(tower_coords)

# Find all towers within range for each grid point
print("Finding nearby towers...")
grid_coords = np.column_stack([grid_x, grid_y])

# Query for all towers within MAX_RANGE_METERS
nearby_indices = tree.query_ball_point(grid_coords, MAX_RANGE_METERS)

# Calculate capacities efficiently
print("Calculating capacities...")
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
        in_range = distances <= MAX_RANGE_METERS
        if np.any(in_range):
            distances = distances[in_range]
            
            # Calculate capacities with decay
            capacities = calculate_capacity_with_decay_vectorized(distances, CAPACITY_MIN, CAPACITY_MAX, MAX_RANGE_METERS, DECAY_STRENGTH)
            
            # Apply noise
            noisy_capacities = apply_noise_vectorized(capacities, distances)
            
            # Take maximum capacity from all towers
            max_capacities[i] = np.max(noisy_capacities)
    
    # Progress update
    if (i + 1) % 50000 == 0:
        print(f"Processed {i+1:,}/{len(grid_lats):,} points...")

# Filter out zero capacity points and create heatmap data
print("Filtering and preparing heatmap data...")
non_zero_mask = max_capacities > 0
heat_data = [[grid_lats[i], grid_lons[i], max_capacities[i]] 
             for i in range(len(grid_lats)) if non_zero_mask[i]]

print(f"Generated {len(heat_data):,} heatmap points with non-zero capacity")

# --- 4. Create the Base Map ---
map_center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=12, tiles='CartoDB dark_matter')

# --- 5. Add the Heatmap Layer ---
if heat_data:
    capacities = [point[2] for point in heat_data]
    max_capacity_found = max(capacities)
    min_capacity_found = min(capacities)
    
    print(f"Capacity range in heatmap: {min_capacity_found:.1f} to {max_capacity_found:.1f}")
    
    HeatMap(
        heat_data, 
        radius=HEATMAP_RADIUS, 
        blur=HEATMAP_BLUR, 
        max_zoom=18,  # Allow heatmap to stay visible at higher zoom levels
        zoom=1,       # Keep heatmap size consistent across zoom levels
        min_opacity=0.1,
        max_opacity=HEATMAP_OPACITY,
        gradient={0.0: 'blue', 0.3: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
    ).add_to(m)
else:
    print("Warning: No heatmap data generated!")

# --- 6. Add Circle Markers for Each Tower ---
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
        popup=f"Tower: {row['unique_name']}<br>Base Capacity: {row['base_capacity']}<br>Max Range: {MAX_RANGE_METERS}m",
        tooltip=row['unique_name']  # Show name on hover
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

# Add range circles for towers (optional - shows coverage area)
# for index, row in df.iterrows():
#     folium.Circle(
#         location=[row['latitude'], row['longitude']],
#         radius=MAX_RANGE_METERS,
#         color='cyan',
#         fill=False,
#         opacity=0.3,
#         weight=1
#     ).add_to(m)

# --- 7. Add Legend and Colorbar ---
# Create legend for tower types
legend_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; width: 150px; height: 90px; 
            background-color: rgba(255, 255, 255, 0.9);
            border:2px solid grey; z-index:9999; 
            font-size:12px; border-radius: 5px;
            padding: 10px;">
<p style="margin: 0; font-weight: bold;">Tower Legend</p>
<p style="margin: 5px 0;"><span style="color: white; font-weight: bold; text-shadow: 1px 1px 1px black;">●</span> Kitchen Towers</p>
<p style="margin: 5px 0;"><span style="color: blue; font-weight: bold;">●</span> Operation Towers</p>
</div>
'''

# Create colorbar for capacity heatmap
if heat_data:
    colorbar_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 100px; 
                background-color: rgba(255, 255, 255, 0.9);
                border:2px solid grey; z-index:9999; 
                font-size:10px; border-radius: 5px;
                padding: 10px;">
    <p style="margin: 0; font-weight: bold; font-size: 12px;">Capacity (Mbps)</p>
    <div style="background: linear-gradient(to right, blue, cyan, yellow, orange, red); 
                height: 20px; width: 180px; margin: 5px 0; border: 1px solid black;"></div>
    <div style="display: flex; justify-content: space-between; width: 180px; font-size: 9px;">
        <span>{min_capacity_found:.0f}</span>
        <span>{(min_capacity_found + max_capacity_found) / 2:.0f}</span>
        <span>{max_capacity_found:.0f}</span>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(colorbar_html))

m.get_root().html.add_child(folium.Element(legend_html))

# --- 8. Save the Map ---
output_file = 'tower_capacity_heatmap.html'
m.save(output_file)

print(f"\nSuccess! Your interactive capacity heatmap has been saved to '{output_file}'")
print(f"Configuration used:")
print(f"  - Capacity range: {CAPACITY_MIN} to {CAPACITY_MAX}")
print(f"  - Max range: {MAX_RANGE_METERS} meters")
print(f"  - Decay strength: {DECAY_STRENGTH}")
print(f"  - Noise strength: {NOISE_STRENGTH}")
print(f"  - Noise distance factor: {NOISE_DISTANCE_FACTOR}")
print(f"  - Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
print(f"  - Heatmap opacity: {HEATMAP_OPACITY}")
print(f"  - Generated {len(heat_data):,} heatmap points")
print(f"  - Tower types: Kitchen (white) and Other (blue)")