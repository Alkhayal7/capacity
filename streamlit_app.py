import streamlit as st
import pandas as pd
import os
from streamlit_folium import st_folium
from heatmap_functions import generate_heatmap, load_tower_data_from_upload
import io

# Page configuration
st.set_page_config(
    page_title="Cell Tower Capacity Heatmap",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Cell Tower Capacity Heatmap Visualizer")

# Initialize session state
if 'map_generated' not in st.session_state:
    st.session_state.map_generated = False
if 'current_map' not in st.session_state:
    st.session_state.current_map = None
if 'heat_data' not in st.session_state:
    st.session_state.heat_data = None
if 'tower_data' not in st.session_state:
    st.session_state.tower_data = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# File upload section
st.header("📤 Upload Tower Data")


# File upload widget
uploaded_file = st.file_uploader(
    "Choose a CSV file with tower coordinates",
    type="csv",
    help="Upload a CSV file containing tower locations and details"
)

# Data validation and processing
if uploaded_file is not None:
    try:
        # Read uploaded file
        uploaded_df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['original_name', 'unique_name', 'type', 'longitude', 'latitude']
        missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Additional validation
        if uploaded_df.empty:
            st.error("❌ The uploaded file is empty")
            st.stop()
            
        if uploaded_df['longitude'].isna().any() or uploaded_df['latitude'].isna().any():
            st.error("❌ Some coordinates are missing (NaN values)")
            st.stop()
            
        if not uploaded_df['longitude'].dtype in ['float64', 'int64'] or not uploaded_df['latitude'].dtype in ['float64', 'int64']:
            st.error("❌ Longitude and latitude must be numeric values")
            st.stop()
        
        # Remove power column if it exists (as done in original)
        if 'power' in uploaded_df.columns:
            uploaded_df = uploaded_df.drop('power', axis=1)
        
        # Store validated data
        st.session_state.uploaded_data = uploaded_df
        
        st.success(f"✅ Successfully loaded {len(uploaded_df)} towers!")
        
        # Show data preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Data Preview")
            st.dataframe(uploaded_df.head(10), height=300)
        
        with col2:
            st.subheader("📈 Summary")
            if 'original_name' in uploaded_df.columns:
                kitchen_count = uploaded_df['original_name'].str.startswith('Kitchen').sum()
                other_count = len(uploaded_df) - kitchen_count
                st.metric("🏠 Kitchen Towers", kitchen_count)
                st.metric("📡 Operation Towers", other_count)
                st.metric("📍 Total Towers", len(uploaded_df))
            
            # Coordinate bounds
            st.write("**Coordinate Bounds:**")
            st.write(f"Lat: {uploaded_df['latitude'].min():.4f} to {uploaded_df['latitude'].max():.4f}")
            st.write(f"Lon: {uploaded_df['longitude'].min():.4f} to {uploaded_df['longitude'].max():.4f}")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()

# Only proceed if data is uploaded
if st.session_state.uploaded_data is None:
    st.info("👆 Please upload a CSV file with tower data to continue")
    st.stop()

# Auto-generate initial map with default parameters if not already generated
if not st.session_state.map_generated and st.session_state.uploaded_data is not None:
    try:
        with st.spinner("Loading initial heatmap..."):
            map_obj, heat_data, tower_data = generate_heatmap(
                capacity_min=20,
                capacity_max=190,
                max_range_meters=400,
                noise_strength=0.1,
                noise_distance_factor=0.001,
                grid_resolution=500,  # Good balance for initial load
                heatmap_radius=8,
                heatmap_blur=5,
                decay_strength=1.0,
                heatmap_opacity=0.7,
                uploaded_data=st.session_state.uploaded_data,
                map_style='CartoDB voyager'
            )
        
        st.session_state.current_map = map_obj
        st.session_state.heat_data = heat_data
        st.session_state.tower_data = tower_data
        st.session_state.map_generated = True
        st.session_state.last_grid_resolution = 500
    except Exception as e:
        st.error(f"Error loading initial map: {str(e)}")

# Sidebar with parameters
st.sidebar.header("📡 Heatmap Control Panel")

# Show data info in sidebar
if st.session_state.uploaded_data is not None:
    with st.sidebar.expander("📊 Data Info"):
        st.write(f"**Towers loaded:** {len(st.session_state.uploaded_data)}")
        if 'original_name' in st.session_state.uploaded_data.columns:
            kitchen_count = st.session_state.uploaded_data['original_name'].str.startswith('Kitchen').sum()
            other_count = len(st.session_state.uploaded_data) - kitchen_count
            st.write(f"🏠 Kitchen towers: {kitchen_count}")
            st.write(f"📡 Operation towers: {other_count}")

st.sidebar.divider()

# Generate button (moved to top for visibility)
generate_button = st.sidebar.button("🚀 Generate Heatmap", type="primary", use_container_width=True, key="generate_top")

# Download section (right under generate button)
if st.session_state.current_map is not None:
    # Generate HTML content
    import tempfile
    import os
    
    # Use a default filename or stored value
    current_resolution = getattr(st.session_state, 'last_grid_resolution', 500)
    output_file = f"capacity_heatmap_{current_resolution}x{current_resolution}.html"
    
    # Save to temporary file and read content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
        st.session_state.current_map.save(tmp_file.name)
        tmp_file.flush()
        
        # Read the content
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Clean up temp file
        os.unlink(tmp_file.name)
    
    # Single download button
    st.sidebar.download_button(
        label="📥 Download HTML Map",
        data=html_content,
        file_name=output_file,
        mime="text/html",
        use_container_width=True,
        help="Download the interactive map as an HTML file"
    )

st.sidebar.divider()

# Capacity parameters
st.sidebar.subheader("⚡ Capacity Settings")
capacity_min = st.sidebar.number_input(
    "Minimum Capacity (Mbps)", 
    min_value=1, max_value=100, value=20,
    help="Minimum capacity at maximum range"
)

capacity_max = st.sidebar.number_input(
    "Maximum Capacity (Mbps)", 
    min_value=50, max_value=500, value=190,
    help="Maximum capacity at tower location"
)

max_range_meters = st.sidebar.number_input(
    "Maximum Range (meters)", 
    min_value=10, max_value=1000, value=400, step=10,
    help="Maximum effective range of towers"
)

decay_strength = st.sidebar.number_input(
    "Decay Strength", 
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Controls how quickly capacity drops with distance (1.0 = normal, >1.0 = faster decay, <1.0 = slower decay)"
)

st.sidebar.divider()

# Noise parameters
st.sidebar.subheader("🌊 Noise Settings")
noise_strength = st.sidebar.slider(
    "Base Noise Strength", 
    min_value=0.0, max_value=1.0, value=0.1, step=0.01,
    help="Base noise strength (0-1, where 1 can reduce capacity to 0)"
)

noise_distance_factor = st.sidebar.number_input(
    "Distance Noise Factor", 
    min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f",
    help="How much noise increases with distance (per meter)"
)

st.sidebar.divider()

# Grid and visualization parameters
st.sidebar.subheader("🗺️ Visualization Settings")
grid_resolution = st.sidebar.number_input(
    "Grid Resolution", 
    min_value=50, max_value=5000, value=1000, step=50,
    help="Number of grid points per dimension (higher = more detailed but slower). Common values: 50, 100, 200, 500, 1000"
)

heatmap_radius = st.sidebar.number_input(
    "Heatmap Point Radius", 
    min_value=1, max_value=20, value=8,
    help="Visual radius for heatmap points"
)

heatmap_blur = st.sidebar.number_input(
    "Heatmap Blur", 
    min_value=1, max_value=20, value=5,
    help="Blur amount for heatmap"
)

heatmap_opacity = st.sidebar.slider(
    "Heatmap Opacity", 
    min_value=0.1, max_value=1.0, value=0.7, step=0.1,
    help="Controls how transparent the heatmap is (lower = more transparent, shows map better)"
)

st.sidebar.divider()

# Map style parameters
st.sidebar.subheader("🗺️ Map Style")
map_style = st.sidebar.selectbox(
    "Map Theme",
    options=[
        "CartoDB voyager",
        "CartoDB dark_matter",
        "OpenStreetMap", 
        "CartoDB positron",
        "Stamen Terrain",
        "Stamen Toner",
    ],
    index=0,  # Default to CartoDB voyager
    help="Choose the background map style"
)

# Process generate button click
if generate_button:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(message):
        status_text.text(f"🔄 {message}")
        # Update progress based on message content
        if "Processing" in message:
            progress_bar.progress(0.1)
        elif "Building spatial index" in message:
            progress_bar.progress(0.2)
        elif "Finding nearby towers" in message:
            progress_bar.progress(0.3)
        elif "Calculating capacities" in message:
            progress_bar.progress(0.4)
        elif "Processed" in message and "points..." in message:
            # Extract progress from processed points
            try:
                parts = message.split("/")
                if len(parts) > 1:
                    processed = int(parts[0].split()[-1].replace(",", ""))
                    total = int(parts[1].split()[0].replace(",", ""))
                    progress = 0.4 + 0.5 * (processed / total)
                    progress_bar.progress(min(progress, 0.9))
            except:
                pass
        elif "Filtering" in message:
            progress_bar.progress(0.95)
    
    try:
        # Generate the heatmap with current parameters
        with st.spinner("Generating heatmap... This may take a while for high resolutions."):
            map_obj, heat_data, tower_data = generate_heatmap(
                capacity_min=capacity_min,
                capacity_max=capacity_max,
                max_range_meters=max_range_meters,
                noise_strength=noise_strength,
                noise_distance_factor=noise_distance_factor,
                grid_resolution=grid_resolution,
                heatmap_radius=heatmap_radius,
                heatmap_blur=heatmap_blur,
                decay_strength=decay_strength,
                heatmap_opacity=heatmap_opacity,
                progress_callback=progress_callback,
                uploaded_data=st.session_state.uploaded_data,
                map_style=map_style
            )
        
        # Update session state with new map
        st.session_state.current_map = map_obj
        st.session_state.heat_data = heat_data
        st.session_state.tower_data = tower_data
        st.session_state.map_generated = True
        st.session_state.last_grid_resolution = grid_resolution
        
        progress_bar.progress(1.0)
        status_text.text("✅ Heatmap updated successfully!")
        
        # Auto-refresh the page to show the new map
        st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error generating heatmap: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# Main content area - Map display
if st.session_state.current_map is not None:
    # Map display with maximum height
    map_data = st_folium(
        st.session_state.current_map, 
        width=None, 
        height=1000,
        returned_objects=["last_object_clicked"]
    )
    
    # Show clicked tower information at the bottom
    if map_data['last_object_clicked'] is not None:
        clicked_data = map_data['last_object_clicked']
        if 'popup' in clicked_data:
            st.info(f"🎯 Selected: {clicked_data['popup']}")
else:
    st.error("Map failed to load. Please check the data file and try again.") 