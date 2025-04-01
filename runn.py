import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime

# Load data with proper type conversion
@st.cache_data
def load_data():
    # Load trip data
    trip_data = pd.read_csv('itms-22mar-2hours.csv')
    
    # Convert trip_delay to numeric, coerce errors to NaN
    trip_data['trip_delay'] = pd.to_numeric(trip_data['trip_delay'], errors='coerce')
    
    # Convert speed to numeric, coerce errors to NaN
    trip_data['speed'] = pd.to_numeric(trip_data['speed'], errors='coerce')
    
    # Process coordinates
    trip_data['lon'] = trip_data['location.coordinates'].str.extract(r'\[([-0-9.]+),')
    trip_data['lat'] = trip_data['location.coordinates'].str.extract(r', ([-0-9.]+)\]')
    trip_data['lon'] = trip_data['lon'].astype(float)
    trip_data['lat'] = trip_data['lat'].astype(float)
    
    # Convert datetime columns
    trip_data['observationDateTime'] = pd.to_datetime(trip_data['observationDateTime'])
    
    return trip_data

trip_data = load_data()

# Dashboard title
st.title('Surat Public Transport Analytics Dashboard')

# Sidebar filters
st.sidebar.header('Filters')
selected_routes = st.sidebar.multiselect(
    'Select Routes', 
    options=trip_data['route_id'].unique(),
    default=trip_data['route_id'].unique()[:3]
)

# Fixing time slider for correct time range
min_time = trip_data['observationDateTime'].min().to_pydatetime()
max_time = trip_data['observationDateTime'].max().to_pydatetime()

time_range = st.sidebar.slider(
    'Select Time Range',
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time),
    format="%Y-%m-%d %H:%M:%S"
)

# Filter data based on selections
filtered_trip = trip_data[
    (trip_data['route_id'].isin(selected_routes)) &
    (trip_data['observationDateTime'] >= time_range[0]) &
    (trip_data['observationDateTime'] <= time_range[1])
].copy()

# Tabs for different views
tab1, tab2, tab3 = st.tabs([
    "Route Maps", 
    "Delay & Speed Analysis", 
    "Vehicle Tracking"
])

with tab1:
    st.header("Route Visualization")

    # Plot current vehicle positions
    st.subheader("Current Vehicle Positions")
    latest_data = filtered_trip.sort_values('observationDateTime').groupby('vehicle_label').last().reset_index()

    view_state = pdk.ViewState(
        latitude=latest_data['lat'].mean(),
        longitude=latest_data['lon'].mean(),
        zoom=11
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        latest_data,
        get_position=["lon", "lat"],
        get_color="[200, 30, 0, 160]",
        get_radius=100,
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[layer]
    ))

    # Heatmap for Delay Areas
    st.subheader("Heatmap of Delay Areas")
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        filtered_trip,
        get_position=["lon", "lat"],
        get_weight="trip_delay",
        radius_pixels=60,
        intensity=1
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view_state,
        layers=[heatmap_layer]
    ))

with tab2:
    st.header("Delay & Speed Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Delay by Route")
        avg_delay = filtered_trip.groupby('route_id')['trip_delay'].mean().reset_index()
        fig = px.bar(
            avg_delay, 
            x='route_id', 
            y='trip_delay',
            labels={'trip_delay': 'Average Delay (seconds)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Speed by Route")
        avg_speed = filtered_trip.groupby('route_id')['speed'].mean().reset_index()
        fig = px.bar(
            avg_speed, 
            x='route_id', 
            y='speed',
            labels={'speed': 'Average Speed (km/h)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Peak Hour Traffic Analysis")
    numeric_cols = ['trip_delay', 'speed']
    peak_data = (
        filtered_trip
        .set_index('observationDateTime')[numeric_cols]
        .resample('10T')
        .mean()
        .reset_index()
    )
    fig = px.line(
        peak_data, 
        x='observationDateTime', 
        y=['trip_delay', 'speed'],
        labels={'value': 'Metric Value', 'observationDateTime': 'Time'},
        title="Peak Traffic Patterns"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Vehicle Tracking")
    selected_vehicle = st.selectbox(
        'Select Vehicle',
        options=trip_data['vehicle_label'].unique()
    )
    vehicle_data = filtered_trip[filtered_trip['vehicle_label'] == selected_vehicle]

    if not vehicle_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Speed", f"{vehicle_data['speed'].iloc[-1]:.1f} km/h")

        with col2:
            st.metric("Current Delay", f"{vehicle_data['trip_delay'].iloc[-1]:.1f} seconds")

        st.subheader("Speed and Delay Over Time")
        fig = px.line(
            vehicle_data,
            x='observationDateTime',
            y=['speed', 'trip_delay'],
            labels={'value': 'Metric Value'},
            title=f"Vehicle {selected_vehicle} Performance"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected vehicle")
