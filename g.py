import streamlit as st
import pandas as pd
import pydeck as pdk
import datetime
import plotly.express as px
import pickle # For loading the model
import xgboost as xgb
from geopy.distance import geodesic
import ast # For debugging purposes

# --- Configure Page to Wide Layout ---
st.set_page_config(layout="wide")

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_itms_data(file_path):
    """Loads and preprocesses the ITMS data."""
    df = pd.read_csv(file_path)

    # Clean 'location.coordinates' and convert to lat/lon
    def extract_coordinates(coord_str):
        try:
            coords = coord_str.strip('[]').split(', ')
            lon, lat = map(float, coords)
            return lat, lon  # Corrected order
        except (ValueError, AttributeError):
            return None, None

    df[['latitude', 'longitude']] = df['location.coordinates'].apply(extract_coordinates).apply(pd.Series)
    df = df.dropna(subset=['latitude', 'longitude'])

    # Convert 'observationDateTime' to datetime objects
    df['observationDateTime'] = pd.to_datetime(df['observationDateTime'], utc=True)
    df['observationDateTime'] = df['observationDateTime'].dt.tz_convert(None)  # Remove timezone awareness

    # Convert 'trip_delay' to numeric, coercing errors to NaN
    df['trip_delay'] = pd.to_numeric(df['trip_delay'], errors='coerce')

    return df

@st.cache_data
def load_shape_data(file_path):
    """Loads the shape data."""
    df = pd.read_csv(file_path)
    return df

# --- Load the Model ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model from a .pkl file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# --- Prediction Function ---
def predict_emissions(model, speed, trip_delay):
    """Predicts CO₂ emissions based on speed and trip delay."""
    trip_delay_hours = trip_delay / 60  # Convert delay to hours
    total_distance_traveled = speed * trip_delay_hours  # Calculate total distance
    fuel_consumption = total_distance_traveled / 3.5  # Estimate fuel consumption
    # co2_emissions = fuel_consumption * 2.31  # Basic calculation

    # Create a DataFrame with the correct feature names
    input_data = pd.DataFrame([[speed, trip_delay_hours, total_distance_traveled, fuel_consumption]],
                                columns=["speed", "trip_delay_hours", "total_distance_traveled", "fuel_consumption"])

    # Use the model to refine predictions
    prediction = model.predict(input_data)
    return prediction[0], fuel_consumption * 2.31 # Returning the model prediction and the basic calculation

# --- Main Application ---
def main():
    st.title("Public Transportation Dashboard")

    # Load data
    itms_df = load_itms_data("itms-22mar-2hours.csv")  # Replace with your file path
    shape_df = load_shape_data("konbert-output-3e875593.csv")  # Replace with your file path

    # Load the model
    model = load_model("co2.pkl")  # Replace with your .pkl file path

    # --- Sidebar for Route Selection ---
    st.sidebar.header("Route Selection")
    selected_route = st.sidebar.selectbox("Select a Route", itms_df['route_id'].unique())

    # Filter data based on selection
    filtered_itms_df = itms_df[itms_df['route_id'] == selected_route].copy()

    tab1, tab2, tab3 = st.tabs(["Overview", "Route Analysis", "Comparative Analysis"])

    with tab1:
        
        # --- Overview Metrics ---
        st.header(f"Overview for Route: {selected_route}")
        if not filtered_itms_df.empty:
            col1, col2, col3 ,col4 = st.columns(4)
            with col1:
                num_vehicles = filtered_itms_df['vehicle_label'].nunique()
                st.metric("Vehicles on Route", value=num_vehicles)
            with col2:
                avg_speed_route = filtered_itms_df['speed'].mean()
                st.metric("Average Speed (km/h)", value=f"{avg_speed_route:.2f}")
            with col3:
                avg_delay_route = filtered_itms_df['trip_delay'].mean()
                st.metric("Average Delay (sec)", value=f"{avg_delay_route:.2f}")
            with col4:
                prediction, co2_emissions = predict_emissions(model, avg_speed_route, avg_delay_route)
                st.metric("predicted_co2_emissions", value=f"{co2_emissions:.2f}")
        else:
            st.info(f"No data available for route: {selected_route}")

    with tab2:
        st.header(f"Detailed Analysis for Route: {selected_route}")
        # --- Heatmaps ---
        if not filtered_itms_df.empty:
            col_delay, col_speed = st.columns(2)
            with col_delay:
                st.subheader("Delay Heatmap")
                delay_by_location = filtered_itms_df.groupby(['latitude', 'longitude'])['trip_delay'].mean().reset_index()
                delay_by_location.rename(columns={'trip_delay': 'average_delay'}, inplace=True)

                heatmap_layer_delay = pdk.Layer(
                    "HeatmapLayer",
                    data=delay_by_location,
                    get_position=["longitude", "latitude"],
                    weights="average_delay",
                    radius=50,
                    intensity=1,
                    threshold=0.05,
                    opacity=0.8,
                    color_range=[
                        [255, 255, 204],  # Light Yellow
                        [255, 237, 160],
                        [254, 217, 118],
                        [254, 178, 76],
                        [253, 141, 60],
                        [252, 78, 42],
                        [227, 26, 28],    # Red
                    ],
                )

                # Route shape for delay map
                route_shape_delay = shape_df[shape_df['shape_id'] == selected_route]
                if not route_shape_delay.empty:
                    shape_layer_delay = pdk.Layer(
                        "LineLayer",
                        data=route_shape_delay,
                        get_position=["shape_pt_lon", "shape_pt_lat"],
                        get_color=[0, 0, 255],  # Blue for route shape
                        get_width=5,
                    )
                    view_state_delay = pdk.ViewState(
                        latitude=route_shape_delay['shape_pt_lat'].mean(),
                        longitude=route_shape_delay['shape_pt_lon'].mean(),
                        zoom=12,
                        pitch=50,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="mapbox://styles/mapbox/streets-v11",
                            initial_view_state=view_state_delay,
                            layers=[shape_layer_delay, heatmap_layer_delay] if shape_layer_delay else [heatmap_layer_delay],
                        ), height=400 # Adjust height as needed
                    )
                else:
                    view_state_delay = pdk.ViewState(
                        latitude=delay_by_location['latitude'].mean(),
                        longitude=delay_by_location['longitude'].mean(),
                        zoom=12,
                        pitch=50,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="mapbox://styles/mapbox/streets-v11",
                            initial_view_state=view_state_delay,
                            layers=[heatmap_layer_delay],
                        ), height=400 # Adjust height as needed
                    )

            with col_speed:
                st.subheader("Speed Heatmap")
                speed_by_location = filtered_itms_df.groupby(['latitude', 'longitude'])['speed'].mean().reset_index()
                speed_by_location.rename(columns={'speed': 'average_speed'}, inplace=True)

                heatmap_layer_speed = pdk.Layer(
                    "HeatmapLayer",
                    data=speed_by_location,
                    get_position=["longitude", "latitude"],
                    weights="average_speed",
                    radius=50,
                    intensity=1,
                    threshold=0.05,
                    opacity=0.8,
                    color_range=[
                        [204, 229, 255],  # Light Blue
                        [153, 204, 255],
                        [102, 178, 255],
                        [51, 153, 255],
                        [0, 128, 255],
                        [0, 102, 204],
                        [0, 76, 153],
                        [0, 51, 102],
                        [0, 25, 51],    # Dark Blue
                        [0, 153, 0],    # Green
                    ],
                )

                # Route shape for speed map
                route_shape_speed = shape_df[shape_df['shape_id'] == selected_route]
                if not route_shape_speed.empty:
                    shape_layer_speed = pdk.Layer(
                        "LineLayer",
                        data=route_shape_speed,
                        get_position=["shape_pt_lon", "shape_pt_lat"],
                        get_color=[0, 0, 255],  # Blue for route shape
                        get_width=5,
                    )
                    view_state_speed = pdk.ViewState(
                        latitude=route_shape_speed['shape_pt_lat'].mean(),
                        longitude=route_shape_speed['shape_pt_lon'].mean(),
                        zoom=12,
                        pitch=50,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="mapbox://styles/mapbox/streets-v11",
                            initial_view_state=view_state_speed,
                            layers=[shape_layer_speed, heatmap_layer_speed] if shape_layer_speed else [heatmap_layer_speed],
                        ), height=400 # Adjust height as needed
                    )
                else:
                    view_state_speed = pdk.ViewState(
                        latitude=speed_by_location['latitude'].mean(),
                        longitude=speed_by_location['longitude'].mean(),
                        zoom=12,
                        pitch=50,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="mapbox://styles/mapbox/streets-v11",
                            initial_view_state=view_state_speed,
                            layers=[heatmap_layer_speed],
                        ), height=400 # Adjust height as needed
                    )
        else:
            st.info(f"No heatmap data available for route: {selected_route}")

        # --- Route Specific Charts and Stats ---
        st.header("Route Details")
        if not filtered_itms_df.empty:
            col_dist, col_speed_dist = st.columns(2)
            with col_dist:
                st.subheader("Distance Traveled by Each Bus")
                distance_travelled = {}
                grouped_by_vehicle = filtered_itms_df.groupby('vehicle_label')

                for vehicle, group in grouped_by_vehicle:
                    group_sorted = group.sort_values(by='observationDateTime')
                    total_distance = 0.0
                    previous_location = None

                    for index, row in group_sorted.iterrows():
                        current_location = (row['latitude'], row['longitude'])
                        if previous_location:
                            distance = geodesic(previous_location, current_location).km
                            total_distance += distance
                        previous_location = current_location
                    distance_travelled[vehicle] = total_distance

                if distance_travelled:
                    distance_df = pd.DataFrame(list(distance_travelled.items()), columns=['Vehicle Label', 'Distance Traveled (km)'])
                    fig_distance = px.bar(distance_df, x='Vehicle Label', y='Distance Traveled (km)',
                                          title="Distance Traveled by Each Bus",
                                          labels={'Vehicle Label': 'Bus', 'Distance Traveled (km)': 'Distance (km)'},
                                          height=500) # Increased height
                    st.plotly_chart(fig_distance)
                else:
                    st.info("No vehicle data available to calculate distance.")

            with col_speed_dist:
                st.subheader("Speed Distribution")
                fig_speed_dist = px.histogram(filtered_itms_df, x='speed',
                                             title=f"Speed Distribution for Route {selected_route}",
                                             labels={'speed': 'Speed (km/h)'},
                                             height=500) # Increased height
                st.plotly_chart(fig_speed_dist)

            col_delay_stats, col_speed_stats = st.columns(2)
            with col_delay_stats:
                st.subheader("Delay Statistics")
                delay_stats = filtered_itms_df['trip_delay'].describe()
                st.dataframe(delay_stats)
            with col_speed_stats:
                st.subheader("Speed Statistics")
                speed_stats = filtered_itms_df['speed'].describe()
                st.dataframe(speed_stats)
        else:
            st.info(f"No detailed data available for route: {selected_route}")

    with tab3:
        st.header("Comparative Analysis")
        # --- Metrics and Emissions Across All Routes ---
        st.subheader("Overall Route Analysis")
        col_overall_metrics, col_emissions = st.columns(2)

        with col_overall_metrics:
            st.subheader("Average Delay and Speed per Route")
            avg_delay_per_route = itms_df.groupby('route_id')['trip_delay'].mean().reset_index()
            avg_delay_per_route.rename(columns={'trip_delay': 'average_delay'}, inplace=True)

            avg_speed_per_route = itms_df.groupby('route_id')['speed'].mean().reset_index()
            avg_speed_per_route.rename(columns={'speed': 'average_speed'}, inplace=True)

            # Merge average delay and speed
            merged_avg = pd.merge(avg_delay_per_route, avg_speed_per_route, on='route_id', suffixes=('_delay', '_speed'))

            fig_combined_avg = px.bar(merged_avg, x='route_id', y=['average_delay', 'average_speed'], barmode='group',
                                        title="Average Delay (seconds) and Speed (km/h) per Route",
                                        labels={'route_id': 'Route ID', 'value': 'Average Value', 'variable': 'Metric'},
                                        height=500) # Increased height
            st.plotly_chart(fig_combined_avg)

        with col_emissions:
            st.subheader("CO₂ Emission Estimation per Route")
            avg_delay_per_route_emission = itms_df.groupby('route_id')['trip_delay'].mean().reset_index()
            avg_delay_per_route_emission.rename(columns={'trip_delay': 'average_delay'}, inplace=True)

            avg_speed_per_route_emission = itms_df.groupby('route_id')['speed'].mean().reset_index()
            avg_speed_per_route_emission.rename(columns={'speed': 'average_speed'}, inplace=True)

            merged_avg_emission = pd.merge(avg_delay_per_route_emission, avg_speed_per_route_emission, on='route_id', suffixes=('_delay', '_speed'))

            if not merged_avg_emission.empty:
                emissions_data = []
                for index, row in merged_avg_emission.iterrows():
                    route_id = row['route_id']
                    avg_delay_seconds = row['average_delay']
                    avg_speed = row['average_speed']

                    # Convert delay to minutes
                    avg_delay_minutes = avg_delay_seconds / 60

                    # Predict emissions
                    if avg_speed > 0 and avg_delay_minutes >= 0:
                        prediction, co2_emissions = predict_emissions(model, avg_speed, avg_delay_minutes)
                        emissions_data.append({'route_id': route_id, 'predicted_co2_emissions': co2_emissions})
                    else:
                        emissions_data.append({'route_id': route_id, 'predicted_co2_emissions': None})

                if emissions_data:
                    emissions_df = pd.DataFrame(emissions_data)
                    fig_emissions = px.bar(emissions_df, x='route_id', y='predicted_co2_emissions',
                                             title="Predicted CO₂ Emissions per Route",
                                             labels={'route_id': 'Route ID', 'predicted_co2_emissions': 'Predicted CO₂ Emissions (kg)'},
                                             height=500) # Increased height
                    st.plotly_chart(fig_emissions)
                else:
                    st.info("Could not estimate CO₂ emissions for any route.")
            else:
                st.info("Average delay or speed data is not available to estimate CO₂ emissions.")

if __name__ == "__main__":
    main()