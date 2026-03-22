import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import aiohttp
import asyncio
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

st.set_page_config(
    page_title="Climate Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Climate Change Monitor")
st.markdown("---")

def generate_sample_data():
    seasonal_temperatures = {
        "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
        "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
        "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
        "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
        "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25}
    }
    
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    
    dates = pd.date_range(start="2010-01-01", periods=3650, freq="D")
    data = []
    
    for city in seasonal_temperatures.keys():
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({
                'city': city,
                'timestamp': date,
                'temperature': round(temperature, 2),
                'season': season
            })
    
    return pd.DataFrame(data)

with st.sidebar:
    st.header("Settings")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with historical data",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['timestamp'], encoding='utf-8')
            st.success(f"Loaded {len(df)} records")
        except:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['timestamp'], encoding='latin1')
                st.success(f"Loaded {len(df)} records")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Click button below to generate sample data")
                df = None
    
    if st.button("Generate Sample Data"):
        df = generate_sample_data()
        st.session_state['df'] = df
        st.success(f"Generated {len(df)} records for {len(df['city'].unique())} cities")
        st.rerun()
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        cities = sorted(df['city'].unique())
        selected_city = st.selectbox("Select city", cities)
        
        api_key = st.text_input(
            "OpenWeatherMap API Key",
            type="password"
        )
        
        st.markdown("---")
        st.subheader("Analysis Parameters")
        window_size = st.slider(
            "Rolling window size (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        sigma_multiplier = st.slider(
            "Sigma multiplier for anomaly detection",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.5
        )
    else:
        selected_city = None
        api_key = None
        window_size = 30
        sigma_multiplier = 2.0


def calculate_rolling_stats(df, window_size):
    df_grouped = df.copy()
    df_grouped['rolling_mean'] = df_grouped['temperature'].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    df_grouped['rolling_std'] = df_grouped['temperature'].rolling(
        window=window_size, center=True, min_periods=1
    ).std()
    return df_grouped


def detect_anomalies(df, sigma_multiplier):
    df['is_anomaly'] = (
            (df['temperature'] > df['rolling_mean'] + sigma_multiplier * df['rolling_std']) |
            (df['temperature'] < df['rolling_mean'] - sigma_multiplier * df['rolling_std'])
    )
    return df


def calculate_seasonal_stats(df):
    seasonal_stats = df.groupby('season').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    seasonal_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    return seasonal_stats


def process_city_parallel(city_data, window_size, sigma_multiplier):
    city_data = city_data.sort_values('timestamp')
    city_data = calculate_rolling_stats(city_data, window_size)
    city_data = detect_anomalies(city_data, sigma_multiplier)
    seasonal_stats = calculate_seasonal_stats(city_data)
    return city_data, seasonal_stats


def parallel_analysis(df, window_size, sigma_multiplier):
    cities = df['city'].unique()

    city_data_dict = {}
    for city in cities:
        city_data_dict[city] = df[df['city'] == city].copy()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for city, data in city_data_dict.items():
            future = executor.submit(
                process_city_parallel,
                data,
                window_size,
                sigma_multiplier
            )
            futures.append((city, future))

        results = {}
        for city, future in futures:
            results[city] = future.result()

    return results


def sequential_analysis(df, window_size, sigma_multiplier):
    results = {}
    cities = df['city'].unique()

    for city in cities:
        city_data = df[df['city'] == city].copy()
        city_data = city_data.sort_values('timestamp')
        city_data = calculate_rolling_stats(city_data, window_size)
        city_data = detect_anomalies(city_data, sigma_multiplier)
        seasonal_stats = calculate_seasonal_stats(city_data)
        results[city] = (city_data, seasonal_stats)

    return results


def get_current_temperature_sync(city, api_key):
    if not api_key:
        return None

    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.now()
            }
        else:
            st.error(f"API error: {response.json().get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


async def get_current_temperature_async(city, api_key):
    if not api_key:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': api_key,
                'units': 'metric',
                'lang': 'ru'
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'temperature': data['main']['temp'],
                        'feels_like': data['main']['feels_like'],
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'description': data['weather'][0]['description'],
                        'timestamp': datetime.now()
                    }
                else:
                    error_data = await response.json()
                    st.error(f"API error: {error_data.get('message', 'Unknown error')}")
                    return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


def check_current_temperature_anomaly(current_temp, seasonal_stats, current_season):
    if current_season not in seasonal_stats.index:
        return None

    mean_temp = seasonal_stats.loc[current_season, 'mean']
    std_temp = seasonal_stats.loc[current_season, 'std']

    is_anomaly = abs(current_temp - mean_temp) > 2 * std_temp

    return {
        'is_anomaly': is_anomaly,
        'mean': mean_temp,
        'std': std_temp,
        'deviation': current_temp - mean_temp
    }


def get_current_season():
    month = datetime.now().month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

        required_columns = ['city', 'timestamp', 'temperature']
        if not all(col in df.columns for col in required_columns):
            st.error("File must contain columns: city, timestamp, temperature")
            st.stop()

        if 'season' not in df.columns:
            month_to_season = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            }
            df['season'] = df['timestamp'].dt.month.map(month_to_season)

        city_data = df[df['city'] == selected_city].copy()
        city_data = city_data.sort_values('timestamp')

        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Overview",
            "Time Series Analysis",
            "Seasonal Analysis",
            "Current Weather"
        ])

        with tab1:
            st.header(f"Data Overview for {selected_city}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average Temperature",
                    f"{city_data['temperature'].mean():.1f}C",
                    delta=None
                )
            with col2:
                st.metric(
                    "Maximum Temperature",
                    f"{city_data['temperature'].max():.1f}C",
                    delta=None
                )
            with col3:
                st.metric(
                    "Minimum Temperature",
                    f"{city_data['temperature'].min():.1f}C",
                    delta=None
                )
            with col4:
                st.metric(
                    "Standard Deviation",
                    f"{city_data['temperature'].std():.1f}C",
                    delta=None
                )

            st.subheader("Temperature Distribution")
            fig_hist = px.histogram(
                city_data,
                x='temperature',
                nbins=50,
                title=f"Temperature Histogram in {selected_city}",
                color_discrete_sequence=['#00ff00']
            )
            fig_hist.add_vline(
                x=city_data['temperature'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {city_data['temperature'].mean():.1f}C"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if st.checkbox("Show performance comparison"):
                st.subheader("Sequential vs Parallel Analysis Comparison")

                start_time = time.time()
                seq_results = sequential_analysis(df, window_size, sigma_multiplier)
                seq_time = time.time() - start_time

                start_time = time.time()
                par_results = parallel_analysis(df, window_size, sigma_multiplier)
                par_time = time.time() - start_time

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Sequential analysis: {seq_time:.2f} seconds")
                with col2:
                    st.success(f"Parallel analysis: {par_time:.2f} seconds")

                speedup = seq_time / par_time
                st.metric("Speedup", f"{speedup:.2f}x", delta=None)

        with tab2:
            st.header(f"Time Series Analysis for {selected_city}")

            city_data_with_stats = calculate_rolling_stats(city_data, window_size)
            city_data_with_stats = detect_anomalies(city_data_with_stats, sigma_multiplier)

            fig_ts = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Temperature with Anomalies", "Rolling Mean and Standard Deviation"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )

            fig_ts.add_trace(
                go.Scatter(
                    x=city_data_with_stats['timestamp'],
                    y=city_data_with_stats['temperature'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )

            anomalies = city_data_with_stats[city_data_with_stats['is_anomaly']]
            if not anomalies.empty:
                fig_ts.add_trace(
                    go.Scatter(
                        x=anomalies['timestamp'],
                        y=anomalies['temperature'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='circle')
                    ),
                    row=1, col=1
                )

            fig_ts.add_trace(
                go.Scatter(
                    x=city_data_with_stats['timestamp'],
                    y=city_data_with_stats['rolling_mean'],
                    mode='lines',
                    name=f'Rolling Mean ({window_size} days)',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )

            fig_ts.add_trace(
                go.Scatter(
                    x=city_data_with_stats['timestamp'],
                    y=city_data_with_stats['rolling_mean'] + sigma_multiplier * city_data_with_stats['rolling_std'],
                    mode='lines',
                    name=f'Upper Bound (+{sigma_multiplier}σ)',
                    line=dict(color='orange', width=1, dash='dash')
                ),
                row=2, col=1
            )

            fig_ts.add_trace(
                go.Scatter(
                    x=city_data_with_stats['timestamp'],
                    y=city_data_with_stats['rolling_mean'] - sigma_multiplier * city_data_with_stats['rolling_std'],
                    mode='lines',
                    name=f'Lower Bound (-{sigma_multiplier}σ)',
                    line=dict(color='orange', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255, 165, 0, 0.1)'
                ),
                row=2, col=1
            )

            fig_ts.update_layout(height=800, showlegend=True)
            fig_ts.update_xaxes(title_text="Date", row=2, col=1)
            fig_ts.update_yaxes(title_text="Temperature (C)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Temperature (C)", row=2, col=1)

            st.plotly_chart(fig_ts, use_container_width=True)

            anomaly_count = anomalies.shape[0]
            total_days = city_data_with_stats.shape[0]
            anomaly_percentage = (anomaly_count / total_days) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Anomalies detected: {anomaly_count} out of {total_days} days ({anomaly_percentage:.2f}%)")
            with col2:
                st.success(f"Sigma multiplier: {sigma_multiplier}")

        with tab3:
            st.header(f"Seasonal Analysis for {selected_city}")

            seasonal_stats = calculate_seasonal_stats(city_data)

            st.subheader("Seasonal Statistics")
            st.dataframe(
                seasonal_stats.style.background_gradient(cmap='RdYlBu_r'),
                use_container_width=True
            )

            fig_seasonal = go.Figure()

            seasons_order = ['winter', 'spring', 'summer', 'autumn']
            seasons_names = {'winter': 'Winter', 'spring': 'Spring', 'summer': 'Summer', 'autumn': 'Autumn'}

            for season in seasons_order:
                season_data = city_data[city_data['season'] == season]
                fig_seasonal.add_trace(go.Violin(
                    y=season_data['temperature'],
                    x=[seasons_names[season]] * len(season_data),
                    name=seasons_names[season],
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightseagreen',
                    opacity=0.7
                ))

            fig_seasonal.update_layout(
                title=f"Temperature Distribution by Season in {selected_city}",
                yaxis_title="Temperature (C)",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig_seasonal, use_container_width=True)

            city_data['year'] = city_data['timestamp'].dt.year
            yearly_stats = city_data.groupby('year')['temperature'].agg(['mean', 'std', 'min', 'max']).reset_index()

            fig_trend = px.line(
                yearly_stats,
                x='year',
                y='mean',
                title=f"Annual Temperature Trend in {selected_city}",
                labels={'year': 'Year', 'mean': 'Average Temperature (C)'}
            )

            fig_trend.add_scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean'] + yearly_stats['std'],
                mode='lines',
                name='+1σ',
                line=dict(dash='dash', color='gray')
            )

            fig_trend.add_scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean'] - yearly_stats['std'],
                mode='lines',
                name='-1σ',
                line=dict(dash='dash', color='gray'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.2)'
            )

            st.plotly_chart(fig_trend, use_container_width=True)

        with tab4:
            st.header(f"Current Weather in {selected_city}")

            if not api_key:
                st.warning("Please enter OpenWeatherMap API key in the sidebar")
            else:
                st.subheader("Synchronous vs Asynchronous Method Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Synchronous Method")
                    start_time = time.time()
                    current_weather_sync = get_current_temperature_sync(selected_city, api_key)
                    sync_time = time.time() - start_time

                    if current_weather_sync:
                        st.success(f"Completed in {sync_time:.3f} sec")
                        st.metric("Temperature", f"{current_weather_sync['temperature']:.1f}C")
                        st.metric("Feels Like", f"{current_weather_sync['feels_like']:.1f}C")
                        st.metric("Humidity", f"{current_weather_sync['humidity']}%")
                        st.metric("Pressure", f"{current_weather_sync['pressure']} hPa")
                        st.write(f"**Description:** {current_weather_sync['description'].capitalize()}")
                    else:
                        st.error("Failed to get data")

                with col2:
                    st.markdown("### Asynchronous Method")


                    async def async_wrapper():
                        start_time = time.time()
                        result = await get_current_temperature_async(selected_city, api_key)
                        async_time = time.time() - start_time
                        return result, async_time


                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    current_weather_async, async_time = loop.run_until_complete(async_wrapper())
                    loop.close()

                    if current_weather_async:
                        st.success(f"Completed in {async_time:.3f} sec")
                        st.metric("Temperature", f"{current_weather_async['temperature']:.1f}C")
                        st.metric("Feels Like", f"{current_weather_async['feels_like']:.1f}C")
                        st.metric("Humidity", f"{current_weather_async['humidity']}%")
                        st.metric("Pressure", f"{current_weather_async['pressure']} hPa")
                        st.write(f"**Description:** {current_weather_async['description'].capitalize()}")
                    else:
                        st.error("Failed to get data")

                st.markdown("---")
                st.subheader("Method Comparison")

                if current_weather_sync and current_weather_async:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Synchronous request time", f"{sync_time:.3f} sec")
                    with col2:
                        st.metric("Asynchronous request time", f"{async_time:.3f} sec")

                    speedup = sync_time / async_time
                    st.metric("Speedup", f"{speedup:.2f}x", delta=None)

                st.markdown("---")
                st.subheader("Current Temperature Analysis")

                if current_weather_sync or current_weather_async:
                    current_temp = (current_weather_sync or current_weather_async)['temperature']
                    current_season = get_current_season()
                    current_season_name = {
                        'winter': 'winter',
                        'spring': 'spring',
                        'summer': 'summer',
                        'autumn': 'autumn'
                    }[current_season]

                    if current_season in seasonal_stats.index:
                        mean_temp = seasonal_stats.loc[current_season, 'mean']
                        std_temp = seasonal_stats.loc[current_season, 'std']
                        is_anomaly = abs(current_temp - mean_temp) > 2 * std_temp

                        st.write(f"**Current season:** {current_season_name.capitalize()}")
                        st.write(f"**Average temperature for {current_season_name}:** {mean_temp:.1f}C")
                        st.write(f"**Standard deviation:** {std_temp:.1f}C")
                        st.write(f"**Deviation from norm:** {current_temp - mean_temp:+.1f}C")

                        if is_anomaly:
                            st.error(
                                f"Current temperature ({current_temp:.1f}C) is ANOMALOUS for {current_season_name}!")
                        else:
                            st.success(
                                f"Current temperature ({current_temp:.1f}C) is within normal range for {current_season_name}")

                        season_data = city_data[city_data['season'] == current_season]

                        fig_current = go.Figure()

                        fig_current.add_trace(go.Violin(
                            y=season_data['temperature'],
                            name=f"Historical Data ({current_season_name})",
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor='lightblue',
                            opacity=0.7
                        ))

                        fig_current.add_trace(go.Scatter(
                            x=[f"Current Temperature"] * len(season_data),
                            y=[current_temp] * len(season_data),
                            mode='markers',
                            name=f"Current: {current_temp:.1f}C",
                            marker=dict(color='red', size=15, symbol='star')
                        ))

                        fig_current.update_layout(
                            title=f"Comparison of Current Temperature with Historical Data for {current_season_name}",
                            yaxis_title="Temperature (C)",
                            height=500
                        )

                        st.plotly_chart(fig_current, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

else:
    st.info("""
    Instructions:

    1. Upload CSV file with historical data
       File must contain columns: city, timestamp, temperature

    2. Enter OpenWeatherMap API key in the sidebar

    3. Select city from dropdown

    4. Configure analysis parameters:
       - Rolling window size
       - Sigma multiplier for anomaly detection

    5. Explore results in tabs:
       - Data Overview
       - Time Series Analysis
       - Seasonal Analysis
       - Current Weather
    """)

st.markdown("---")
st.markdown("Climate Change Monitor | Developed with Streamlit, Plotly and OpenWeatherMap API")
