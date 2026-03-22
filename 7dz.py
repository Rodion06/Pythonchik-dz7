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
        "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
        "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
        "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
        "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15}
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

with st.sidebar:
    st.header("Settings")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with historical data",
        type=['csv']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Sample Data", use_container_width=True):
            df = generate_sample_data()
            st.session_state['df'] = df
            st.session_state['data_source'] = 'generated'
            st.success(f"Generated {len(df)} records for {len(df['city'].unique())} cities")
            st.rerun()
    
    with col2:
        if 'df' in st.session_state and st.button("Clear Data", use_container_width=True):
            del st.session_state['df']
            if 'data_source' in st.session_state:
                del st.session_state['data_source']
            st.rerun()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
            if 'season' not in df.columns:
                month_to_season = {
                    12: 'winter', 1: 'winter', 2: 'winter',
                    3: 'spring', 4: 'spring', 5: 'spring',
                    6: 'summer', 7: 'summer', 8: 'summer',
                    9: 'autumn', 10: 'autumn', 11: 'autumn'
                }
                df['season'] = df['timestamp'].dt.month.map(month_to_season)
            st.session_state['df'] = df
            st.session_state['data_source'] = 'uploaded'
            st.success(f"Loaded {len(df)} records")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        cities = sorted(df['city'].unique())
        selected_city = st.selectbox("Select city", cities)
        
        api_key = st.text_input(
            "OpenWeatherMap API Key",
            type="password",
            value=""
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

if 'df' in st.session_state:
    df = st.session_state['df']
    
    if selected_city:
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
                st.metric("Average Temperature", f"{city_data['temperature'].mean():.1f}C")
            with col2:
                st.metric("Maximum Temperature", f"{city_data['temperature'].max():.1f}C")
            with col3:
                st.metric("Minimum Temperature", f"{city_data['temperature'].min():.1f}C")
            with col4:
                st.metric("Standard Deviation", f"{city_data['temperature'].std():.1f}C")
            
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
                st.metric("Speedup", f"{speedup:.2f}x")
        
        with tab2:
            st.header(f"Time Series Analysis for {selected_city}")
            
            city_data_with_stats = calculate_rolling_stats(city_data, window_size)
            city_data_with_stats = detect_anomalies(city_data_with_stats, sigma_multiplier)
            
            fig_ts = make_subplots(rows=2, cols=1,
                subplot_titles=("Temperature with Anomalies", "Rolling Mean and Standard Deviation"),
                vertical_spacing=0.15, row_heights=[0.6, 0.4])
            
            fig_ts.add_trace(go.Scatter(x=city_data_with_stats['timestamp'],
                y=city_data_with_stats['temperature'], mode='lines',
                name='Temperature', line=dict(color='blue', width=1), opacity=0.7), row=1, col=1)
            
            anomalies = city_data_with_stats[city_data_with_stats['is_anomaly']]
            if not anomalies.empty:
                fig_ts.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['temperature'],
                    mode='markers', name='Anomalies', marker=dict(color='red', size=8)), row=1, col=1)
            
            fig_ts.add_trace(go.Scatter(x=city_data_with_stats['timestamp'],
                y=city_data_with_stats['rolling_mean'], mode='lines',
                name=f'Rolling Mean ({window_size} days)', line=dict(color='green', width=2)), row=2, col=1)
            
            fig_ts.add_trace(go.Scatter(x=city_data_with_stats['timestamp'],
                y=city_data_with_stats['rolling_mean'] + sigma_multiplier * city_data_with_stats['rolling_std'],
                mode='lines', name=f'Upper Bound (+{sigma_multiplier}σ)',
                line=dict(color='orange', width=1, dash='dash')), row=2, col=1)
            
            fig_ts.add_trace(go.Scatter(x=city_data_with_stats['timestamp'],
                y=city_data_with_stats['rolling_mean'] - sigma_multiplier * city_data_with_stats['rolling_std'],
                mode='lines', name=f'Lower Bound (-{sigma_multiplier}σ)',
                line=dict(color='orange', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(255, 165, 0, 0.1)'), row=2, col=1)
            
            fig_ts.update_layout(height=800, showlegend=True)
            fig_ts.update_xaxes(title_text="Date", row=2, col=1)
            fig_ts.update_yaxes(title_text="Temperature (C)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Temperature (C)", row=2, col=1)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            anomaly_count = anomalies.shape[0]
            total_days = city_data_with_stats.shape[0]
            st.info(f"Anomalies detected: {anomaly_count} out of {total_days} days ({anomaly_count/total_days*100:.2f}%)")
        
        with tab3:
            st.header(f"Seasonal Analysis for {selected_city}")
            seasonal_stats = calculate_seasonal_stats(city_data)
            st.subheader("Seasonal Statistics")
            st.dataframe(seasonal_stats.style.background_gradient(cmap='RdYlBu_r'), use_container_width=True)
            
            fig_seasonal = go.Figure()
            seasons_order = ['winter', 'spring', 'summer', 'autumn']
            seasons_names = {'winter': 'Winter', 'spring': 'Spring', 'summer': 'Summer', 'autumn': 'Autumn'}
            
            for season in seasons_order:
                season_data = city_data[city_data['season'] == season]
                if len(season_data) > 0:
                    fig_seasonal.add_trace(go.Violin(y=season_data['temperature'],
                        x=[seasons_names[season]] * len(season_data), name=seasons_names[season],
                        box_visible=True, meanline_visible=True, fillcolor='lightseagreen', opacity=0.7))
            
            fig_seasonal.update_layout(title=f"Temperature Distribution by Season in {selected_city}",
                yaxis_title="Temperature (C)", showlegend=False, height=500)
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            city_data['year'] = city_data['timestamp'].dt.year
            yearly_stats = city_data.groupby('year')['temperature'].agg(['mean', 'std']).reset_index()
            fig_trend = px.line(yearly_stats, x='year', y='mean',
                title=f"Annual Temperature Trend in {selected_city}",
                labels={'year': 'Year', 'mean': 'Average Temperature (C)'})
            fig_trend.add_scatter(x=yearly_stats['year'], y=yearly_stats['mean'] + yearly_stats['std'],
                mode='lines', name='+1σ', line=dict(dash='dash', color='gray'))
            fig_trend.add_scatter(x=yearly_stats['year'], y=yearly_stats['mean'] - yearly_stats['std'],
                mode='lines', name='-1σ', line=dict(dash='dash', color='gray'),
                fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)')
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
                
                if current_weather_sync and current_weather_async:
                    st.markdown("---")
                    st.subheader("Method Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Synchronous time", f"{sync_time:.3f} sec")
                    with col2:
                        st.metric("Asynchronous time", f"{async_time:.3f} sec")
                    st.metric("Speedup", f"{sync_time/async_time:.2f}x")
                
                st.markdown("---")
                st.subheader("Current Temperature Analysis")
                current_weather = current_weather_sync or current_weather_async
                if current_weather:
                    current_temp = current_weather['temperature']
                    current_season = get_current_season()
                    current_season_name = {'winter': 'Winter', 'spring': 'Spring', 'summer': 'Summer', 'autumn': 'Autumn'}[current_season]
                    
                    seasonal_stats = calculate_seasonal_stats(city_data)
                    if current_season in seasonal_stats.index:
                        mean_temp = seasonal_stats.loc[current_season, 'mean']
                        std_temp = seasonal_stats.loc[current_season, 'std']
                        is_anomaly = abs(current_temp - mean_temp) > 2 * std_temp
                        
                        st.write(f"**Current season:** {current_season_name}")
                        st.write(f"**Historical average:** {mean_temp:.1f}C")
                        st.write(f"**Deviation:** {current_temp - mean_temp:+.1f}C")
                        
                        if is_anomaly:
                            st.error(f"Current temperature ({current_temp:.1f}C) is ANOMALOUS!")
                        else:
                            st.success(f"Current temperature ({current_temp:.1f}C) is within normal range")
                        
                        season_data = city_data[city_data['season'] == current_season]
                        fig_current = go.Figure()
                        fig_current.add_trace(go.Violin(y=season_data['temperature'],
                            name=f"Historical Data", box_visible=True, meanline_visible=True,
                            fillcolor='lightblue', opacity=0.7))
                        fig_current.add_trace(go.Scatter(x=["Current"], y=[current_temp],
                            mode='markers', name=f"Current: {current_temp:.1f}C",
                            marker=dict(color='red', size=15, symbol='star')))
                        fig_current.update_layout(title=f"Current vs Historical Temperature",
                            yaxis_title="Temperature (C)", height=500)
                        st.plotly_chart(fig_current, use_container_width=True)

else:
    st.info("""
    ## Instructions:
    1. Click **"Generate Sample Data"** in the sidebar to create test data
    2. Or upload your own CSV file with columns: city, timestamp, temperature
    3. Enter OpenWeatherMap API key to get current weather
    4. Select city and explore the 4 tabs
    """)

st.markdown("---")
st.markdown("Climate Change Monitor | Streamlit + Plotly + OpenWeatherMap")
