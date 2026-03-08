import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from inference_wind import (
    build_base_weather_dataframe, 
    recursive_24h_forecast,
    HOURS_REQUIRED,
    run_test_benchmarks
)

# Page configuration
st.set_page_config(
    page_title="Wind Power Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the "Dashboard" look
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .main-title { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; font-size: 28px; color: #FFFFF; margin-top: 20px; }
    .sub-title { font-family: 'Helvetica Neue', sans-serif; font-weight: 400; font-size: 16px; color: #FFFFF; margin-bottom: 20px; }
    
    /* Carousel Styling */
    .carousel-container { display: flex; overflow-x: auto; gap: 10px; padding: 15px 5px; background-color: #f4f6fa; border-radius: 12px; margin-bottom: 20px; }
    .weather-card { min-width: 80px; text-align: center; padding: 5px; border-right: 1px solid #e0e0e0; }
    .weather-card:last-child { border-right: none; }
    .card-time { font-size: 11px; color: #666; font-weight: 600; text-transform: uppercase;}
    .card-mw { font-size: 14px; font-weight: 700; color: #000; }
    .card-wind { font-size: 10px; color: #555; }
    .footer-text { font-size: 11px; color: #666; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Wind Power Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">(Port Alama Wind Farm)</div>', unsafe_allow_html=True)

# Run the backend logic
with st.spinner('Running AI Inference...'):
    try:
        base_df = build_base_weather_dataframe()
        
        # Unpack both the base and corrected predictions
        base_preds, corrected_preds = recursive_24h_forecast(base_df)
        
        # Get the last 24h wind speeds for the UI
        wind_speeds = base_df['Wind Spd (km/h)'].tail(HOURS_REQUIRED).values
        
    except Exception as e:
        st.error(f"Backend Error: {e}")
        st.stop()

# Generate time labels
current_hour = datetime.now().hour
hours_label = []
for i in range(HOURS_REQUIRED):
    h = (current_hour + i + 1) % 24
    suffix = "AM" if h < 12 else "PM"
    h_display = 12 if h % 12 == 0 else h % 12
    hours_label.append(f"{h_display}{suffix}")

# Render carousel
carousel_html = '<div class="carousel-container">'
for i in range(len(corrected_preds)):
    mw_val = f"{corrected_preds[i]:.1f} mw"
    wind_val = f"{wind_speeds[i] if i < len(wind_speeds) else 0:.0f}km/h"
    
    carousel_html += f"""<div class="weather-card"><div class="card-time">{hours_label[i]}</div><div class="card-mw">{mw_val}</div><div class="card-wind">{wind_val}</div></div>"""
carousel_html += '</div>'
st.markdown(carousel_html, unsafe_allow_html=True)

fig = go.Figure()

# Corrected Prediction Area
fig.add_trace(go.Scatter(
    x=hours_label,
    y=corrected_preds,
    mode='lines+markers',
    name='Corrected prediction',
    line=dict(color='#3344a0', width=2),
    marker=dict(size=6, color='#3344a0'),
    fill='tozeroy',
    fillcolor='rgba(100, 120, 240, 0.3)',
    hovertemplate='<b>Corrected:</b> %{y:.2f} MW<extra></extra>'
))

# Base Model Line
fig.add_trace(go.Scatter(
    x=hours_label,
    y=base_preds,
    mode='lines+markers',
    name='Base model prediction',
    line=dict(color='#3344a0', width=2, dash='dash'),
    marker=dict(size=6, color='#3344a0'),
    hovertemplate='<b>Base Model:</b> %{y:.2f} MW<extra></extra>'
))

fig.add_vline(x=0, line_width=2, line_color="#3344a0")

# Layout - Configuring the Legend Box
fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor='white',
    paper_bgcolor='white',
    
    xaxis=dict(
        showgrid=True, 
        gridcolor='#f0f0f0', 
        showline=False, 
        tickfont=dict(size=10, color='#666'),
        range=[-0.1, 23.5],
    ),
    
    yaxis=dict(
        visible=True,
        showgrid=False,
        showline=False,
        tickfont=dict(size=10, color='#666'),
        range=[0, max(max(base_preds), max(corrected_preds)) * 1.8],
        ticklen=0,
    ),
    
    # Legend styling
    legend=dict(
        orientation="v",
        yanchor="top", 
        y=0.98,
        xanchor="right", 
        x=0.99,
        bgcolor="white",
        bordercolor="#d3d3d3",
        borderwidth=1,
        font=dict(size=12, color="#333")
    ),
    
    hovermode="x unified"
)

st.plotly_chart(fig, width='stretch')
st.markdown('<div class="footer-text">This dashboard shows predictions as average per generator for Port Alama wind farm.</div>', unsafe_allow_html=True)

# MODEL BENCHMARKING SECTION
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="main-title" style="font-size: 22px;">Model Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Run Live Performance Tests on T1 and T3 Data</div>', unsafe_allow_html=True)

if st.button("Run Benchmarks on Test Sets"):
    with st.spinner("Loading test datasets and calculating metrics..."):
        try:
            # Run the backend function
            rmse_t1, r2_t1, rmse_t3, r2_t3 = run_test_benchmarks()
            
            # Display the numbers nicely in columns
            col1, col2 = st.columns(2)
            with col1:
                st.info("**T1 Test Set Results**")
                st.metric("Ensemble RMSE", f"{rmse_t1:.4f}")
                st.metric("Ensemble R²", f"{r2_t1:.4f}")
                
            with col2:
                st.info("**T3 Test Set Results**")
                st.metric("Ensemble RMSE", f"{rmse_t3:.4f}")
                st.metric("Ensemble R²", f"{r2_t3:.4f}")
                
            # Render the Bar Chart
            fig_bench = go.Figure()

            # RMSE Bars
            fig_bench.add_trace(go.Bar(
                x=['T1 Test Set', 'T3 Test Set'],
                y=[rmse_t1, rmse_t3],
                name='RMSE',
                marker_color='#3344a0'
            ))

            # R2 Bars
            fig_bench.add_trace(go.Bar(
                x=['T1 Test Set', 'T3 Test Set'],
                y=[r2_t1, r2_t3],
                name='R² Score',
                marker_color='#ff7f0e',
            ))

            fig_bench.update_layout(
                title="Model Benchmark: Tuned Ensembles",
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showline=True, linewidth=1, linecolor='#d3d3d3', tickfont=dict(color='black')),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Score", tickfont=dict(color='black')),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black'))
            )
            
            st.plotly_chart(fig_bench, use_container_width=True)

        except Exception as e:
            st.error(f"Error running benchmarks: {e}")
            st.warning("Ensure the paths to 'wind_power_error_test_1.csv' and 'wind_power_error_test_2.csv' are correct in your inference script.")