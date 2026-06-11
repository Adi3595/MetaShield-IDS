# dashboard.py
import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime

# API endpoint
API_URL = "http://localhost:5000"

st.set_page_config(
    page_title="MetaShield Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ MetaShield - Real-Time Threat Detection")
st.markdown("Few-shot learning cybersecurity system")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    st.header("Learn New Attack")
    attack_name = st.text_input("Attack Name")
    num_examples = st.slider("Number of examples", 1, 5, 2)
    
    if st.button("📚 Learn Attack") and attack_name:
        # Generate synthetic examples
        examples = [np.random.randn(78).tolist() for _ in range(num_examples)]
        
        response = requests.post(
            f"{API_URL}/learn",
            json={"attack_name": attack_name, "examples": examples}
        )
        
        if response.status_code == 200:
            st.success(f"✅ Learned {attack_name}")
        else:
            st.error("❌ Failed to learn attack")

# Main content
col1, col2, col3 = st.columns(3)

# Get stats from API
try:
    stats_response = requests.get(f"{API_URL}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        
        with col1:
            st.metric("Known Attacks", len(stats['known_attacks']))
        
        with col2:
            st.metric("Total Alerts", stats['total_alerts'])
        
        with col3:
            st.metric("Recent Alerts", len(stats['recent_alerts']))
        
        # Known attacks
        st.header("📋 Known Attacks")
        if stats['known_attacks']:
            df = pd.DataFrame({
                'Attack Type': stats['known_attacks'],
                'Status': ['Active'] * len(stats['known_attacks'])
            })
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No attacks learned yet")
        
        # Recent alerts
        st.header("🚨 Recent Alerts")
        if stats['recent_alerts']:
            alerts_df = pd.DataFrame(stats['recent_alerts'])
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No recent alerts")
        
        # Real-time monitoring
        st.header("📊 Real-Time Monitoring")
        
        # Create placeholder for live chart
        chart_placeholder = st.empty()
        
        # Simulate live data
        if st.button("▶️ Start Monitoring", key="start"):
            alert_history = []
            
            for i in range(50):  # 50 iterations
                # Generate random flow
                flow = np.random.randn(78)
                
                # Detect
                response = requests.post(
                    f"{API_URL}/detect",
                    json={"features": flow.tolist()}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    alert_history.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'threat': result['attack'] if result['threat_detected'] else 'Normal',
                        'confidence': result['confidence']
                    })
                    
                    # Keep last 20
                    if len(alert_history) > 20:
                        alert_history.pop(0)
                    
                    # Create chart
                    df = pd.DataFrame(alert_history)
                    if not df.empty:
                        fig = go.Figure()
                        
                        # Color based on threat
                        colors = ['red' if x != 'Normal' else 'green' for x in df['threat']]
                        
                        fig.add_trace(go.Bar(
                            x=df['time'],
                            y=df['confidence'],
                            marker_color=colors,
                            name='Confidence'
                        ))
                        
                        fig.update_layout(
                            title='Live Detection Confidence',
                            xaxis_title='Time',
                            yaxis_title='Confidence',
                            yaxis_range=[0, 1],
                            height=400
                        )
                        
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(1)  # Update every second
                
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to API server. Make sure it's running!")
    st.code("python api_server.py")

# Footer
st.markdown("---")
st.markdown("Powered by Few-Shot Meta-Learning | Trained on CIC-IDS2017")