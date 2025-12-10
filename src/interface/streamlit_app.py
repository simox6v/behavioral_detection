"""
Behavioral Detection System - Professional Dashboard
Advanced Threat Classification Platform
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configuration de l'importation
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit non disponible")

# Configuration du logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Threat Categories
THREAT_CATEGORIES = {
    'ransomware': {
        'name': 'Ransomware Simulation',
        'color': '#ef4444',
        'icon': 'lock',
        'severity': 'critical',
        'description': 'Encryption behavior detected'
    },
    'keylogger': {
        'name': 'Keylogger Simulation', 
        'color': '#f59e0b',
        'icon': 'keyboard',
        'severity': 'high',
        'description': 'Keystroke capture activity'
    },
    'portscan': {
        'name': 'Port Scan',
        'color': '#8b5cf6',
        'icon': 'network',
        'severity': 'medium',
        'description': 'Network reconnaissance detected'
    },
    'suspicious_write': {
        'name': 'Suspicious Write Pattern',
        'color': '#3b82f6',
        'icon': 'file',
        'severity': 'medium',
        'description': 'Abnormal file write behavior'
    },
    'benign': {
        'name': 'Normal Activity',
        'color': '#22c55e',
        'icon': 'check',
        'severity': 'low',
        'description': 'No threat detected'
    }
}


# ==================== PROFESSIONAL CSS DESIGN ====================
PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    .pro-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    
    .pro-header h1 {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .pro-header .subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    .status-active {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-inactive {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.active { background: #22c55e; }
    .status-dot.inactive { background: #ef4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .threat-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
        height: 100%;
    }
    
    .threat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    
    .threat-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.75rem;
    }
    
    .threat-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .threat-name {
        color: #f8fafc;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .threat-count {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .threat-delta {
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .section-header {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(71, 85, 105, 0.4);
    }
    
    .alert-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
    }
    
    .alert-card:hover {
        background: #273548;
    }
    
    .alert-ransomware { border-left-color: #ef4444; background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, #1e293b 100%); }
    .alert-keylogger { border-left-color: #f59e0b; background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, #1e293b 100%); }
    .alert-portscan { border-left-color: #8b5cf6; background: linear-gradient(90deg, rgba(139, 92, 246, 0.1) 0%, #1e293b 100%); }
    .alert-suspicious_write { border-left-color: #3b82f6; background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, #1e293b 100%); }
    .alert-benign { border-left-color: #22c55e; background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, #1e293b 100%); }
    
    .alert-time {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .alert-type {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .alert-content {
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .alert-meta {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.5rem;
    }
    
    .alert-meta span {
        color: #94a3b8;
        font-size: 0.8rem;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(71, 85, 105, 0.4);
    }
    
    .severity-badge {
        font-size: 0.65rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    
    .severity-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .severity-high { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .severity-medium { background: rgba(139, 92, 246, 0.2); color: #8b5cf6; }
    .severity-low { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    
    .feature-item {
        background: #1e293b;
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .feature-name {
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 0.5rem;
    }
    
    .feature-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .pro-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(71, 85, 105, 0.3);
        margin-top: 2rem;
    }
</style>
"""


def create_app():
    """Create the professional Streamlit application"""
    
    st.set_page_config(
        page_title="Threat Classification System | Security Platform",
        page_icon="shield",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    
    # ==================== SESSION STATE INIT ====================
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if 'threat_counts' not in st.session_state:
        st.session_state.threat_counts = {
            'ransomware': 0,
            'keylogger': 0,
            'portscan': 0,
            'suspicious_write': 0,
            'benign': 0
        }
    
    if 'threat_deltas' not in st.session_state:
        st.session_state.threat_deltas = {
            'ransomware': 0,
            'keylogger': 0,
            'portscan': 0,
            'suspicious_write': 0,
            'benign': 0
        }
    
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'current_memory_mb': 0.0
        }
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 2rem;">
            <h2 style="color: #f8fafc; font-size: 1.1rem; font-weight: 600; margin: 0;">
                THREAT CLASSIFICATION
            </h2>
            <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">
                Advanced Security Platform v1.0
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Model Configuration</p>', unsafe_allow_html=True)
        
        model_name = st.selectbox(
            "Classification Model",
            ["Multi-Class Classifier", "Isolation Forest", "Random Forest", "XGBoost", "Neural Network"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        alert_threshold = st.slider(
            "Alert Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Confidence threshold for threat classification"
        )
        
        refresh_interval = st.slider(
            "Refresh Interval (s)",
            1, 10, 2,
            help="Dashboard refresh rate in seconds"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Classification Control</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("START", use_container_width=True, type="primary", disabled=st.session_state.running):
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("STOP", use_container_width=True, disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        status_class = "active" if st.session_state.running else "inactive"
        status_text = "System Active" if st.session_state.running else "System Stopped"
        
        st.markdown(f"""
        <div style="background: #0f172a; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <p style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">System Status</p>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div class="status-dot {status_class}"></div>
                <span style="color: {'#22c55e' if st.session_state.running else '#ef4444'}; font-weight: 500;">
                    {status_text}
                </span>
            </div>
            <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.75rem;">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Threat Legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Threat Categories</p>', unsafe_allow_html=True)
        
        for key, threat in THREAT_CATEGORIES.items():
            if key != 'benign':
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <div style="width: 12px; height: 12px; border-radius: 3px; background: {threat['color']};"></div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">{threat['name']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== MAIN CONTENT ====================
    
    status_badge_class = "status-active" if st.session_state.running else "status-inactive"
    status_badge_text = "CLASSIFICATION ACTIVE" if st.session_state.running else "CLASSIFICATION STOPPED"
    status_dot_class = "active" if st.session_state.running else "inactive"
    
    st.markdown(f"""
    <div class="pro-header">
        <h1>Threat Classification System</h1>
        <p class="subtitle">Real-time behavioral analysis and threat classification platform</p>
        <div class="status-badge {status_badge_class}">
            <div class="status-dot {status_dot_class}"></div>
            {status_badge_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== UPDATE STATS ====================
    if st.session_state.running:
        # Reset deltas
        for key in st.session_state.threat_deltas:
            st.session_state.threat_deltas[key] = 0
        
        # Simulate classifications
        num_events = np.random.randint(2, 8)
        
        # Weighted random selection (most should be benign)
        threat_types = ['ransomware', 'keylogger', 'portscan', 'suspicious_write', 'benign']
        weights = [0.05, 0.08, 0.12, 0.15, 0.60]  # More likely benign
        
        for _ in range(num_events):
            threat = np.random.choice(threat_types, p=weights)
            st.session_state.threat_counts[threat] += 1
            st.session_state.threat_deltas[threat] += 1
            st.session_state.stats['total_detections'] += 1
        
        st.session_state.stats['avg_latency_ms'] = np.random.uniform(0.1, 3.0)
        st.session_state.stats['current_memory_mb'] = np.random.uniform(100, 200)
        
        # Add to history
        if len(st.session_state.detection_history) >= 60:
            st.session_state.detection_history.pop(0)
        
        threats_count = sum(st.session_state.threat_deltas[t] for t in threat_types if t != 'benign')
        st.session_state.detection_history.append({
            'time': datetime.now(),
            'total': num_events,
            'threats': threats_count,
            'ransomware': st.session_state.threat_deltas['ransomware'],
            'keylogger': st.session_state.threat_deltas['keylogger'],
            'portscan': st.session_state.threat_deltas['portscan'],
            'suspicious_write': st.session_state.threat_deltas['suspicious_write']
        })
        
        # Generate alerts for threats
        for threat_type in ['ransomware', 'keylogger', 'portscan', 'suspicious_write']:
            if st.session_state.threat_deltas[threat_type] > 0:
                threat_info = THREAT_CATEGORIES[threat_type]
                st.session_state.alerts.insert(0, {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'type': threat_type,
                    'name': threat_info['name'],
                    'color': threat_info['color'],
                    'severity': threat_info['severity'],
                    'description': threat_info['description'],
                    'confidence': np.random.uniform(0.75, 0.98),
                    'latency': np.random.uniform(0.1, 2.0)
                })
        
        st.session_state.alerts = st.session_state.alerts[:10]
    
    # ==================== THREAT CLASSIFICATION CARDS ====================
    st.markdown('<p class="section-header">Threat Classifications</p>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    threat_order = ['ransomware', 'keylogger', 'portscan', 'suspicious_write', 'benign']
    
    for i, threat_key in enumerate(threat_order):
        threat = THREAT_CATEGORIES[threat_key]
        count = st.session_state.threat_counts[threat_key]
        delta = st.session_state.threat_deltas.get(threat_key, 0)
        
        with cols[i]:
            delta_html = ""
            if st.session_state.running and delta > 0:
                delta_html = f'<div class="threat-delta" style="color: {threat["color"]};">+{delta}</div>'
            
            st.markdown(f"""
            <div class="threat-card" style="border-top: 3px solid {threat['color']};">
                <div class="threat-header">
                    <div class="threat-icon" style="background: {threat['color']}20;">
                        <span style="color: {threat['color']};">●</span>
                    </div>
                    <div>
                        <div class="threat-name">{threat['name']}</div>
                        <span class="severity-badge severity-{threat['severity']}">{threat['severity']}</span>
                    </div>
                </div>
                <div class="threat-count">{count:,}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CHARTS ====================
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown('<p class="section-header">Classification Activity</p>', unsafe_allow_html=True)
        
        if st.session_state.detection_history:
            times = [h['time'] for h in st.session_state.detection_history]
            ransomware = [h['ransomware'] for h in st.session_state.detection_history]
            keylogger = [h['keylogger'] for h in st.session_state.detection_history]
            portscan = [h['portscan'] for h in st.session_state.detection_history]
            suspicious = [h['suspicious_write'] for h in st.session_state.detection_history]
        else:
            times = pd.date_range(end=datetime.now(), periods=30, freq='2s')
            ransomware = [0] * 30
            keylogger = [0] * 30
            portscan = [0] * 30
            suspicious = [0] * 30
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times, y=ransomware, mode='lines', name='Ransomware',
            line=dict(color='#ef4444', width=2), stackgroup='one'
        ))
        fig.add_trace(go.Scatter(
            x=times, y=keylogger, mode='lines', name='Keylogger',
            line=dict(color='#f59e0b', width=2), stackgroup='one'
        ))
        fig.add_trace(go.Scatter(
            x=times, y=portscan, mode='lines', name='Port Scan',
            line=dict(color='#8b5cf6', width=2), stackgroup='one'
        ))
        fig.add_trace(go.Scatter(
            x=times, y=suspicious, mode='lines', name='Suspicious Write',
            line=dict(color='#3b82f6', width=2), stackgroup='one'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(71, 85, 105, 0.3)', tickfont=dict(color='#94a3b8', size=10)),
            yaxis=dict(showgrid=True, gridcolor='rgba(71, 85, 105, 0.3)', tickfont=dict(color='#94a3b8', size=10)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#94a3b8', size=11)),
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown('<p class="section-header">Threat Distribution</p>', unsafe_allow_html=True)
        
        labels = []
        values = []
        colors = []
        
        for key in ['ransomware', 'keylogger', 'portscan', 'suspicious_write', 'benign']:
            count = st.session_state.threat_counts[key]
            if count > 0 or key == 'benign':
                labels.append(THREAT_CATEGORIES[key]['name'])
                values.append(max(count, 1) if key == 'benign' else count)
                colors.append(THREAT_CATEGORIES[key]['color'])
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker_colors=colors,
            textinfo='percent',
            textfont=dict(color='#f8fafc', size=11),
            hovertemplate='%{label}: %{value}<extra></extra>'
        )])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SYSTEM METRICS ====================
    st.markdown('<p class="section-header">System Metrics</p>', unsafe_allow_html=True)
    
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric("TOTAL EVENTS", f"{st.session_state.stats['total_detections']:,}")
    
    with metrics_cols[1]:
        total_threats = sum(st.session_state.threat_counts[t] for t in ['ransomware', 'keylogger', 'portscan', 'suspicious_write'])
        st.metric("TOTAL THREATS", f"{total_threats:,}")
    
    with metrics_cols[2]:
        st.metric("LATENCY", f"{st.session_state.stats['avg_latency_ms']:.2f} ms")
    
    with metrics_cols[3]:
        st.metric("MEMORY", f"{st.session_state.stats['current_memory_mb']:.0f} MB")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== ALERTS ====================
    st.markdown('<p class="section-header">Recent Threat Alerts</p>', unsafe_allow_html=True)
    
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.markdown(f"""
            <div class="alert-card alert-{alert['type']}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="alert-time">{alert['time']}</div>
                    <span class="alert-type" style="background: {alert['color']}30; color: {alert['color']};">{alert['name']}</span>
                </div>
                <div class="alert-content">{alert['description']}</div>
                <div class="alert-meta">
                    <span>Confidence: {alert['confidence']:.1%}</span>
                    <span>Latency: {alert['latency']:.2f}ms</span>
                    <span class="severity-badge severity-{alert['severity']}">{alert['severity']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #64748b;">
            No threat alerts recorded. Start classification to detect threats.
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== FOOTER ====================
    st.markdown("""
    <div class="pro-footer">
        <p>Threat Classification System | Advanced Security Platform</p>
        <p style="color: #475569; font-size: 0.7rem; margin-top: 0.5rem;">
            Educational Project - Simulation Only
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== AUTO REFRESH ====================
    if st.session_state.running:
        time.sleep(refresh_interval)
        st.rerun()


def main():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: pip install streamlit plotly")
        return
    create_app()


if __name__ == "__main__":
    main()
