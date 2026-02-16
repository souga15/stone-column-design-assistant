"""
Stone Column Design Assistant V4 - Complete Professional Edition
Advanced AI-powered geotechnical design tool with comprehensive analytics
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import time
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Stone Column Design Assistant V6", page_icon="🗿", layout="wide")

# --- ANIMATION LOADER FUNCTION ---
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a cool engineering/AI animation for the sidebar
lottie_ai = load_lottieurl("https://lottie.host/8b211c47-f571-4fb9-8b8d-29007e2a975f/eIfZ40Yv68.json")

# STYLING
st.markdown("""
<style>
    @view-transition { navigation: auto; }
    .block-container { animation: smoothFadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
    @keyframes smoothFadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: gradient-animation 4s ease infinite; background-size: 200% 200%;
        color: #111;
    }
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 { color: #ffffff; font-size: 2.8rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    .main-header p { color: #ffffff; font-size: 1.2rem; margin-top: 0.5rem; font-weight: 500;}
    
    .stMetric { 
        background: linear-gradient(135deg, #1e2129 0%, #2d3139 100%); 
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #99f2c8; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
        transition: transform 0.3s ease, box-shadow 0.3s ease; 
    }
    .stMetric:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(153,242,200,0.3); }
    h2, h3 { color: #99f2c8; font-weight: 600; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(153,242,200,0.3); margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_all():
    try:
        model = tf.keras.models.load_model("stone_column_ann_model.h5", compile=False)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y
    except Exception as e:
        return None, None, None

model, scaler_X, scaler_y = load_all()

PARAM_RANGES = {'cu': (5.0, 40.0, 15.0), 'D': (0.06, 0.8, 0.4), 'L': (0.7, 12.0, 6.0), 'sD': (2.0, 4.0, 2.5), 'Eenc': (0.0, 20.0, 0.0)}
PARAM_INFO = {
    'cu': {'name': 'Undrained Shear Strength', 'unit': 'kPa'},
    'D': {'name': 'Column Diameter', 'unit': 'm'},
    'L': {'name': 'Column Length', 'unit': 'm'},
    'sD': {'name': 'Spacing Ratio (s/D)', 'unit': '-'},
    'Eenc': {'name': 'Encasement Stiffness', 'unit': 'kN/m'}
}

# FUNCTIONS
def predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc):
    x = np.array([[cu, D, L, sD, Eenc]], dtype=np.float32)
    x_scaled = scaler_X.transform(x)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    sigma, P10 = pred[0], pred[1]
    
    # Calculate FS
    if P10 <= 0 or sigma <= 0: FS = 0.0
    else:
        FS = sigma / P10
        corr = 1.0
        if cu < 10: corr *= (0.7 + 0.03 * cu)
        if D < 0.25: corr *= (0.8 + 0.8 * D)
        if sD < 2.5: corr *= (0.85 + 0.06 * sD)
        if L < 3.0: corr *= (0.75 + 0.083 * L)
        FS = max(0.5, min(FS * corr, 5.0))
    return sigma, P10, FS

def calc_derived(cu, D, L, sD, sigma, P10):
    A_col = np.pi * (D/2)**2
    return {
        'spacing': sD * D, 'slenderness': L/D, 'area_repl': 1/(sD**2),
        'improv_factor': sigma/cu if cu > 0 else 0, 'col_area': A_col,
        'load_kN': P10 * A_col, 'load_per_len': (P10 * A_col)/L if L > 0 else 0
    }

def validate_design(cu, D, L, sD):
    warnings, reliable = [], True
    if cu < 10: warnings.append(f"⚠️ Weak soil (cu={cu:.1f} kPa)"); reliable = False
    if D < 0.25: warnings.append(f"⚠️ Small diameter (D={D:.2f} m)"); reliable = False
    if L < 3.0: warnings.append(f"⚠️ Short column (L={L:.1f} m)"); reliable = False
    if sD < 2.5: warnings.append(f"⚠️ Tight spacing (s/D={sD:.1f})"); reliable = False
    return reliable, warnings

# HEADER
st.markdown('<div class="main-header"><h1>🗿 Stone Column AI Assistant</h1>'
            '<p>Professional Edition | Deep Learning Geotechnical Analytics</p></div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_animation")
        
    st.header("Design Parameters")
    cu = st.number_input(f"{PARAM_INFO['cu']['name']} ({PARAM_INFO['cu']['unit']})", PARAM_RANGES['cu'][0], PARAM_RANGES['cu'][1], PARAM_RANGES['cu'][2], 0.5)
    D = st.number_input(f"{PARAM_INFO['D']['name']} ({PARAM_INFO['D']['unit']})", PARAM_RANGES['D'][0], PARAM_RANGES['D'][1], PARAM_RANGES['D'][2], 0.01, format="%.2f")
    L = st.number_input(f"{PARAM_INFO['L']['name']} ({PARAM_INFO['L']['unit']})", PARAM_RANGES['L'][0], PARAM_RANGES['L'][1], PARAM_RANGES['L'][2], 0.1, format="%.1f")
    sD = st.number_input(f"{PARAM_INFO['sD']['name']} ({PARAM_INFO['sD']['unit']})", PARAM_RANGES['sD'][0], PARAM_RANGES['sD'][1], PARAM_RANGES['sD'][2], 0.1, format="%.1f")
    Eenc = st.number_input(f"{PARAM_INFO['Eenc']['name']} ({PARAM_INFO['Eenc']['unit']})", PARAM_RANGES['Eenc'][0], PARAM_RANGES['Eenc'][1], PARAM_RANGES['Eenc'][2], 0.5, format="%.1f")
    
    st.success(f"**Calculated Spacing:** {sD * D:.2f} m")
    st.markdown("---")
    sens = st.checkbox("Sensitivity Analysis", True)
    heat = st.checkbox("Interaction Heatmap", True)

if model is None:
    st.warning("Model files not found. Please ensure .h5 and .pkl files are in the directory.")
    st.stop()

# VISIBLE LOADING STATE
with st.spinner("🧠 Running Neural Network Analysis..."):
    time.sleep(0.4) # Brief pause to make the loading spinner visible
    sigma, P10, FS = predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc)
    der = calc_derived(cu, D, L, sD, sigma, P10)
    reliable, warns = validate_design(cu, D, L, sD)

# TOAST NOTIFICATION
st.toast('Analysis Updated Successfully!', icon='✅')

# RESULTS
st.header("Prediction Results")
if warns:
    for w in warns: st.error(w)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultimate Stress", f"{sigma:.2f} kPa")
c2.metric("Service Load", f"{P10:.2f} kPa")
c3.metric("Factor of Safety", f"{FS:.2f}", delta="Safe" if FS >= 2.0 else "Low", delta_color="normal" if FS >= 2.0 else "inverse")
c4.metric("Slenderness", f"{der['slenderness']:.1f}")

st.markdown("---")

# SENSITIVITY
if sens:
    st.header("Sensitivity Analysis")
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    sel = st.selectbox("Parameter:", list(opts.keys()))
    key = opts[sel]
    
    vals = np.linspace(PARAM_RANGES[key][0], PARAM_RANGES[key][1], 50)
    preds = []
    for v in vals:
        t = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}; t[key] = v
        x = scaler_X.transform(np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]]))
        p = scaler_y.inverse_transform(model.predict(x, verbose=0))[0]
        preds.append([p[0], p[1], compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])])
    preds = np.array(preds)
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Ultimate Stress", "Service Load", "Factor of Safety"])
    colors = ['#99f2c8', '#26c6da', '#ffa726']
    for i in range(3):
        fig.add_trace(go.Scatter(x=vals, y=preds[:,i], mode='lines', line=dict(color=colors[i], width=3), fill='tozeroy'), row=1, col=i+1)
        fig.add_vline(x={'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}[key], line_dash="dash", line_color="red", row=1, col=i+1)
    fig.update_layout(height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# HEATMAP
if heat:
    st.header("Parameter Interaction")
    c1, c2, c3 = st.columns(3)
    p1n = c1.selectbox("X-Axis:", list(opts.keys()), key='h1')
    p2n = c2.selectbox("Y-Axis:", list(opts.keys()), index=1, key='h2')
    outn = c3.selectbox("Predict:", ["Ultimate Stress", "Service Load", "Factor of Safety"])
    
    if p1n != p2n:
        p1k, p2k = opts[p1n], opts[p2n]
        p1v, p2v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], 25), np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], 25)
        Z = np.zeros((25, 25))
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)
        
        with st.spinner("Generating High-Res Map..."):
            for i, v2 in enumerate(p2v):
                for j, v1 in enumerate(p1v):
                    t = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}; t[p1k], t[p2k] = v1, v2
                    x = scaler_X.transform(np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]]))
                    p = scaler_y.inverse_transform(model.predict(x, verbose=0))[0]
                    Z[i,j] = p[oidx] if oidx < 2 else compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
            
        fig = go.Figure(go.Heatmap(z=Z, x=p1v, y=p2v, colorscale='Tealrose'))
        fig.add_scatter(x=[{'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}[p1k]], y=[{'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}[p2k]], 
                        mode='markers+text', marker=dict(size=15, color='white', symbol='x'), text=['Current'], textposition='top center')
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
