"""
Stone Column Design Assistant V4 - Complete Professional Edition
Advanced AI-powered geotechnical design tool with comprehensive analytics
Uses 2-output model (Ultimate Stress, Service Load) with computed Factor of Safety
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from datetime import datetime


try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Stone Column Design Assistant V4",
    page_icon="🗿",
    layout="wide"
)

# STYLING
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #856404;
        font-size: 0.85rem;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)



# LOAD MODEL AND SCALERS

@st.cache_resource
def load_all():
    if not TF_AVAILABLE:
        return None, None, None
    try:
        model = tf.keras.models.load_model("stone_column_ann_model.h5", compile=False)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model/scalers: {str(e)}")
        return None, None, None


model, scaler_X, scaler_y = load_all()



# PARAMETERS

PARAM_RANGES = {
    'cu':   (5.0,  40.0, 15.0),
    'D':    (0.06,  0.8,  0.4),
    'L':    (0.7,  12.0,  6.0),
    'sD':   (2.0,   4.0,  2.5),
    'Eenc': (0.0,  20.0,  0.0),
}

PARAM_INFO = {
    'cu':   {'name': 'Undrained Shear Strength', 'unit': 'kPa'},
    'D':    {'name': 'Column Diameter',           'unit': 'm'},
    'L':    {'name': 'Column Length',             'unit': 'm'},
    'sD':   {'name': 'Spacing Ratio (s/D)',       'unit': '-'},
    'Eenc': {'name': 'Encasement Stiffness',      'unit': 'kN/m'},
}



# CORE FUNCTIONS

def predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc):
    x = np.array([[cu, D, L, sD, Eenc]], dtype=np.float32)
    x_scaled = scaler_X.transform(x)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    sigma, P10 = float(pred[0]), float(pred[1])
    FS = compute_FS(sigma, P10, cu, D, L, sD)
    return sigma, P10, FS


def compute_FS(sigma, P10, cu, D, L, sD):
    if P10 <= 0 or sigma <= 0:
        return 0.0
    FS = sigma / P10
    corr = 1.0
    if cu < 10:
        corr *= (0.7 + 0.03 * cu)
    if D < 0.25:
        corr *= (0.8 + 0.8 * D)
    if sD < 2.5:
        corr *= (0.85 + 0.06 * sD)
    if L < 3.0:
        corr *= (0.75 + 0.083 * L)
    return max(0.5, min(FS * corr, 5.0))


def calc_derived(cu, D, L, sD, sigma, P10):
    A_col = np.pi * (D / 2) ** 2
    return {
        'spacing':       sD * D,
        'slenderness':   L / D,
        'area_repl':     1 / (sD ** 2),
        'improv_factor': sigma / cu if cu > 0 else 0,
        'col_area':      A_col,
        'load_kN':       P10 * A_col,
        'load_per_len':  (P10 * A_col) / L if L > 0 else 0,
    }


def validate_design(cu, D, L, sD):
    warnings_list = []
    reliable = True
    if cu < 10:
        warnings_list.append(f"⚠️ Weak soil (cu={cu:.1f} kPa)")
        reliable = False
    if D < 0.25:
        warnings_list.append(f"⚠️ Small diameter (D={D:.2f} m)")
        reliable = False
    if L < 3.0:
        warnings_list.append(f"⚠️ Short column (L={L:.1f} m)")
        reliable = False
    if sD < 2.5:
        warnings_list.append(f"⚠️ Tight spacing (s/D={sD:.1f})")
        reliable = False
    if L / D > 20:
        warnings_list.append(f"⚠️ High slenderness (L/D={L / D:.1f})")
    return reliable, warnings_list


def assess_safety(FS, reliable):
    if not reliable:
        return "UNRELIABLE", "Outside validated range", "error"
    if FS >= 2.5:
        return "Excellent", "Exceeds requirements", "success"
    elif FS >= 2.0:
        return "Adequate", "Meets requirements", "success"
    elif FS >= 1.5:
        return "Marginal", "Consider optimization", "warning"
    else:
        return "Insufficient", "Revision required", "error"


def run_sensitivity(inp, key, scaler_X, scaler_y, model):
    """Return arrays of (vals, sigma_arr, P10_arr, FS_arr) over the range of `key`."""
    vals = np.linspace(PARAM_RANGES[key][0], PARAM_RANGES[key][1], 60)
    results = []
    for v in vals:
        t = inp.copy()
        t[key] = v
        x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]], dtype=np.float32)
        xs = scaler_X.transform(x)
        ps = model.predict(xs, verbose=0)[0]
        p = scaler_y.inverse_transform(ps.reshape(1, -1))[0]
        fs = compute_FS(float(p[0]), float(p[1]), t['cu'], t['D'], t['L'], t['sD'])
        results.append([float(p[0]), float(p[1]), fs])
    return vals, np.array(results)


def run_heatmap(inp, p1k, p2k, oidx, scaler_X, scaler_y, model, res=40):
    p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
    p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
    Z = np.zeros((len(p2v), len(p1v)))
    for i, v2 in enumerate(p2v):
        for j, v1 in enumerate(p1v):
            t = inp.copy()
            t[p1k] = v1
            t[p2k] = v2
            x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]], dtype=np.float32)
            xs = scaler_X.transform(x)
            ps = model.predict(xs, verbose=0)[0]
            p = scaler_y.inverse_transform(ps.reshape(1, -1))[0]
            if oidx < 2:
                Z[i, j] = float(p[oidx])
            else:
                Z[i, j] = compute_FS(float(p[0]), float(p[1]), t['cu'], t['D'], t['L'], t['sD'])
    return p1v, p2v, Z


def run_surface(inp, p1k, p2k, oidx, scaler_X, scaler_y, model, res=20):
    p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
    p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
    P1, P2 = np.meshgrid(p1v, p2v)
    Z = np.zeros_like(P1, dtype=float)
    for i in range(P1.shape[0]):
        for j in range(P1.shape[1]):
            t = inp.copy()
            t[p1k] = P1[i, j]
            t[p2k] = P2[i, j]
            x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]], dtype=np.float32)
            xs = scaler_X.transform(x)
            ps = model.predict(xs, verbose=0)[0]
            p = scaler_y.inverse_transform(ps.reshape(1, -1))[0]
            if oidx < 2:
                Z[i, j] = float(p[oidx])
            else:
                Z[i, j] = compute_FS(float(p[0]), float(p[1]), t['cu'], t['D'], t['L'], t['sD'])
    return p1v, p2v, Z


# 
# HEADER
# 
st.markdown(
    '<div class="main-header">'
    '<h1>🗿 Stone Column Design Assistant V4</h1>'
    '<p>AI-Powered Geotechnical Design with Comprehensive Analytics</p>'
    '</div>',
    unsafe_allow_html=True,
)



# SIDEBAR

with st.sidebar:
    st.header("⚙️ Design Parameters")
    st.markdown("---")

    cu = st.number_input(
        f"{PARAM_INFO['cu']['name']} ({PARAM_INFO['cu']['unit']})",
        min_value=PARAM_RANGES['cu'][0], max_value=PARAM_RANGES['cu'][1],
        value=PARAM_RANGES['cu'][2], step=0.5,
    )
    D = st.number_input(
        f"{PARAM_INFO['D']['name']} ({PARAM_INFO['D']['unit']})",
        min_value=PARAM_RANGES['D'][0], max_value=PARAM_RANGES['D'][1],
        value=PARAM_RANGES['D'][2], step=0.01, format="%.2f",
    )
    L = st.number_input(
        f"{PARAM_INFO['L']['name']} ({PARAM_INFO['L']['unit']})",
        min_value=PARAM_RANGES['L'][0], max_value=PARAM_RANGES['L'][1],
        value=PARAM_RANGES['L'][2], step=0.1, format="%.1f",
    )
    sD = st.number_input(
        f"{PARAM_INFO['sD']['name']} ({PARAM_INFO['sD']['unit']})",
        min_value=PARAM_RANGES['sD'][0], max_value=PARAM_RANGES['sD'][1],
        value=PARAM_RANGES['sD'][2], step=0.1, format="%.1f",
    )
    Eenc = st.number_input(
        f"{PARAM_INFO['Eenc']['name']} ({PARAM_INFO['Eenc']['unit']})",
        min_value=PARAM_RANGES['Eenc'][0], max_value=PARAM_RANGES['Eenc'][1],
        value=PARAM_RANGES['Eenc'][2], step=0.5, format="%.1f",
    )

    st.info(f"**Spacing:** {sD * D:.2f} m")
    st.markdown("---")

    st.header("📊 Analysis Options")
    sens   = st.checkbox("Sensitivity Analysis", value=True)
    heat   = st.checkbox("Interaction Heatmap",  value=True)
    surf3d = st.checkbox("3D Surface Plot",       value=True)


# 
# CHECK MODEL
# 
if not TF_AVAILABLE:
    st.error("TensorFlow is not installed. Please run: `pip install tensorflow`")
    st.stop()

if model is None or scaler_X is None or scaler_y is None:
    st.warning(
        "⚠️ Model files not found. Please ensure the following files exist in the working directory:\n"
        "- `stone_column_ann_model.h5`\n"
        "- `scaler_X.pkl`\n"
        "- `scaler_y.pkl`"
    )
    st.stop()


# 
# PREDICT
# 
inp = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}
sigma, P10, FS = predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc)
der = calc_derived(cu, D, L, sD, sigma, P10)
reliable, warns = validate_design(cu, D, L, sD)


# 
# RESULTS
# 
st.header("📈 Prediction Results")

if warns:
    st.error("**Reliability Warnings:**")
    for w in warns:
        st.write(w)
    st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultimate Stress (σ)", f"{sigma:.2f} kPa",  help="AI predicted")
c2.metric("Service Load (P10)",  f"{P10:.2f} kPa",    help="AI predicted")
c3.metric(
    "Factor of Safety",
    f"{FS:.2f}",
    delta="Safe" if FS >= 2.0 else "Low",
    delta_color="normal" if FS >= 2.0 else "inverse",
    help="Computed: σ / P10",
)
c4.metric("Slenderness (L/D)", f"{der['slenderness']:.1f}")

st.subheader("Design Assessment")
status, msg, col = assess_safety(FS, reliable)
if col == "success":
    st.success(f"**{status}:** {msg}")
elif col == "warning":
    st.warning(f"**{status}:** {msg}")
else:
    st.error(f"**{status}:** {msg}")

st.subheader("Design Information")
ci1, ci2 = st.columns(2)
ci1.info(
    f"**Column Configuration:**\n"
    f"Type: {'Encased' if Eenc > 0 else 'Unencased'}\n\n"
    f"Area Replacement Ratio: {der['area_repl']:.3f}\n\n"
    f"Column Spacing: {der['spacing']:.2f} m\n\n"
    f"Slenderness (L/D): {der['slenderness']:.1f}\n\n"
    f"Column Area: {der['col_area']:.4f} m²"
)
eff = "High" if der['improv_factor'] > 3 else "Moderate" if der['improv_factor'] > 2 else "Low"
ci2.info(
    f"**Performance Metrics:**\n"
    f"Improvement Factor: {der['improv_factor']:.2f}×\n\n"
    f"Service Load: {der['load_kN']:.2f} kN\n\n"
    f"Load per Length: {der['load_per_len']:.2f} kN/m\n\n"
    f"Efficiency: {eff}\n\n"
    f"Reliability: {'✅ OK' if reliable else '❌ UNRELIABLE'}"
)

st.markdown("---")


# 
# SENSITIVITY ANALYSIS
# 
if sens:
    st.header("🔍 Sensitivity Analysis")
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    sel = st.selectbox("Select Parameter to Vary:", list(opts.keys()), key='sens_sel')
    key_sel = opts[sel]

    with st.spinner("Running sensitivity analysis..."):
        vals, preds = run_sensitivity(inp, key_sel, scaler_X, scaler_y, model)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Ultimate Stress (kPa)", "Service Load (kPa)", "Factor of Safety"],
    )
    colors = ['#667eea', '#26c6da', '#ffa726']
    labels = ['Ultimate Stress', 'Service Load', 'Factor of Safety']
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=vals, y=preds[:, i],
                mode='lines', name=labels[i],
                line=dict(color=colors[i], width=3),
                fill='tozeroy',
                fillcolor=colors[i].replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in colors[i] else colors[i],
            ),
            row=1, col=i + 1,
        )
        fig.add_vline(x=inp[key_sel], line_dash="dash", line_color="red", row=1, col=i + 1)

    # FS threshold line only on col 3
    fig.add_hline(y=2.0, line_dash="dot", line_color="green", row=1, col=3)

    fig.update_layout(
        height=450, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_text=f"Sensitivity: {sel}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Range Statistics")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Min σ", f"{preds[:, 0].min():.2f} kPa")
    sc1.metric("Max σ", f"{preds[:, 0].max():.2f} kPa")
    sc2.metric("Min P10", f"{preds[:, 1].min():.2f} kPa")
    sc2.metric("Max P10", f"{preds[:, 1].max():.2f} kPa")
    sc3.metric("Min FS", f"{preds[:, 2].min():.2f}")
    sc3.metric("Max FS", f"{preds[:, 2].max():.2f}")

    st.markdown("---")


# 
# INTERACTION HEATMAP
# 
if heat:
    st.header("🌡️ Parameter Interaction Heatmap")
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    hc1, hc2, hc3 = st.columns(3)
    p1n  = hc1.selectbox("X-axis:", list(opts.keys()), key='h1')
    p2n  = hc2.selectbox("Y-axis:", list(opts.keys()), index=1, key='h2')
    outn = hc3.selectbox("Output:", ["Ultimate Stress", "Service Load", "Factor of Safety"], key='hout')

    if p1n == p2n:
        st.warning("Please select two different parameters for X and Y axes.")
    else:
        p1k  = opts[p1n]
        p2k  = opts[p2n]
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)

        with st.spinner("Computing heatmap..."):
            p1v, p2v, Z = run_heatmap(inp, p1k, p2k, oidx, scaler_X, scaler_y, model, res=40)

        fig = go.Figure(go.Heatmap(z=Z, x=p1v, y=p2v, colorscale='Plasma'))
        fig.add_trace(go.Scatter(
            x=[inp[p1k]], y=[inp[p2k]],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=3, color='white')),
            text=['Current'],
            textposition='top center',
            name='Current Design',
        ))
        fig.update_layout(
            title=f"{outn}: {p1n} vs {p2n}",
            xaxis_title=p1n,
            yaxis_title=p2n,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

        oi = np.unravel_index(np.argmax(Z), Z.shape)
        st.success(
            f"**Optimal point:** {p1n} = {p1v[oi[1]]:.2f},  "
            f"{p2n} = {p2v[oi[0]]:.2f},  "
            f"{outn} = {Z[oi]:.2f}"
        )

    st.markdown("---")


# 
# 3D SURFACE
# 
if surf3d:
    st.header("📐 3D Surface Plot")
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    sc1, sc2, sc3 = st.columns(3)
    p1n  = sc1.selectbox("X-axis:", list(opts.keys()), key='3d1')
    p2n  = sc2.selectbox("Y-axis:", list(opts.keys()), index=1, key='3d2')
    outn = sc3.selectbox("Z-axis:", ["Ultimate Stress", "Service Load", "Factor of Safety"], key='3dout')

    if p1n == p2n:
        st.warning("Please select two different parameters for X and Y axes.")
    else:
        res  = st.slider("Grid Resolution", min_value=10, max_value=30, value=20, key='3dres')
        p1k  = opts[p1n]
        p2k  = opts[p2n]
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)

        with st.spinner("Generating 3D surface..."):
            p1v, p2v, Z = run_surface(inp, p1k, p2k, oidx, scaler_X, scaler_y, model, res=res)

        fig = go.Figure(go.Surface(z=Z, x=p1v, y=p2v, colorscale='Viridis', opacity=0.9))
        fig.update_layout(
            scene=dict(
                xaxis_title=p1n,
                yaxis_title=p2n,
                zaxis_title=outn,
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            title=f"3D Surface: {outn}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


# 
# EXPORT
# 
st.header("📥 Export Results")
with st.expander("View & Download Results"):
    rows = []
    for k in PARAM_INFO:
        rows.append({
            'Category': 'Input',
            'Parameter': PARAM_INFO[k]['name'],
            'Value': inp[k],
            'Unit': PARAM_INFO[k]['unit'],
        })
    rows.append({'Category': 'Output', 'Parameter': 'Ultimate Stress',  'Value': round(sigma, 4), 'Unit': 'kPa'})
    rows.append({'Category': 'Output', 'Parameter': 'Service Load',     'Value': round(P10,   4), 'Unit': 'kPa'})
    rows.append({'Category': 'Output', 'Parameter': 'Factor of Safety', 'Value': round(FS,    4), 'Unit': '-'})
    rows.append({'Category': 'Output', 'Parameter': 'Slenderness L/D',  'Value': round(der['slenderness'], 2), 'Unit': '-'})
    rows.append({'Category': 'Output', 'Parameter': 'Improvement Factor','Value': round(der['improv_factor'], 2), 'Unit': '-'})
    rows.append({'Category': 'Output', 'Parameter': 'Service Load (kN)', 'Value': round(der['load_kN'], 2), 'Unit': 'kN'})
    rows.append({'Category': 'Output', 'Parameter': 'Safety Status',     'Value': status, 'Unit': ''})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name=f"stone_column_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# 
# RECOMMENDATIONS
# 
st.header("💡 Recommendations")
rc1, rc2 = st.columns(2)

with rc1:
    st.subheader("Structural")
    if FS < 1.5:
        st.error(
            "**Critical Issues Detected**\n\n"
            "- Increase column diameter or length\n"
            "- Reduce spacing ratio\n"
            "- Consider geotextile encasement"
        )
    elif FS < 2.0:
        st.warning(
            "**Optimization Suggested**\n\n"
            "- Adjust column dimensions\n"
            "- Reduce spacing ratio by 10–15%\n"
            "- Review soil parameters"
        )
    else:
        st.success(
            "**Design is Structurally Sound**\n\n"
            "- Meets safety criteria\n"
            "- Can be optimized for cost efficiency"
        )

with rc2:
    st.subheader("Economic")
    cost_index = (D ** 2) * L / (sD ** 2)
    if cost_index > 2.0:
        st.warning(
            "**High Material Usage**\n\n"
            "- Consider reducing diameter if FS allows\n"
            "- Increase spacing ratio\n"
            f"- Cost index: {cost_index:.2f} (high)"
        )
    else:
        st.success(
            "**Cost-Efficient Design**\n\n"
            "- Material usage is optimized\n"
            f"- Cost index: {cost_index:.2f} (acceptable)"
        )


# 
# FINAL SUMMARY
# 
st.header("📋 Final Summary")
fs1, fs2, fs3 = st.columns(3)

fs1.metric("Safety Status",  status)
fs1.metric("Factor of Safety", f"{FS:.2f}")
fs1.metric("Column Type", "Encased" if Eenc > 0 else "Unencased")

fs2.metric("Ultimate Stress σ",    f"{sigma:.1f} kPa")
fs2.metric("Service Load P10",     f"{P10:.1f} kPa")
fs2.metric("Improvement Factor",   f"{der['improv_factor']:.2f}×")

fs3.metric("Column Spacing",       f"{der['spacing']:.2f} m")
fs3.metric("Slenderness L/D",      f"{der['slenderness']:.1f}")
fs3.metric("Area Replacement Ratio", f"{der['area_repl']:.3f}")



# FOOTER

st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    "⚠️ <strong>Disclaimer:</strong> This tool provides AI-assisted preliminary design estimates only. "
    "Final design must be verified by a qualified geotechnical engineer using site-specific data and applicable design codes."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer">'
    "Stone Column Design Assistant V4 © 2026 &nbsp;|&nbsp; "
    "FS computed as σ / P10 with engineering correction factors"
    "</div>",
    unsafe_allow_html=True,
)
