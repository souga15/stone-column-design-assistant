"""
Stone Column Design Assistant - Professional Version
Advanced AI-powered geotechnical design tool with comprehensive analytics
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Stone Column Design Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #66bb6a;
        --warning-color: #ffa726;
        --danger-color: #ef5350;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: gradient-animation 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1e2129 0%, #2d3139 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }
    
    h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102,126,234,0.3);
        margin-top: 2rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #2d3139 0%, #3d4149 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--secondary-color);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained TensorFlow model
    Returns: model object or None if loading fails
    """
    model_path = "stone_column_ann_model.h5"
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the current directory.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Parameter ranges: (min, max, default)
PARAMETER_RANGES = {
    'cu': (5.0, 40.0, 15.0),
    'D': (0.06, 0.8, 0.4),
    'L': (0.7, 12.0, 6.0),
    'sD': (2.0, 4.0, 2.5),
    'Eenc': (0.0, 20.0, 0.0)
}

# Parameter metadata
PARAMETER_INFO = {
    'cu': {
        'name': 'Undrained Shear Strength',
        'unit': 'kPa',
        'description': 'Soil resistance to deformation without drainage'
    },
    'D': {
        'name': 'Column Diameter',
        'unit': 'm',
        'description': 'Diameter of the stone column'
    },
    'L': {
        'name': 'Column Length',
        'unit': 'm',
        'description': 'Depth of the stone column'
    },
    'sD': {
        'name': 'Spacing Ratio (s/D)',
        'unit': '-',
        'description': 'Spacing between columns divided by diameter'
    },
    'Eenc': {
        'name': 'Encasement Stiffness',
        'unit': 'kN/m',
        'description': 'Stiffness of encasing material (0 for unencased)'
    }
}

# Output labels
OUTPUT_LABELS = ["Ultimate Stress", "Service Load", "Factor of Safety"]
OUTPUT_UNITS = ["kPa", "kN", "-"]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def predict_outcomes(model, cu, D, L, sD, Eenc):
    """
    Make predictions using the trained model
    Returns: (sigma, P10, FS)
    """
    x = np.array([[cu, D, L, sD, Eenc]], dtype=np.float32)
    predictions = model.predict(x, verbose=0)[0]
    return predictions[0], predictions[1], predictions[2]

def calculate_derived_parameters(cu, D, L, sD, Eenc, sigma, P10):
    """
    Calculate additional derived parameters
    """
    spacing = sD * D
    slenderness = L / D
    area_replacement = 1 / (sD ** 2)
    improvement_factor = sigma / cu if cu > 0 else 0
    load_per_length = P10 / L if L > 0 else 0
    
    return {
        'spacing': spacing,
        'slenderness': slenderness,
        'area_replacement': area_replacement,
        'improvement_factor': improvement_factor,
        'load_per_length': load_per_length
    }

def assess_safety(FS):
    """
    Assess design safety based on factor of safety
    Returns: (status, message, color)
    """
    if FS >= 2.5:
        return "Excellent", "Design exceeds requirements with strong safety buffer", "success"
    elif FS >= 2.0:
        return "Adequate", "Design meets standard safety requirements", "success"
    elif FS >= 1.5:
        return "Marginal", "Consider increasing column dimensions or reducing spacing", "warning"
    else:
        return "Insufficient", "Immediate design revision required", "error"

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>Stone Column Design Assistant</h1>
    <p>Advanced AI-Powered Geotechnical Design Tool with Comprehensive Analytics</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - INPUT PARAMETERS
# =============================================================================

with st.sidebar:
    st.header("Design Parameters")
    st.markdown("---")
    
    # Input fields
    cu = st.number_input(
        f"{PARAMETER_INFO['cu']['name']} ({PARAMETER_INFO['cu']['unit']})",
        min_value=PARAMETER_RANGES['cu'][0],
        max_value=PARAMETER_RANGES['cu'][1],
        value=PARAMETER_RANGES['cu'][2],
        step=0.5,
        help=PARAMETER_INFO['cu']['description']
    )
    
    D = st.number_input(
        f"{PARAMETER_INFO['D']['name']} ({PARAMETER_INFO['D']['unit']})",
        min_value=PARAMETER_RANGES['D'][0],
        max_value=PARAMETER_RANGES['D'][1],
        value=PARAMETER_RANGES['D'][2],
        step=0.01,
        format="%.2f",
        help=PARAMETER_INFO['D']['description']
    )
    
    L = st.number_input(
        f"{PARAMETER_INFO['L']['name']} ({PARAMETER_INFO['L']['unit']})",
        min_value=PARAMETER_RANGES['L'][0],
        max_value=PARAMETER_RANGES['L'][1],
        value=PARAMETER_RANGES['L'][2],
        step=0.1,
        format="%.1f",
        help=PARAMETER_INFO['L']['description']
    )
    
    sD = st.number_input(
        f"{PARAMETER_INFO['sD']['name']} ({PARAMETER_INFO['sD']['unit']})",
        min_value=PARAMETER_RANGES['sD'][0],
        max_value=PARAMETER_RANGES['sD'][1],
        value=PARAMETER_RANGES['sD'][2],
        step=0.1,
        format="%.1f",
        help=PARAMETER_INFO['sD']['description']
    )
    
    Eenc = st.number_input(
        f"{PARAMETER_INFO['Eenc']['name']} ({PARAMETER_INFO['Eenc']['unit']})",
        min_value=PARAMETER_RANGES['Eenc'][0],
        max_value=PARAMETER_RANGES['Eenc'][1],
        value=PARAMETER_RANGES['Eenc'][2],
        step=0.5,
        format="%.1f",
        help=PARAMETER_INFO['Eenc']['description']
    )
    
    st.info(f"**Calculated Spacing:** {sD * D:.2f} m")
    
    st.markdown("---")
    st.header("Analysis Options")
    
    enable_sensitivity = st.checkbox("Sensitivity Analysis", value=True)
    enable_heatmap = st.checkbox("Interaction Heatmap", value=True)
    enable_3d = st.checkbox("3D Surface Plot", value=True)
    enable_comparison = st.checkbox("Multi-Parameter Comparison", value=False)
    
    st.markdown("---")
    with st.expander("User Guide"):
        st.markdown("""
        **How to Use:**
        1. Enter design parameters in the fields above
        2. View prediction results in the main panel
        3. Enable analysis options for detailed insights
        4. Export results for documentation
        
        **Tips:**
        - Hover over input fields for descriptions
        - Use sensitivity analysis to optimize design
        - Check interaction heatmap for parameter relationships
        """)

# =============================================================================
# MAIN PREDICTIONS
# =============================================================================

if model is None:
    st.warning("Model not loaded. Please ensure the model file exists.")
    st.stop()

# Collect inputs
input_params = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}

# Make predictions
try:
    sigma, P10, FS = predict_outcomes(model, cu, D, L, sD, Eenc)
    derived = calculate_derived_parameters(cu, D, L, sD, Eenc, sigma, P10)
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.stop()

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

st.header("Prediction Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Ultimate Stress",
        value=f"{sigma:.2f} kPa",
        help="Maximum stress capacity of the stone column"
    )

with col2:
    st.metric(
        label="Service Load",
        value=f"{P10:.2f} kN",
        help="Load at 10% settlement threshold"
    )

with col3:
    st.metric(
        label="Factor of Safety",
        value=f"{FS:.2f}",
        delta="Safe" if FS >= 2.0 else "Low",
        delta_color="normal" if FS >= 2.0 else "inverse",
        help="Overall safety factor"
    )

with col4:
    st.metric(
        label="Slenderness (L/D)",
        value=f"{derived['slenderness']:.1f}",
        help="Length to diameter ratio"
    )

# Safety Assessment
st.subheader("Design Assessment")
status, message, color = assess_safety(FS)

if color == "success":
    st.success(f"**{status}:** {message}")
elif color == "warning":
    st.warning(f"**{status}:** {message}")
else:
    st.error(f"**{status}:** {message}")

# Design Information
st.subheader("Design Information")
col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Column Configuration:**
    - Type: {'Encased' if Eenc > 0 else 'Unencased'}
    - Area Replacement Ratio: {derived['area_replacement']:.3f}
    - Column Spacing: {derived['spacing']:.2f} m
    - Slenderness Ratio: {derived['slenderness']:.1f}
    """)

with col2:
    efficiency = "High" if derived['improvement_factor'] > 3 else "Moderate" if derived['improvement_factor'] > 2 else "Low"
    st.info(f"""
    **Performance Metrics:**
    - Stress Improvement Factor: {derived['improvement_factor']:.2f}x
    - Load per Unit Length: {derived['load_per_length']:.2f} kN/m
    - Design Efficiency: {efficiency}
    """)

st.markdown("---")

# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

if enable_sensitivity:
    st.header("Sensitivity Analysis")
    st.write("Analyze how each parameter affects the design outcomes")
    
    param_options = {
        PARAMETER_INFO[k]['name']: k for k in PARAMETER_INFO
    }
    
    selected_param = st.selectbox(
        "Select parameter to analyze:",
        list(param_options.keys()),
        help="Parameter to vary while keeping others constant"
    )
    param_key = param_options[selected_param]
    
    # Generate parameter range
    param_min, param_max, _ = PARAMETER_RANGES[param_key]
    param_values = np.linspace(param_min, param_max, 60)
    
    # Calculate predictions
    predictions = []
    for val in param_values:
        temp_inputs = input_params.copy()
        temp_inputs[param_key] = val
        x = np.array([[temp_inputs['cu'], temp_inputs['D'], temp_inputs['L'], 
                      temp_inputs['sD'], temp_inputs['Eenc']]])
        pred = model.predict(x, verbose=0)[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=OUTPUT_LABELS,
        horizontal_spacing=0.1
    )
    
    colors = ['#667eea', '#26c6da', '#ffa726']
    
    for i, (col_num, label) in enumerate([(1, OUTPUT_LABELS[0]), (2, OUTPUT_LABELS[1]), (3, OUTPUT_LABELS[2])]):
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=predictions[:, i],
                mode='lines',
                name=label,
                line=dict(color=colors[i], width=3),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(colors[i][j:j+2], 16) for j in (1, 3, 5)) + [0.2])}'
            ),
            row=1, col=col_num
        )
        
        # Add current value line
        fig.add_vline(
            x=input_params[param_key],
            line_dash="dash",
            line_color="red",
            line_width=2,
            row=1, col=col_num
        )
    
    # Add safety threshold for FS
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="green",
        line_width=2,
        row=1, col=3
    )
    
    # Update layout
    for i in range(1, 4):
        fig.update_xaxes(title_text=selected_param, row=1, col=i, gridcolor='rgba(255,255,255,0.1)')
    
    fig.update_yaxes(title_text=f"{OUTPUT_LABELS[0]} ({OUTPUT_UNITS[0]})", row=1, col=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text=f"{OUTPUT_LABELS[1]} ({OUTPUT_UNITS[1]})", row=1, col=2, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text=f"{OUTPUT_LABELS[2]} ({OUTPUT_UNITS[2]})", row=1, col=3, gridcolor='rgba(255,255,255,0.1)')
    
    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Statistical Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Minimum Ultimate Stress", f"{predictions[:, 0].min():.2f} kPa")
        st.metric("Maximum Ultimate Stress", f"{predictions[:, 0].max():.2f} kPa")
        st.metric("Range", f"{predictions[:, 0].max() - predictions[:, 0].min():.2f} kPa")
    
    with col2:
        st.metric("Minimum Service Load", f"{predictions[:, 1].min():.2f} kN")
        st.metric("Maximum Service Load", f"{predictions[:, 1].max():.2f} kN")
        st.metric("Range", f"{predictions[:, 1].max() - predictions[:, 1].min():.2f} kN")
    
    with col3:
        st.metric("Minimum Factor of Safety", f"{predictions[:, 2].min():.2f}")
        st.metric("Maximum Factor of Safety", f"{predictions[:, 2].max():.2f}")
        st.metric("Range", f"{predictions[:, 2].max() - predictions[:, 2].min():.2f}")
    
    st.markdown("---")

# =============================================================================
# INTERACTION HEATMAP
# =============================================================================

if enable_heatmap:
    st.header("Parameter Interaction Analysis")
    st.write("Explore how two parameters jointly affect design outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    param_options = {PARAMETER_INFO[k]['name']: k for k in PARAMETER_INFO}
    
    with col1:
        param1_name = st.selectbox("X-axis parameter:", list(param_options.keys()), key='hm_p1')
    with col2:
        param2_name = st.selectbox("Y-axis parameter:", list(param_options.keys()), index=1, key='hm_p2')
    with col3:
        output_choice = st.selectbox("Output to visualize:", OUTPUT_LABELS)
    
    if param1_name != param2_name:
        param1_key = param_options[param1_name]
        param2_key = param_options[param2_name]
        
        # Generate grid
        resolution = 40
        p1_values = np.linspace(PARAMETER_RANGES[param1_key][0], PARAMETER_RANGES[param1_key][1], resolution)
        p2_values = np.linspace(PARAMETER_RANGES[param2_key][0], PARAMETER_RANGES[param2_key][1], resolution)
        
        output_idx = OUTPUT_LABELS.index(output_choice)
        Z = np.zeros((len(p2_values), len(p1_values)))
        
        # Calculate predictions for grid
        for i, p2_val in enumerate(p2_values):
            for j, p1_val in enumerate(p1_values):
                temp_inputs = input_params.copy()
                temp_inputs[param1_key] = p1_val
                temp_inputs[param2_key] = p2_val
                x = np.array([[temp_inputs['cu'], temp_inputs['D'], temp_inputs['L'], 
                             temp_inputs['sD'], temp_inputs['Eenc']]])
                pred = model.predict(x, verbose=0)[0]
                Z[i, j] = pred[output_idx]
        
        # Create heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
    z=Z,
    x=p1_values,
    y=p2_values,
    colorscale='Plasma',
    colorbar=dict(
        title=dict(
            text=f"{output_choice} ({OUTPUT_UNITS[output_idx]})",
            side="right"
        ),
        thickness=20
    ),
    hovertemplate='%{x:.2f}<br>%{y:.2f}<br>%{z:.2f}<extra></extra>'
))

        
        # Add current design marker
        fig.add_scatter(
            x=[input_params[param1_key]],
            y=[input_params[param2_key]],
            mode='markers+text',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=3, color='white')
            ),
            text=['Current Design'],
            textposition='top center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name='Current Design',
            showlegend=True
        )
        
        fig.update_layout(
            title=f"<b>{output_choice} Interaction Map</b>",
            xaxis_title=param1_name,
            yaxis_title=param2_name,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal configuration
        optimal_idx = np.unravel_index(np.argmax(Z), Z.shape)
        optimal_p1 = p1_values[optimal_idx[1]]
        optimal_p2 = p2_values[optimal_idx[0]]
        optimal_val = Z[optimal_idx]
        
        st.success(f"""
        **Optimal Configuration for Maximum {output_choice}:**
        - {param1_name}: {optimal_p1:.2f} {PARAMETER_INFO[param1_key]['unit']}
        - {param2_name}: {optimal_p2:.2f} {PARAMETER_INFO[param2_key]['unit']}
        - {output_choice}: {optimal_val:.2f} {OUTPUT_UNITS[output_idx]}
        """)
    else:
        st.warning("Please select two different parameters for interaction analysis")
    
    st.markdown("---")

# =============================================================================
# 3D SURFACE PLOT
# =============================================================================

if enable_3d:
    st.header("3D Surface Visualization")
    st.write("Interactive three-dimensional view of parameter relationships")
    
    col1, col2, col3 = st.columns(3)
    
    param_options = {PARAMETER_INFO[k]['name']: k for k in PARAMETER_INFO}
    
    with col1:
        param_3d_1 = st.selectbox("X-axis:", list(param_options.keys()), key='3d_p1')
    with col2:
        param_3d_2 = st.selectbox("Y-axis:", list(param_options.keys()), index=1, key='3d_p2')
    with col3:
        output_3d = st.selectbox("Z-axis:", OUTPUT_LABELS)
    
    if param_3d_1 != param_3d_2:
        resolution_3d = st.slider(
            "Surface Resolution",
            min_value=15,
            max_value=40,
            value=25,
            help="Higher values create smoother surfaces but take longer to render"
        )
        
        p1_key = param_options[param_3d_1]
        p2_key = param_options[param_3d_2]
        
        # Generate mesh grid
        p1_vals = np.linspace(PARAMETER_RANGES[p1_key][0], PARAMETER_RANGES[p1_key][1], resolution_3d)
        p2_vals = np.linspace(PARAMETER_RANGES[p2_key][0], PARAMETER_RANGES[p2_key][1], resolution_3d)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        
        output_idx = OUTPUT_LABELS.index(output_3d)
        Z_3d = np.zeros_like(P1)
        
        # Calculate predictions for surface
        for i in range(P1.shape[0]):
            for j in range(P1.shape[1]):
                temp_inputs = input_params.copy()
                temp_inputs[p1_key] = P1[i, j]
                temp_inputs[p2_key] = P2[i, j]
                x = np.array([[temp_inputs['cu'], temp_inputs['D'], temp_inputs['L'], 
                             temp_inputs['sD'], temp_inputs['Eenc']]])
                pred = model.predict(x, verbose=0)[0]
                Z_3d[i, j] = pred[output_idx]
        
        # Create 3D surface
        fig_3d = go.Figure()
        
        fig_3d.add_trace(go.Surface(
            z=Z_3d,
            x=p1_vals,
            y=p2_vals,
            colorscale='Viridis',
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            colorbar=dict(
                title=f"{output_3d}<br>({OUTPUT_UNITS[output_idx]})",
                thickness=20
            )
        ))
        
        fig_3d.update_layout(
            title=f"<b>{output_3d} Surface Plot</b>",
            scene=dict(
                xaxis_title=param_3d_1,
                yaxis_title=param_3d_2,
                zaxis_title=f"{output_3d} ({OUTPUT_UNITS[output_idx]})",
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("Please select two different parameters for 3D visualization")
    
    st.markdown("---")

# =============================================================================
# EXPORT RESULTS
# =============================================================================

st.header("Export Results")

with st.expander("View and Download Results", expanded=False):
    # Create comprehensive results dataframe
    results_data = {
        'Parameter': [PARAMETER_INFO[k]['name'] for k in PARAMETER_INFO.keys()] + 
                     ['', 'Ultimate Stress', 'Service Load', 'Factor of Safety', '',
                      'Column Spacing', 'Slenderness Ratio', 'Area Replacement Ratio',
                      'Improvement Factor', 'Load per Length'],
        'Value': [input_params[k] for k in PARAMETER_INFO.keys()] + 
                 ['', sigma, P10, FS, '',
                  derived['spacing'], derived['slenderness'], derived['area_replacement'],
                  derived['improvement_factor'], derived['load_per_length']],
        'Unit': [PARAMETER_INFO[k]['unit'] for k in PARAMETER_INFO.keys()] + 
                ['', 'kPa', 'kN', '-', '', 'm', '-', '-', '-', 'kN/m']
    }
    
    results_df = pd.DataFrame(results_data)
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stone_column_results_{timestamp}.csv"
    
    # Download button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        help="Download all results and parameters to CSV file"
    )
    
    # Additional export info
    st.info(f"""
    **Export Information:**
    - Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - Design Status: {status}
    - Column Type: {'Encased' if Eenc > 0
        else 'Unencased'}
    - Model Version: ANN v1.0
    """)

# =============================================================================
# DESIGN RECOMMENDATIONS
# =============================================================================

st.header("AI Design Recommendations")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.subheader("Structural Recommendations")

    if FS < 1.5:
        st.error("""
        **Critical Design Issues Detected**
        - Increase column diameter (D)
        - Reduce spacing ratio (s/D)
        - Consider adding geosynthetic encasement
        - Increase column length (L)
        """)
    elif FS < 2.0:
        st.warning("""
        **Design Optimization Suggested**
        - Slightly increase diameter or length
        - Reduce spacing by 10–15%
        - Consider partial encasement
        """)
    else:
        st.success("""
        **Design is Structurally Sound**
        - Current configuration meets safety criteria
        - Further optimization possible for cost efficiency
        """)

with rec_col2:
    st.subheader("Economic Optimization")

    cost_index = (D**2) * L / (sD**2)

    if cost_index > 2.0:
        st.warning("""
        **High Material Usage**
        - Consider reducing diameter
        - Increase spacing slightly if FS allows
        """)
    else:
        st.success("""
        **Cost-Efficient Design**
        - Material usage is optimized
        - Good balance between safety and economy
        """)

# =============================================================================
# DESIGN SUMMARY CARD
# =============================================================================

st.header("Final Design Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Design Status", status)
    st.metric("Safety Factor", f"{FS:.2f}")
    st.metric("Column Type", "Encased" if Eenc > 0 else "Unencased")

with summary_col2:
    st.metric("Ultimate Stress", f"{sigma:.1f} kPa")
    st.metric("Service Load", f"{P10:.1f} kN")
    st.metric("Improvement Factor", f"{derived['improvement_factor']:.2f}x")

with summary_col3:
    st.metric("Spacing", f"{derived['spacing']:.2f} m")
    st.metric("Slenderness", f"{derived['slenderness']:.1f}")
    st.metric("Area Replacement", f"{derived['area_replacement']:.3f}")

# =============================================================================
# DISCLAIMER
# =============================================================================

st.markdown("---")
st.markdown("""
<div class="info-card">
<b>⚠ Engineering Disclaimer</b><br><br>
This tool provides AI-assisted preliminary design guidance only.  
Final design decisions must be verified by a qualified geotechnical engineer using:
<ul>
<li>Site-specific investigation data</li>
<li>Relevant national/international design codes</li>
<li>Professional engineering judgment</li>
</ul>
The developers assume no liability for misuse of this tool.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div style="text-align:center; opacity:0.7; margin-top:2rem;">
Stone Column Design Assistant © 2026<br>
Developed for academic and professional geotechnical applications
</div>
""", unsafe_allow_html=True)
