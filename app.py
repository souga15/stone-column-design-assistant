import streamlit as st
import numpy as np
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("stone_column_ann_model.h5")

st.set_page_config(page_title="Stone Column Design Assistant", layout="centered")

st.title("Stone Column Design Assistant")
st.write("Enter design parameters to predict stone column performance.")

# -----------------------------
# User Inputs
# -----------------------------
cu = st.number_input("Undrained shear strength cu (kPa)", min_value=5.0, max_value=40.0, value=15.0, step=0.5)
D  = st.number_input("Column diameter D (m)", min_value=0.06, max_value=0.8, value=0.4, step=0.01)
L  = st.number_input("Column length L (m)", min_value=0.7, max_value=12.0, value=6.0, step=0.1)
sD = st.number_input("Spacing ratio s/D", min_value=2.0, max_value=4.0, value=2.5, step=0.1)
Eenc = st.number_input("Encasement stiffness (kN/m)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict performance"):
    x = np.array([[cu, D, L, sD, Eenc]])
    preds = model.predict(x, verbose=0)[0]

    sigma, P10, FS = preds

    st.success("Prediction Results")
    st.write(f"**Ultimate stress:** {sigma:.2f} kPa")
    st.write(f"**Service load:** {P10:.2f} kN")
    st.write(f"**Factor of safety:** {FS:.2f}")
