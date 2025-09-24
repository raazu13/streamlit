import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import subprocess
import sys

# Define the model and scaler file names
MODEL_FILE = 'random_forest_model.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'feature_names.pkl'

# --- 1. Training Logic ---
def run_training_script():
    """Runs the train_and_save_model.py script to generate model files."""
    st.info("Model files not found. Running the training script to generate them...")
    
    # Use subprocess to run the training script
    # The output is captured to be displayed in the Streamlit app
    process = subprocess.run([sys.executable, 'train_and_save_model.py'], capture_output=True, text=True)
    
    if process.returncode == 0:
        st.success("Training script executed successfully.")
        st.code(process.stdout)
    else:
        st.error("Error during training script execution.")
        st.code(process.stderr)
        st.stop()
        
# Check if model files exist, if not, run the training script
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(FEATURES_FILE):
    run_training_script()

# --- 2. Load the pre-trained model and preprocessing objects ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        feature_names = joblib.load(FEATURES_FILE)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found even after training.")
        st.error("This is an unexpected error. Please check your file paths.")
        return None, None, None

model, scaler, feature_names = load_resources()

# Exit if resources couldn't be loaded
if model is None or scaler is None or feature_names is None:
    st.stop()


# --- 3. Set up the Streamlit page layout and title ---
st.set_page_config(layout="wide")

# Main content
st.title('Exoplanet Classification App 🪐')
st.markdown("""
    This app predicts whether an exoplanet candidate is **'CONFIRMED'** or a **'FALSE POSITIVE'**
    using a Random Forest model trained on a subset of the Kepler Exoplanet Search Results data.
""")


# --- 4. User Input ---
st.header('Enter Exoplanet Candidate Data')
st.markdown("Please input the following physical parameters:")

# Create input widgets in the main content area
col1, col2, col3 = st.columns(3)
with col1:
    period = st.number_input('Orbital Period (days)', value=10.2, format="%.4f")
    prad = st.number_input('Planetary Radius (Earth Radii)', value=1.5, format="%.2f")
with col2:
    teq = st.number_input('Equilibrium Temperature (K)', value=500, format="%d")
    insol = st.number_input('Stellar Insolation (Earth Flux)', value=100.0, format="%.2f")
with col3:
    srad = st.number_input('Stellar Radius (Solar Radii)', value=1.0, format="%.2f")
    slogg = st.number_input('Stellar Surface Gravity (log10(cm/s²))', value=4.5, format="%.2f")

# Add a submit button
submitted = st.button("Predict")


# --- 5. Prediction logic and results display ---
if submitted:
    st.subheader("Prediction Result")

    # Create a DataFrame from the user's input
    new_data = pd.DataFrame([[period, prad, teq, insol, srad, slogg]], columns=feature_names)
    
    # Scale the new data using the same scaler from training
    new_data_scaled = scaler.transform(new_data)
    
    # Make the prediction and get probabilities
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)

    # Display the result based on the prediction
    if prediction[0] == 1:
        st.success('Prediction: CONFIRMED Exoplanet 🤩')
        st.write(f"Confidence Score: **{prediction_proba[0][1]:.2%}**")
    else:
        st.error('Prediction: FALSE POSITIVE 😞')
        st.write(f"Confidence Score: **{prediction_proba[0][0]:.2%}**")

    st.markdown("---")
    st.subheader("Input Data Summary")
    st.dataframe(new_data)
