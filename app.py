import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt # Import the new library

# --- Load the Trained Model and Scaler ---
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler files not found. Make sure 'model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Red Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)


# --- Font and Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    h1, h3 {
        font-family: 'Montserrat', sans-serif !important;
    }
    .vega-actions {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Web App Interface ---
st.title("🍷 Red Wine Quality Predictor")
st.write(
    "This app predicts the quality of a red wine ('good' or 'not good') based on its chemical properties. "
    "Adjust the sliders below to match your wine's characteristics and click 'Predict'!"
)

# Sliders and prediction logic (no changes here)
col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.slider('Fixed Acidity (g/dm³)', 4.0, 16.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity (g/dm³)', 0.1, 1.6, 0.7)
    citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, 0.0)
    residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.9, 16.0, 1.9)
    chlorides = st.slider('Chlorides (g/dm³)', 0.01, 0.62, 0.076)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide (mg/dm³)', 1, 72, 11)
with col2:
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide (mg/dm³)', 6, 289, 34)
    density = st.slider('Density (g/cm³)', 0.9900, 1.0040, 0.9978, step=0.0001, format="%.4f")
    ph = st.slider('pH', 2.70, 4.00, 3.51)
    sulphates = st.slider('Sulphates (g/dm³)', 0.30, 2.00, 0.56)
    alcohol = st.slider('Alcohol (% vol.)', 8.0, 15.0, 9.4)

if st.button('Predict Wine Quality'):
    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]], columns=feature_names)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    st.subheader("Prediction Result")
    if prediction[0] == 'good':
        st.success(f"This wine is predicted to be of **GOOD** quality.")
    else:
        st.error(f"This wine is predicted to be of **NOT GOOD** quality.")
    st.write("Prediction Confidence:")
    st.info(f"Confidence for 'Good' quality: **{prediction_proba[0][1]*100:.2f}%**")
    st.info(f"Confidence for 'Not Good' quality: **{prediction_proba[0][0]*100:.2f}%**")

# --- Feature Importance Section (Using Matplotlib) ---
st.write("---") 

with st.expander("Click here to see what makes a quality wine"):
    st.write("This chart shows which chemical properties have the biggest impact on wine quality according to the prediction model.")
    
    feature_importances = pd.DataFrame({
        'feature': scaler.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # --- THIS IS THE NEW MATPLOTLIB CHART CODE ---
    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()
    
    # Create the horizontal bar chart
    ax.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue')
    
    # Invert y-axis to have the most important feature on top
    ax.invert_yaxis()
    
    # Set labels
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    
    # Use Streamlit to display the Matplotlib figure
    st.pyplot(fig)
