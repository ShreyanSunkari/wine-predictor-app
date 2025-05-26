import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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


# --- Sliders (Simplified Version) ---
# We have removed the 'key' arguments and all session state logic for now.
col1_sliders, col2_sliders = st.columns(2)
with col1_sliders:
    fixed_acidity = st.slider('Fixed Acidity (g/dm³)', 4.0, 16.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity (g/dm³)', 0.1, 1.6, 0.7)
    citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, 0.0)
    residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.9, 16.0, 1.9)
    chlorides = st.slider('Chlorides (g/dm³)', 0.01, 0.62, 0.076)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide (mg/dm³)', 1, 72, 11)
with col2_sliders:
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide (mg/dm³)', 6, 289, 34)
    density = st.slider('Density (g/cm³)', 0.9900, 1.0040, 0.9978, step=0.0001, format="%.4f")
    # I've renamed the variable 'ph' to 'ph_slider' to be extra safe.
    ph_slider = st.slider('pH', 2.70, 4.00, 3.51)
    sulphates = st.slider('Sulphates (g/dm³)', 0.30, 2.00, 0.56)
    alcohol = st.slider('Alcohol (% vol.)', 8.0, 15.0, 9.4)


# --- Prediction Logic ---
if st.button('Predict Wine Quality'):
    # This robust logic creates a dictionary from the slider values and builds a DataFrame.
    input_dict = {
        'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity,
        'citric acid': citric_acid, 'residual sugar': residual_sugar,
        'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide, 'density': density,
        'pH': ph_slider, 'sulphates': sulphates, 'alcohol': alcohol
    }
    input_data = pd.DataFrame([input_dict])

    # Ensure the columns are in the exact order the model expects
    ordered_feature_names = scaler.feature_names_in_
    input_data = input_data[ordered_feature_names]

    # Scale the data and make a prediction
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 'good':
        st.success(f"This wine is predicted to be of **GOOD** quality.")
    else:
        st.error(f"This wine is predicted to be of **NOT GOOD** quality.")
    st.write("Prediction Confidence:")
    st.info(f"Confidence for 'Good' quality: **{prediction_proba[0][1]*100:.2f}%**")
    st.info(f"Confidence for 'Not Good' quality: **{prediction_proba[0][0]*100:.2f}%**")

# --- Feature Importance Section ---
st.write("---")
with st.expander("Click here to see what makes a quality wine"):
    st.write("This chart shows which chemical properties have the biggest impact on wine quality according to the prediction model.")
    feature_importances = pd.DataFrame({
        'feature': scaler.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    fig, ax = plt.subplots()
    ax.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
