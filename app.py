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


# --- Preset Buttons (NEW FEATURE) ---

# Define default values for all sliders
default_values = {
    'fixed_acidity': 7.4, 'volatile_acidity': 0.7, 'citric_acid': 0.0,
    'residual_sugar': 1.9, 'chlorides': 0.076, 'free_sulfur_dioxide': 11.0,
    'total_sulfur_dioxide': 34.0, 'density': 0.9978, 'ph': 3.51,
    'sulphates': 0.56, 'alcohol': 9.4
}

# Initialize session state for each slider if it doesn't exist
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Define the callback functions to update session state with preset values
def load_good_wine_preset():
    # A real example of a "good" wine from the dataset
    st.session_state.fixed_acidity = 6.6
    st.session_state.volatile_acidity = 0.52
    st.session_state.citric_acid = 0.08
    st.session_state.residual_sugar = 2.4
    st.session_state.chlorides = 0.07
    st.session_state.free_sulfur_dioxide = 13.0
    st.session_state.total_sulfur_dioxide = 32.0
    st.session_state.density = 0.9955
    st.session_state.ph = 3.42
    st.session_state.sulphates = 0.62
    st.session_state.alcohol = 11.4

def load_not_good_wine_preset():
    # A real example of a "not good" wine from the dataset
    st.session_state.fixed_acidity = 7.4
    st.session_state.volatile_acidity = 0.7
    st.session_state.citric_acid = 0.0
    st.session_state.residual_sugar = 1.9
    st.session_state.chlorides = 0.076
    st.session_state.free_sulfur_dioxide = 11.0
    st.session_state.total_sulfur_dioxide = 34.0
    st.session_state.density = 0.9978
    st.session_state.ph = 3.51
    st.session_state.sulphates = 0.56
    st.session_state.alcohol = 9.4

st.write("---")
st.subheader("Try an Example Preset")
col1_ex, col2_ex = st.columns(2)
with col1_ex:
    st.button(
        "Load 'Good' Wine Example",
        on_click=load_good_wine_preset,
        use_container_width=True
    )
with col2_ex:
    st.button(
        "Load 'Not Good' Wine Example",
        on_click=load_not_good_wine_preset,
        use_container_width=True
    )
st.write("---")

# --- Sliders (MODIFIED TO USE SESSION STATE) ---
col1_sliders, col2_sliders = st.columns(2)
with col1_sliders:
    fixed_acidity = st.slider('Fixed Acidity (g/dm³)', 4.0, 16.0, key='fixed_acidity')
    volatile_acidity = st.slider('Volatile Acidity (g/dm³)', 0.1, 1.6, key='volatile_acidity')
    citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, key='citric_acid')
    residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.9, 16.0, key='residual_sugar')
    chlorides = st.slider('Chlorides (g/dm³)', 0.01, 0.62, key='chlorides')
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide (mg/dm³)', 1, 72, key='free_sulfur_dioxide')
with col2_sliders:
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide (mg/dm³)', 6, 289, key='total_sulfur_dioxide')
    density = st.slider('Density (g/cm³)', 0.9900, 1.0040, step=0.0001, format="%.4f", key='density')
    ph = st.slider('pH', 2.70, 4.00, key='ph')
    sulphates = st.slider('Sulphates (g/dm³)', 0.30, 2.00, key='sulphates')
    alcohol = st.slider('Alcohol (% vol.)', 8.0, 15.0, key='alcohol')


# --- Prediction Logic ---
if st.button('Predict Wine Quality'):
    # When predicting, we now use the values from session state
    feature_names = list(default_values.keys())
    input_values = [st.session_state[key] for key in feature_names]

    input_data = pd.DataFrame([input_values], columns=feature_names)
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
