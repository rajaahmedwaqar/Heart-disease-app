"""
Heart Disease Prediction System - Streamlit web app
Loads the model, imputer, scaler and feature list saved by the notebook (Step 6.4).
Takes exactly four inputs and shows the model's prediction of heart disease.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


# ---------- Load the files created by the notebook ----------
@st.cache_resource
def load_files():
    model = pickle.load(open("heart_disease_model.pkl", "rb"))
    imputer = pickle.load(open("imputer.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("selected_features.pkl", "rb"))
    return model, imputer, scaler, features


try:
    model, imputer, scaler, features = load_files()
except FileNotFoundError as error:
    st.error(f"A required file is missing: {error.filename}. "
             "Upload all .pkl files to the same folder as app.py.")
    st.stop()

if len(features) != 4:
    st.error("The saved feature list must contain exactly 4 attributes.")
    st.stop()

# ---------- Page ----------
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the four medical values below. The model (Support Vector Classifier trained on the "
         "UCI Cleveland heart disease data) predicts whether heart disease is present.")

thal_options = {"Normal (3)": 3, "Fixed defect (6)": 6, "Reversible defect (7)": 7}

thal_label = st.selectbox("Thallium stress-test result (thal)", list(thal_options.keys()))
ca = st.selectbox("Number of major blood vessels coloured by fluoroscopy (ca)", [0, 1, 2, 3])
thalach = st.number_input("Maximum heart rate achieved, beats per minute (thalach)",
                          min_value=60, max_value=220, value=150, step=1)
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)",
                          min_value=0.0, max_value=7.0, value=1.0, step=0.1)

if st.button("Predict"):
    values = {"thal": thal_options[thal_label], "ca": ca,
              "thalach": thalach, "oldpeak": oldpeak}

    # Check the inputs are present and valid numbers
    if any(v is None or not np.isfinite(float(v)) for v in values.values()):
        st.error("Please enter a valid value in every field.")
        st.stop()

    # Same four columns, same order, same preprocessing as training
    user_input = pd.DataFrame([values])[features]
    prepared = pd.DataFrame(scaler.transform(imputer.transform(user_input)), columns=features)
    prediction = int(model.predict(prepared)[0])

    st.subheader("Model prediction of heart disease")
    if prediction == 1:
        st.error("Heart disease predicted (1)")
    else:
        st.success("No heart disease predicted (0)")

st.caption("Educational project (AIC354 Assignment 2). Not a medical diagnosis tool.")
