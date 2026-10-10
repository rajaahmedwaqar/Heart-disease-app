"""
Heart Disease Prediction System - Streamlit web app
Loads the model, imputer, scaler and feature list saved by the notebook (Step 6.4).
Takes exactly four inputs and shows the model's prediction of heart disease.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# ---------- Styling (colours, cards, buttons) ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 820px;}

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 18px; padding: 2rem 1.5rem; text-align: center; color: #ffffff;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18); margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.1rem; color: #ffffff;}
.hero p {margin: 0.5rem 0 0 0; font-size: 1.05rem; color: #d6e4ea;}
.pulse {display: inline-block; animation: beat 1.3s infinite;}
@keyframes beat {0%,100% {transform: scale(1);} 50% {transform: scale(1.25);}}

.section-title {font-size: 1.15rem; font-weight: 700; margin: 0.6rem 0 0.2rem 0; color: #c0392b;}

div.stButton > button {
    width: 100%; background: linear-gradient(90deg, #c0392b, #e74c3c); color: #ffffff;
    border: none; border-radius: 12px; padding: 0.75rem 1rem; font-size: 1.1rem; font-weight: 700;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 6px 16px rgba(192,57,43,0.4); color: #ffffff;
}

.result {border-radius: 16px; padding: 1.4rem; text-align: center; margin-top: 1rem;}
.result.bad  {background: #fdecea; border: 2px solid #e74c3c; color: #922b21;}
.result.good {background: #e8f6ef; border: 2px solid #27ae60; color: #1e6b41;}
.result .label {font-size: 0.95rem; opacity: 0.85;}
.result .value {font-size: 1.7rem; font-weight: 800; margin-top: 0.2rem;}

.chips {display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; margin-top: 0.8rem;}
.chip {background: rgba(128,128,128,0.15); border-radius: 999px; padding: 0.25rem 0.8rem; font-size: 0.9rem;}
.small {text-align: center; font-size: 0.85rem; opacity: 0.7; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)


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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this app")
    st.write("A Support Vector Classifier trained on the UCI **Cleveland** heart disease data "
             "(303 patients) predicts whether heart disease is present from **four** inputs.")
    st.write("**Test accuracy:** 0.84  \n**Test patients:** 61")
    st.write("**How to use**  \n1. Enter the four values  \n2. Click **Predict**")
    st.warning("Educational project. Not a medical diagnosis tool.")
    st.caption("FA24-BSE-070 · M.Ahmed Waqar · AIC354")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1><span class="pulse">❤️</span> Heart Disease Prediction</h1>
  <p>Enter four medical values and the trained model predicts whether heart disease is present.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
st.markdown('<div class="section-title">🩺 Patient values</div>', unsafe_allow_html=True)

thal_options = {"Normal (3)": 3, "Fixed defect (6)": 6, "Reversible defect (7)": 7}

left, right = st.columns(2)
with left:
    thal_label = st.selectbox("Thallium stress-test result (thal)", list(thal_options.keys()),
                              help="Result of the thallium stress test: 3 normal, 6 fixed defect, 7 reversible defect.")
    ca = st.selectbox("Major vessels coloured by fluoroscopy (ca)", [0, 1, 2, 3],
                      help="Number of major blood vessels (0 to 3) coloured by fluoroscopy.")
with right:
    thalach = st.number_input("Maximum heart rate, bpm (thalach)",
                              min_value=60, max_value=220, value=150, step=1,
                              help="Highest heart rate reached during the stress test (60 to 220).")
    oldpeak = st.number_input("ST depression (oldpeak)",
                              min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                              help="ST depression induced by exercise relative to rest (0.0 to 7.0).")

st.write("")
predict_clicked = st.button("🔍 Predict")

# ---------- Prediction ----------
if predict_clicked:
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

    if prediction == 1:
        css_class, text = "bad", "⚠️ Heart disease predicted (1)"
    else:
        css_class, text = "good", "✅ No heart disease predicted (0)"

    st.markdown(f"""
    <div class="result {css_class}">
      <div class="label">Model prediction of heart disease</div>
      <div class="value">{text}</div>
    </div>
    <div class="chips">
      <span class="chip">thal: {values['thal']}</span>
      <span class="chip">ca: {values['ca']}</span>
      <span class="chip">thalach: {values['thalach']}</span>
      <span class="chip">oldpeak: {values['oldpeak']}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="small">Educational project (AIC354 Assignment 2). '
            'Not a medical diagnosis tool.</div>', unsafe_allow_html=True)
