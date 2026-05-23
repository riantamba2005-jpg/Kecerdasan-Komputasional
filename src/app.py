import numpy as np
import joblib
import streamlit as st
from pathlib import Path


@st.cache_resource
def load_artifacts():
    base_path = Path(__file__).resolve().parent.parent
    model_path = base_path / "models" / "diabetes_ann.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def user_input_form():
    st.sidebar.header("Input Data Kesehatan")
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.number_input("BloodPressure", min_value=0, max_value=140, value=70)
    skin_thickness = st.sidebar.number_input("SkinThickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=33)

    submit_button = st.sidebar.button("Prediksi Risiko Diabetes")

    features = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
    ]).reshape(1, -1)

    return features, submit_button


def main():
    st.set_page_config(page_title="Sistem Prediksi Risiko Diabetes", layout="centered")
    st.title("Sistem Prediksi Risiko Diabetes Melitus")
    st.markdown(
        "Aplikasi web ini menggunakan model Artificial Neural Network (ANN) dengan algoritma Backpropagation untuk memprediksi risiko diabetes.")

    model, scaler = load_artifacts()
    if model is None or scaler is None:
        st.error(
            "Model belum tersedia. Jalankan `python src/train.py` terlebih dahulu untuk membuat model dan scaler di folder `models/`."
        )
        return

    features, submit_button = user_input_form()
    st.write("### Data Kesehatan Pengguna")
    st.write(
        {
            "Pregnancies": int(features[0, 0]),
            "Glucose": int(features[0, 1]),
            "BloodPressure": int(features[0, 2]),
            "SkinThickness": int(features[0, 3]),
            "Insulin": int(features[0, 4]),
            "BMI": float(features[0, 5]),
            "DiabetesPedigreeFunction": float(features[0, 6]),
            "Age": int(features[0, 7]),
        }
    )

    if submit_button:
        scaled_input = scaler.transform(features)
        prediction_prob = float(model.predict_proba(scaled_input)[:, 1][0])
        risk_label = "Risiko Tinggi" if prediction_prob >= 0.5 else "Risiko Rendah"
        confidence = round(prediction_prob * 100, 2)

        st.success(f"Prediksi: {risk_label}")
        st.write(f"Probabilitas diabetes: {confidence}%")
        st.info(
            "Interpretasi: nilai probabilitas >= 50% menunjukkan risiko yang lebih tinggi untuk diabetes melitus."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Informasi Model")
    st.sidebar.markdown("- ANN dengan 3 hidden layer")
    st.sidebar.markdown("- Scikit-Learn MLPClassifier dengan backpropagation")
    st.sidebar.markdown("- Fungsi aktivasi ReLU di hidden layer")
    st.sidebar.markdown("- Fungsi aktivasi logistic di output layer")
    st.sidebar.markdown("- Optimizer: SGD dengan learning rate 0.01")
    st.sidebar.markdown("- Loss: binary crossentropy")


if __name__ == "__main__":
    main()
