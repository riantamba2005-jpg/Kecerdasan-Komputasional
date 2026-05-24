import numpy as np
import joblib
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Sistem Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide",
)


def local_css():
    st.markdown(
        """
        <style>
        .main-container {background: linear-gradient(180deg, #f8fafc 0%, #e2f0f9 100%);}
        .title {color: #0b3d91;}
        .card {background: white; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08);}
        .metric-card {background: #0b3d91; color: white; border-radius: 16px; padding: 16px;}
        .small-note {color: #475569;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts(model_path, scaler_path, model_mtime, scaler_mtime):
    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def encode_smoking(smoking_history: str) -> int:
    smoking_map = {
        "never": 0,
        "current": 1,
        "formerly smoked": 2,
        "former": 2,
        "no info": 0,
        "unknown": 0,
    }
    return smoking_map.get(smoking_history.lower().strip(), 0)


def build_feature_vector(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age,
    hba1c,
    hypertension,
    heart_disease,
    gender,
    smoking_history,
):
    gender = gender.title().strip()
    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 if gender == "Female" else 0
    smoking_encoded = encode_smoking(smoking_history)

    return np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
        hba1c,
        int(hypertension),
        int(heart_disease),
        gender_male,
        gender_female,
        smoking_encoded,
    ]).reshape(1, -1)


def user_input_form():
    with st.sidebar:
        st.header("Input Data Kesehatan")
        age = st.number_input("Usia (Age)", min_value=0, max_value=120, value=33)
        gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Unknown"])
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        glucose = st.number_input("Glukosa (Glucose)", min_value=0, max_value=300, value=120)
        hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.7, format="%.1f")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        smoking_history = st.selectbox(
            "Kebiasaan Merokok",
            ["Never", "Current", "Formerly Smoked", "No Info"],
        )

        st.markdown("---")
        with st.expander("Input tambahan (opsional)"):
            pregnancies = st.number_input("Kehamilan (Pregnancies)", min_value=0, max_value=20, value=0, step=1)
            blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
            dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        submit_button = st.button("Prediksi Risiko Diabetes")

    features = build_feature_vector(
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
        hba1c,
        hypertension,
        heart_disease,
        gender,
        smoking_history,
    )

    inputs = {
        "Age": int(age),
        "Gender": gender,
        "BMI": float(bmi),
        "Glucose": int(glucose),
        "HbA1c Level": float(hba1c),
        "Hypertension": bool(hypertension),
        "Heart Disease": bool(heart_disease),
        "Smoking History": smoking_history,
        "Pregnancies": int(pregnancies),
        "BloodPressure": int(blood_pressure),
        "SkinThickness": int(skin_thickness),
        "Insulin": int(insulin),
        "DiabetesPedigreeFunction": float(dpf),
    }

    return features, submit_button, inputs


def main():
    local_css()
    base_path = Path(__file__).resolve().parent.parent
    model_path = base_path / "models" / "diabetes_ann.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"
    model_mtime = model_path.stat().st_mtime if model_path.exists() else None
    scaler_mtime = scaler_path.stat().st_mtime if scaler_path.exists() else None
    model, scaler = load_artifacts(model_path, scaler_path, model_mtime, scaler_mtime)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("# Sistem Prediksi Risiko Diabetes Melitus")
    st.markdown("### Aplikasi web ini menggunakan Artificial Neural Network (ANN) untuk memprediksi risiko diabetes berdasarkan data kesehatan pengguna.")
    st.markdown("</div>", unsafe_allow_html=True)

    if model is None or scaler is None:
        st.warning(
            "Model belum tersedia. Jalankan `python src/train.py` terlebih dahulu untuk membuat model dan scaler di folder `models/`."
        )
        return

    features, submit_button, inputs = user_input_form()

    left, right = st.columns([2, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Ringkasan Input")
        personal, health = st.columns(2)

        with personal:
            st.markdown("**Data Pribadi**")
            st.write(f"- **Usia**: {inputs['Age']}")
            st.write(f"- **Jenis Kelamin**: {inputs['Gender']}")
            st.write(f"- **BMI**: {inputs['BMI']}")
            st.write(f"- **Glukosa**: {inputs['Glucose']}")
            st.write(f"- **HbA1c Level**: {inputs['HbA1c Level']}")

        with health:
            st.markdown("**Kondisi Kesehatan**")
            st.write(f"- **Hypertension**: {'Ya' if inputs['Hypertension'] else 'Tidak'}")
            st.write(f"- **Heart Disease**: {'Ya' if inputs['Heart Disease'] else 'Tidak'}")
            st.write(f"- **Smoking History**: {inputs['Smoking History']}")
            st.write(f"- **Pregnancies**: {inputs['Pregnancies']}")
            st.write(f"- **Blood Pressure**: {inputs['BloodPressure']}")
            st.write(f"- **Skin Thickness**: {inputs['SkinThickness']}")
            st.write(f"- **Insulin**: {inputs['Insulin']}")
            st.write(f"- **DPF**: {inputs['DiabetesPedigreeFunction']}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Hasil Prediksi")
        if submit_button:
            try:
                scaled_input = scaler.transform(features.astype(float))
            except Exception as exc:
                st.error(f"Terjadi kesalahan transformasi input: {exc}")
                return
            prediction_prob = float(model.predict_proba(scaled_input)[:, 1][0])
            confidence = round(prediction_prob * 100, 2)

            if prediction_prob < 0.4:
                risk_label = "Risiko Rendah"
                risk_message = "Hasil prediksi menunjukkan risiko rendah. Pertahankan gaya hidup sehat."
                metric_color = "green"
            elif prediction_prob < 0.7:
                risk_label = "Risiko Sedang"
                risk_message = "Hasil prediksi menunjukkan risiko sedang. Perhatikan pola makan dan kontrol kesehatan secara teratur."
                metric_color = "yellow"
            else:
                risk_label = "Risiko Tinggi"
                risk_message = "Hasil prediksi menunjukkan risiko tinggi. Disarankan untuk segera konsultasi dengan tenaga medis."
                metric_color = "red"

            st.metric("Prediksi", risk_label)
            st.metric("Probabilitas", f"{confidence}%")
            st.write("---")
            if risk_label == "Risiko Tinggi":
                st.error(risk_message)
            elif risk_label == "Risiko Sedang":
                st.warning(risk_message)
            else:
                st.success(risk_message)
        else:
            st.write("Tekan tombol prediksi di sidebar untuk melihat hasil.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini memudahkan pengguna untuk memprediksi kemungkinan diabetes dengan input data kesehatan penting. "
        "Gunakan angka yang realistis agar hasil prediksi lebih representatif."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Detail Model dan Informasi"):
        st.write("- ANN 3 hidden layer")
        st.write("- Scikit-Learn MLPClassifier dengan backpropagation")
        st.write("- Aktivasi ReLU pada hidden layer")
        st.write("- Aktivasi logistic pada output layer")
        st.write("- Optimizer: SGD, learning rate 0.01")
        st.write("- Model dan scaler dimuat dari folder `models/`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Cara Pakai")
    st.sidebar.write("1. Masukkan data kesehatan di sidebar.")
    st.sidebar.write("2. Klik tombol Prediksi Risiko Diabetes.")
    st.sidebar.write("3. Lihat hasil prediksi di panel utama.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Info Tambahan")
    st.sidebar.markdown("- Pastikan model sudah dilatih dengan `python src/train.py`.")
    st.sidebar.markdown("- Jika hasil tidak muncul, restart server Streamlit.")


if __name__ == "__main__":
    main()
