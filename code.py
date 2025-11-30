import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
import math
import os
import io

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Rujukan Cerdas (TACC Hybrid)",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Fungsi Load Data ---
@st.cache_data
def load_dataset(uploaded_file=None):
    """
    Memuat data dengan prioritas:
    1. File yang diupload user via UI (jika ada).
    2. File 'disease_diagnosis.csv' yang ada di GitHub/Folder Lokal.
    """
    df = pd.DataFrame()

    # A. Cek Upload User (Prioritas Utama)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast("Menggunakan file dari upload.", icon="📂")
            return df
        except Exception as e:
            st.error(f"Error membaca file upload: {e}")

    # B. Cek File Lokal / GitHub (Prioritas Kedua)
    # Streamlit Cloud akan membaca ini sebagai file lokal
    local_path = "disease_diagnosis.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            # st.toast("Dataset utama dimuat.", icon="✅")
            return df
        except Exception as e:
            st.error(f"Error membaca file dataset: {e}")

    # C. Jika file tidak ditemukan sama sekali
    return df

# --- 3. Preprocessing & Feature Engineering ---
def calculate_vital_score(row):
    """
    Menghitung Vital Score (M).
    Skor bertambah jika tanda vital di luar batas normal.
    """
    score = 0
    # Heart Rate (Normal: 60-100)
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr > 100 or hr < 60:
            score += 2
    except: pass
    
    # Suhu (Normal: 36.5 - 37.5)
    try:
        temp = float(row['Body_Temperature_C'])
        if temp > 38.0 or temp < 36.0:
            score += 2
        elif temp > 37.5:
            score += 1
    except: pass
        
    # Saturasi Oksigen (Normal: > 95)
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 < 95:
            score += 3
        elif o2 < 98:
            score += 1
    except: pass

    # Tekanan Darah (Systolic > 140 atau < 90)
    try:
        val = str(row['Blood_Pressure_mmHg'])
        if '/' in val:
            systolic = int(val.split('/')[0])
            if systolic > 140 or systolic < 90:
                score += 2
    except:
        pass 
    
    return score

def preprocess_data(df):
    """
    Memproses Data Mentah -> Format Model TACC
    Variabel: Age (A), Severity_Score (C), Vital_Score (M)
    Target: Referral_Required
    """
    processed = df.copy()
    
    # 1. Vital Score (M)
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    
    # 2. Severity Score (C) - Menggantikan Comorbidity
    severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
    processed['Severity'] = processed['Severity'].astype(str).str.strip() 
    processed['Severity_Score'] = processed['Severity'].map(severity_map).fillna(0)
    
    # 3. Target Variable
    # Logika: Jika Severity = Severe, maka Butuh Rujukan (1)
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if x['Severity'] == 'Severe' else 0, axis=1
    )
    
    return processed[['Age', 'Severity_Score', 'Vital_Score', 'Referral_Required']]

# --- 4. Pelatihan Model (H2O + LogReg) ---
@st.cache_resource
def train_hybrid_model(df_processed):
    """
    Melatih model H2O, mengambil prediksi (S), lalu melatih Logistic Regression.
    """
    # Init H2O (Limit memory untuk Cloud agar tidak crash)
    try:
        h2o.init(max_mem_size='512M')
    except:
        st.error("Gagal inisialisasi H2O. Pastikan file 'packages.txt' ada di GitHub.")
        return None, None, None

    # Convert ke H2O Frame
    hf = h2o.H2OFrame(df_processed)
    
    y = 'Referral_Required'
    # Fitur Input (Tanpa Time)
    x = ['Age', 'Severity_Score', 'Vital_Score']
    
    hf[y] = hf[y].asfactor()
    
    # 1. Train H2O AutoML (GBM/XGBoost)
    # Waktu training dibatasi 60 detik agar loading cepat di web
    aml = H2OAutoML(max_models=3, seed=42, include_algos=['GBM', 'XGBoost'], max_runtime_secs=60) 
    aml.train(x=x, y=y, training_frame=hf)
    
    best_model_h2o = aml.leader
    
    # 2. Dapatkan Skor Prediksi ML (S) dari data latih
    preds = best_model_h2o.predict(hf)
    ml_scores = preds['p1'].as_data_frame().values.flatten()
    
    # 3. Train Logistic Regression
    X_logreg = df_processed[x].copy()
    X_logreg['ML_Score'] = ml_scores 
    Y_logreg = df_processed[y]
    
    log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    log_reg.fit(X_logreg, Y_logreg)
    
    # 4. Ambil Koefisien
    coeffs = {
        'Intercept (beta_0)': log_reg.intercept_[0],
        'Age (beta_1)': log_reg.coef_[0][0],
        'Severity_Score (beta_2)': log_reg.coef_[0][1],
        'Vital_Score (beta_3)': log_reg.coef_[0][2],
        'ML_Score (beta_4)': log_reg.coef_[0][3]
    }
    
    return best_model_h2o, log_reg, coeffs

# --- 5. Fungsi Kalkulasi Probabilitas ---
def calculate_final_prob(age, severity, vital, ml_score, coeffs):
    # Rumus P(Y=1)
    logit = (coeffs['Intercept (beta_0)'] + 
             coeffs['Age (beta_1)'] * age + 
             coeffs['Severity_Score (beta_2)'] * severity + 
             coeffs['Vital_Score (beta_3)'] * vital + 
             coeffs['ML_Score (beta_4)'] * ml_score)
    
    prob = 1 / (1 + math.exp(-logit))
    return prob

# --- MAIN APPLICATION LOGIC ---

# Sidebar: Upload File
st.sidebar.header("📁 Data Input")
st.sidebar.caption("Dataset 'disease_diagnosis.csv' dimuat otomatis dari GitHub. Anda bisa menggantinya di sini.")
uploaded_file = st.sidebar.file_uploader("Upload CSV Baru (Opsional)", type=["csv"])

# 1. Load Data
df_raw = load_dataset(uploaded_file)

if not df_raw.empty:
    df_model = preprocess_data(df_raw)

    # 2. Sidebar Controls
    st.sidebar.title("⚙️ Kontrol Model")
    if st.sidebar.button("🔄 Latih Ulang Model"):
        h2o.cluster().shutdown()
        st.cache_resource.clear()
        st.rerun()

    with st.spinner("Sedang melatih model Hybrid (H2O + LogReg)..."):
        gbm_model, logreg_model, coefficients = train_hybrid_model(df_model)

    # 3. Tampilkan Rumus
    st.sidebar.markdown("### 🧮 Rumus Prediksi (TACC)")
    if coefficients:
        formula_latex = f"""
        $$
        P(Y=1) = \\frac{{1}}{{1 + e^{{-z}}}}
        $$
        Dimana $z$ adalah:
        $$
        {coefficients['Intercept (beta_0)']:.3f} + 
        {coefficients['Age (beta_1)']:.3f}(A) + 
        {coefficients['Severity_Score (beta_2)']:.3f}(Sev) + 
        {coefficients['Vital_Score (beta_3)']:.3f}(M) + 
        {coefficients['ML_Score (beta_4)']:.3f}(S)
        $$
        """
        st.sidebar.latex(formula_latex)
        st.sidebar.caption("**A**: Age, **Sev**: Severity, **M**: Vital Score, **S**: ML Score")

    # 4. User Interface Utama
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🩺 Triage Pasien")
        st.write("Masukkan parameter klinis pasien:")
        
        with st.form("referral_form"):
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur Pasien (Tahun)", 0, 120, 45)
                p_hr = st.number_input("Detak Jantung (bpm)", 30, 250, 85)
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (°C)", 34.0, 43.0, 37.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 96)
                p_severity_label = st.selectbox("Tingkat Keparahan (Severity)", 
                                                options=["Mild", "Moderate", "Severe"])
            
            submit_btn = st.form_submit_button("Analisis Keputusan")
        
        if submit_btn:
            # A. Hitung Komponen Manual
            sev_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
            p_sev_score = sev_map[p_severity_label]
            
            row_vital = {'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                         'Oxygen_Saturation_%': p_o2, 'Blood_Pressure_mmHg': p_bp}
            p_vital_score = calculate_vital_score(row_vital)
            
            # B. Prediksi Machine Learning (S)
            input_h2o = pd.DataFrame({
                'Age': [p_age], 
                'Severity_Score': [p_sev_score], 
                'Vital_Score': [p_vital_score]
            })
            hf_sample = h2o.H2OFrame(input_h2o)
            ml_pred = gbm_model.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            # C. Kalkulasi Final
            final_prob = calculate_final_prob(p_age, p_sev_score, p_vital_score, s_score, coefficients)
            
            # D. Tampilan Hasil
            st.markdown("---")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Vital Score (M)", f"{p_vital_score}")
            res_col2.metric("Severity (Sev)", f"{p_sev_score}")
            res_col3.metric("ML Score (S)", f"{s_score:.3f}")
            res_col4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.markdown(f"Probabilitas **{final_prob:.1%}** melebihi batas {threshold}. Disarankan rujuk ke RS.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.markdown(f"Probabilitas **{final_prob:.1%}**. Pasien dapat ditangani di Klinik/Puskesmas.")

    with col2:
        st.subheader("📁 Dataset & Statistik")
        st.write(f"Total Data: **{len(df_raw)}** baris")
        st.dataframe(df_raw[['Age', 'Severity', 'Diagnosis']].head(15), hide_index=True)
        
        if coefficients:
            st.write("---")
            st.write("**Bobot Koefisien (Impact):**")
            coef_chart = pd.DataFrame.from_dict(coefficients, orient='index', columns=['Value'])
            coef_chart = coef_chart.drop('Intercept (beta_0)')
            st.bar_chart(coef_chart)

else:
    st.error("File 'disease_diagnosis.csv' tidak ditemukan. Harap upload ke GitHub atau melalui sidebar.")