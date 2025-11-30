import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Rujukan Cerdas (Fixed Model)",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Fungsi Load Data (Hanya Lokal) ---
@st.cache_data
def load_fixed_dataset():
    """
    Hanya memuat file 'disease_diagnosis.csv' dari folder lokal/GitHub.
    """
    local_path = "disease_diagnosis.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            return df
        except Exception as e:
            st.error(f"File database rusak: {e}")
    else:
        st.error("DATABASE TIDAK DITEMUKAN. Pastikan file 'disease_diagnosis.csv' ada di folder aplikasi.")
    return pd.DataFrame()

# --- 3. Feature Engineering (High Accuracy) ---

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """Ekstraksi Red Flags dari gejala."""
    symptoms_list = []
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    
    return {
        'Flag_Breath': 1 if 'breath' in text_sym else 0,
        'Flag_Fever': 1 if 'fever' in text_sym else 0,
        'Flag_Pain': 1 if 'pain' in text_sym or 'ache' in text_sym else 0,
        'Symptom_Count': len(symptoms_list)
    }

def calculate_vital_score(row):
    score = 0
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr > 100 or hr < 60: score += 2
    except: pass
    try:
        temp = float(row['Body_Temperature_C'])
        if temp > 38.0 or temp < 36.0: score += 2
        elif temp > 37.5: score += 1
    except: pass
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 < 95: score += 3
        elif o2 < 98: score += 1
    except: pass
    try:
        val = str(row['Blood_Pressure_mmHg'])
        if '/' in val:
            systolic = int(val.split('/')[0])
            if systolic > 140 or systolic < 90: score += 2
    except: pass 
    return score

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Hitung Skor
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    
    # 2. Extract Flags
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 3. Target (Severity)
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 'Vital_Score', 'Oxygen_Raw', 'Temp_Raw', 
        'Flag_Breath', 'Flag_Fever', 'Flag_Pain', 'Referral_Required'
    ]]

# --- 4. Pelatihan Model (Auto-Run) ---
@st.cache_resource
def train_fixed_model(df_processed):
    # Init H2O Hemat RAM
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        st.error("Gagal inisialisasi H2O. Pastikan Java terinstal.")
        return None, None, None, None

    hf = h2o.H2OFrame(df_processed)
    y = 'Referral_Required'
    x = ['Age', 'Vital_Score', 'Oxygen_Raw', 'Temp_Raw', 'Flag_Breath', 'Flag_Fever', 'Flag_Pain']
    
    hf[y] = hf[y].asfactor()
    
    # Train AutoML
    aml = H2OAutoML(
        max_models=5, 
        seed=42, # SEED TETAP = HASIL KONSISTEN
        include_algos=['GBM', 'DRF'], 
        max_runtime_secs=60,
        verbosity='error',
        balance_classes=True
    ) 
    
    try:
        aml.train(x=x, y=y, training_frame=hf)
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None, None, None

    best_model_h2o = aml.leader
    
    # Prediksi S
    preds = best_model_h2o.predict(hf)
    ml_scores = preds['p1'].as_data_frame().values.flatten()
    
    # Logistic Regression
    X_logreg = df_processed[x].copy()
    X_logreg['ML_Score'] = ml_scores 
    Y_logreg = df_processed[y]
    
    log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_logreg, Y_logreg)
    
    # Evaluasi AUC
    y_prob_final = log_reg.predict_proba(X_logreg)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_logreg, y_prob_final)
    roc_auc = auc(fpr, tpr)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(X_logreg.columns):
        coeffs[col] = log_reg.coef_[0][i]
        
    return best_model_h2o, log_reg, coeffs, metrics

# --- 5. Fungsi Kalkulasi Probabilitas ---
def calculate_final_prob(input_data_dict, ml_score, coeffs):
    logit = coeffs['Intercept']
    for feature, value in input_data_dict.items():
        if feature in coeffs:
            logit += coeffs[feature] * value
    logit += coeffs['ML_Score'] * ml_score
    
    prob = 1 / (1 + math.exp(-logit))
    return prob

# --- MAIN APP ---

# 1. Load Data (Langsung, tanpa sidebar upload)
df_raw = load_fixed_dataset()

if not df_raw.empty:
    # Persiapan Opsi Gejala
    s1_options = get_column_options(df_raw, 'Symptom_1')
    s2_options = ["-"] + get_column_options(df_raw, 'Symptom_2')
    s3_options = ["-"] + get_column_options(df_raw, 'Symptom_3')
    
    # Preprocessing
    df_model = preprocess_data(df_raw)

    # 2. Training Model (Otomatis berjalan sekali di awal)
    # Tidak ada tombol "Latih Ulang", langsung jalan.
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    with st.spinner("Menginisialisasi Sistem Cerdas (Initial Setup)..."):
        gbm_model, logreg_model, coeffs, metrics = train_fixed_model(df_model)
        st.session_state.model_trained = True

    # --- SIDEBAR: Hasil Model Tetap ---
    st.sidebar.title("ℹ️ Status Model")
    st.sidebar.success("Model Aktif")
    st.sidebar.caption(f"Basis Data: {len(df_raw)} Pasien")
    
    if metrics:
        st.sidebar.markdown("---")
        st.sidebar.metric("Akurasi (AUC)", f"{metrics['auc']:.4f}")
        
        # Plot ROC Kecil
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='green', lw=2)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_title('ROC Curve', fontsize=10)
        ax.axis('off') # Hilangkan axis biar bersih
        st.sidebar.pyplot(fig)

    if coeffs:
        st.sidebar.markdown("### 🧮 Bobot Fitur")
        # Visualisasi koefisien sederhana
        coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Value'])
        coef_df = coef_df.drop('Intercept').sort_values(by='Value', ascending=False).head(5)
        st.sidebar.table(coef_df)

    # --- UI UTAMA ---
    st.title("🏥 Sistem Triage & Rujukan")
    st.markdown("Masukkan parameter klinis pasien untuk mendapatkan analisis rujukan otomatis.")
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.form("referral_form"):
            st.subheader("1. Tanda Vital")
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur (Tahun)", 0, 120, 45)
                p_hr = st.number_input("Detak Jantung (bpm)", 30, 250, 85)
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (°C)", 34.0, 43.0, 37.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 96)

            st.subheader("2. Keluhan & Gejala")
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala Utama", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=s2_options)
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis Keputusan", type="primary")
        
        if submit_btn:
            # 1. Proses Input
            selected_symptoms_raw = [p_sym1, p_sym2, p_sym3]
            valid_symptoms = [s for s in selected_symptoms_raw if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            row_vital = {'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                         'Oxygen_Saturation_%': p_o2, 'Blood_Pressure_mmHg': p_bp}
            p_vital_score = calculate_vital_score(row_vital)
            
            input_dict = {
                'Age': p_age,
                'Vital_Score': p_vital_score,
                'Oxygen_Raw': p_o2,
                'Temp_Raw': p_temp,
                'Flag_Breath': flags['Flag_Breath'],
                'Flag_Fever': flags['Flag_Fever'],
                'Flag_Pain': flags['Flag_Pain']
            }
            
            # 2. Prediksi ML
            input_df = pd.DataFrame([input_dict])
            hf_sample = h2o.H2OFrame(input_df)
            ml_pred = gbm_model.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            # 3. Kalkulasi Final
            final_prob = calculate_final_prob(input_dict, s_score, coeffs)
            
            # 4. Tampilkan Hasil
            st.markdown("---")
            st.subheader("📋 Hasil Analisis")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Saturasi O2", f"{p_o2}%")
            k2.metric("Indikasi Berat", "Ada" if flags['Flag_Breath'] else "Tidak")
            k3.metric("ML Score (S)", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.write("**Rekomendasi:** Pasien memiliki risiko tinggi. Segera rujuk ke fasilitas lanjut.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write("**Rekomendasi:** Pasien stabil. Dapat ditangani dengan rawat jalan/obat.")

    with col2:
        st.subheader("Sampel Data Referensi")
        st.dataframe(df_raw[['Age', 'Symptom_1', 'Severity', 'Diagnosis']].head(15), hide_index=True)

else:
    st.error("Gagal memulai aplikasi. File dataset hilang.")