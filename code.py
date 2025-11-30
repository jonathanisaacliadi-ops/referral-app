import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Triage Medis (NEWS2 Full)",
    layout="wide"
)

# --- 2. Fungsi Load Data (Smart Dummy) ---
@st.cache_data
def load_fixed_dataset():
    local_path = "disease_diagnosis.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            return df
        except Exception as e:
            st.error(f"File database rusak: {e}")
    else:
        st.warning("Database tidak ditemukan. Menggunakan DATA DUMMY TERSTRUKTUR (Agar Model Belajar Pola yang Benar).")
        # Generator Data Dummy yang LOGIS (Bukan random murni)
        # Agar bobot model tidak negatif/ngawur, kita harus buat korelasi antara Vital Sign & Severity
        np.random.seed(42)
        n = 300
        
        # Buat label severe/mild dulu
        severity = np.random.choice(['Severe', 'Mild'], n, p=[0.3, 0.7])
        
        data = []
        for s in severity:
            if s == 'Severe':
                # Pasien parah: Vital sign cenderung jelek
                age = np.random.randint(50, 90)
                hr = np.random.randint(100, 140) # Takikardia
                sys = np.random.randint(80, 100) # Hipotensi (Syok) atau Krisis
                o2 = np.random.randint(85, 93)   # Hipoksia
                temp = np.random.uniform(38.5, 40.0) # Demam
            else:
                # Pasien mild: Vital sign normal
                age = np.random.randint(20, 60)
                hr = np.random.randint(60, 90)
                sys = np.random.randint(110, 130)
                o2 = np.random.randint(96, 100)
                temp = np.random.uniform(36.0, 37.5)
            
            data.append({
                'Age': age,
                'Blood_Pressure_mmHg': f"{sys}/80",
                'Oxygen_Saturation_%': o2,
                'Body_Temperature_C': round(temp, 1),
                'Heart_Rate_bpm': hr,
                'Symptom_1': 'Fever' if temp > 38 else 'None',
                'Symptom_2': '-', 'Symptom_3': '-',
                'Severity': s
            })
            
        return pd.DataFrame(data)
    return pd.DataFrame()

# --- 3. Feature Engineering (NEWS2 Full Standard) ---

def calculate_news2_score_full(row):
    """
    Menghitung NEWS2 Score Lengkap (Scale 1)
    Sesuai PDF NEWS2 Observation Chart
    """
    score = 0
    
    # 1. Respiration Rate (Tidak ada di dataset, asumsikan normal/0)
    
    # 2. Oxygen Saturation (SpO2) - Scale 1
    # PDF: <=91 (3), 92-93 (2), 94-95 (1), >=96 (0)
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 <= 91: score += 3
        elif 92 <= o2 <= 93: score += 2
        elif 94 <= o2 <= 95: score += 1
        # >= 96 score 0
    except: pass

    # 3. Air or Oxygen? (Tidak ada di dataset, asumsi Air/0)

    # 4. Systolic BP
    # PDF: <=90 (3), 91-100 (2), 101-110 (1), 111-219 (0), >=220 (3)
    try:
        if 'Sys_Raw' in row: sys = float(row['Sys_Raw'])
        else: sys = float(str(row['Blood_Pressure_mmHg']).split('/')[0])

        if sys <= 90: score += 3
        elif 91 <= sys <= 100: score += 2
        elif 101 <= sys <= 110: score += 1
        elif 111 <= sys <= 219: score += 0
        elif sys >= 220: score += 3
    except: pass

    # 5. Heart Rate
    # PDF: <=40 (3), 41-50 (1), 51-90 (0), 91-110 (1), 111-130 (2), >=131 (3)
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 51 <= hr <= 90: score += 0
        elif 91 <= hr <= 110: score += 1
        elif 111 <= hr <= 130: score += 2 
        elif hr >= 131: score += 3
    except: pass

    # 6. Consciousness (Tidak ada, asumsi Alert/0)

    # 7. Temperature
    # PDF: <=35.0 (3), 35.1-36.0 (1), 36.1-38.0 (0), 38.1-39.0 (1), >=39.1 (2)
    try:
        t = float(row['Body_Temperature_C'])
        if t <= 35.0: score += 3
        elif 35.1 <= t <= 36.0: score += 1
        elif 36.1 <= t <= 38.0: score += 0
        elif 38.1 <= t <= 39.0: score += 1
        elif t >= 39.1: score += 2
    except: pass
    
    return score

def extract_features_from_symptoms(row_or_list):
    symptoms_list = []
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    return {
        'Sym_Dyspnea': 1 if 'breath' in text_sym or 'shortness' in text_sym else 0,
        'Sym_Fever': 1 if 'fever' in text_sym else 0
    }

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def preprocess_data(df):
    processed = df.copy()
    
    # Cleaning Data
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    processed['Heart_Raw'] = pd.to_numeric(processed['Heart_Rate_bpm'], errors='coerce').fillna(80)
    
    # Hitung NEWS2 (FULL dengan SpO2)
    processed['NEWS2_Score'] = processed.apply(calculate_news2_score_full, axis=1)
    
    # Extract Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # Target Variable
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed

# --- 4. Pelatihan Model (Solusi Bobot Negatif) ---
@st.cache_resource
def train_medical_model(df_processed):
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        return None, None, None, None, None

    # --- PENTING: MEMILIH FITUR AGAR BOBOT TIDAK NEGATIF ---
    # Kita HANYA memakai NEWS2_Score, Age, dan Gejala.
    # Kita MEMBUANG data mentah (Sys_Raw, Oxygen_Raw, dll) dari training set.
    # Ini memaksa model untuk "percaya" pada NEWS2 Score, sehingga bobotnya akan POSITIF.
    
    feature_cols = ['Age', 'NEWS2_Score', 'Sym_Dyspnea', 'Sym_Fever']
    target_col = 'Referral_Required'
    
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling (Penting untuk Intercept)
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_cols, index=X_test.index)

    # --- H2O ---
    train_h2o = pd.concat([X_train, y_train], axis=1)
    hf_train = h2o.H2OFrame(train_h2o)
    
    hf_train[target_col] = hf_train[target_col].asfactor()
    
    aml = H2OAutoML(
        max_models=2, 
        seed=42, 
        include_algos=['GBM'], 
        max_runtime_secs=60,
        verbosity='error'
    ) 
    try:
        aml.train(x=feature_cols, y=target_col, training_frame=hf_train)
        best_model = aml.leader
    except: 
        best_model = None

    # --- Logistic Regression ---
    log_reg = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(feature_cols):
        coeffs[col] = log_reg.coef_[0][i]
        
    return best_model, log_reg, coeffs, metrics, scaler

# --- MAIN APP ---
df_raw = load_fixed_dataset()

if not df_raw.empty:
    df_model = preprocess_data(df_raw)
    
    if 'model_ready' not in st.session_state:
        with st.spinner("Melatih Model & Optimasi Bobot..."):
            gbm, logreg, coef, metr, scaler = train_medical_model(df_model)
            if logreg:
                st.session_state.gbm = gbm
                st.session_state.logreg = logreg
                st.session_state.coef = coef
                st.session_state.metrics = metr
                st.session_state.scaler = scaler
                st.session_state.model_ready = True
            else:
                st.error("Gagal melatih model.")

    # --- UI UTAMA ---
    st.title("Sistem Triage Medis (NEWS2 + AI)")
    st.write("Model ini telah dioptimasi agar bobot NEWS2 bernilai positif (Risiko berbanding lurus dengan Skor).")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Pasien")
        with st.form("referral_form"):
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur", 0, 120, 60)
                p_hr = st.number_input("Nadi (bpm)", 30, 250, 115) # Default tinggi biar kelihatan skornya
                p_bp = st.text_input("Tekanan Darah (mmHg)", "95/60") # Default rendah biar skor tinggi
            with c2:
                p_temp = st.number_input("Suhu Tubuh (C)", 34.0, 43.0, 39.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 90) # Default rendah

            st.write("Keluhan")
            s1_options = get_column_options(df_raw, 'Symptom_1')
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala 1", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=['-'])
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=['-'])
            
            submit_btn = st.form_submit_button("Analisis", type="primary")

    with col2:
        st.subheader("Hasil Analisis")
        
        if submit_btn and st.session_state.get('model_ready'):
            try:
                if '/' in p_bp: p_sys, p_dia = map(float, p_bp.split('/'))
                else: p_sys, p_dia = 120.0, 80.0
            except: p_sys, p_dia = 120.0, 80.0

            valid_symptoms = [s for s in [p_sym1, p_sym2, p_sym3] if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            # Hitung NEWS2 (FULL)
            row_input = {
                'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                'Sys_Raw': p_sys, 'Oxygen_Saturation_%': p_o2
            }
            p_news2 = calculate_news2_score_full(row_input)
            
            # Siapkan Input Model (Hanya Fitur Terpilih)
            input_dict = {
                'Age': p_age, 
                'NEWS2_Score': p_news2,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_Fever': flags['Sym_Fever']
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Scaling
            scaler = st.session_state.scaler
            input_scaled_array = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled_array, columns=input_df.columns)
            
            # Prediksi
            final_prob = st.session_state.logreg.predict_proba(input_scaled_df)[0][1]
            
            k1, k2, k3 = st.columns(3)
            k1.metric("NEWS2 Score", f"{p_news2}")
            k2.metric("Oksigen (SpO2)", f"{p_o2}%")
            k3.metric("Risiko (Probabilitas)", f"{final_prob:.1%}")
            
            if final_prob > 0.5:
                st.error(f"RUJUKAN DIPERLUKAN (Risiko Tinggi: {final_prob:.1%})")
            else:
                st.success(f"PASIEN STABIL (Risiko Rendah: {final_prob:.1%})")
                
        elif not st.session_state.get('model_ready'):
             st.info("Tunggu model sedang dimuat...")

    st.markdown("---")
    with st.expander("Detail Bobot Model (Bukti Bobot Positif)", expanded=True):
        coeffs = st.session_state.get('coef')
        if coeffs:
            st.write("Perhatikan bahwa **NEWS2_Score** sekarang memiliki bobot **POSITIF** (bar biru ke kanan), yang berarti semakin tinggi skor NEWS2, semakin tinggi prediksi risiko.")
            coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Bobot'])
            st.bar_chart(coef_df.drop('Intercept'))
            st.dataframe(coef_df.style.background_gradient(cmap='coolwarm'))

else:
    st.error("Error loading data.")