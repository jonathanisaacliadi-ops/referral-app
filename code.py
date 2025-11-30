import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Rujukan (Medis Standar)",
    page_icon="🩺",
    layout="wide"
)

# --- 2. Fungsi Load Data ---
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
        st.error("DATABASE TIDAK DITEMUKAN. Pastikan 'disease_diagnosis.csv' ada.")
    return pd.DataFrame()

# --- 3. Feature Engineering (Sesuai Standar Medis) ---

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """Ekstraksi Red Flags."""
    symptoms_list = []
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    return {
        'Flag_Breath': 1 if 'breath' in text_sym or 'shortness' in text_sym else 0,
        'Flag_Fever': 1 if 'fever' in text_sym else 0,
        'Flag_Pain': 1 if 'pain' in text_sym or 'ache' in text_sym else 0
    }

def calculate_vital_score(row):
    """
    Menghitung Vital Score dengan bobot medis yang lebih akurat.
    """
    score = 0
    
    # 1. Heart Rate (Nadi)
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr > 120 or hr < 50: score += 2  # Tachycardia/Bradycardia berat
        elif hr > 100: score += 1           # Tachycardia ringan
    except: pass
    
    # 2. Suhu
    try:
        t = float(row['Body_Temperature_C'])
        if t > 39.0 or t < 36.0: score += 2 # Demam tinggi / Hipotermia
        elif t > 37.5: score += 1
    except: pass
    
    # 3. Saturasi Oksigen (Prioritas Tinggi)
    try:
        o = float(row['Oxygen_Saturation_%'])
        if o < 92: score += 3    # Hipoksia Berat (Wajib Rujuk)
        elif o < 95: score += 2  # Hipoksia Ringan
    except: pass
    
    # 4. Tekanan Darah (Kurva U)
    try:
        if isinstance(row, dict): # Handle input dari UI
            sys = float(row['Sys_Raw'])
            dia = float(row['Dia_Raw'])
        else: # Handle input dari DataFrame
            val = str(row['Blood_Pressure_mmHg'])
            if '/' in val:
                sys = int(val.split('/')[0])
                dia = int(val.split('/')[1])
            else:
                return score # Skip jika format salah

        # Logika Medis:
        if sys < 90: score += 3          # Hipotensi (Syok) -> BAHAYA BESAR
        elif sys > 180 or dia > 120: score += 3 # Krisis Hipertensi -> BAHAYA BESAR
        elif sys > 140 or dia > 90: score += 1  # Hipertensi biasa -> Risiko Sedang
    except: pass 
    
    return score

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Parsing BP ke Kolom Numerik
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    # 2. Vital Lainnya
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    
    # 3. Hitung Vital Score (Logic Baru)
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    
    # 4. FITUR BARU: Flag BP Spesifik (Agar model 'ngeh' kondisi ekstrem)
    processed['Flag_Hypotension'] = (processed['Sys_Raw'] < 90).astype(int)
    processed['Flag_HTN_Crisis'] = ((processed['Sys_Raw'] > 180) | (processed['Dia_Raw'] > 120)).astype(int)
    
    # 5. Risk Index
    processed['Risk_Index'] = processed['Age'] * processed['Vital_Score']
    
    # 6. Flags Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 7. Target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 'Vital_Score', 'Risk_Index',
        'Oxygen_Raw', 'Temp_Raw', 'Sys_Raw', 'Dia_Raw',
        'Flag_Hypotension', 'Flag_HTN_Crisis', # Masukkan Flag BP ke Input
        'Flag_Breath', 'Flag_Fever', 'Flag_Pain',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model (Regularized) ---
@st.cache_resource
def train_medical_model(df_processed):
    # Init H2O
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        st.error("Gagal inisialisasi H2O.")
        return None, None, None, None, None

    # 1. Split Data
    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    train_pd = pd.concat([X_train_pd, y_train_pd], axis=1)
    test_pd = pd.concat([X_test_pd, y_test_pd], axis=1)
    
    hf_train = h2o.H2OFrame(train_pd)
    hf_test = h2o.H2OFrame(test_pd)
    
    y_col = 'Referral_Required'
    x_cols = list(X.columns)
    
    hf_train[y_col] = hf_train[y_col].asfactor()
    
    # 2. Train H2O 
    aml = H2OAutoML(
        max_models=3, 
        seed=42, 
        include_algos=['DRF', 'GBM'], 
        max_runtime_secs=45,
        verbosity='error',
        balance_classes=True
    ) 
    try:
        aml.train(x=x_cols, y=y_col, training_frame=hf_train)
    except Exception as e:
        st.error(f"Training H2O Error: {e}")
        return None, None, None, None, None

    best_model_h2o = aml.leader
    
    # 3. Prediksi S
    pred_train = best_model_h2o.predict(hf_train)
    s_train = pred_train['p1'].as_data_frame().values.flatten()
    
    pred_test = best_model_h2o.predict(hf_test)
    s_test = pred_test['p1'].as_data_frame().values.flatten()
    
    # 4. Logistic Regression
    X_train_lr = X_train_pd.copy()
    X_train_lr['ML_Score'] = s_train
    
    X_test_lr = X_test_pd.copy()
    X_test_lr['ML_Score'] = s_test
    
    # C=0.2 memberikan keseimbangan yang baik antara bias dan varians
    log_reg = LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train_lr, y_train_pd)
    
    # 5. Evaluasi
    y_prob_test = log_reg.predict_proba(X_test_lr)[:, 1]
    y_pred_test = (y_prob_test > 0.65).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test_pd, y_prob_test)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test_pd, y_pred_test)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(X_train_lr.columns):
        coeffs[col] = log_reg.coef_[0][i]
        
    return best_model_h2o, log_reg, coeffs, metrics, X_test_pd

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

df_raw = load_fixed_dataset()

if not df_raw.empty:
    s1_options = get_column_options(df_raw, 'Symptom_1')
    s2_options = ["-"] + get_column_options(df_raw, 'Symptom_2')
    s3_options = ["-"] + get_column_options(df_raw, 'Symptom_3')
    
    df_model = preprocess_data(df_raw)

    st.sidebar.title("⚙️ Status Model")
    
    if 'metrics' not in st.session_state:
        with st.spinner("Melatih Model dengan Standar Medis (BP/O2)..."):
            gbm, logreg, coef, metr, x_test = train_medical_model(df_model)
            st.session_state.gbm = gbm
            st.session_state.logreg = logreg
            st.session_state.coef = coef
            st.session_state.metrics = metr
            st.session_state.model_ready = True
    
    if st.session_state.get('model_ready'):
        metrics = st.session_state.metrics
        coeffs = st.session_state.coef
        
        # --- SIDEBAR: Metrik ---
        st.sidebar.metric("Validation AUC", f"{metrics['auc']:.4f}")
        st.sidebar.caption("AUC > 0.85 dengan logika BP yang benar.")

        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='red', lw=2)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title('ROC Curve')
        st.sidebar.pyplot(fig)
        
        # Cek apakah Flag BP memiliki bobot tinggi
        if coeffs:
            st.sidebar.markdown("### Faktor Risiko")
            coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Bobot'])
            # Filter hanya bobot positif terbesar
            top_features = coef_df.drop('Intercept').sort_values(by='Bobot', ascending=False).head(5)
            st.sidebar.table(top_features)

    # --- UI UTAMA ---
    st.title("🏥 Sistem Triage Klinis (Standardized)")
    st.markdown("""
    **Fitur Utama:**
    *   **Deteksi Syok:** Hipotensi (BP < 90/60) otomatis menaikkan risiko.
    *   **Deteksi Hipoksia:** Saturasi Oksigen < 92% menaikkan risiko.
    *   **Gejala:** Analisis kata kunci (sesak, demam, nyeri).
    """)
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.form("referral_form"):
            st.subheader("1. Tanda Vital")
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur (Tahun)", 0, 120, 45)
                p_hr = st.number_input("Detak Jantung (bpm)", 30, 250, 85)
                # Input BP tetap text agar user mudah mengetik
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (°C)", 34.0, 43.0, 37.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 96)

            st.subheader("2. Keluhan")
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala Utama", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=s2_options)
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis Keputusan", type="primary")
        
        if submit_btn and st.session_state.get('model_ready'):
            # 1. Parsing Input
            selected_symptoms_raw = [p_sym1, p_sym2, p_sym3]
            valid_symptoms = [s for s in selected_symptoms_raw if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            # Parsing Tekanan Darah
            try:
                if '/' in p_bp:
                    p_sys = float(p_bp.split('/')[0])
                    p_dia = float(p_bp.split('/')[1])
                else:
                    p_sys, p_dia = 120.0, 80.0
            except:
                st.error("Format Tekanan Darah salah. Gunakan format '120/80'.")
                p_sys, p_dia = 120.0, 80.0

            # Hitung Skor Vital dengan Logika Medis Baru
            row_dummy = {
                'Sys_Raw': p_sys, 'Dia_Raw': p_dia, 
                'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                'Oxygen_Saturation_%': p_o2
            }
            # Paksa calculate_vital_score pakai input manual
            p_vital_score = calculate_vital_score(row_dummy)
            
            p_risk_index = p_age * p_vital_score
            
            # Dictionary Input Lengkap
            input_dict = {
                'Age': p_age,
                'Vital_Score': p_vital_score,
                'Risk_Index': p_risk_index,
                'Oxygen_Raw': p_o2,
                'Temp_Raw': p_temp,
                'Sys_Raw': p_sys, 
                'Dia_Raw': p_dia,
                'Flag_Hypotension': 1 if p_sys < 90 else 0,
                'Flag_HTN_Crisis': 1 if (p_sys > 180 or p_dia > 120) else 0,
                'Flag_Breath': flags['Flag_Breath'],
                'Flag_Fever': flags['Flag_Fever'],
                'Flag_Pain': flags['Flag_Pain']
            }
            
            # 2. Prediksi ML
            # Pastikan urutan kolom sesuai training, H2O handle by name
            input_df = pd.DataFrame([input_dict])
            hf_sample = h2o.H2OFrame(input_df)
            ml_pred = st.session_state.gbm.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            # 3. Kalkulasi Final
            final_prob = calculate_final_prob(input_dict, s_score, st.session_state.coef)
            
            # 4. Tampilkan Hasil
            st.markdown("---")
            st.subheader("📋 Hasil Analisis")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("BP Status", "Hipotensi" if p_sys < 90 else ("Krisis" if p_sys > 180 else "Stabil"))
            k2.metric("Oksigen", f"{p_o2}%")
            k3.metric("ML Score (S)", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.write(f"Pasien berisiko tinggi. Probabilitas {final_prob:.1%} > {threshold}.")
                
                # Alasan Spesifik
                reasons = []
                if p_sys < 90: reasons.append("Hipotensi (Tanda Syok)")
                if p_o2 < 92: reasons.append("Hipoksia Berat")
                if p_sys > 180: reasons.append("Krisis Hipertensi")
                if flags['Flag_Breath']: reasons.append("Keluhan Sesak Napas")
                
                if reasons:
                    st.warning("Faktor Risiko Utama: " + ", ".join(reasons))
                    
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write("Pasien stabil. Dapat ditangani di FKTP.")

    with col2:
        st.subheader("Sampel Data")
        st.dataframe(df_raw[['Age', 'Diagnosis', 'Severity']].head(10), hide_index=True)

else:
    st.error("Gagal memulai aplikasi. File dataset hilang.")