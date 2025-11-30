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
    page_title="Sistem Rujukan (BP Enhanced)",
    page_icon="🩸",
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

# --- 3. Feature Engineering (Parsing Tekanan Darah) ---

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """Ekstraksi Red Flags (Categorical)."""
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
    """Mengubah angka vital menjadi SKOR (Kategori)."""
    score = 0
    try:
        if float(row['Heart_Rate_bpm']) > 100 or float(row['Heart_Rate_bpm']) < 60: score += 2
    except: pass
    try:
        if float(row['Body_Temperature_C']) > 38.0 or float(row['Body_Temperature_C']) < 36.0: score += 2
    except: pass
    try:
        if float(row['Oxygen_Saturation_%']) < 95: score += 3
    except: pass
    
    # Skor BP (Deteksi Hipotensi/Hipertensi Ekstrem)
    try:
        bp_str = str(row['Blood_Pressure_mmHg'])
        if '/' in bp_str:
            sys = int(bp_str.split('/')[0])
            dia = int(bp_str.split('/')[1])
            # Syok (Hipotensi) atau Krisis Hipertensi
            if sys < 90 or sys > 180 or dia > 110: 
                score += 3 
            elif sys > 140 or sys < 100:
                score += 1
    except: pass 
    return score

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Parsing Tekanan Darah (PENTING: Memecah string '120/80' jadi 2 kolom)
    # Kita pisahkan kolom 'Blood_Pressure_mmHg' menjadi 'Sys_Raw' dan 'Dia_Raw'
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    
    # Handle jika kolom kedua tidak ada (misal data kotor)
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    # 2. Vital Lainnya
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    
    # 3. Hitung Vital Score (Agregat)
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    
    # 4. Risk Index
    processed['Risk_Index'] = processed['Age'] * processed['Vital_Score']
    
    # 5. Flags Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 6. Target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    # FILTER INPUT: Masukkan Sys_Raw dan Dia_Raw agar model belajar dari angka BP
    return processed[[
        'Age', 'Vital_Score', 'Risk_Index',
        'Oxygen_Raw', 'Temp_Raw', 'Sys_Raw', 'Dia_Raw', # <--- Fitur BP Masuk Sini
        'Flag_Breath', 'Flag_Fever', 'Flag_Pain',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model ---
@st.cache_resource
def train_bp_model(df_processed):
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
    
    # 2. Train H2O (Termasuk fitur BP)
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
    
    # 4. Logistic Regression (Regularized)
    X_train_lr = X_train_pd.copy()
    X_train_lr['ML_Score'] = s_train
    
    X_test_lr = X_test_pd.copy()
    X_test_lr['ML_Score'] = s_test
    
    log_reg = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train_lr, y_train_pd)
    
    # 5. Evaluasi
    y_prob_test = log_reg.predict_proba(X_test_lr)[:, 1]
    y_pred_test = (y_prob_test > 0.65).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test_pd, y_prob_test)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test_pd, y_pred_test)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    # Simpan Koefisien
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
        with st.spinner("Melatih Model dengan Fitur Tekanan Darah..."):
            gbm, logreg, coef, metr, x_test = train_bp_model(df_model)
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
        st.sidebar.caption("AUC > 0.80 menandakan model memahami risiko tekanan darah.")

        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='red', lw=2)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title('ROC Curve')
        st.sidebar.pyplot(fig)
        
        # Tampilkan Bobot: Lihat apakah Sys_Raw/Dia_Raw muncul di top factors
        st.sidebar.markdown("### Top Factors")
        coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Bobot'])
        st.sidebar.table(coef_df.drop('Intercept').sort_values(by='Bobot', ascending=False).head(5))

    # --- UI UTAMA ---
    st.title("🏥 Sistem Triage Klinis (BP Support)")
    st.write("Model kini memperhitungkan **Tekanan Darah (Sistolik & Diastolik)** secara spesifik.")
    
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
            
            # Parsing Tekanan Darah dari User Input
            try:
                if '/' in p_bp:
                    p_sys = float(p_bp.split('/')[0])
                    p_dia = float(p_bp.split('/')[1])
                else:
                    p_sys = 120.0
                    p_dia = 80.0
            except:
                st.error("Format Tekanan Darah salah. Gunakan format '120/80'.")
                p_sys, p_dia = 120.0, 80.0

            # Hitung Skor Vital
            row_vital = {'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                         'Oxygen_Saturation_%': p_o2, 'Blood_Pressure_mmHg': p_bp}
            p_vital_score = calculate_vital_score(row_vital)
            
            p_risk_index = p_age * p_vital_score
            
            # Dictionary Input Lengkap (Termasuk Sys/Dia)
            input_dict = {
                'Age': p_age,
                'Vital_Score': p_vital_score,
                'Risk_Index': p_risk_index,
                'Oxygen_Raw': p_o2,
                'Temp_Raw': p_temp,
                'Sys_Raw': p_sys, # Input Baru
                'Dia_Raw': p_dia, # Input Baru
                'Flag_Breath': flags['Flag_Breath'],
                'Flag_Fever': flags['Flag_Fever'],
                'Flag_Pain': flags['Flag_Pain']
            }
            
            # 2. Prediksi ML
            # Pastikan urutan kolom sama dengan saat training (diurus oleh H2O otomatis by name)
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
            k1.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            k2.metric("Indikasi Sesak", "Ya" if flags['Flag_Breath'] else "Tidak")
            k3.metric("ML Score (S)", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.write("**Analisis:** Risiko tinggi terdeteksi.")
                if p_sys < 90: st.warning("⚠️ Pasien Hipotensi (Syok) - Butuh penanganan segera.")
                if p_sys > 180: st.warning("⚠️ Pasien Krisis Hipertensi.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write("**Analisis:** Kondisi stabil.")

    with col2:
        st.subheader("Sampel Data")
        st.dataframe(df_raw.head(10), hide_index=True)

else:
    st.error("Gagal memulai aplikasi. File dataset hilang.")