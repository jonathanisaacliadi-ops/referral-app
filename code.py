import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Rujukan Cerdas (Validated)",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Fungsi Load Data (Fixed) ---
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

# --- 3. Feature Engineering ---

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
        'Flag_Breath': 1 if 'breath' in text_sym else 0,
        'Flag_Fever': 1 if 'fever' in text_sym else 0,
        'Flag_Pain': 1 if 'pain' in text_sym or 'ache' in text_sym else 0
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
            sys = int(val.split('/')[0])
            if sys > 140 or sys < 90: score += 2
    except: pass 
    return score

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Skor Vital
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    
    # 2. Flags Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 3. Target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    # PENGHAPUSAN FITUR RAW AGAR TIDAK LEAKAGE/OVERFIT
    return processed[[
        'Age', 'Vital_Score', 
        'Flag_Breath', 'Flag_Fever', 'Flag_Pain', 
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model (VALIDASI KETAT) ---
@st.cache_resource
def train_validated_model(df_processed):
    # Init H2O Hemat RAM
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        st.error("Gagal inisialisasi H2O.")
        return None, None, None, None

    # 1. SPLIT DATA (80% Train, 20% Test)
    # Kita pisahkan secara manual agar AUC dihitung dari data 'Test' yang murni
    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    # Split pandas dataframe
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Gabungkan kembali untuk format H2O
    train_pd = pd.concat([X_train_pd, y_train_pd], axis=1)
    test_pd = pd.concat([X_test_pd, y_test_pd], axis=1)
    
    hf_train = h2o.H2OFrame(train_pd)
    hf_test = h2o.H2OFrame(test_pd)
    
    y_col = 'Referral_Required'
    x_cols = ['Age', 'Vital_Score', 'Flag_Breath', 'Flag_Fever', 'Flag_Pain']
    
    hf_train[y_col] = hf_train[y_col].asfactor()
    hf_test[y_col] = hf_test[y_col].asfactor()
    
    # 2. Train H2O pada Data TRAIN saja
    aml = H2OAutoML(
        max_models=3, 
        seed=42, 
        include_algos=['GBM', 'DRF'], 
        max_runtime_secs=45,
        verbosity='error',
        balance_classes=True
    ) 
    aml.train(x=x_cols, y=y_col, training_frame=hf_train)
    best_model_h2o = aml.leader
    
    # 3. Prediksi S (ML Score)
    # Kita butuh skor S untuk Train (buat latih LogReg) dan Test (buat validasi LogReg)
    pred_train_h2o = best_model_h2o.predict(hf_train)
    s_train = pred_train_h2o['p1'].as_data_frame().values.flatten()
    
    pred_test_h2o = best_model_h2o.predict(hf_test)
    s_test = pred_test_h2o['p1'].as_data_frame().values.flatten()
    
    # 4. Logistic Regression
    # Latih LogReg menggunakan data TRAIN + Skor S Train
    X_train_lr = X_train_pd.copy()
    X_train_lr['ML_Score'] = s_train
    
    log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train_lr, y_train_pd)
    
    # 5. Validasi AUC pada Data TEST (Data Asing)
    # Kita uji performa LogReg pada data yang belum pernah dilihatnya
    X_test_lr = X_test_pd.copy()
    X_test_lr['ML_Score'] = s_test
    
    y_prob_test = log_reg.predict_proba(X_test_lr)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_pd, y_prob_test)
    roc_auc = auc(fpr, tpr)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # Simpan Koefisien
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(X_train_lr.columns):
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

df_raw = load_fixed_dataset()

if not df_raw.empty:
    s1_options = get_column_options(df_raw, 'Symptom_1')
    s2_options = ["-"] + get_column_options(df_raw, 'Symptom_2')
    s3_options = ["-"] + get_column_options(df_raw, 'Symptom_3')
    
    df_model = preprocess_data(df_raw)

    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Proses Training Otomatis di Awal
    with st.spinner("Memvalidasi Model (Train/Test Split)..."):
        gbm_model, logreg_model, coeffs, metrics = train_validated_model(df_model)
        st.session_state.model_trained = True

    # --- Sidebar ---
    st.sidebar.title("ℹ️ Status Validasi")
    st.sidebar.success("Model Terlatih")
    
    if metrics:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Performa (Test Set)")
        st.sidebar.caption("AUC dihitung dari 20% data yang disembunyikan saat pelatihan.")
        
        st.sidebar.metric("Validation AUC", f"{metrics['auc']:.4f}")
        
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='purple', lw=2)
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_title('ROC Curve (Test Data)', fontsize=10)
        ax.set_xlabel('False Positive')
        ax.set_ylabel('True Positive')
        st.sidebar.pyplot(fig)

    if coeffs:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚖️ Bobot Logistik")
        coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Value'])
        coef_df = coef_df.drop('Intercept').sort_values(by='Value', ascending=False)
        st.sidebar.dataframe(coef_df, height=200)

    # --- UI Utama ---
    st.title("🏥 Sistem Triage Klinis (Validated)")
    st.markdown("Analisis rujukan berdasarkan model **Hybrid TACC** yang telah divalidasi.")
    
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
            
            submit_btn = st.form_submit_button("Analisis", type="primary")
        
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
            k1.metric("Vital Score", f"{p_vital_score}")
            k2.metric("Indikasi Sesak", "Ya" if flags['Flag_Breath'] else "Tidak")
            k3.metric("ML Score", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.write(f"Probabilitas ({final_prob:.1%}) > Threshold ({threshold}).")
                st.write("**Saran:** Pasien berisiko tinggi. Segera rujuk ke RS.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write(f"Probabilitas ({final_prob:.1%}) <= Threshold ({threshold}).")
                st.write("**Saran:** Kondisi stabil. Tangani di FKTP.")

    with col2:
        st.subheader("Sampel Data")
        st.dataframe(df_raw[['Age', 'Symptom_1', 'Diagnosis', 'Severity']].head(15), hide_index=True)

else:
    st.error("Gagal memulai. Pastikan file 'disease_diagnosis.csv' ada.")