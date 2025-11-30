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
    page_title="Sistem Triage (NEWS2 Compliant)",
    page_icon="📋",
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

# --- 3. Feature Engineering ---

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """Ekstraksi Gejala."""
    symptoms_list = []
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    return {
        'Sym_Dyspnea': 1 if 'breath' in text_sym or 'shortness' in text_sym else 0,
        'Sym_ChestPain': 1 if 'chest' in text_sym and 'pain' in text_sym else 0,
        'Sym_Fever': 1 if 'fever' in text_sym else 0
    }

def calculate_news2_score_strict(row):
    """
    Menghitung Skor NEWS2 STRICTLY berdasarkan gambar chart.
    Catatan: Dataset tidak memiliki Respirasi & Kesadaran, jadi skor parsial.
    """
    score = 0
    
    # --- A. Respirasi (Tidak ada di dataset, diasumsikan normal/0) ---
    # Jika nanti ada data 'Respiration_Rate', logikanya:
    # <=8 (3), 9-11 (1), 12-20 (0), 21-24 (2), >=25 (3)
    
    # --- B. Saturasi Oksigen (SpO2 Scale 1) ---
    # Sesuai Gambar: <=91 (3), 92-93 (2), 94-95 (1), >=96 (0)
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 <= 91: score += 3
        elif 92 <= o2 <= 93: score += 2
        elif 94 <= o2 <= 95: score += 1
        else: score += 0 # >= 96
    except: pass

    # --- C. Tekanan Darah Sistolik ---
    # Sesuai Gambar: <=90 (3), 91-100 (2), 101-110 (1), 111-219 (0), >=220 (0/Tidak berwarna di chart)
    # *Koreksi:* Pada NEWS2 standard, Hipotensi dinilai, Hipertensi TIDAK dinilai (0).
    try:
        if isinstance(row, dict):
            sys = float(row['Sys_Raw'])
        else:
            sys = float(str(row['Blood_Pressure_mmHg']).split('/')[0])

        if sys <= 90: score += 3
        elif 91 <= sys <= 100: score += 2
        elif 101 <= sys <= 110: score += 1
        else: score += 0 # 111 ke atas (termasuk 220) adalah 0 di NEWS2
    except: pass

    # --- D. Nadi (Pulse) ---
    # Sesuai Gambar: <=40 (3), 41-50 (1), 51-90 (0), 91-100 (1), 101-110 (2), 111-130 (2), >=131 (3)
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 51 <= hr <= 90: score += 0
        elif 91 <= hr <= 100: score += 1
        elif 101 <= hr <= 130: score += 2 # Gabungan range 101-110 dan 111-130
        elif hr >= 131: score += 3
    except: pass

    # --- E. Suhu (Temperature) ---
    # Sesuai Gambar: <=35.0 (3), 35.1-36.0 (1), 36.1-38.0 (0), 38.1-39.0 (1), >=39.1 (2)
    try:
        t = float(row['Body_Temperature_C'])
        if t <= 35.0: score += 3
        elif 35.1 <= t <= 36.0: score += 1
        elif 36.1 <= t <= 38.0: score += 0
        elif 38.1 <= t <= 39.0: score += 1
        elif t >= 39.1: score += 2
    except: pass
    
    return score

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Parsing BP
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    # Diastolik tidak dipakai di scoring NEWS2, tapi berguna untuk ML
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    # 2. Raw Vitals
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    processed['Heart_Raw'] = pd.to_numeric(processed['Heart_Rate_bpm'], errors='coerce').fillna(80)
    
    # 3. Hitung NEWS2 Score (STRICT)
    processed['NEWS2_Score'] = processed.apply(calculate_news2_score_strict, axis=1)
    
    # 4. Tambahan Fitur Medis (Untuk membantu ML mengenali yang tidak tercover NEWS2)
    # NEWS2 mengabaikan Tensi Tinggi (>180), tapi untuk rujukan, ini penting (Krisis Hipertensi)
    # Kita buat Flag terpisah agar ML bisa mempelajarinya tanpa merusak skor NEWS2 murni.
    processed['Flag_HTN_Crisis'] = (processed['Sys_Raw'] >= 180).astype(int)
    
    # 5. Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 6. Target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 'NEWS2_Score', 
        'Sys_Raw', 'Dia_Raw', 'Oxygen_Raw', 'Temp_Raw', 'Heart_Raw', # Raw data membantu presisi ML
        'Flag_HTN_Crisis',
        'Sym_Dyspnea', 'Sym_ChestPain', 'Sym_Fever',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model ---
@st.cache_resource
def train_news2_model(df_processed):
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        return None, None, None, None

    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_pd = pd.concat([X_train, y_train], axis=1)
    test_pd = pd.concat([X_test, y_test], axis=1)
    
    hf_train = h2o.H2OFrame(train_pd)
    hf_test = h2o.H2OFrame(test_pd)
    
    y_col = 'Referral_Required'
    x_cols = list(X.columns)
    
    hf_train[y_col] = hf_train[y_col].asfactor()
    
    aml = H2OAutoML(
        max_models=3, 
        seed=42, 
        include_algos=['DRF', 'GBM'], 
        max_runtime_secs=60,
        verbosity='error',
        balance_classes=True
    ) 
    try:
        aml.train(x=x_cols, y=y_col, training_frame=hf_train)
    except: return None, None, None, None

    best_model = aml.leader
    
    s_train = best_model.predict(hf_train)['p1'].as_data_frame().values.flatten()
    s_test = best_model.predict(hf_test)['p1'].as_data_frame().values.flatten()
    
    X_train['ML_Score'] = s_train
    X_test['ML_Score'] = s_test
    
    log_reg = LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train, y_train)
    
    y_prob = log_reg.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.65).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(X_train.columns):
        coeffs[col] = log_reg.coef_[0][i]
        
    return best_model, log_reg, coeffs, metrics

# --- 5. Kalkulasi ---
def calculate_final_prob(input_dict, ml_score, coeffs):
    logit = coeffs['Intercept']
    for feat, val in input_dict.items():
        if feat in coeffs:
            logit += coeffs[feat] * val
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
        with st.spinner("Training Model with Official NEWS2 Weights..."):
            gbm, logreg, coef, metr = train_news2_model(df_model)
            st.session_state.gbm = gbm
            st.session_state.logreg = logreg
            st.session_state.coef = coef
            st.session_state.metrics = metr
            st.session_state.model_ready = True
    
    if st.session_state.get('model_ready'):
        metrics = st.session_state.metrics
        st.sidebar.metric("Validasi AUC", f"{metrics['auc']:.4f}")
        
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='blue', lw=2)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        st.sidebar.pyplot(fig)
        
        # Tampilkan Bobot NEWS2 di Sidebar sebagai info
        st.sidebar.info("""
        **NEWS2 Logic Applied:**
        * O2 <= 91: +3
        * BP <= 90: +3
        * HR <= 40 / >= 131: +3
        * Temp <= 35.0: +3
        """)

    # --- UI UTAMA ---
    st.title("🏥 Sistem Triage Klinis (NEWS2 Standard)")
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.form("referral_form"):
            st.subheader("1. Tanda Vital")
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur", 0, 120, 45)
                p_hr = st.number_input("Nadi (bpm)", 30, 250, 80)
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (°C)", 34.0, 43.0, 36.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 98)

            st.subheader("2. Keluhan")
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala 1", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=s2_options)
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis", type="primary")
        
        if submit_btn and st.session_state.get('model_ready'):
            # Parsing
            try:
                if '/' in p_bp: p_sys, p_dia = map(float, p_bp.split('/'))
                else: p_sys, p_dia = 120.0, 80.0
            except: p_sys, p_dia = 120.0, 80.0

            valid_symptoms = [s for s in [p_sym1, p_sym2, p_sym3] if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            # NEWS2 Calculation (Manual untuk Display)
            row_dummy = {
                'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                'Oxygen_Saturation_%': p_o2, 'Sys_Raw': p_sys
            }
            p_news2 = calculate_news2_score_strict(row_dummy)
            
            # Input ML
            input_dict = {
                'Age': p_age,
                'NEWS2_Score': p_news2,
                'Sys_Raw': p_sys, 'Dia_Raw': p_dia,
                'Oxygen_Raw': p_o2, 'Temp_Raw': p_temp, 'Heart_Raw': p_hr,
                'Flag_HTN_Crisis': 1 if p_sys >= 180 else 0,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_ChestPain': flags['Sym_ChestPain'],
                'Sym_Fever': flags['Sym_Fever']
            }
            
            # Predict
            input_df = pd.DataFrame([input_dict])
            hf_sample = h2o.H2OFrame(input_df)
            ml_pred = st.session_state.gbm.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            final_prob = calculate_final_prob(input_dict, s_score, st.session_state.coef)
            
            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("NEWS2 Score", f"{p_news2}", help="Skor deteksi dini perburukan klinis")
            k2.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            k3.metric("ML Score", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            # Keputusan
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                
                # Logic Penjelasan NEWS2
                if p_news2 >= 7: st.write("- **Kritis (NEWS2 ≥ 7):** Risiko henti jantung tinggi. Respon tim emergensi segera.")
                elif p_news2 >= 5: st.write("- **Risiko Sedang (NEWS2 ≥ 5):** Observasi ketat tiap jam atau rujuk.")
                elif p_news2 >= 1: st.write("- **Risiko Rendah:** Pantau rutin.")
                
                if p_sys >= 180: st.warning("⚠️ **Krisis Hipertensi:** Tensi sangat tinggi.")
                
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write("Kondisi stabil. Skor NEWS2 rendah.")

    with col2:
        st.subheader("Data Pasien")
        st.dataframe(df_raw[['Age', 'Diagnosis', 'Severity']].head(15), hide_index=True)

else:
    st.error("Gagal memulai aplikasi.")