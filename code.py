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
    page_title="Sistem Triage Medis (NEWS2 Standard)",
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

# --- 3. Feature Engineering (Berdasarkan NEWS2 & AHA) ---

def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """
    Mendeteksi Red Flags Gejala berdasarkan Triage ESI (Emergency Severity Index).
    """
    symptoms_list = []
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    
    # Kategori Gejala
    return {
        'Sym_Dyspnea': 1 if 'breath' in text_sym or 'shortness' in text_sym else 0, # Sesak
        'Sym_ChestPain': 1 if 'chest' in text_sym and 'pain' in text_sym else 0,    # Nyeri Dada
        'Sym_Fever': 1 if 'fever' in text_sym else 0,
        'Sym_General': 1 if 'fatigue' in text_sym or 'ache' in text_sym else 0
    }

def calculate_news2_score(row):
    """
    Menghitung Skor Vital berdasarkan standar NEWS2 (National Early Warning Score 2)
    dan Guidelines Hipertensi AHA.
    """
    score = 0
    
    # --- 1. Respirasi / Oksigen (NEWS2) ---
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 <= 91: score += 3      # Gagal napas (Bahaya)
        elif 92 <= o2 <= 93: score += 2
        elif 94 <= o2 <= 95: score += 1
        # >= 96 score 0
    except: pass

    # --- 2. Tekanan Darah Sistolik (NEWS2 + JNC8) ---
    try:
        if isinstance(row, dict):
            sys = float(row['Sys_Raw'])
            dia = float(row['Dia_Raw'])
        else:
            bp_str = str(row['Blood_Pressure_mmHg'])
            sys = float(bp_str.split('/')[0])
            dia = float(bp_str.split('/')[1])

        # Hipotensi (Syok) - NEWS2
        if sys <= 90: score += 3
        elif 91 <= sys <= 100: score += 2
        elif 101 <= sys <= 110: score += 1
        # Hipertensi Krisis - AHA (Tambahan untuk dataset ini)
        elif sys >= 180 or dia >= 120: score += 2 
    except: pass

    # --- 3. Nadi / Heart Rate (NEWS2) ---
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 51 <= hr <= 90: score += 0
        elif 91 <= hr <= 110: score += 1
        elif 111 <= hr <= 130: score += 2
        elif hr >= 131: score += 3
    except: pass

    # --- 4. Suhu (NEWS2) ---
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
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    # 2. Raw Vitals
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    
    # 3. Hitung NEWS2 Score (Standar Medis)
    processed['NEWS2_Score'] = processed.apply(calculate_news2_score, axis=1)
    
    # 4. Gejala Flags
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 5. Target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    # Feature Selection: Fokus pada Skor Medis Standar + Gejala
    return processed[[
        'Age', 
        'NEWS2_Score', # Skor Gabungan Medis
        'Sys_Raw', 'Oxygen_Raw', # Data mentah kritis tetap dimasukkan untuk presisi ML
        'Sym_Dyspnea', 'Sym_ChestPain', 'Sym_Fever',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model ---
@st.cache_resource
def train_medical_model(df_processed):
    try:
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        st.error("Gagal inisialisasi H2O.")
        return None, None, None, None

    # Split Data
    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_pd = pd.concat([X_train_pd, y_train_pd], axis=1)
    test_pd = pd.concat([X_test_pd, y_test_pd], axis=1)
    
    hf_train = h2o.H2OFrame(train_pd)
    hf_test = h2o.H2OFrame(test_pd)
    
    y_col = 'Referral_Required'
    x_cols = list(X.columns)
    
    hf_train[y_col] = hf_train[y_col].asfactor()
    
    # Train H2O
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
    except:
        return None, None, None, None

    best_model_h2o = aml.leader
    
    # Prediksi S
    s_train = best_model_h2o.predict(hf_train)['p1'].as_data_frame().values.flatten()
    s_test = best_model_h2o.predict(hf_test)['p1'].as_data_frame().values.flatten()
    
    # Logistic Regression (Balanced C)
    X_train_lr = X_train_pd.copy()
    X_train_lr['ML_Score'] = s_train
    
    X_test_lr = X_test_pd.copy()
    X_test_lr['ML_Score'] = s_test
    
    log_reg = LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train_lr, y_train_pd)
    
    # Evaluasi
    y_prob_test = log_reg.predict_proba(X_test_lr)[:, 1]
    y_pred_test = (y_prob_test > 0.65).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test_pd, y_prob_test)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test_pd, y_pred_test)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(X_train_lr.columns):
        coeffs[col] = log_reg.coef_[0][i]
        
    return best_model_h2o, log_reg, coeffs, metrics

# --- 5. Kalkulasi Final ---
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

    st.sidebar.title("⚙️ Status Validasi")
    
    if 'metrics' not in st.session_state:
        with st.spinner("Menerapkan Standar NEWS2 & AHA Guidelines..."):
            gbm, logreg, coef, metr = train_medical_model(df_model)
            st.session_state.gbm = gbm
            st.session_state.logreg = logreg
            st.session_state.coef = coef
            st.session_state.metrics = metr
            st.session_state.model_ready = True
    
    if st.session_state.get('model_ready'):
        metrics = st.session_state.metrics
        coeffs = st.session_state.coef
        
        st.sidebar.metric("Akurasi Medis (AUC)", f"{metrics['auc']:.4f}")
        
        # Plot ROC
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='blue', lw=2)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title('ROC Curve')
        st.sidebar.pyplot(fig)
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Dasar Pembobotan:**
        1. **NEWS2 Score**: Saturasi O2, Nadi, Suhu, TD Sistolik.
        2. **JNC 8**: Hipertensi Krisis.
        3. **ESI Triage**: Gejala Sesak & Nyeri Dada.
        """)

    # --- UI UTAMA ---
    st.title("🏥 Sistem Triage & Rujukan (NEWS2)")
    st.markdown("Sistem pendukung keputusan klinis berbasis Machine Learning dan standar medis internasional.")
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.form("referral_form"):
            st.subheader("1. Tanda Vital (NEWS2 Parameter)")
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur", 0, 120, 45)
                p_hr = st.number_input("Nadi (bpm)", 30, 250, 80)
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (°C)", 34.0, 43.0, 36.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 98)

            st.subheader("2. Keluhan & Gejala")
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala 1", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=s2_options)
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis Keputusan", type="primary")
        
        if submit_btn and st.session_state.get('model_ready'):
            # 1. Parsing & Scoring
            try:
                if '/' in p_bp:
                    p_sys, p_dia = map(float, p_bp.split('/'))
                else: p_sys, p_dia = 120.0, 80.0
            except: p_sys, p_dia = 120.0, 80.0

            valid_symptoms = [s for s in [p_sym1, p_sym2, p_sym3] if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            # Hitung NEWS2 Score
            row_dummy = {
                'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                'Oxygen_Saturation_%': p_o2, 'Sys_Raw': p_sys, 'Dia_Raw': p_dia
            }
            p_news2 = calculate_news2_score(row_dummy)
            
            # Input untuk Model
            input_dict = {
                'Age': p_age,
                'NEWS2_Score': p_news2,
                'Sys_Raw': p_sys,
                'Oxygen_Raw': p_o2,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_ChestPain': flags['Sym_ChestPain'],
                'Sym_Fever': flags['Sym_Fever']
            }
            
            # 2. Prediksi ML
            input_df = pd.DataFrame([input_dict])
            hf_sample = h2o.H2OFrame(input_df)
            ml_pred = st.session_state.gbm.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            # 3. Final Probability
            final_prob = calculate_final_prob(input_dict, s_score, st.session_state.coef)
            
            # 4. Tampilan
            st.markdown("---")
            st.subheader("📋 Hasil Analisis")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("NEWS2 Score", f"{p_news2}")
            k2.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            k3.metric("ML Score (S)", f"{s_score:.3f}")
            k4.metric("Probabilitas", f"{final_prob:.1%}")
            
            # Logic Triage Sederhana untuk UI
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.write("**Rekomendasi Klinis:**")
                if p_news2 >= 7: st.write("- **NEWS2 Tinggi (Critical):** Risiko henti jantung/napas. Resusitasi segera.")
                elif p_news2 >= 5: st.write("- **NEWS2 Sedang:** Pantau ketat setiap jam, pertimbangkan rawat inap.")
                if p_sys < 90: st.write("- **Hipotensi:** Pasang akses infus, guyur cairan.")
                if p_o2 < 92: st.write("- **Hipoksia:** Berikan Oksigen segera.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.write("**Rekomendasi Klinis:** Kondisi stabil. Rawat jalan dengan obat simptomatik.")

    with col2:
        st.subheader("Data Referensi")
        st.dataframe(df_raw.head(15), hide_index=True)
        
        if st.session_state.get('coef'):
            st.write("---")
            st.write("**Bobot Variabel (Model Logistik):**")
            coef_df = pd.DataFrame.from_dict(st.session_state.coef, orient='index', columns=['Bobot'])
            st.bar_chart(coef_df.drop('Intercept').sort_values(by='Bobot', ascending=False))

else:
    st.error("Gagal memulai aplikasi.")