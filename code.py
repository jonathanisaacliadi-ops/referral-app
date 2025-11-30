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
    page_title="Sistem Triage Medis",
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
        st.error("DATABASE TIDAK DITEMUKAN. Pastikan disease_diagnosis.csv ada.")
    return pd.DataFrame()

# --- 3. Feature Engineering ---
def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

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
        'Sym_ChestPain': 1 if 'chest' in text_sym and 'pain' in text_sym else 0,
        'Sym_Fever': 1 if 'fever' in text_sym else 0
    }

def calculate_news2_score_strict(row):
    score = 0
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 <= 91: score += 3
        elif 92 <= o2 <= 93: score += 2
        elif 94 <= o2 <= 95: score += 1
    except: pass

    try:
        if isinstance(row, dict): sys = float(row['Sys_Raw'])
        else: sys = float(str(row['Blood_Pressure_mmHg']).split('/')[0])

        if sys <= 90: score += 3
        elif 91 <= sys <= 100: score += 2
        elif 101 <= sys <= 110: score += 1
    except: pass

    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 91 <= hr <= 100: score += 1
        elif 101 <= hr <= 130: score += 2 
        elif hr >= 131: score += 3
    except: pass

    try:
        t = float(row['Body_Temperature_C'])
        if t <= 35.0: score += 3
        elif 35.1 <= t <= 36.0: score += 1
        elif 38.1 <= t <= 39.0: score += 1
        elif t >= 39.1: score += 2
    except: pass
    
    return score

def preprocess_data(df):
    processed = df.copy()
    
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    processed['Heart_Raw'] = pd.to_numeric(processed['Heart_Rate_bpm'], errors='coerce').fillna(80)
    
    processed['NEWS2_Score'] = processed.apply(calculate_news2_score_strict, axis=1)
    processed['Flag_HTN_Crisis'] = (processed['Sys_Raw'] >= 180).astype(int)
    
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 'NEWS2_Score', 
        'Sys_Raw', 'Dia_Raw', 'Oxygen_Raw', 'Temp_Raw', 'Heart_Raw',
        'Flag_HTN_Crisis',
        'Sym_Dyspnea', 'Sym_ChestPain', 'Sym_Fever',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model ---
@st.cache_resource
def train_medical_model(df_processed):
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
    df_model = preprocess_data(df_raw)
    
    # Train Model (Hidden Spinner)
    if 'metrics' not in st.session_state:
        with st.spinner("Memproses Model AI & Standar NEWS2..."):
            gbm, logreg, coef, metr = train_medical_model(df_model)
            st.session_state.gbm = gbm
            st.session_state.logreg = logreg
            st.session_state.coef = coef
            st.session_state.metrics = metr
            st.session_state.model_ready = True

    # --- UI UTAMA ---
    st.title("Sistem Triage & Rujukan Klinis")
    st.write("Sistem pendukung keputusan klinis berbasis Machine Learning dan standar medis internasional (NEWS2, JNC8).")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Klinis Pasien")
        with st.form("referral_form"):
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Umur", 0, 120, 45)
                p_hr = st.number_input("Nadi (bpm)", 30, 250, 80)
                p_bp = st.text_input("Tekanan Darah (mmHg)", "120/80")
            with c2:
                p_temp = st.number_input("Suhu Tubuh (C)", 34.0, 43.0, 36.5)
                p_o2 = st.number_input("Saturasi Oksigen (%)", 50, 100, 98)

            st.write("Keluhan & Gejala")
            s1_options = get_column_options(df_raw, 'Symptom_1')
            s2_options = ["-"] + get_column_options(df_raw, 'Symptom_2')
            s3_options = ["-"] + get_column_options(df_raw, 'Symptom_3')
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: p_sym1 = st.selectbox("Gejala 1", options=s1_options)
            with sc2: p_sym2 = st.selectbox("Gejala 2", options=s2_options)
            with sc3: p_sym3 = st.selectbox("Gejala 3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis Keputusan", type="primary")

    with col2:
        st.subheader("Hasil Analisis")
        
        if submit_btn and st.session_state.get('model_ready'):
            try:
                if '/' in p_bp: p_sys, p_dia = map(float, p_bp.split('/'))
                else: p_sys, p_dia = 120.0, 80.0
            except: p_sys, p_dia = 120.0, 80.0

            valid_symptoms = [s for s in [p_sym1, p_sym2, p_sym3] if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            row_dummy = {
                'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                'Oxygen_Saturation_%': p_o2, 'Sys_Raw': p_sys, 'Dia_Raw': p_dia
            }
            p_news2 = calculate_news2_score_strict(row_dummy)
            
            input_dict = {
                'Age': p_age, 'NEWS2_Score': p_news2,
                'Sys_Raw': p_sys, 'Dia_Raw': p_dia,
                'Oxygen_Raw': p_o2, 'Temp_Raw': p_temp, 'Heart_Raw': p_hr,
                'Flag_HTN_Crisis': 1 if p_sys >= 180 else 0,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_ChestPain': flags['Sym_ChestPain'],
                'Sym_Fever': flags['Sym_Fever']
            }
            
            input_df = pd.DataFrame([input_dict])
            hf_sample = h2o.H2OFrame(input_df)
            ml_pred = st.session_state.gbm.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            final_prob = calculate_final_prob(input_dict, s_score, st.session_state.coef)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("NEWS2 Score", f"{p_news2}")
            k2.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            k3.metric("Risiko Rujukan", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"RUJUKAN DIPERLUKAN (Risiko {final_prob:.1%})")
                st.write("Indikasi Klinis:")
                if p_news2 >= 5: st.warning(f"- Skor NEWS2 {p_news2} (Bahaya Klinis Akut)")
                if p_sys <= 90: st.warning("- Hipotensi (Risiko Syok)")
                if p_o2 <= 91: st.warning("- Hipoksia Berat")
                if flags['Sym_Dyspnea']: st.warning("- Keluhan Sesak Napas")
            else:
                st.success(f"TIDAK PERLU RUJUKAN (Risiko {final_prob:.1%})")
                st.write("Kondisi stabil. Rawat jalan dengan obat simptomatik.")
        else:
            st.info("Silakan isi data pasien di sebelah kiri dan klik 'Analisis Keputusan'.")

    # --- MENU BAWAH (COLLAPSIBLE) ---
    st.markdown("---")
    with st.expander("Detail Model, Rumus & Data", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Performa & Metrik", "Rumus & Bobot", "Dataset"])
        
        metrics = st.session_state.get('metrics')
        coeffs = st.session_state.get('coef')

        variable_map = {
            'Intercept': 'Intercept (Nilai Dasar)',
            'Age': 'Usia Pasien (Age)',
            'NEWS2_Score': 'Skor Peringatan Dini (NEWS2_Score)',
            'Sys_Raw': 'Tekanan Darah Sistolik (Sys_Raw)',
            'Dia_Raw': 'Tekanan Darah Diastolik (Dia_Raw)',
            'Oxygen_Raw': 'Saturasi Oksigen (Oxygen_Raw)',
            'Temp_Raw': 'Suhu Tubuh (Temp_Raw)',
            'Heart_Raw': 'Detak Jantung (Heart_Raw)',
            'Flag_HTN_Crisis': 'Indikator Krisis Hipertensi (Flag_HTN_Crisis)',
            'Sym_Dyspnea': 'Gejala Sesak Napas (Sym_Dyspnea)',
            'Sym_ChestPain': 'Gejala Nyeri Dada (Sym_ChestPain)',
            'Sym_Fever': 'Gejala Demam (Sym_Fever)',
            'ML_Score': 'Skor Prediksi AI (ML_Score)'
        }

        with tab1:
            if metrics:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Skor AUC", f"{metrics['auc']:.4f}")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.plot(metrics['fpr'], metrics['tpr'], color='blue', lw=2)
                    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    ax.set_title('ROC Curve')
                    st.pyplot(fig)
                with c2:
                    st.write("Confusion Matrix:")
                    cm = metrics['cm']
                    
                    # VARIABLE LABEL UNTUK KOTAK
                    group_names = ['TN (Stabil)', 'FP (Salah Rujuk)', 'FN (Bahaya)', 'TP (Rujuk)']
                    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
                    
                    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
                    labels = np.asarray(labels).reshape(2,2)
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax_cm)
                    ax_cm.set_xlabel('Prediksi Model')
                    ax_cm.set_ylabel('Data Aktual')
                    st.pyplot(fig_cm)

        with tab2:
            if coeffs:
                st.markdown("#### Bobot Variabel")
                coef_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['Bobot'])
                plot_df = coef_df.drop('Intercept')
                plot_df.index = plot_df.index.map(lambda x: variable_map.get(x, x))
                plot_df = plot_df.sort_values(by='Bobot', ascending=False)
                
                st.bar_chart(plot_df)
                
                coef_df.index = coef_df.index.map(lambda x: variable_map.get(x, x))
                st.dataframe(coef_df.style.format("{:.4f}"))

        with tab3:
            st.markdown(f"Total Data: {len(df_raw)} Pasien")
            st.dataframe(df_raw.head(20))

else:
    st.error("Gagal memulai aplikasi.")