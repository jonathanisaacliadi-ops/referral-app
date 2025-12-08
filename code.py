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
import time
import sys
import tempfile 

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
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"File database rusak: {e}")
    else:
        st.error("DATABASE TIDAK DITEMUKAN. Pastikan file 'disease_diagnosis.csv' sudah diupload.")
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
        'Sym_Fever': 1 if 'fever' in text_sym else 0
    }

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
    
    processed['Flag_HTN_Crisis'] = (processed['Sys_Raw'] >= 180).astype(int)
    
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # Target tetap mengikuti data asli (biarkan AI belajar dari data apa adanya)
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 
        'Sys_Raw', 'Dia_Raw', 'Oxygen_Raw', 'Temp_Raw', 'Heart_Raw', 
        'Flag_HTN_Crisis',
        'Sym_Dyspnea', 'Sym_Fever',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model (H2O + LogReg Revisi) ---
@st.cache_resource
def train_medical_model(df_processed):
    use_gbm = True
    best_model = None
    error_msg = ""
    
    # ... (Bagian Inisialisasi H2O TETAP SAMA seperti sebelumnya) ...
    try:
        try:
            h2o.cluster().shutdown(prompt=False)
            time.sleep(3) 
        except:
            pass 
        h2o.init(max_mem_size='600M', nthreads=1, ice_root=tempfile.mkdtemp(), verbose=False) 
    except Exception as e:
        error_msg = str(e)
        print(f"H2O Init Failed: {e}", file=sys.stderr)
        use_gbm = False

    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        return None, None, None, None 

    s_train = None
    s_test = None

    # ... (Bagian Training H2O GBM TETAP SAMA) ...
    if use_gbm:
        try:
            train_pd = pd.concat([X_train, y_train], axis=1)
            hf_train = h2o.H2OFrame(train_pd)
            y_col = 'Referral_Required'
            hf_train[y_col] = hf_train[y_col].asfactor()
            features_gbm = list(X.columns)
            
            aml = H2OAutoML(
                max_models=2, 
                seed=42, 
                include_algos=['GBM'], 
                max_runtime_secs=90, 
                verbosity='error',
                balance_classes=True
            ) 
            aml.train(x=features_gbm, y=y_col, training_frame=hf_train)
            best_model = aml.leader
            
            if best_model:
                hf_test = h2o.H2OFrame(X_test)
                s_train = best_model.predict(hf_train)['p1'].as_data_frame().values.flatten()
                s_test = best_model.predict(hf_test)['p1'].as_data_frame().values.flatten()
            else:
                use_gbm = False 
        except Exception as e:
            print(f"H2O Training Failed: {e}", file=sys.stderr)
            use_gbm = False 

    # === BAGIAN INI YANG DIUBAH (HAPUS DEMAM DARI LOGREG) ===
    def get_lr_features(df_orig, ml_scores, use_ml):
        df_new = pd.DataFrame(index=df_orig.index)
        
        # 1. Komponen AI (Deep Component)
        # Menangkap pola kompleks termasuk Demam, Oksigen, Nadi, dll
        if use_ml and ml_scores is not None:
            df_new['ML_Score'] = ml_scores
        
        # 2. Komponen Safety Net (Wide Component - Prioritas Medis)
        # Hanya menyisakan variabel 'Red Flag' sesuai standar NEWS2
        df_new['Age'] = df_orig['Age']
        df_new['Flag_HTN_Crisis'] = df_orig['Flag_HTN_Crisis'] # Safety Net 1
        df_new['Sym_Dyspnea'] = df_orig['Sym_Dyspnea']         # Safety Net 2 (Pernapasan)
        
        # Note: Sym_Fever DIHAPUS dari sini karena bobotnya rendah (NEWS2 Score < 3)
        # Demam biarkan ditangani oleh ML_Score saja.
        
        return df_new
    # ========================================================

    X_train_lr = get_lr_features(X_train, s_train, use_gbm)
    X_test_lr = get_lr_features(X_test, s_test, use_gbm)
    
    scaler = StandardScaler()
    
    try:
        X_train_scaled_array = scaler.fit_transform(X_train_lr)
        X_test_scaled_array = scaler.transform(X_test_lr)
    except:
        return None, None, None, None
        
    cols_lr = X_train_lr.columns
    X_train_final = pd.DataFrame(X_train_scaled_array, columns=cols_lr, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_scaled_array, columns=cols_lr, index=X_test.index)
    
    log_reg = LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', max_iter=2000, random_state=42)
    log_reg.fit(X_train_final, y_train)
    
    y_prob = log_reg.predict_proba(X_test_final)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(cols_lr):
        coeffs[col] = log_reg.coef_[0][i]
    
    coeffs['scaler_mean'] = scaler.mean_.tolist()
    coeffs['scaler_scale'] = scaler.scale_.tolist()
    coeffs['scaler_cols'] = list(cols_lr)
    coeffs['use_gbm'] = use_gbm 
    coeffs['error_msg'] = error_msg 
        
    return best_model, log_reg, coeffs, metrics

# --- 5. Kalkulasi Prediksi Baru (Update: Hapus Demam dari Manual Calculation) ---
def calculate_final_prob(input_dict, ml_score, coeffs):
    # 1. ATURAN EMAS KEAMANAN (Golden Safety Rules - Override Mutlak)
    critical_reasons = []
    if input_dict['Oxygen_Raw'] <= 90: critical_reasons.append("Saturasi Oksigen Kritis (<=90%)")
    if input_dict['Temp_Raw'] >= 39.5: critical_reasons.append("Hiperpireksia (>=39.5°C)")
    if input_dict['Sys_Raw'] <= 90: critical_reasons.append("Hipotensi Berat (<=90 mmHg)")
    if input_dict['Heart_Raw'] >= 140: critical_reasons.append("Takikardia Ekstrem (>=140 bpm)")
    
    if critical_reasons:
        return 0.999, critical_reasons 
        
    # 2. Kalkulasi Model Hybrid
    try:
        means = np.array(coeffs['scaler_mean'])
        scales = np.array(coeffs['scaler_scale'])
        cols = coeffs['scaler_cols']
        use_gbm = coeffs.get('use_gbm', False)
    except KeyError:
        return 0.5, []
    
    # Mapping data input ke kolom yang diharapkan LogReg
    # 'Sym_Fever' TIDAK DIMASUKKAN ke dictionary ini
    data_row = {
        'Age': input_dict['Age'],
        'Flag_HTN_Crisis': input_dict['Flag_HTN_Crisis'],
        'Sym_Dyspnea': input_dict['Sym_Dyspnea']
    }
    
    if use_gbm:
        data_row['ML_Score'] = ml_score
    
    input_values = []
    # Loop ini otomatis hanya akan mengambil Age, HTN, Dyspnea, dan ML_Score
    # karena 'cols' berasal dari training yang sudah dikurangi kolomnya.
    for c in cols:
        input_values.append(data_row.get(c, 0))
    
    input_values = np.array(input_values).reshape(1, -1)
    
    input_scaled = (input_values - means) / scales
    
    logit = coeffs['Intercept']
    for i, col_name in enumerate(cols):
        logit += coeffs[col_name] * input_scaled[0][i]
            
    prob = 1 / (1 + math.exp(-logit))
    return prob, []

# --- MAIN APP ---

df_raw = load_fixed_dataset()

if not df_raw.empty:
    df_model = preprocess_data(df_raw)
    
    if 'model_ready' not in st.session_state:
        with st.spinner("Memproses Model Mencoba H2O"):
            gbm, logreg, coef, metr = train_medical_model(df_model)
            
            if logreg is not None:
                st.session_state.gbm = gbm
                st.session_state.logreg = logreg
                st.session_state.coef = coef
                st.session_state.metrics = metr
                st.session_state.model_ready = True
            else:
                st.error("Gagal melatih model dasar.")

    st.title("Sistem Triage & Rujukan Klinis")
    st.write("Sistem pendukung keputusan klinis berbasis Machine Learning (GBM + LogReg).")
    
    if st.session_state.get('model_ready'):
        use_gbm = st.session_state.coef.get('use_gbm', False)
        if not use_gbm:
            error_details = st.session_state.coef.get('error_msg', 'Unknown Error')
            st.warning("⚠️ Mode Terbatas: Komponen AI Lanjut (GBM) tidak aktif. Prediksi menggunakan Model Standar (LogReg).")
            with st.expander("Lihat Detail Error Teknis (Untuk Debugging)"):
                st.code(error_details)
    
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
            
            input_dict_full = {
                'Age': p_age, 
                'Sys_Raw': p_sys, 'Dia_Raw': p_dia,
                'Oxygen_Raw': p_o2, 'Temp_Raw': p_temp, 'Heart_Raw': p_hr,
                'Flag_HTN_Crisis': 1 if p_sys >= 180 else 0,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_Fever': flags['Sym_Fever']
            } 
            
            s_score = 0.5
            use_gbm = st.session_state.coef.get('use_gbm', False)
            
            if use_gbm and st.session_state.gbm:
                input_gbm = input_dict_full.copy()
                input_df_gbm = pd.DataFrame([input_gbm])
                hf_sample = h2o.H2OFrame(input_df_gbm)
                try:
                    ml_pred = st.session_state.gbm.predict(hf_sample)
                    s_score = ml_pred['p1'].as_data_frame().values[0][0]
                except:
                    s_score = 0.5 

            # Kalkulasi Probabilitas (Sekarang mengembalikan 2 nilai: prob dan list alasan kritis)
            final_prob, critical_reasons = calculate_final_prob(input_dict_full, s_score, st.session_state.coef)
            
            k1, k2 = st.columns(2)
            k1.metric("Risiko Rujukan", f"{final_prob:.1%}") 
            k2.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            
            threshold = 0.5 
            if final_prob > threshold:
                st.error(f"RUJUKAN DIPERLUKAN (Risiko {final_prob:.1%})")
                st.write("Indikasi Klinis:")
                
                # Tampilkan alasan kritis dari override jika ada
                if critical_reasons:
                    for reason in critical_reasons:
                        st.warning(f"- {reason} [CRITICAL]")
                else:
                    # Alasan reguler dari AI
                    if p_sys >= 180: st.warning("- Krisis Hipertensi (JNC8)")
                    if flags['Sym_Dyspnea']: st.warning("- Keluhan Sesak Napas")
                    if flags['Sym_Fever']: st.warning("- Gejala Demam")
                    if s_score > 0.7: st.warning("- Pola Vital Mencurigakan (AI)")
            else:
                st.success(f"TIDAK PERLU RUJUKAN (Risiko {final_prob:.1%})")
                st.write("Kondisi stabil. Rawat jalan dengan obat simptomatik.")
                
        elif not st.session_state.get('model_ready'):
             st.info("Silakan isi data pasien di sebelah kiri dan klik 'Analisis Keputusan'.")

    # --- MENU BAWAH (UI Asli) ---
    st.markdown("---")
    with st.expander("Detail Model, Rumus & Data", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Performa & Metrik", "Rumus & Bobot", "Dataset"])
        
        metrics = st.session_state.get('metrics')
        coeffs = st.session_state.get('coef')

        # Map variable name untuk display
        variable_map = {
            'Intercept': 'Intercept (Nilai Dasar)',
            'Age': 'Usia Pasien (Age)',
            'ML_Score': 'Skor Prediksi AI (ML_Score)',
            'Sym_Dyspnea': 'Gejala Sesak Napas',
            'Sym_Fever': 'Gejala Demam',
            'Flag_HTN_Crisis': 'Krisis Hipertensi'
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
                st.markdown("#### Bobot Variabel (Scaled)")
                # Bersihkan dictionary dari data scaler internal sebelum plotting
                plot_coeffs = coeffs.copy()
                for k in ['scaler_mean', 'scaler_scale', 'scaler_cols', 'use_gbm', 'error_msg']:
                    if k in plot_coeffs: del plot_coeffs[k]

                coef_df = pd.DataFrame.from_dict(plot_coeffs, orient='index', columns=['Bobot'])
                plot_df = coef_df.drop('Intercept', errors='ignore')
                plot_df.index = plot_df.index.map(lambda x: variable_map.get(x, x))
                plot_df = plot_df.sort_values(by='Bobot', ascending=False)
                
                st.bar_chart(plot_df)
                
                coef_df.index = coef_df.index.map(lambda x: variable_map.get(x, x))
                st.dataframe(coef_df.style.format("{:.4f}"))

        with tab3:
            st.markdown(f"Total Data: {len(df_raw)} Pasien")
            st.dataframe(df_raw)

else:
    st.error("Gagal memulai aplikasi.")