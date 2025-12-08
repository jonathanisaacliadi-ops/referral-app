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
    page_title="Sistem Triage Medis (Hybrid AI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Fungsi Load Data ---
@st.cache_data
def load_fixed_dataset():
    # Pastikan nama file sesuai dengan yang Anda upload
    local_path = "disease_diagnosis.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            # Bersihkan nama kolom dari spasi berlebih
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"File database rusak: {e}")
    else:
        st.error("DATABASE TIDAK DITEMUKAN. Pastikan file 'disease_diagnosis.csv' ada di folder yang sama.")
    return pd.DataFrame()

# --- 3. Feature Engineering & Preprocessing ---
def get_column_options(df, col_name):
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def extract_features_from_symptoms(row_or_list):
    """
    Mengekstrak fitur gejala dari input user atau baris dataframe.
    """
    symptoms_list = []
    # Jika input adalah Row Dataframe
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip().lower() for s in items if str(s) != 'nan']
    # Jika input adalah List dari UI
    else:
        symptoms_list = [str(s).strip().lower() for s in row_or_list if s != "-" and s is not None]

    text_sym = " ".join(symptoms_list)
    
    # Kata kunci sederhana untuk mendeteksi gejala
    return {
        'Sym_Dyspnea': 1 if 'breath' in text_sym or 'shortness' in text_sym else 0,
        'Sym_Fever': 1 if 'fever' in text_sym else 0,
        # Sym_Cough bisa ditambahkan untuk GBM, tapi tidak masuk LogReg manual
        'Sym_Cough': 1 if 'cough' in text_sym else 0 
    }

def preprocess_data(df):
    processed = df.copy()
    
    # 1. Parsing Tekanan Darah
    bp_split = processed['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
    processed['Sys_Raw'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
    if bp_split.shape[1] > 1:
        processed['Dia_Raw'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    else:
        processed['Dia_Raw'] = 80

    # 2. Parsing Vital Signs Lain
    processed['Oxygen_Raw'] = pd.to_numeric(processed['Oxygen_Saturation_%'], errors='coerce').fillna(98)
    processed['Temp_Raw'] = pd.to_numeric(processed['Body_Temperature_C'], errors='coerce').fillna(36.5)
    processed['Heart_Raw'] = pd.to_numeric(processed['Heart_Rate_bpm'], errors='coerce').fillna(80)
    
    # 3. Feature Flag Khusus (Safety Net)
    processed['Flag_HTN_Crisis'] = (processed['Sys_Raw'] >= 180).astype(int)
    
    # 4. Ekstrak Gejala
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    # 5. Target Variable
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    # Return kolom yang relevan untuk Model Level 1 (GBM)
    # GBM akan "melihat" semuanya: Usia, Demam, Batuk, Oksigen, dll.
    return processed[[
        'Age', 
        'Sys_Raw', 'Dia_Raw', 'Oxygen_Raw', 'Temp_Raw', 'Heart_Raw', 
        'Flag_HTN_Crisis',
        'Sym_Dyspnea', 'Sym_Fever', 'Sym_Cough',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model (Arsitektur Hybrid Revisi) ---
@st.cache_resource
def train_medical_model(df_processed):
    use_gbm = True
    best_model = None
    error_msg = ""
    
    # A. Inisialisasi H2O
    try:
        try:
            h2o.cluster().shutdown(prompt=False)
            time.sleep(2) 
        except:
            pass 
        h2o.init(max_mem_size='600M', nthreads=1, ice_root=tempfile.mkdtemp(), verbose=False) 
    except Exception as e:
        error_msg = str(e)
        print(f"H2O Init Failed: {e}", file=sys.stderr)
        use_gbm = False

    # B. Split Data
    X = df_processed.drop('Referral_Required', axis=1)
    y = df_processed['Referral_Required']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        return None, None, None, None 

    s_train = None
    s_test = None

    # C. Training Level 1: H2O GBM (Deep Component)
    if use_gbm:
        try:
            train_pd = pd.concat([X_train, y_train], axis=1)
            hf_train = h2o.H2OFrame(train_pd)
            y_col = 'Referral_Required'
            hf_train[y_col] = hf_train[y_col].asfactor()
            features_gbm = list(X.columns) # GBM memakai SEMUA fitur (termasuk Age & Fever)
            
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
                # Dapatkan probabilitas prediksi dari GBM untuk menjadi input LogReg
                s_train = best_model.predict(hf_train)['p1'].as_data_frame().values.flatten()
                s_test = best_model.predict(hf_test)['p1'].as_data_frame().values.flatten()
            else:
                use_gbm = False 
        except Exception as e:
            print(f"H2O Training Failed: {e}", file=sys.stderr)
            use_gbm = False 

    # D. Training Level 2: Logistic Regression (Safety Net / Wide Component)
    # FITUR SELECTION PENTING:
    # Kita hanya mengambil ML_Score, Dyspnea, dan HTN.
    # Age dan Fever DIHAPUS dari sini agar tidak redundan.
    def get_lr_features(df_orig, ml_scores, use_ml):
        df_new = pd.DataFrame(index=df_orig.index)
        
        # 1. Masukkan otak AI (ML Score)
        if use_ml and ml_scores is not None:
            df_new['ML_Score'] = ml_scores
        
        # 2. Masukkan Safety Net (Hanya Gejala Akut Prioritas NEWS2)
        df_new['Flag_HTN_Crisis'] = df_orig['Flag_HTN_Crisis']
        df_new['Sym_Dyspnea'] = df_orig['Sym_Dyspnea'] 
        
        # Catatan: Age dan Sym_Fever TIDAK dimasukkan di sini.
        return df_new

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
    
    # Evaluasi
    y_prob = log_reg.predict_proba(X_test_final)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'cm': cm}
    
    # Simpan Koefisien untuk Laporan
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(cols_lr):
        coeffs[col] = log_reg.coef_[0][i]
    
    coeffs['scaler_mean'] = scaler.mean_.tolist()
    coeffs['scaler_scale'] = scaler.scale_.tolist()
    coeffs['scaler_cols'] = list(cols_lr)
    coeffs['use_gbm'] = use_gbm 
    coeffs['error_msg'] = error_msg 
        
    return best_model, log_reg, coeffs, metrics

# --- 5. Fungsi Prediksi (Final Logic) ---
def calculate_final_prob(input_dict, ml_score, coeffs):
    # 1. ATURAN EMAS KEAMANAN (Golden Safety Rules)
    # Override mutlak jika kondisi tanda vital sangat ekstrim
    critical_reasons = []
    if input_dict['Oxygen_Raw'] <= 90: critical_reasons.append("Saturasi Oksigen Kritis (<=90%)")
    if input_dict['Temp_Raw'] >= 39.5: critical_reasons.append("Hiperpireksia (>=39.5°C)")
    if input_dict['Sys_Raw'] <= 90: critical_reasons.append("Hipotensi Berat (<=90 mmHg)")
    if input_dict['Heart_Raw'] >= 140: critical_reasons.append("Takikardia Ekstrem (>=140 bpm)")
    
    if critical_reasons:
        return 0.999, critical_reasons # Langsung Rujuk
        
    # 2. Kalkulasi Model Hybrid (Jika Tanda Vital Masih di Zona Abu-abu)
    try:
        means = np.array(coeffs['scaler_mean'])
        scales = np.array(coeffs['scaler_scale'])
        cols = coeffs['scaler_cols']
        use_gbm = coeffs.get('use_gbm', False)
    except KeyError:
        return 0.5, []
    
    # Mapping Data Input ke Variabel Model LogReg
    # Hanya memasukkan variabel yang dilatih di Level 2 (ML Score, HTN, Dyspnea)
    data_row = {
        'Flag_HTN_Crisis': input_dict['Flag_HTN_Crisis'],
        'Sym_Dyspnea': input_dict['Sym_Dyspnea']
    }
    
    if use_gbm:
        data_row['ML_Score'] = ml_score
    
    # Buat array input sesuai urutan kolom saat training
    input_values = []
    for c in cols:
        input_values.append(data_row.get(c, 0))
    
    input_values = np.array(input_values).reshape(1, -1)
    
    # Standarisasi (Z-Score)
    input_scaled = (input_values - means) / scales
    
    # Hitung Logit
    logit = coeffs['Intercept']
    for i, col_name in enumerate(cols):
        logit += coeffs[col_name] * input_scaled[0][i]
            
    # Sigmoid
    prob = 1 / (1 + math.exp(-logit))
    return prob, []

# --- MAIN APPLICATION UI ---

df_raw = load_fixed_dataset()

if not df_raw.empty:
    df_model = preprocess_data(df_raw)
    
    if 'model_ready' not in st.session_state:
        with st.spinner("Sedang Melatih Model Hybrid (H2O + LogReg)... Mohon Tunggu..."):
            gbm, logreg, coef, metr = train_medical_model(df_model)
            
            if logreg is not None:
                st.session_state.gbm = gbm
                st.session_state.logreg = logreg
                st.session_state.coef = coef
                st.session_state.metrics = metr
                st.session_state.model_ready = True
            else:
                st.error("Gagal melatih model dasar.")

    st.title("Sistem Pendukung Keputusan Triage Klinis")
    st.caption("Menggunakan Arsitektur Hybrid: H2O GBM (Pola Non-Linear) + Logistic Regression (Safety Net NEWS2)")
    
    if st.session_state.get('model_ready'):
        use_gbm = st.session_state.coef.get('use_gbm', False)
        if not use_gbm:
            error_details = st.session_state.coef.get('error_msg', 'Unknown Error')
            st.warning("⚠️ Mode Terbatas: Komponen AI Lanjut (GBM) tidak aktif karena limitasi server. Prediksi menggunakan Logistic Regression murni.")
    
    st.markdown("---")

    col1, col2 = st.columns([1, 1.2])

    # --- KOLOM KIRI: INPUT DATA ---
    with col1:
        st.subheader("Data Pasien")
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
            
            submit_btn = st.form_submit_button("Analisis Risiko", type="primary")

    # --- KOLOM KANAN: HASIL ANALISIS ---
    with col2:
        st.subheader("Hasil & Rekomendasi")
        
        if submit_btn and st.session_state.get('model_ready'):
            # Preprocessing Input User
            try:
                if '/' in p_bp: p_sys, p_dia = map(float, p_bp.split('/'))
                else: p_sys, p_dia = 120.0, 80.0
            except: p_sys, p_dia = 120.0, 80.0

            valid_symptoms = [s for s in [p_sym1, p_sym2, p_sym3] if s != "-"]
            flags = extract_features_from_symptoms(valid_symptoms)
            
            # Siapkan Dictionary Lengkap untuk Input GBM
            input_dict_full = {
                'Age': p_age, 
                'Sys_Raw': p_sys, 'Dia_Raw': p_dia,
                'Oxygen_Raw': p_o2, 'Temp_Raw': p_temp, 'Heart_Raw': p_hr,
                'Flag_HTN_Crisis': 1 if p_sys >= 180 else 0,
                'Sym_Dyspnea': flags['Sym_Dyspnea'],
                'Sym_Fever': flags['Sym_Fever'],
                'Sym_Cough': flags['Sym_Cough']
            } 
            
            # Tahap 1: Prediksi AI (GBM)
            # AI melihat semuanya: Usia, Suhu, Demam, dll.
            s_score = 0.5
            use_gbm = st.session_state.coef.get('use_gbm', False)
            
            if use_gbm and st.session_state.gbm:
                # Konversi input dict ke H2O Frame
                # Kita perlu memastikan kolomnya sama persis dengan saat training
                # Trik: Buat DataFrame dengan kolom yang sesuai
                input_df_gbm = pd.DataFrame([input_dict_full])
                # H2O butuh tipe data yang pas, biasanya otomatis terhandle
                hf_sample = h2o.H2OFrame(input_df_gbm)
                try:
                    ml_pred = st.session_state.gbm.predict(hf_sample)
                    s_score = ml_pred['p1'].as_data_frame().values[0][0]
                except Exception as e:
                    s_score = 0.5 # Fallback
            
            # Tahap 2: Prediksi Manual (LogReg)
            # LogReg hanya melihat: ML_Score + Dyspnea + HTN
            final_prob, critical_reasons = calculate_final_prob(input_dict_full, s_score, st.session_state.coef)
            
            # Tampilan Hasil UI
            k1, k2, k3 = st.columns(3)
            k1.metric("Risiko Total", f"{final_prob:.1%}") 
            k2.metric("Skor AI (GBM)", f"{s_score:.2f}")
            k3.metric("Tanda Vital", f"SpO2: {int(p_o2)}%")
            
            threshold = 0.5 
            if final_prob > threshold:
                st.error(f"⚠️ RUJUKAN DIPERLUKAN")
                st.markdown("##### Indikasi Klinis Utama:")
                
                # Alasan Kritis (Override)
                if critical_reasons:
                    for reason in critical_reasons:
                        st.write(f"🔴 **{reason}** (Bahaya Nyawa/Organ)")
                else:
                    # Alasan Model Hybrid
                    if p_sys >= 180: st.write("🟠 **Krisis Hipertensi** (Risiko Stroke)")
                    if flags['Sym_Dyspnea']: st.write("🟠 **Keluhan Sesak Napas** (Gangguan Airway/Breathing)")
                    if s_score > 0.6: st.write(f"🔵 **Pola Klinis Kompleks** (Deteksi AI: Usia/Suhu/Nadi mendukung perburukan)")
            else:
                st.success(f"✅ PASIEN STABIL (Rawat Jalan)")
                st.write("Tidak ditemukan tanda bahaya akut mayor. Berikan terapi simptomatik.")
                
        elif not st.session_state.get('model_ready'):
             st.info("Silakan tunggu model selesai dilatih...")

    # --- MENU BAWAH: PENJELASAN MODEL ---
    st.markdown("---")
    with st.expander("🔍 Bedah Model: Rumus & Metrik (Untuk Laporan)", expanded=False):
        tab1, tab2 = st.tabs(["Rumus Matematika (Hybrid)", "Performa Model"])
        
        metrics = st.session_state.get('metrics')
        coeffs = st.session_state.get('coef')

        # Map variable name untuk display yang cantik
        variable_map = {
            'Intercept': 'Intercept (Base Risk)',
            'ML_Score': 'Skor Prediksi AI (GBM)',
            'Sym_Dyspnea': 'Gejala Sesak Napas (Safety Net)',
            'Flag_HTN_Crisis': 'Krisis Hipertensi (Safety Net)'
        }

        with tab1:
            st.markdown("### Formulasi Matematis Final")
            st.markdown(r"""
            Sistem menggunakan pendekatan **Stacked Ensemble** di mana model Machine Learning (GBM) menangkap pola kompleks (termasuk Usia, Demam, dll), 
            kemudian hasilnya dikoreksi oleh lapisan keamanan (Logistic Regression) yang fokus pada kegawatan akut (NEWS2).
            """)
            
            st.latex(r"P(Rujuk) = \frac{1}{1 + e^{-z}}")
            st.latex(r"z = \beta_0 + (\beta_1 \cdot Z_{ML\_Score}) + (\beta_2 \cdot Z_{Dyspnea}) + (\beta_3 \cdot Z_{HTN\_Crisis})")
            
            st.info("Catatan: Variabel **Usia** dan **Demam** tidak muncul di rumus ini karena sudah dihitung secara otomatis di dalam komponen $Z_{ML\\_Score}$.")

            if coeffs:
                st.markdown("#### Bobot Koefisien Aktual (Training Data)")
                # Bersihkan dictionary dari data scaler
                plot_coeffs = coeffs.copy()
                for k in ['scaler_mean', 'scaler_scale', 'scaler_cols', 'use_gbm', 'error_msg']:
                    if k in plot_coeffs: del plot_coeffs[k]

                coef_df = pd.DataFrame.from_dict(plot_coeffs, orient='index', columns=['Bobot (Log-Odds)'])
                plot_df = coef_df.drop('Intercept', errors='ignore')
                plot_df.index = plot_df.index.map(lambda x: variable_map.get(x, x))
                plot_df = plot_df.sort_values(by='Bobot (Log-Odds)', ascending=True)
                
                st.bar_chart(plot_df)

        with tab2:
            if metrics:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Area Under Curve (AUC)", f"{metrics['auc']:.4f}")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, label=f'AUC = {metrics["auc"]:.2f}')
                    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
                    ax.set_title('ROC Curve')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                with c2:
                    st.write("Confusion Matrix")
                    cm = metrics['cm']
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                    ax_cm.set_xlabel('Prediksi')
                    ax_cm.set_ylabel('Aktual')
                    st.pyplot(fig_cm)

else:
    st.error("Gagal memuat aplikasi.")