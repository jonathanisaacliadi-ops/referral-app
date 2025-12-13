import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    page_title="Sistem Triage Medis (Final Lengkap + ROC)",
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
    
    # Flag High BP (Gray Zone) > 140/90 masuk ke model
    processed['Flag_High_BP'] = ((processed['Sys_Raw'] > 140) | (processed['Dia_Raw'] > 90)).astype(int)
    
    # Flag Crisis (>180) untuk Golden Rule (nanti di-drop dari training)
    processed['Flag_HTN_Crisis'] = (processed['Sys_Raw'] >= 180).astype(int)
    
    flags = processed.apply(extract_features_from_symptoms, axis=1)
    flags_df = pd.DataFrame(flags.tolist(), index=processed.index)
    processed = pd.concat([processed, flags_df], axis=1)
    
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    return processed[[
        'Age', 
        'Sys_Raw', 'Dia_Raw', 'Oxygen_Raw', 'Temp_Raw', 'Heart_Raw', 
        'Flag_High_BP',     
        'Flag_HTN_Crisis',  
        'Sym_Dyspnea', 'Sym_Fever',
        'Referral_Required'
    ]]

# --- 4. Pelatihan Model (H2O + LogReg) ---
@st.cache_resource
def train_medical_model(df_processed):
    use_gbm = True
    best_model = None
    error_msg = ""
    
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

    # Drop Flag_HTN_Crisis dari training karena sudah jadi Golden Rule
    X = df_processed.drop(['Referral_Required', 'Flag_HTN_Crisis'], axis=1)
    y = df_processed['Referral_Required']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        return None, None, None, None 

    s_train = None
    s_test = None

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

    def get_lr_features(df_orig, ml_scores, use_ml):
        df_new = pd.DataFrame(index=df_orig.index)
        if use_ml and ml_scores is not None:
            df_new['ML_Score'] = ml_scores
        
        # Fitur LogReg: Flag High BP & Dyspnea
        df_new['Flag_High_BP'] = df_orig['Flag_High_BP'] 
        df_new['Sym_Dyspnea'] = df_orig['Sym_Dyspnea'] 
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
    
    y_prob = log_reg.predict_proba(X_test_final)[:, 1]
    
    # --- METRICS CALCULATION ---
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    def calc_metrics(y_true, y_pred):
        return {
            'acc': accuracy_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, zero_division=0),
            'rec': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'cm': confusion_matrix(y_true, y_pred)
        }

    # 1. Default Threshold (0.5)
    y_pred_def = (y_prob >= 0.5).astype(int)
    m_def = calc_metrics(y_test, y_pred_def)

    # 2. ROC Optimal
    youden_idx = np.argmax(tpr - fpr)
    thresh_roc = thresholds[youden_idx]
    y_pred_roc = (y_prob >= thresh_roc).astype(int)
    m_roc = calc_metrics(y_test, y_pred_roc)

    # 3. Max Accuracy
    accuracy_list = []
    for th in thresholds:
        y_pred_temp = (y_prob >= th).astype(int)
        accuracy_list.append(accuracy_score(y_test, y_pred_temp))
    
    max_acc_idx = np.argmax(accuracy_list)
    thresh_acc = thresholds[max_acc_idx]
    y_pred_acc = (y_prob >= thresh_acc).astype(int)
    m_acc = calc_metrics(y_test, y_pred_acc)
    
    metrics = {
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
        
        # Default
        'cm_default': m_def['cm'], 'acc_default': m_def['acc'],
        'prec_default': m_def['prec'], 'rec_default': m_def['rec'], 'f1_default': m_def['f1'],
        
        # ROC Optimal
        'cm_roc': m_roc['cm'], 'acc_roc': m_roc['acc'],
        'prec_roc': m_roc['prec'], 'rec_roc': m_roc['rec'], 'f1_roc': m_roc['f1'],
        'thresh_roc': thresh_roc,
        
        # Max Accuracy
        'cm_acc': m_acc['cm'], 'acc_max': m_acc['acc'],
        'prec_max': m_acc['prec'], 'rec_max': m_acc['rec'], 'f1_max': m_acc['f1'],
        'thresh_acc': thresh_acc
    }
    
    coeffs = {'Intercept': log_reg.intercept_[0]}
    for i, col in enumerate(cols_lr):
        coeffs[col] = log_reg.coef_[0][i]
    
    coeffs['scaler_mean'] = scaler.mean_.tolist()
    coeffs['scaler_scale'] = scaler.scale_.tolist()
    coeffs['scaler_cols'] = list(cols_lr)
    coeffs['use_gbm'] = use_gbm 
    coeffs['error_msg'] = error_msg 
        
    return best_model, log_reg, coeffs, metrics


def calculate_final_prob(input_dict, ml_score, coeffs):
    critical_reasons = []
    
    # --- GOLDEN RULE LOGIC ---
    
    # 1. KRISIS HIPERTENSI
    if input_dict['Sys_Raw'] >= 180:
        critical_reasons.append("Krisis Hipertensi (Sistolik >= 180 mmHg) [GOLDEN RULE]")
    
    # 2. SATURASI OKSIGEN KRITIS (Threshold <= 88)
    if input_dict['Oxygen_Raw'] <= 88: 
        critical_reasons.append("Saturasi Oksigen Kritis (<=88%) [GOLDEN RULE]")
        
    # 3. Tanda Vital Ekstrem Lainnya
    if input_dict['Temp_Raw'] >= 39.5: critical_reasons.append("Hiperpireksia (>=39.5°C)")
    if input_dict['Heart_Raw'] >= 140: critical_reasons.append("Takikardia Ekstrem (>=140 bpm)")
    
    if critical_reasons:
        return 0.999, critical_reasons 

    # --- PERHITUNGAN MODEL ---
    try:
        means = np.array(coeffs['scaler_mean'])
        scales = np.array(coeffs['scaler_scale'])
        cols = coeffs['scaler_cols']
        use_gbm = coeffs.get('use_gbm', False)
    except KeyError:
        return 0.5, []
    
    data_row = {
        'Flag_High_BP': input_dict['Flag_High_BP'],
        'Sym_Dyspnea': input_dict['Sym_Dyspnea']
    }
    
    if use_gbm:
        data_row['ML_Score'] = ml_score
    
    input_values = []
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

with st.sidebar:
    st.header("Kontrol")
    if st.button("Reset / Latih Ulang Model"):
        st.cache_resource.clear()
        if 'model_ready' in st.session_state:
            del st.session_state['model_ready']
        st.rerun()

df_raw = load_fixed_dataset()

if not df_raw.empty:
    df_model = preprocess_data(df_raw)
    
    if 'model_ready' not in st.session_state:
        with st.spinner("Memproses Model..."):
            gbm, logreg, coef, metr = train_medical_model(df_model)
            
            if logreg is not None:
                st.session_state.gbm = gbm
                st.session_state.logreg = logreg
                st.session_state.coef = coef
                st.session_state.metrics = metr
                st.session_state.model_ready = True
            else:
                st.error("Gagal melatih model dasar.")

    st.title("Sistem Triage Medis")
    st.write("Aturan Emas: O2 ≤ 88% atau Sistolik ≥ 180 mmHg = Rujuk Otomatis.")
    
    if st.session_state.get('model_ready'):
        use_gbm = st.session_state.coef.get('use_gbm', False)
        if not use_gbm:
            error_details = st.session_state.coef.get('error_msg', 'Unknown Error')
            st.warning("⚠️ Mode Terbatas: LogReg Only.")
    
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
                'Flag_High_BP': 1 if (p_sys > 140 or p_dia > 90) else 0, 
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

            final_prob, critical_reasons = calculate_final_prob(input_dict_full, s_score, st.session_state.coef)
            
            k1, k2 = st.columns(2)
            k1.metric("Risiko Rujukan", f"{final_prob:.1%}") 
            k2.metric("Tekanan Darah", f"{int(p_sys)}/{int(p_dia)}")
            
            metrics_data = st.session_state.get('metrics', {})
            threshold = 0.5
            best_thresh = metrics_data.get('thresh_acc', 0.5)
            
            st.caption(f"Threshold Default: 0.5 | Recommended: {best_thresh:.3f}")

            if final_prob > threshold:
                st.error(f"RUJUKAN DIPERLUKAN (Risiko {final_prob:.1%} > {threshold})")
                st.write("Indikasi Klinis:")
                
                if critical_reasons:
                    for reason in critical_reasons:
                        st.warning(f"- {reason} [CRITICAL]")
                else:
                    if (p_sys > 140 or p_dia > 90): st.warning("- Hipertensi (High BP Flag)")
                    if flags['Sym_Dyspnea']: st.warning("- Keluhan Sesak Napas")
                    if flags['Sym_Fever']: st.warning("- Gejala Demam")
                    if s_score > 0.7: st.warning("- Pola Vital Mencurigakan (AI)")
            else:
                st.success(f"TIDAK PERLU RUJUKAN (Risiko {final_prob:.1%} <= {threshold})")
                st.write("Kondisi stabil. Rawat jalan dengan obat simptomatik.")
                
        elif not st.session_state.get('model_ready'):
             st.info("Silakan isi data pasien di sebelah kiri dan klik 'Analisis Keputusan'.")


    st.markdown("---")
    with st.expander("Detail Model & Metrik", expanded=False):
        tab1, tab2 = st.tabs(["Performa & Metrik", "Bobot Variabel"])
        
        metrics = st.session_state.get('metrics')
        coeffs = st.session_state.get('coef')

        variable_map = {
            'Intercept': 'Intercept (Nilai Bias)',
            'ML_Score': 'Skor AI (GBM)',
            'Sym_Dyspnea': 'Gejala Sesak',
            'Flag_High_BP': 'Tensi Tinggi (>140/90)'
        }

        with tab1:
            if metrics:
                # 1. AUC
                st.metric("Skor AUC", f"{metrics['auc']:.4f}")
                
                st.markdown("---")
                st.write("### Perbandingan Performa")
                
                cm_def = metrics.get('cm_default')
                cm_roc = metrics.get('cm_roc')
                cm_acc = metrics.get('cm_acc')
                
                col_cm1, col_cm2, col_cm3 = st.columns(3)
                
                def make_labels(cm):
                    if cm is None: return np.array([["",""],["",""]])
                    names = ['TN', 'FP', 'FN', 'TP']
                    counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
                    labels = [f"{v1}\n{v2}" for v1, v2 in zip(names, counts)]
                    return np.asarray(labels).reshape(2,2)

                # --- METRIK F1, REC, PREC ---
                with col_cm1:
                    st.write(f"**A. Default (0.5)**")
                    st.caption(f"Acc: {metrics.get('acc_default', 0):.1%} | Rec: {metrics.get('rec_default', 0):.1%}")
                    st.caption(f"Prec: {metrics.get('prec_default', 0):.1%} | F1: {metrics.get('f1_default', 0):.1%}")
                    if cm_def is not None:
                        fig1, ax1 = plt.subplots(figsize=(3, 2.5))
                        sns.heatmap(cm_def, annot=make_labels(cm_def), fmt='', cmap='Blues', cbar=False, ax=ax1)
                        st.pyplot(fig1)

                with col_cm2:
                    st.write(f"**B. ROC Opt ({metrics.get('thresh_roc', 0):.3f})**")
                    st.caption(f"Acc: {metrics.get('acc_roc', 0):.1%} | Rec: {metrics.get('rec_roc', 0):.1%}")
                    st.caption(f"Prec: {metrics.get('prec_roc', 0):.1%} | F1: {metrics.get('f1_roc', 0):.1%}")
                    if cm_roc is not None:
                        fig2, ax2 = plt.subplots(figsize=(3, 2.5))
                        sns.heatmap(cm_roc, annot=make_labels(cm_roc), fmt='', cmap='Purples', cbar=False, ax=ax2)
                        st.pyplot(fig2)

                with col_cm3:
                    st.write(f"**C. Max Acc ({metrics.get('thresh_acc', 0):.3f})**")
                    st.caption(f"Acc: {metrics.get('acc_max', 0):.1%} | Rec: {metrics.get('rec_max', 0):.1%}")
                    st.caption(f"Prec: {metrics.get('prec_max', 0):.1%} | F1: {metrics.get('f1_max', 0):.1%}")
                    if cm_acc is not None:
                        fig3, ax3 = plt.subplots(figsize=(3, 2.5))
                        sns.heatmap(cm_acc, annot=make_labels(cm_acc), fmt='', cmap='Greens', cbar=False, ax=ax3)
                        st.pyplot(fig3)
                
                # --- [FIXED] GRAFIK ROC DIKEMBALIKAN DI SINI ---
                st.markdown("---")
                st.write("#### Kurva ROC")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(metrics['fpr'], metrics['tpr'], color='blue', lw=2, label='ROC curve')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_title('ROC Curve')
                st.pyplot(fig)

        # --- TABEL DAN INTERCEPT ---
        with tab2:
            if coeffs:
                st.markdown("#### Detail Bobot Variabel")
                
                # Copy data agar aman
                plot_data = coeffs.copy()
                table_data = coeffs.copy()
                
                # A. DATA UNTUK GRAFIK (Tanpa Intercept, Tanpa Metadata)
                for k in ['scaler_mean', 'scaler_scale', 'scaler_cols', 'use_gbm', 'error_msg', 'Intercept']:
                    if k in plot_data: del plot_data[k]
                
                # B. DATA UNTUK TABEL (Ada Intercept, Tanpa Metadata)
                for k in ['scaler_mean', 'scaler_scale', 'scaler_cols', 'use_gbm', 'error_msg']:
                    if k in table_data: del table_data[k]

                # 1. Plot Grafik
                df_plot = pd.DataFrame.from_dict(plot_data, orient='index', columns=['Bobot'])
                df_plot['Bobot'] = pd.to_numeric(df_plot['Bobot'], errors='coerce')
                df_plot = df_plot.dropna().sort_values(by='Bobot', ascending=False)
                df_plot.index = df_plot.index.map(lambda x: variable_map.get(x, x))
                st.bar_chart(df_plot)
                
                # 2. Tampilkan Tabel (Dengan Intercept)
                df_table = pd.DataFrame.from_dict(table_data, orient='index', columns=['Bobot'])
                df_table['Bobot'] = pd.to_numeric(df_table['Bobot'], errors='coerce')
                df_table = df_table.dropna().sort_values(by='Bobot', ascending=False)
                df_table.index = df_table.index.map(lambda x: variable_map.get(x, x))
                
                st.write("### Tabel Angka Presisi (Termasuk Intercept)")
                st.dataframe(df_table.style.format("{:.4f}"))

else:
    st.error("Gagal memulai aplikasi.")