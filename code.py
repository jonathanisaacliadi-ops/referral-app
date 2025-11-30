import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import os
import io

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Rujukan Cerdas (Final)",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Fungsi Load Data ---
@st.cache_data
def load_dataset(uploaded_file=None):
    df = pd.DataFrame()

    # A. Cek Upload User
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast("Menggunakan file yang diupload.", icon="📂")
            return df
        except Exception as e:
            st.error(f"Error membaca file upload: {e}")

    # B. Cek File Lokal / GitHub
    local_path = "disease_diagnosis.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            return df
        except Exception as e:
            st.error(f"Error membaca file dataset: {e}")

    return df

# --- 3. Preprocessing & Feature Engineering ---

def get_column_options(df, col_name):
    """Mengambil nilai unik dari satu kolom spesifik."""
    if df.empty or col_name not in df.columns: return []
    items = df[col_name].unique()
    clean_items = [str(s).strip() for s in items if str(s) != 'nan' and str(s).strip() != '']
    return sorted(list(set(clean_items)))

def calculate_symptom_score(row_or_list):
    """Menghitung skor gejala (Weighted)."""
    score = 0
    symptoms_list = []

    # Jika input adalah Row DataFrame (saat training)
    if isinstance(row_or_list, pd.Series):
        items = [row_or_list.get('Symptom_1'), row_or_list.get('Symptom_2'), row_or_list.get('Symptom_3')]
        symptoms_list = [str(s).strip() for s in items if str(s) != 'nan']
    # Jika input adalah List (saat prediksi UI)
    else:
        symptoms_list = [str(s).strip() for s in row_or_list if s != "-" and s is not None]

    for sym in symptoms_list:
        if not sym: continue
        sym_lower = sym.lower()
        # Gejala Berat (Red Flags) - Bobot +3
        if sym_lower in ['shortness of breath', 'chest pain', 'high fever', 'severe cough', 'cyanosis']:
            score += 3
        # Gejala Biasa - Bobot +1
        else:
            score += 1
            
    return score

def calculate_vital_score(row):
    """Menghitung Vital Score (M)."""
    score = 0
    # Heart Rate
    try:
        hr = float(row['Heart_Rate_bpm'])
        if hr > 100 or hr < 60: score += 2
    except: pass
    
    # Suhu
    try:
        temp = float(row['Body_Temperature_C'])
        if temp > 38.0 or temp < 36.0: score += 2
        elif temp > 37.5: score += 1
    except: pass
        
    # Saturasi Oksigen
    try:
        o2 = float(row['Oxygen_Saturation_%'])
        if o2 < 95: score += 3
        elif o2 < 98: score += 1
    except: pass

    # Tekanan Darah
    try:
        val = str(row['Blood_Pressure_mmHg'])
        if '/' in val:
            systolic = int(val.split('/')[0])
            if systolic > 140 or systolic < 90: score += 2
    except: pass 
    
    return score

def preprocess_data(df):
    """Memproses Data Mentah -> Format Model."""
    processed = df.copy()
    
    # 1. Hitung Skor
    processed['Vital_Score'] = processed.apply(calculate_vital_score, axis=1)
    processed['Symptom_Score'] = processed.apply(calculate_symptom_score, axis=1)
    
    # 2. Target Variable
    # Severity Score dihapus dari input, hanya digunakan untuk menentukan target
    processed['Referral_Required'] = processed.apply(
        lambda x: 1 if str(x['Severity']).strip() == 'Severe' else 0, axis=1
    )
    
    # Return features (Tanpa Severity_Score di X untuk mencegah kebocoran data)
    return processed[['Age', 'Vital_Score', 'Symptom_Score', 'Referral_Required']]

# --- 4. Pelatihan Model (Versi Hemat RAM & Anti-Leakage) ---
@st.cache_resource
def train_hybrid_model(df_processed):
    # 1. Inisialisasi H2O Ringan
    try:
        # Batasi RAM 400M dan 1 Thread untuk mencegah Crash di Cloud
        h2o.init(max_mem_size='400M', nthreads=1) 
    except:
        st.error("Gagal inisialisasi H2O. Pastikan Java terinstal.")
        return None, None, None, None

    hf = h2o.H2OFrame(df_processed)
    y = 'Referral_Required'
    # FITUR INPUT (X): Hanya Age, Vital, dan Symptom.
    x = ['Age', 'Vital_Score', 'Symptom_Score']
    
    hf[y] = hf[y].asfactor()
    
    # 2. Train AutoML (Hanya GBM & DRF, Tanpa XGBoost)
    aml = H2OAutoML(
        max_models=3, 
        seed=42, 
        include_algos=['GBM', 'DRF'], # XGBoost dihapus karena boros memori
        max_runtime_secs=45,
        verbosity='error'
    ) 
    
    try:
        aml.train(x=x, y=y, training_frame=hf)
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None, None, None

    best_model_h2o = aml.leader
    
    # 3. Prediksi Skor ML (S)
    preds = best_model_h2o.predict(hf)
    ml_scores = preds['p1'].as_data_frame().values.flatten()
    
    # 4. Logistic Regression
    X_logreg = df_processed[x].copy()
    X_logreg['ML_Score'] = ml_scores 
    Y_logreg = df_processed[y]
    
    log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    log_reg.fit(X_logreg, Y_logreg)
    
    # 5. Evaluasi AUC
    y_prob_final = log_reg.predict_proba(X_logreg)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_logreg, y_prob_final)
    roc_auc = auc(fpr, tpr)
    
    metrics = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    coeffs = {
        'Intercept (beta_0)': log_reg.intercept_[0],
        'Age (beta_1)': log_reg.coef_[0][0],
        'Vital_Score (beta_2)': log_reg.coef_[0][1],
        'Symptom_Score (beta_3)': log_reg.coef_[0][2],
        'ML_Score (beta_4)': log_reg.coef_[0][3]
    }
    
    return best_model_h2o, log_reg, coeffs, metrics

# --- 5. Fungsi Kalkulasi Probabilitas ---
def calculate_final_prob(age, vital, sym_score, ml_score, coeffs):
    logit = (coeffs['Intercept (beta_0)'] + 
             coeffs['Age (beta_1)'] * age + 
             coeffs['Vital_Score (beta_2)'] * vital + 
             coeffs['Symptom_Score (beta_3)'] * sym_score + 
             coeffs['ML_Score (beta_4)'] * ml_score)
    
    prob = 1 / (1 + math.exp(-logit))
    return prob

# --- MAIN APP ---

st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Opsional)", type=["csv"])

# Load Data
df_raw = load_dataset(uploaded_file)

if not df_raw.empty:
    # Ambil opsi untuk Dropdown Gejala
    s1_options = get_column_options(df_raw, 'Symptom_1')
    s2_options = ["-"] + get_column_options(df_raw, 'Symptom_2')
    s3_options = ["-"] + get_column_options(df_raw, 'Symptom_3')
    
    df_model = preprocess_data(df_raw)

    st.sidebar.title("⚙️ Kontrol Model")
    if st.sidebar.button("🔄 Latih Ulang Model"):
        try:
            h2o.cluster().shutdown()
        except: pass
        st.cache_resource.clear()
        st.rerun()

    with st.spinner("Melatih model (Mode Stabil & Akurat)..."):
        gbm_model, logreg_model, coefficients, metrics = train_hybrid_model(df_model)

    # --- Sidebar: Metrik & Rumus ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Evaluasi Model")
    
    if metrics:
        st.sidebar.metric("ROC AUC Score", f"{metrics['auc']:.4f}")
        
        # Plot ROC
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, label=f'AUC = {metrics["auc"]:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.sidebar.pyplot(fig)

    st.sidebar.markdown("### 🧮 Rumus Prediksi")
    if coefficients:
        formula_latex = f"""
        $$
        P(Y=1) = \\frac{{1}}{{1 + e^{{-z}}}}
        $$
        Dimana z adalah:
        $$
        {coefficients['Intercept (beta_0)']:.2f} + 
        {coefficients['Age (beta_1)']:.2f}(Age) + 
        {coefficients['Vital_Score (beta_2)']:.2f}(Vital) + 
        {coefficients['Symptom_Score (beta_3)']:.2f}(Sym) + 
        {coefficients['ML_Score (beta_4)']:.2f}(S)
        $$
        """
        st.sidebar.latex(formula_latex)

    # --- UI Utama ---
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.header("🩺 Triage Pasien")
        st.write("Masukkan parameter klinis pasien:")
        
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
            st.caption("Pilih gejala sesuai data historis yang tersedia.")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                p_sym1 = st.selectbox("Gejala Utama", options=s1_options)
            with sc2:
                p_sym2 = st.selectbox("Gejala ke-2", options=s2_options)
            with sc3:
                p_sym3 = st.selectbox("Gejala ke-3", options=s3_options)
            
            submit_btn = st.form_submit_button("Analisis Keputusan")
        
        if submit_btn:
            selected_symptoms_raw = [p_sym1, p_sym2, p_sym3]
            valid_symptoms = [s for s in selected_symptoms_raw if s != "-"]
            
            # Hitung Skor Input
            row_vital = {'Heart_Rate_bpm': p_hr, 'Body_Temperature_C': p_temp, 
                         'Oxygen_Saturation_%': p_o2, 'Blood_Pressure_mmHg': p_bp}
            p_vital_score = calculate_vital_score(row_vital)
            p_sym_score = calculate_symptom_score(valid_symptoms)
            
            # Prediksi ML (H2O)
            input_h2o = pd.DataFrame({
                'Age': [p_age], 
                'Vital_Score': [p_vital_score],
                'Symptom_Score': [p_sym_score]
            })
            hf_sample = h2o.H2OFrame(input_h2o)
            ml_pred = gbm_model.predict(hf_sample)
            s_score = ml_pred['p1'].as_data_frame().values[0][0]
            
            # Kalkulasi Probabilitas Akhir
            final_prob = calculate_final_prob(p_age, p_vital_score, p_sym_score, s_score, coefficients)
            
            # Tampilan Hasil
            st.markdown("---")
            st.subheader("📋 Hasil Analisis")
            
            if valid_symptoms:
                st.write("**Gejala:** " + ", ".join(valid_symptoms))
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Vital Score", f"{p_vital_score}")
            r2.metric("Symptom Score", f"{p_sym_score}")
            r3.metric("ML Score (S)", f"{s_score:.3f}")
            r4.metric("Probabilitas", f"{final_prob:.1%}")
            
            threshold = 0.65
            if final_prob > threshold:
                st.error(f"🚨 **RUJUKAN DIPERLUKAN**")
                st.markdown(f"Probabilitas **{final_prob:.1%}** melebihi batas {threshold}. Disarankan rujuk ke RS.")
            else:
                st.success(f"✅ **TIDAK PERLU RUJUKAN**")
                st.markdown(f"Probabilitas **{final_prob:.1%}**. Pasien dapat ditangani di FKTP.")

    with col2:
        st.subheader("📊 Statistik Dataset")
        st.write(f"Total Data: {len(df_raw)} pasien")
        st.dataframe(df_raw[['Age', 'Symptom_1', 'Diagnosis', 'Severity']].head(10), hide_index=True)
        
        if coefficients:
            st.write("---")
            st.write("**Bobot Fitur (Koefisien):**")
            coef_chart = pd.DataFrame.from_dict(coefficients, orient='index', columns=['Value'])
            coef_chart = coef_chart.drop('Intercept (beta_0)')
            st.bar_chart(coef_chart)

else:
    st.error("File 'disease_diagnosis.csv' tidak ditemukan.")