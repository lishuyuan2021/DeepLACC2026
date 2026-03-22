import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import platform

# 1. Solve OpenMP conflict & Paths
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global Plotting Settings
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ==============================================================================
# 1. Define Model Architecture (Strictly matched: 28 features, 128 nodes, 3 layers)
# ==============================================================================
class DeepSurvNet(nn.Module):
    def __init__(self, in_features=28): 
        super(DeepSurvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 1) # Output: Log Hazard Ratio
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. Resource Loading
# ==============================================================================
@st.cache_resource
def load_resources():
    # STRICT feature sequence from your Training Script
    features_list = [
        "age", "raceWhite", "raceBlack", "raceAmerican Indian/Alaska Native",
        "adjuvant.chemotherapyYes", "CEAPositive", "primary.siteAscending Colon", 
        "primary.siteHepatic Flexure", "primary.siteTransverse Colon", "primary.siteSplenic Flexure",
        "primary.siteDescending Colon", "primary.siteSigmoid Colon", 
        "primary.siteRectosigmoid Junction", "One.primary.onlyYes", 
        "First.malignant.pri.Yes", "tumor.depositsPositive", "regional.nodes.positive", 
        "Perineural.InvasionYes", "regional.nodes.examined", "TN.stagepT4N0", 
        "TN.stageypT0-2N+", "TN.stageypT0-2N0", "TN.stageypT3N+", "TN.stageypT3N0", 
        "TN.stageypT4N+", "TN.stageypT4N0", "GradeModerately differentiated", 
        "GradePoorly differentiated/Undifferentiated"
    ]
    
    # Load Weights
    model = DeepSurvNet(28)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    # Load Scalers, Background, and Baseline
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    # Display Names for SHAP plot
    feature_labels_en = [
        "Age", "Race: White", "Race: Black", "Race: Native",
        "Adjuvant Chemotherapy", "CEA Positive", "Site: Ascending", 
        "Site: Hepatic Flex", "Site: Transverse", "Site: Splenic Flex",
        "Site: Descending", "Site: Sigmoid", "Site: Rectosigmoid", 
        "Only One Primary", "First Malignant Tumor", "Tumor Deposits (+)", 
        "Positive Nodes Count", "Perineural Invasion (PNI)", "Nodes Examined", 
        "Stage: pT4N0", "Stage: ypT0-2N+", "Stage: ypT0-2N0", "Stage: ypT3N+", 
        "Stage: ypT3N0", "Stage: ypT4N+", "Stage: ypT4N0", 
        "Grade: Moderate", "Grade: Poor/Undiff"
    ]
    
    return model, scalers, bg_data, base_surv, features_list, feature_labels_en

model, scalers, bg_data, base_surv, features_list, feature_labels_en = load_resources()

# ==============================================================================
# 3. UI Layout
# ==============================================================================
st.set_page_config(page_title="DeepSurv-LACC-App", layout="wide")
st.title("Prognosis Prediction System for Locally Advanced Colon Cancer (LACC)")
st.markdown("Individualized risk calculation powered by **Native PyTorch DeepSurv**. Data Source: SEER database.")

with st.sidebar:
    st.header("1. Demographics & Clinical History")
    age = st.slider("Age at Diagnosis", 18, 95, 60)
    race = st.selectbox("Race", ["Asian/Pacific Islander", "White", "Black", "American Indian/Alaska Native"])
    first_malig = st.selectbox("Was colon cancer the FIRST primary cancer?", ["Yes", "No"])

    st.header("2. Diagnosis")
    cea = st.selectbox("CEA Status (Pre-op)", ["Negative", "Positive"])
    site = st.selectbox("Primary Site", ["Cecum", "Ascending Colon", "Hepatic Flexure", "Transverse Colon", 
                                        "Splenic Flexure", "Descending Colon", "Sigmoid Colon", "Rectosigmoid Junction"])
    primary_only = st.selectbox("Is this the ONLY primary site?", ["Yes", "No"])
    
    st.header("3. Pathological Characteristics")
    nodes_pos = st.number_input("Positive Nodes Count", 0, 80, 1)
    nodes_exam = st.number_input("Total Examined Nodes", 1, 90, 15)
    tn_stage = st.selectbox("Combined TN Stage", [
        "pT4N+", "pT4N0", "ypT0-2N+", "ypT0-2N0", "ypT3N+", "ypT3N0", "ypT4N+", "ypT4N0"
    ])
    grade = st.selectbox("Histological Grade", ["Grade I (Well)", "Grade II (Moderate)", "Grade III/IV (Poor/Undiff)"])
    pni = st.selectbox("Perineural Invasion (PNI)", ["No", "Yes"])
    deposits = st.selectbox("Tumor Deposits (TD)", ["No", "Yes"])
    
    st.header("4. Postoperative Management")
    therapy = st.selectbox("Adjuvant Chemotherapy (AC)", ["Untreated/Refused", "Treated"])

# ==============================================================================
# 4. Analysis Logic
# ==============================================================================
if st.sidebar.button("🚀 Analyze Survival Risk", type="primary"):
    # Initialize a dictionary with all 28 features set to 0.0
    input_dict = {f: 0.0 for f in features_list}
    
    # A. Numeric Standardization
    input_dict["age"] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    input_dict["regional.nodes.positive"] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    input_dict["regional.nodes.examined"] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    
    # B. Categorical One-Hot Mapping
    if race == "White": input_dict["raceWhite"] = 1.0
    elif race == "Black": input_dict["raceBlack"] = 1.0
    elif race == "American Indian/Alaska Native": input_dict["raceAmerican Indian/Alaska Native"] = 1.0
    
    if therapy == "Treated": input_dict["adjuvant.chemotherapyYes"] = 1.0
    if cea == "Positive": input_dict["CEAPositive"] = 1.0
    if pni == "Yes": input_dict["Perineural.InvasionYes"] = 1.0
    if deposits == "Yes": input_dict["tumor.depositsPositive"] = 1.0
    if primary_only == "Yes": input_dict["One.primary.onlyYes"] = 1.0
    if first_malig == "Yes": input_dict["First.malignant.pri.Yes"] = 1.0
    
    site_map = {
        "Ascending Colon": "primary.siteAscending Colon", "Hepatic Flexure": "primary.siteHepatic Flexure", 
        "Transverse Colon": "primary.siteTransverse Colon", "Splenic Flexure": "primary.siteSplenic Flexure",
        "Descending Colon": "primary.siteDescending Colon", "Sigmoid Colon": "primary.siteSigmoid Colon", 
        "Rectosigmoid Junction": "primary.siteRectosigmoid Junction"
    }
    if site in site_map: input_dict[site_map[site]] = 1.0
    
    tn_map = {
        "pT4N0": "TN.stagepT4N0", "ypT0-2N+": "TN.stageypT0-2N+", 
        "ypT0-2N0": "TN.stageypT0-2N0", "ypT3N+": "TN.stageypT3N+", 
        "ypT3N0": "TN.stageypT3N0", "ypT4N+": "TN.stageypT4N+", 
        "ypT4N0": "TN.stageypT4N0"
    }
    if tn_stage in tn_map: input_dict[tn_map[tn_stage]] = 1.0
    
    if grade == "Grade II (Moderate)": input_dict["GradeModerately differentiated"] = 1.0
    elif grade == "Grade III/IV (Poor/Undiff)": input_dict["GradePoorly differentiated/Undifferentiated"] = 1.0

    # C. Convert Dictionary to NumPy using strict sequence
    input_vec = np.array([input_dict[f] for f in features_list], dtype=np.float32)
    input_tensor = torch.from_numpy(input_vec).view(1, -1)

    # D. Inference
    with torch.no_grad():
        log_h = model(input_tensor).item()
        rr = np.exp(log_h)

    # Risk Stratification (From training percentiles)
    T_L, T_H = 0.9051, 1.1772 
    if rr < T_L: g, color, sug = "Low Risk", "#28a745", "Patients with low-risk disease have a relatively favorable prognosis compared to all other LACC cases. It is recommended to consider whether to pursue further treatment based on individual circumstances and to undergo regular follow-up examinations."
    elif rr <= T_H: g, color, sug = "Medium Risk", "#fd7e14", "For patients at moderate risk, it is recommended that follow-up treatment plans be determined based on clinical guidelines and the patient’s specific condition; close follow-up and regular checkups are strongly recommended."
    else: g, color, sug = "High Risk", "#dc3545", "High-risk patients! It is strongly recommended that follow-up treatment be considered in accordance with guidelines and the patient’s specific condition, with close follow-up and regular check-ups, and a comprehensive physical examination if necessary."

    # OS Probabilities
    surv_p = {t: (base_surv[str(t)] ** rr) * 100 for t in [12, 36, 60, 120]}

    # Layout for Results
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white'>
                <p style='margin:0; font-size:16px;'>Prediction Stratification</p>
                <h2 style='color:white; margin:5px 0;'>{g}</h2>
                <p style='margin:0; font-size:14px; opacity:0.8;'>Relative Risk (RR): {rr:.3f}</p>
            </div>""", unsafe_allow_html=True)
        
        st.write("")
        st.subheader("OS Probability Projections")
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.metric("1-Year OS", f"{surv_p[12]:.1f}%")
        with r1_c2: st.metric("3-Year OS", f"{surv_p[36]:.1f}%")
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.metric("5-Year OS", f"{surv_p[60]:.1f}%")
        with r2_c2: st.metric("10-Year OS", f"{surv_p[120]:.1f}%")
        st.divider()
        st.info(f"**💡 Clinical Suggestion:** \n\n {sug}")

    with col2:
        st.subheader("Feature Contribution Attribution (SHAP)")
        with st.spinner("AI decoding contribution factors..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_val_raw = explainer.shap_values(input_tensor)
            
            # Data structure robust extraction
            s_val = np.squeeze(shap_val_raw[0]) if isinstance(shap_val_raw, list) else np.squeeze(shap_val_raw)
            ev = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

            fig, ax = plt.subplots(figsize=(10, 8))
            e_obj = shap.Explanation(
                values=s_val, 
                base_values=float(ev), # Scaler expectation
                data=input_vec, 
                feature_names=feature_labels_en
            )
            shap.plots.waterfall(e_obj, max_display=12, show=False)
            plt.title("Prognostic Indicator Contributions (Waterfall Plot)", pad=20)
            st.pyplot(fig)

st.divider()
st.caption("Native PyTorch | CRC Dept, Shanghai Changhai Hospital. Use for academic research only.")