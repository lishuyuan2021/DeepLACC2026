import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# 1. Solve OpenMP conflict for Windows local runs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. Define Model Architecture (Aligned with 28 features, 128 nodes, 3 layers)
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
    model = DeepSurvNet(28)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    # Feature list matching the training sequence
    feature_list = [
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
    return model, scalers, bg_data, base_surv, feature_list

model, scalers, bg_data, base_surv, feature_list = load_resources()

# ==============================================================================
# 3. UI Layout
# ==============================================================================
st.set_page_config(page_title="DeepSurv-LACC-App", layout="wide")
st.title("Prognosis Prediction System for Locally Advanced Colon Cancer (LACC)")
st.markdown("Individualized risk calculation powered by **Native PyTorch DeepSurv**. Data Source: SEER database.")

with st.sidebar:
    st.header("1. Demographics")
    age = st.slider("Age at diagnosis", 18, 79, 60)
    race = st.selectbox("Race", ["Asian/Pacific Islander", "White", "Black", "American Indian/Alaska Native"])
    
    st.header("2. Diagnosis & Pathology")
    therapy = st.selectbox("Adjuvant Chemotherapy (AC)", ["Untreated", "Treated"])
    cea = st.selectbox("CEA Status (Pre-op)", ["Negative", "Positive"])
    pni = st.selectbox("Perineural Invasion (PNI)", ["No", "Yes"])
    deposits = st.selectbox("Tumor Deposits (TD)", ["No", "Yes"])
    
    st.header("3. Surgical Stage")
    nodes_pos = st.number_input("Number of Positive Nodes", 0, 80, 1)
    nodes_exam = st.number_input("Total Examined Nodes", 1, 90, 15)
    tn_stage = st.selectbox("TNM Combined Stage", [
        "pT4N+", "pT4N0", "ypT0-2N+", "ypT0-2N0", "ypT3N+", "ypT3N0", "ypT4N+", "ypT4N0"
    ])
    grade = st.selectbox("Histological Grade", ["Grade I (Well)", "Grade II (Moderate)", "Grade III/IV (Poor)"])
    site = st.selectbox("Primary Site", ["Cecum/Others", "Ascending Colon", "Hepatic Flexure", "Transverse Colon", 
                                        "Splenic Flexure", "Descending Colon", "Sigmoid Colon", "Rectosigmoid Junction"])
    
    st.header("4. History")
    primary_only = st.selectbox("Is this the ONLY primary site?", ["Yes", "No"])
    first_malig = st.selectbox("Was colon cancer the first primary cancer in lifetime?", ["Yes", "No"])

# 执行预测 (严格字符对齐)
if st.sidebar.button("🚀 Analyze Survival Risk", type="primary"):
    input_vec = np.zeros(28)
    input_vec[0] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    if race == "White": input_vec[1] = 1
    elif race == "Black": input_vec[2] = 1
    elif race == "American Indian/Alaska Native": input_vec[3] = 1
    
    # 修正逻辑：必须匹配 Treated / Positive
    if therapy == "Treated": input_vec[4] = 1
    if cea == "Positive": input_vec[5] = 1
    
    site_map = {"Ascending Colon": 6, "Hepatic Flexure": 7, "Transverse Colon": 8, "Splenic Flexure": 9, "Descending Colon": 10, "Sigmoid Colon": 11, "Rectosigmoid Junction": 12}
    if site in site_map: input_vec[site_map[site]] = 1
    
    # 修正：Yes 映射为 1
    if primary_only == "Yes": input_vec[13] = 1
    if first_malig == "Yes": input_vec[14] = 1
    
    if deposits == "Yes": input_vec[15] = 1
    input_vec[16] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    if pni == "Yes": input_vec[17] = 1
    input_vec[18] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    
    tn_map = {"pT4N0": 19, "ypT0-2N+": 20, "ypT0-2N0": 21, "ypT3N+": 22, "ypT3N0": 23, "ypT4N+": 24, "ypT4N0": 25}
    if tn_stage in tn_map: input_vec[tn_map[tn_stage]] = 1
    
    if grade == "Grade II (Moderate)": input_vec[26] = 1
    elif grade == "Grade III/IV (Poor)": input_vec[27] = 1

    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)
    with torch.no_grad():
        log_h = model(input_tensor).item()
        rr = np.exp(log_h)

    # Risk Stratification based on training 40%/80% percentiles
    T_L, T_H = 0.9495, 1.3921 
    if rr < T_L: group, color, suggestion = "Low Risk", "#28a745", "Recommend routine guidelines; standard follow-up."
    elif rr <= T_H: group, color, suggestion = "Medium Risk", "#fd7e14", "Close follow-up and follow-up examinations should be considered."
    else: group, color, suggestion = "High Risk", "#dc3545", "High risk! It is strongly recommended to provide appropriate postoperative care in accordance with the guidelines and the patient’s specific condition, and to conduct close follow-up."

    # Calculated Probabilities from baseline
    surv_probs = {t: (base_surv[str(t)] ** rr) * 100 for t in [12, 36, 60, 120]}

    # Layout for Results
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white'>
                <p style='margin:0; font-size:16px;'>Prediction Stratification</p>
                <h2 style='color:white; margin:5px 0; font-weight: bold; border:none;'>{group}</h2>
                <p style='margin:0; font-size:14px; opacity:0.8;'>Relative Risk (RR): {rr:.3f}</p>
            </div>""", unsafe_allow_html=True)
        
        st.write("")
        st.subheader("OS Probability Projections")
        
        # Grid Display: 1, 3, 5, 10 Years
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.metric("1-Year OS", f"{surv_probs[12]:.1f}%")
        with r1_c2: st.metric("3-Year OS", f"{surv_probs[36]:.1f}%")
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.metric("5-Year OS", f"{surv_probs[60]:.1f}%")
        with r2_c2: st.metric("10-Year OS", f"{surv_probs[120]:.1f}%")
        
        st.divider()
        st.info(f"**Management Strategy:** \n\n {suggestion}")

    with col2:
        st.subheader("Feature Contribution Interpretation (SHAP)")
        with st.spinner("AI engine decoding contribution factors..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_vals = explainer.shap_values(input_tensor)
            
            # Robust extraction of SHAP array
            s_val = np.squeeze(shap_vals[0]) if isinstance(shap_vals, list) else np.squeeze(shap_vals)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            exp_obj = shap.Explanation(
                values=s_val, 
                base_values=explainer.expected_value[0], 
                data=input_vec, 
                feature_names=feature_list
            )
            shap.plots.waterfall(exp_obj, max_display=12, show=False)
            plt.title("Prognostic Attribution Score (Waterfall)", pad=15)
            st.pyplot(fig)

st.divider()
st.caption("AI Engine: DeepSurv | CRC Department, Shanghai Changhai Hospital")