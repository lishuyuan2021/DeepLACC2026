import os
# 核心解决 OMP 报错冲突 (必须放在第一行)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import platform

# 获取基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 0. 绘图环境与中文字体处理
# ==============================================================================
plt.rcParams['axes.unicode_minus'] = False 
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ==============================================================================
# 1. 定义模型结构 (需与训练脚本完全对齐: 28特征 -> 128x3)
# ==============================================================================
class DeepSurvNet(nn.Module):
    def __init__(self, in_features=28): 
        super(DeepSurvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 1) 
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. 资源加载 (严格锁定特征顺序)
# ==============================================================================
@st.cache_resource
def load_resources():
    # 严格按照训练代码1中的 features 顺序排列
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
    
    # 加载模型
    model = DeepSurvNet(28)
    model_path = os.path.join(BASE_DIR, "deepsurv_weights.pt")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 加载标准化参数与背景数据
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    # 为了瀑布图美观，设置特征显示标签
    feature_labels_en = [
        "Age", "Race: White", "Race: Black", "Race: Native",
        "Adjuvant Chemotherapy", "CEA Positive", "Site: Ascending", 
        "Site: Hepatic Flex", "Site: Transverse", "Site: Splenic Flex",
        "Site: Descending", "Site: Sigmoid", "Site: Rectosigmoid", 
        "One primary only", "First malignant pri", "Tumor deposits (+)", 
        "Positive nodes count", "Perineural Invasion", "Nodes examined", 
        "Stage: pT4N0", "Stage: ypT0-2N+", "Stage: ypT0-2N0", "Stage: ypT3N+", 
        "Stage: ypT3N0", "Stage: ypT4N+", "Stage: ypT4N0", 
        "Grade: Moderate", "Grade: Poor/Undiff"
    ]
    
    return model, scalers, bg_data, base_surv, features_list, feature_labels_en

model, scalers, bg_data, base_surv, features_list, feature_labels_en = load_resources()

# ==============================================================================
# 3. 页面布局与侧边栏
# ==============================================================================
st.set_page_config(page_title="LACC Prognosis System", layout="wide",initial_sidebar_state="expanded")
st.title("基于深度学习的局部晚期结肠癌 (LACC) 个体化生存预测工具")
st.markdown("Developed by: 上海长海医院肛肠外科团队")

with st.sidebar:
    st.header("1. 基本人口学信息")
    age = st.slider("患者确诊年龄", 18, 95, 60)
    race_cn = st.selectbox("患者种族", ["亚裔/太平洋岛民", "白种人", "黑种人", "印第安人/阿拉斯加人"])
    first_malig_cn = st.selectbox("恶性肿瘤病史", ["无", "有"])

    st.header("2. 临床诊断参数")
    cea_cn = st.selectbox("术前 CEA 状态", ["阴性", "阳性"])
    site_cn = st.selectbox("肿瘤原发部位", ["盲肠", "升结肠", "结肠肝曲", "横结肠", 
                                        "结肠脾曲", "降结肠", "乙状结肠", "直乙交界部"])
    primary_only_cn = st.selectbox("是否仅该处唯一原发灶", ["是", "否"])
    
    st.header("3. 肿瘤病理特征")
    nodes_pos = st.number_input("阳性淋巴结数量", 0, 80, 1)
    nodes_exam = st.number_input("清扫淋巴结总数", 1, 90, 15)
    tn_stage_cn = st.selectbox("综合分期 (TN.stage)", [
        "pT4N+", "pT4N0", "ypT0-2N+", "ypT0-2N0", "ypT3N+", "ypT3N0", "ypT4N+", "ypT4N0"
    ])
    grade_cn = st.selectbox("组织分化等级", ["高分化", "中分化", "低分化/未分化"])
    pni_cn = st.selectbox("神经侵犯 (PNI)", ["无", "有"])
    deposits_cn = st.selectbox("癌结节 (TD)", ["无", "有"])

    
    st.header("4. 术后治疗措施")
    therapy_cn = st.selectbox("术后辅助化疗", ["未接受/拒绝", "接受"])

# ==============================================================================
# 4. 执行预测 (核心修改点：使用字段名对齐，防止解释错位)
# ==============================================================================
if st.sidebar.button("🚀 点击分析预后", type="primary"):
    # 创建零值字典
    input_dict = {f: 0.0 for f in features_list}
    
    # A. 连续变量标准化
    input_dict["age"] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    input_dict["regional.nodes.positive"] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    input_dict["regional.nodes.examined"] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    
    # B. 类别变量映射 (根据特征名设为 1.0)
    if race_cn == "白种人": input_dict["raceWhite"] = 1.0
    elif race_cn == "黑种人": input_dict["raceBlack"] = 1.0
    elif race_cn == "印第安人/阿拉斯加人": input_dict["raceAmerican Indian/Alaska Native"] = 1.0
    
    if therapy_cn == "接受": input_dict["adjuvant.chemotherapyYes"] = 1.0
    if cea_cn == "阳性": input_dict["CEAPositive"] = 1.0
    if pni_cn == "有": input_dict["Perineural.InvasionYes"] = 1.0
    if deposits_cn == "有": input_dict["tumor.depositsPositive"] = 1.0
    
    if primary_only_cn == "是": input_dict["One.primary.onlyYes"] = 1.0
    # 注意：此处必须对应代码1中的 "First.malignant.pri.Yes" 逻辑
    if first_malig_cn == "无": input_dict["First.malignant.pri.Yes"] = 1.0

    site_map = {
        "升结肠": "primary.siteAscending Colon", "结肠肝曲": "primary.siteHepatic Flexure", 
        "横结肠": "primary.siteTransverse Colon", "结肠脾曲": "primary.siteSplenic Flexure",
        "降结肠": "primary.siteDescending Colon", "乙状结肠": "primary.siteSigmoid Colon", 
        "直乙交界部": "primary.siteRectosigmoid Junction"
    }
    if site_cn in site_map: input_dict[site_map[site_cn]] = 1.0
    
    tn_map = {
        "pT4N0": "TN.stagepT4N0", "ypT0-2N+": "TN.stageypT0-2N+", 
        "ypT0-2N0": "TN.stageypT0-2N0", "ypT3N+": "TN.stageypT3N+", 
        "ypT3N0": "TN.stageypT3N0", "ypT4N+": "TN.stageypT4N+", 
        "ypT4N0": "TN.stageypT4N0"
    }
    if tn_stage_cn in tn_map: input_dict[tn_map[tn_stage_cn]] = 1.0
    
    if grade_cn == "中分化": input_dict["GradeModerately differentiated"] = 1.0
    elif grade_cn == "低分化/未分化": input_dict["GradePoorly differentiated/Undifferentiated"] = 1.0

    # C. 按 list 顺序导出为 NumPy 数组，确保输入模型的数据维顺序 100% 正确
    input_vec = np.array([input_dict[f] for f in features_list], dtype=np.float32)
    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)

    # D. 模型计算
    with torch.no_grad():
        log_h = model(input_tensor).item()
        rr = np.exp(log_h)

    # 风险分层 (锁定值：代码1输出结果)
    T_L, T_H = 0.9051, 1.1772 
    if rr < T_L: g, color, sug = "低风险 (Low)", "#28a745", "低风险患者，预后在所有LACC中相对较好，推荐按并根据具体情况考虑是否后续治疗，定期随访复查。"
    elif rr <= T_H: g, color, sug = "中风险 (Medium)", "#fd7e14", "中风险患者，建议依据指南及患者具体情况考虑后续治疗方案，同时强烈建议密切随访复查。"
    else: g, color, sug = "高风险 (High)", "#dc3545", "高风险患者！强烈建议依据指南及患者具体情况考虑后续治疗，并密切随访复查，必要时完善全身检查。"

    surv_p = {t: (base_surv[str(t)] ** rr) * 100 for t in [12, 36, 60, 120]}

    # UI 显示
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white'>
                <p style='margin:0; font-size:16px;'>生存风险实时评级</p>
                <h2 style='color:white; margin:10px 0;'>{g}</h2>
                <p style='margin:0; font-size:14px; opacity:0.8;'>相对风险 (RR): {rr:.3f}</p>
            </div>""", unsafe_allow_html=True)
        
        st.write("")
        st.subheader("存活概率预测")
        r1_c1, r1_c2 = st.columns(2); r2_c1, r2_c2 = st.columns(2)
        with r1_c1: st.metric("1年期 OS", f"{surv_p[12]:.1f}%")
        with r1_c2: st.metric("3年期 OS", f"{surv_p[36]:.1f}%")
        with r2_c1: st.metric("5年期 OS", f"{surv_p[60]:.1f}%")
        with r2_c2: st.metric("10年期 OS", f"{surv_p[120]:.1f}%")
        st.divider(); st.success(f"**💡 管理建议:** \n\n {sug}")

    with col2:
        st.subheader("指标临床贡献度 (SHAP)")
        with st.spinner("归因计算中..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_val_raw = explainer.shap_values(input_tensor)
            
            # 数据结构稳健性处理
            final_shap = np.squeeze(shap_val_raw[0]) if isinstance(shap_val_raw, list) else np.squeeze(shap_val_raw)
            ev = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

            fig, ax = plt.subplots(figsize=(10, 8))
            e_obj = shap.Explanation(
                values=final_shap, 
                base_values=float(ev), # 必须转为浮点数标量
                data=input_vec, 
                feature_names=feature_labels_en
            )
            shap.plots.waterfall(e_obj, max_display=12, show=False)
            plt.title("Pathological Indicator Contributions", pad=20)
            st.pyplot(fig)

st.divider()
st.caption("注：本系统结果仅供临床科研参考。")