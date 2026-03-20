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

# 获取路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 0. 绘图环境彻底修复
# ==============================================================================
# 彻底解决负号显示为方框的问题
plt.rcParams['axes.unicode_minus'] = False 
# 自动探测字体，确保标题能正常显示
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
else:
    # 针对服务器端环境，保持默认，后续将 SHAP 特征名转为英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ==============================================================================
# 1. 定义模型结构 (务必与重训规格一致: 28特征, 128节点, 3隐藏层)
# ==============================================================================
class DeepSurvNet(nn.Module):
    def __init__(self, in_features=28): 
        super(DeepSurvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 1) # 输出 Log Hazard Ratio
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. 资源加载
# ==============================================================================
@st.cache_resource
def load_resources():
    model = DeepSurvNet(28)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    # 【重点修改】图片用英文标签显示，彻底杜绝云端部署的方框问题
    # 英文名称采用学术缩写，临床易读
    feature_labels_en = [
        "Age", "Race: White", "Race: Black", "Race: Native",
        "Adjuvant therapy", "CEA (+)", "Site: Ascending", 
        "Site: Hepatic Flex", "Site: Transverse", "Site: Splenic Flex",
        "Site: Descending", "Site: Sigmoid", "Site: Rectosigmoid", 
        "Only one primary?", "First malignancy?", "Tumor deposits (+)", 
        "Positive nodes count", "PNI (+)", "Nodes examined", "TN: pT4N0", 
        "TN: ypT0-2N+", "TN: ypT0-2N0", "TN: ypT3N+", "TN: ypT3N0", 
        "TN: ypT4N+", "TN: ypT4N0", "Grade: II", "Grade: III/IV"
    ]
    return model, scalers, bg_data, base_surv, feature_labels_en

model, scalers, bg_data, base_surv, feature_labels_en = load_resources()

# ==============================================================================
# 3. 页面布局
# ==============================================================================
st.set_page_config(page_title="LACC 个体预后评估系统", layout="wide")
st.title("局部晚期结肠癌 (LACC) 个体化生存预测工具")
st.markdown("Developed by: 长海医院肛肠外科 | 基于 SEER 数据库构建")

with st.sidebar:
    st.header("1. 基本人口学信息")
    age = st.slider("患者确诊年龄", 18, 79, 60) # 建议改为85，增加覆盖范围
    race_cn = st.selectbox("患者种族", ["亚裔/太平洋岛民", "白种人", "黑种人", "印第安人/阿拉斯加人"])
    
    st.header("2. 诊断与病理参数")
    therapy_cn = st.selectbox("术后辅助化疗 (AC)", ["未接受/拒绝", "接受"])
    cea_cn = st.selectbox("CEA 状态 (术前)", ["阴性", "阳性"])
    pni_cn = st.selectbox("神经侵犯 (PNI)", ["无", "有"])
    deposits_cn = st.selectbox("癌结节 (TD)", ["无", "有"])
    
    st.header("3. 手术分期与级别")
    nodes_pos = st.number_input("阳性淋巴结数量", 0, 80, 1)
    nodes_exam = st.number_input("检出淋巴结总数", 1, 90, 15)
    tn_stage_cn = st.selectbox("联合分期 (TN.stage)", [
        "pT4N+", "pT4N0", "ypT0-2N+", "ypT0-2N0", "ypT3N+", "ypT3N0", "ypT4N+", "ypT4N0"
    ])
    grade_cn = st.selectbox("分化等级 (Grade)", ["高分化", "中分化", "低分化/未分化"])
    site_cn = st.selectbox("肿瘤原发部位", ["盲肠", "升结肠", "结肠肝曲", "横结肠", 
                                        "结肠脾曲", "降结肠", "乙状结肠", "直乙交界部"])
    
    st.header("4. 病史特征")
    primary_only_cn = st.selectbox("是否仅该处唯一原发灶", ["是", "否"])
    first_malig_cn = st.selectbox("其他恶性肿瘤病史", ["无", "有"])

# 执行预测 (核心修正处)
if st.sidebar.button("🚀 点击分析预后", type="primary"):
    input_vec = np.zeros(28)
    # [0] age
    input_vec[0] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    # [1-3] race
    if race_cn == "白种人": input_vec[1] = 1
    elif race_cn == "黑种人": input_vec[2] = 1
    elif race_cn == "印第安人/阿拉斯加人": input_vec[3] = 1
    # [4-5] AC & CEA (字符严格匹配)
    if therapy_cn == "接受": input_vec[4] = 1
    if cea_cn == "阳性": input_vec[5] = 1
    # [6-12] site
    site_map = {"升结肠": 6, "结肠肝曲": 7, "横结肠": 8, "结肠脾曲": 9, "降结肠": 10, "乙状结肠": 11, "直乙接合部": 12}
    if site_cn in site_map: input_vec[site_map[site_cn]] = 1
    # [13-14] history
    if primary_only_cn == "是": input_vec[13] = 1
    if first_malig_cn == "无": input_vec[14] = 1
    # [15-18] Patho
    if deposits_cn == "有": input_vec[15] = 1
    input_vec[16] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    if pni_cn == "有": input_vec[17] = 1
    input_vec[18] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    # [19-25] TN Stage (映射必须完全同步)
    tn_map = {"pT4N0": 19, "ypT0-2N+": 20, "ypT0-2N0": 21, "ypT3N+": 22, "ypT3N0": 23, "ypT4N+": 24, "ypT4N0": 25}
    if tn_stage_cn in tn_map: input_vec[tn_map[tn_stage_cn]] = 1
    # [26-27] Grade
    if grade_cn == "中分化": input_vec[26] = 1
    elif grade_cn == "低分化/未分化": input_vec[27] = 1

    # 计算
    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)
    with torch.no_grad():
        log_h = model(input_tensor).item()
        rr = np.exp(log_h)

    # 分层数值 (锁定自 train_final 输出)
    T_L, T_H = 0.8594, 1.2597 
    if rr < T_L: g, color, sug = "低风险组 (Low)", "#28a745", "低危，预后预期较好。推荐按照标准随访频率监测即可。"
    elif rr <= T_H: g, color, sug = "中风险组 (Medium)", "#fd7e14", "中危，存在相关复发指标，建议加强影像学复查及血清肿瘤指标检测。"
    else: g, color, sug = "高风险组 (High)", "#dc3545", "高位！建议加强全身评估，强化后续管理方案，密切随访。"

    # 生存计算
    surv_p = {t: (base_surv[str(t)] ** rr) * 100 for t in [12, 36, 60, 120]}

    # UI 展示
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white'>
                <p style='margin:0; font-size:16px;'>生存风险实时评级</p>
                <h2 style='color:white; margin:10px 0; font-family:"Microsoft YaHei";'>{g}</h2>
                <p style='margin:0; font-size:14px; opacity:0.8;'>相对风险指数 (Relative Risk): {rr:.3f}</p>
            </div>""", unsafe_allow_html=True)
        
        st.write("")
        st.subheader("预计 OS 存活概率")
        
        # 1-3-5-10 网格展示
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.metric("1年 存活率", f"{surv_p[12]:.1f}%")
        with r1_c2: st.metric("3年 存活率", f"{surv_p[36]:.1f}%")
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.metric("5年 存活率", f"{surv_p[60]:.1f}%")
        with r2_c2: st.metric("10年 存活率", f"{surv_p[120]:.1f}%")
        
        st.divider()
        st.success(f"**💡 临床临床管理决策参考:** \n\n {sug}")

    with col2:
        st.subheader("核心指标影响贡献度分解 (SHAP)")
        with st.spinner("AI 引擎正进行归因推演..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_val_raw = explainer.shap_values(input_tensor)
            
            # 处理多分类输出数组差异
            final_shap = np.squeeze(shap_val_raw[0]) if isinstance(shap_val_raw, list) else np.squeeze(shap_val_raw)

            fig, ax = plt.subplots(figsize=(10, 8))
            e_obj = shap.Explanation(
                values=final_shap, 
                base_values=explainer.expected_value[0], 
                data=input_vec, 
                feature_names=feature_labels_en # 使用英文标签解决乱码
            )
            shap.plots.waterfall(e_obj, max_display=12, show=False)
            plt.title("Pathological indicator contribution", pad=15)
            st.pyplot(fig)

st.divider()
st.caption("AI 技术平台：Native PyTorch | 系统开发小组：上海长海医院肛肠外科团队")