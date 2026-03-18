import matplotlib.font_manager as fm

# 获取当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 针对服务器端中文乱码的终极补丁
font_path = os.path.join(BASE_DIR, 'SimHei.ttf') # 确保你的 GitHub 仓库里传了这个 ttf 文件

if os.path.exists(font_path):
    # 强制加载自定义字体
    fe = fm.FontEntry(fname=font_path, name='SimHei')
    fm.font_manager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False # 同时也修好了负号方块
else:
    # 备选
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import platform

# ==============================================================================
# 0. 绘图符号与中文彻底修复补丁
# ==============================================================================
# 彻底强制解决负号显示为方框的问题
plt.rcParams['axes.unicode_minus'] = False  # 用 ASCII 减号代替 Unicode 负号

# 根据系统环境选择字体
if platform.system() == "Windows":
    # 优先顺序：微软雅黑 -> 黑体 -> 楷体
    for font in ['Microsoft YaHei', 'SimHei', 'SimSun', 'STSong']:
        plt.rcParams['font.sans-serif'] = [font]
        try:
            plt.plot([0,1],[0,1]) # 简单测试
            break
        except:
            continue
else:
    # 针对 Streamlit Cloud (Linux) 环境的无文字模式兜底
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 1. 彻底解决多环境冲突与 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. 定义模型结构 (务必与 28特征, 128节点规格一致)
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
    # 初始化 28 特征模型
    model = DeepSurvNet(28)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "deepsurv_weights.pt"), map_location='cpu'))
    model.eval()
    
    # 加载辅助文件
    scalers = pd.read_csv(os.path.join(BASE_DIR, "scalers.csv"), index_col='variable')
    bg_data = pd.read_csv(os.path.join(BASE_DIR, "bg_data.csv"))
    base_surv = pd.read_csv(os.path.join(BASE_DIR, "baseline_surv.csv")).iloc[0].to_dict()
    
    # SHAP 图表显示的汉化特征列表 (严格对应 28 位向量顺序)
    feature_names_cn = [
        "年龄", "白种人", "黑种人", "印第安/阿拉斯加人",
        "术后辅助化疗(AC)", "CEA水平: 阳性", "部位: 升结肠", 
        "部位: 肝曲", "部位: 横结肠", "部位: 脾曲",
        "部位: 降结肠", "部位: 乙状结肠", 
        "部位: 直乙接合处", "单一原发癌: 否", 
        "有过其他恶性肿瘤: 是", "肿瘤沉积(TD): 阳性", "阳性淋巴结数", 
        "神经侵犯(PNI): 有", "淋巴结检出总数", "分期: pT4N0", 
        "分期: ypT0-2N+", "分期: ypT0-2N0", "分期: ypT3N+", "分期: ypT3N0", 
        "分期: ypT4N+", "分期: ypT4N0", "分级: 中分化", 
        "分级: 低/未分化"
    ]
    return model, scalers, bg_data, base_surv, feature_names_cn

model, scalers, bg_data, base_surv, feature_names_cn = load_resources()

# ==============================================================================
# 3. 页面布局与输入
# ==============================================================================
st.set_page_config(page_title="LACC 预后评估系统", layout="wide")
st.title("局部晚期结肠癌 (LACC) 个体化预后风险评估系统 (中文版)")
st.markdown("基于 **原生 PyTorch 深度学习** 模型，多中心 SEER 数据构建。")

with st.sidebar:
    st.header("1. 基本人口学信息")
    age = st.slider("确诊年龄 (Age)", 18, 79, 60)
    race_cn = st.selectbox("患者种族 (Race)", ["亚裔/太平洋岛民", "白种人", "黑种人", "印第安人/阿拉斯加人"])
    
    st.header("2. 临床诊断与分期")
    therapy_cn = st.selectbox("术后辅助化疗 (AC)", ["不接受/拒绝", "接受"])
    cea_cn = st.selectbox("CEA 状态 (术前)", ["阴性 (<5 ng/ml)", "阳性 (≥5 ng/ml)"])
    pni_cn = st.selectbox("神经侵犯 (PNI)", ["无 (No)", "有 (Yes)"])
    deposits_cn = st.selectbox("癌结节 (TD)", ["无 (Negative)", "有 (Positive)"])
    
    st.header("3. 关键病理参数")
    nodes_pos = st.number_input("阳性淋巴结数", 0, 50, 1)
    nodes_exam = st.number_input("清扫淋巴结总数", 1, 80, 15)
    tn_stage_cn = st.selectbox("TN分期", [
        "pT4N+", "pT4N0", "ypT0-2N+", "ypT0-2N0", "ypT3N+", "ypT3N0", "ypT4N+", "ypT4N0"
    ])
    
    grade_cn = st.selectbox("分化等级 (Grade)", ["高分化 (Grade I)", "中分化 (Grade II)", "低分化/未分化 (GIII/IV)"])
    site_cn = st.selectbox("肿瘤原发部位", ["盲肠/其他", "升结肠", "结肠肝曲", "横结肠", 
                                        "结肠脾曲", "降结肠", "乙状结肠", "直乙交界部"])
    
    st.header("4. 病史特征")
    primary_only_cn = st.selectbox("是否仅该处唯一原发灶", ["是", "否"])
    first_malig_cn = st.selectbox("是否有恶性肿瘤病史", ["否", "是"])

# ==============================================================================
# 4. 执行预测逻辑
# ==============================================================================
if st.sidebar.button("🚀 开始个体化预后预测", type="primary"):
    # 创建 28 位预测向量
    input_vec = np.zeros(28)
    
    # 填充标准化数值和独热码 (此处严格根据模型 features 顺序)
    input_vec[0] = (age - scalers.loc['age', 'mean']) / scalers.loc['age', 'sd']
    if race_cn == "白种人": input_vec[1] = 1
    elif race_cn == "黑种人": input_vec[2] = 1
    elif race_cn == "印第安人/阿拉斯加人": input_vec[3] = 1
    if therapy_cn == "已接受": input_vec[4] = 1
    if cea_cn == "阳性 (≥5 ng/ml)": input_vec[5] = 1
    
    site_map = {"升结肠": 6, "结肠肝曲": 7, "横结肠": 8, "结肠脾曲": 9, "降结肠": 10, "乙状结肠": 11, "直乙接合部": 12}
    if site_cn in site_map: input_vec[site_map[site_cn]] = 1
    
    if primary_only_cn == "是": input_vec[13] = 1 
    if first_malig_cn == "否": input_vec[14] = 1 # 特征为 Yes
    
    if deposits_cn == "有 (Positive)": input_vec[15] = 1
    input_vec[16] = (nodes_pos - scalers.loc['regional.nodes.positive', 'mean']) / scalers.loc['regional.nodes.positive', 'sd']
    if pni_cn == "有 (Yes)": input_vec[17] = 1
    input_vec[18] = (nodes_exam - scalers.loc['regional.nodes.examined', 'mean']) / scalers.loc['regional.nodes.examined', 'sd']
    
    tn_map = {"pT4N0": 19, "ypT0-2N+": 20, "ypT0-2N0": 21, "ypT3N+": 22, "ypT3N0": 23, "ypT4N+": 24, "ypT4N0": 25}
    if tn_stage_cn in tn_map: input_vec[tn_map[tn_stage_cn]] = 1
    
    if grade_cn == "中分化 (Grade II)": input_vec[26] = 1
    elif grade_cn == "低分化/未分化 (GIII/IV)": input_vec[27] = 1

    # 进行神经网络计算
    input_tensor = torch.from_numpy(input_vec).float().view(1, -1)
    with torch.no_grad():
        log_h = model(input_tensor).item()
        rr = np.exp(log_h)

    # 40%/80% 风险判定逻辑 (请使用重训结果对应的值)
    T_L, T_H = 0.8594, 1.2597 
    if rr < T_L: 
        g, color, sug = "低风险 (Low Risk)", "#28a745", "低风险组，按指南要求定期复查。预后较好。"
    elif rr <= T_H: 
        g, color, sug = "中风险 (Medium Risk)", "#fd7e14", "中风险组，由于具备相关高危因素，建议密切复查及随访。"
    else: 
        g, color, sug = "高风险 (High Risk)", "#dc3545", "高风险组！建议根据实际情况选择后续治疗措施并密切复查随访。"

    # 生存概率换算
    surv_prob = {t: (base_surv[str(t)] ** rr) * 100 for t in [12, 36, 60, 120]}

    # --- 开始界面呈现 ---
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        st.markdown(f"""
            <div style='background-color:{color}; padding:25px; border-radius:20px; text-align:center; color:white'>
                <p style='margin:0; font-size:16px;'>生存预后风险评级</p>
                <h2 style='color:white; margin:10px 0; font-family:"Microsoft YaHei";'>{g}</h2>
                <p style='margin:0; font-size:14px; opacity:0.8;'>相对风险指数 (RR): {rr:.3f}</p>
            </div>""", unsafe_allow_html=True)
        
        st.write("")
        st.subheader("📊 预计各时间节点总生存率")
        row1_1, row1_2 = st.columns(2)
        with row1_1: st.metric("1年生存率", f"{surv_prob[12]:.1f}%")
        with row1_2: st.metric("3年生存率", f"{surv_prob[36]:.1f}%")
        row2_1, row2_2 = st.columns(2)
        with row2_1: st.metric("5年生存率", f"{surv_prob[60]:.1f}%")
        with row2_2: st.metric("10年生存率", f"{surv_prob[120]:.1f}%")
        
        st.divider()
        st.success(f"**💉 个体化诊疗决策支持:** \n\n {sug}")

    with col2:
        st.subheader("💡 各病理指标对当前预后的具体贡献")
        with st.spinner("AI 正在拆解神经网络权重并匹配背景特征..."):
            explainer = shap.DeepExplainer(model, torch.from_numpy(bg_data.values).float())
            shap_v = explainer.shap_values(input_tensor)
            
            # 特殊处理 shap 多类情况
            final_shap = np.squeeze(shap_v[0]) if isinstance(shap_v, list) else np.squeeze(shap_v)

            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 建立 Explanation 对象
            e_obj = shap.Explanation(
                values=final_shap, 
                base_values=explainer.expected_value[0], 
                data=input_vec, 
                feature_names=feature_names_cn 
            )
            
            # 使用官方 waterfall 绘制
            # 注意: 此处 axes.unicode_minus=False 会解决符号变方框的问题
            shap.plots.waterfall(e_obj, max_display=12, show=False)
            plt.title("特征风险贡献度剖析 (SHAP Value Waterfall)", pad=15)
            st.pyplot(fig)

st.divider()
st.caption("AI 支持：上海长海医院肛肠外科 | 数据来源：SEER 数据库 | 模型架构：原生 PyTorch 深度神经网络 | 版本：2026.03")