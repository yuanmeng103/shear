import streamlit as st
import joblib
import numpy as np

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="焊钉连接件抗剪承载力计算平台",
    page_icon="⚙️",
    layout="centered"
)

# ---------------- CSS 样式 ----------------
st.markdown("""
<style>
    /* 背景图片 + 半透明遮罩 */
    .stApp {
        background-image: url("E:/随手/shear/1.jpg");  /* 注意：使用正斜杠 / */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }
    body {
        font-family: 'SimSun', serif;
    }
    .title {
        font-family: 'SimSun', serif;
        font-weight: bold;
        font-size: 38px;
        text-align: center;
        color: #1F3A93;
    }
    .subtitle {
        font-family: 'SimSun', serif;
        font-weight: bold;
        font-size: 22px;
        color: #2C3E50;
    }
    .content {
        font-family: 'SimSun', serif;
        font-size: 20px;
        line-height: 1.8;
        text-align: justify;
    }
    .stNumberInput label {
        font-family: 'SimSun', serif !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }
    .stNumberInput input {
        font-family: 'Times New Roman', serif !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }
    .result {
        font-family: 'Times New Roman', serif;
        font-weight: bold;
        font-size: 26px;
        color: #B03A2E;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 标题 ----------------
st.markdown("<div class='title'>焊钉连接件抗剪承载力计算平台</div>", unsafe_allow_html=True)
st.write("")

# ---------------- 插入图片 ----------------
st.image("E:/随手/shear/1.jpg", caption="推出试验示意图", use_column_width=True)

# ---------------- 平台介绍 ----------------
st.markdown("""
<div class='subtitle'>🔹 平台介绍</div>
<div class='content'>
本平台基于机器学习算法（XGBoost），结合大量焊钉推出试验数据开发，能够快速预测单钉与群钉连接件的抗剪承载力。
用户只需输入几何与材料参数，即可获得预测结果。平台旨在提升焊钉连接计算的智能化与工程实用性，
无需安装复杂软件，在线即可完成计算与分析。
</div>
""", unsafe_allow_html=True)

st.write("---")

# ---------------- 加载模型 ----------------
single_model = joblib.load("single_model.pkl")
group_model = joblib.load("group_model.pkl")

# ---------------- 参数输入 ----------------
st.markdown("<div class='subtitle'>🔹 参数输入</div>", unsafe_allow_html=True)

model_type = st.radio("请选择模型类型：", ("单钉模型", "群钉模型"))

# 输入参数
d = st.number_input("焊钉直径 d (mm)", min_value=10.0, max_value=30.0, step=0.1)
h = st.number_input("焊钉高度 h (mm)", min_value=50.0, max_value=500.0, step=0.1)
Ec = st.number_input("混凝土弹性模量 E_c (GPa)", min_value=20.0, max_value=60.0, step=1.0)
fcu = st.number_input("混凝土立方体抗压强度 f_cu (MPa)", min_value=20.0, max_value=70.0, step=1.0)
fsy = st.number_input("焊钉钢材屈服强度 f_sy (MPa)", min_value=200.0, max_value=700.0, step=10.0)
fsu = st.number_input("焊钉钢材极限抗拉强度 f_su (MPa)", min_value=200.0, max_value=600.0, step=10.0)

if model_type == "群钉模型":
    lz = st.number_input("纵向间距 l_z (mm)", min_value=0.0, max_value=300.0, step=1.0)
    nz = st.number_input("焊钉层数 n_z", min_value=0.0, max_value=10.0, step=1.0)
    lh = st.number_input("横向间距 l_h (mm)", min_value=0.0, max_value=300.0, step=1.0)
else:
    lz, lh, nz = None, None, None

# ---------------- 预测按钮 ----------------
if st.button("计算抗剪承载力"):
    try:
        if model_type == "单钉模型":
            X = np.array([[d, h, Ec, fcu, fsy, fsu]])
            y_pred = single_model.predict(X)[0]
        else:
            # 注意：这里的输入顺序和群钉模型训练时保持一致
            X = np.array([[d, h, lz, nz, lh, Ec, fcu, fsy, fsu]])
            y_pred = group_model.predict(X)[0]

        st.markdown(f"<div class='result'>预测抗剪承载力：{y_pred:.2f} kN</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"计算出现错误：{e}")
