import streamlit as st      
import joblib
import xgboost as xgb
import numpy as np
import base64
import os

def set_background(image_name):
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)

    if not os.path.exists(image_path):
        st.error(f"找不到背景图片: {image_path}")
        return

    # 1️⃣ 读取图片并生成 base64
    with open(image_path, "rb") as f:
        data = f.read()
    img_base64 = base64.b64encode(data).decode()  # ✅ 一定要在 f-string 前生成

    # 2️⃣ 注入 CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* 背景浅化 */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.3); /* 越大越浅 */
            z-index: -1;
        }}

        /* 控件半透明背景 */
        .stBlock {{
            background: rgba(255, 255, 255, 0.3);
            padding: 1rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- 调用背景图 ----------------
set_background("1.jpg")  # 这里写你的图片名

def load_xgb_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} 不存在")
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# 加载单钉与群钉模型
single_model = load_xgb_model(r"E:\shear\single_model.json")
group_model  = load_xgb_model(r"E:\shear\group_model.json")

# 全局样式：统一字体、大小、加粗，并缩小参数说明与输入框的间距
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'SimSun', 'Times New Roman', serif !important;
}

/* 平台标题 */
.stTitle {
    font-size: 32px !important;
    font-weight: bold !important;
}

/* 平台说明文字 */
.stMarkdown div[style*="line-height"] {
    font-size: 24px !important;
}

/* ---- 输入框区域 ---- */
input, select, textarea, label, div, span {
    font-family: 'Times New Roman', 'SimSun', serif !important;
    font-size: 24px !important;
}

/* 参数说明与输入框间距 */
.stNumberInput > label, .stMarkdown {
    margin-bottom: 2px !important;
}

/* 输入框内部间距缩小 */
.stNumberInput>div>div>div>input {
    font-size: 24px !important;      /* 控制字体大小 */
    padding: 6px 12px !important;    /* 控制内部上下左右间距 */
    height: 48px !important; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---- selectbox 高度和宽度 ---- */
div[data-baseweb="select"] > div {
    min-height: 50px !important;  /* 控制外框高度 */
    width: 220px !important;      /* 控制宽度 */
}

/* selectbox 显示区域字体和高度 */
div[data-baseweb="select"] input {
    font-size: 24px !important;   /* 字体大小 */
    height: 48px !important;      /* 高度 */
    padding: 6px 12px !important; /* 内部间距 */
}

/* 下拉选项字体大小 */
div[data-baseweb="select"] ul li {
    font-size: 24px !important;
}

/* ---- st.success 输出框内字体大小 ---- */
div[data-testid="stSuccess"] div[data-testid="stMarkdownContainer"] {
    font-size: 28px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# 平台标题
st.title("焊钉连接件抗剪承载力计算平台(SCCPWS)")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "2.png")
with open(file_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

# --- 优雅布局 ---
st.markdown(f"""
<div style="
    background-color: #f8f9fa;
    border-radius: 15px;
    padding: 25px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
">
    <div style="flex: 1; font-size: 24px; line-height: 1.8; text-align: justify; color: #333;">
        基于机器学习算法（XGBoost），结合639个单钉推出试验和193个群钉推出试验的数据库，
        部署为在线计算平台，
        该平台能够快速预测单钉与群钉连接件的抗剪承载力。
        用户只需输入几何与材料参数，即可获得预测结果。
    </div>
    <div style="flex: 0 0 260px; margin-left: 40px;">
        <img src="data:image/png;base64,{encoded}"
             style="width:100%; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<p style="font-size:24px; font-weight:bold;">请选择模型类型：</p>', unsafe_allow_html=True)

model_type = st.selectbox(
    "模型选择",  # 非空 label
    ("单钉模型", "群钉模型"),
    label_visibility="collapsed"  # 隐藏原 label
)

# 输入参数（论文风格下标）
st.markdown("#### 输入参数")

# 单钉参数
st.markdown('<p style="font-size:26px;">焊钉直径 <i>d</i> <span style="font-style:normal;">(mm)</span></p>', unsafe_allow_html=True)
d = st.number_input("d", min_value=0.0, max_value=30.0, step=0.1, label_visibility="collapsed")

st.markdown('<p style="font-size:26px;">焊钉高度 <i>h</i> <span style="font-style:normal;">(mm)</span></p>', unsafe_allow_html=True)
h = st.number_input("h", min_value=50.0, max_value=500.0, step=0.1, label_visibility="collapsed")

st.markdown('<p style="font-size:26px;">混凝土弹性模量 <i>E</i><sub>c</sub> <span style="font-style:normal;">(GPa)</span></p>', unsafe_allow_html=True)
Ec = st.number_input("Ec", min_value=20.0, max_value=60.0, step=1.0, key="Ec", label_visibility="collapsed")

st.markdown('<p style="font-size:26px;">混凝土立方体抗压强度 <i>f</i><sub>cu</sub> <span style="font-style:normal;">(MPa)</span></p>', unsafe_allow_html=True)
fcu = st.number_input("fcu", min_value=20.0, max_value=70.0, step=1.0, key="fcu", label_visibility="collapsed")

st.markdown('<p style="font-size:26px;">焊钉钢材的屈服强度 <i>f</i><sub>sy</sub> <span style="font-style:normal;">(MPa)</span></p>', unsafe_allow_html=True)
fsy = st.number_input("fsy", min_value=200.0, max_value=700.0, step=10.0, key="fsy", label_visibility="collapsed")

st.markdown('<p style="font-size:26px;">焊钉钢材的极限抗拉强度 <i>f</i><sub>su</sub> <span style="font-style:normal;">(MPa)</span></p>', unsafe_allow_html=True)
fsu = st.number_input("fsu", min_value=200.0, max_value=600.0, step=10.0, key="fsu", label_visibility="collapsed")

# 群钉特有参数
if model_type == "群钉模型":
    st.markdown('<p style="font-size:26px;">纵向间距 <i>l</i><sub>z</sub> <span style="font-style:normal;">(mm)</span></p>', unsafe_allow_html=True)
    lz = st.number_input("lz", min_value=0.0, max_value=300.0, step=1.0, key="lz", label_visibility="collapsed")
    
    st.markdown('<p style="font-size:26px;">焊钉层数 <i>n</i><sub>z</sub> </p>', unsafe_allow_html=True)
    nz = st.number_input("nz", min_value=0.0, max_value=10.0, step=1.0, key="nz", label_visibility="collapsed")
    
    st.markdown('<p style="font-size:26px;">横向间距 <i>l</i><sub>h</sub> <span style="font-style:normal;">(mm)</span></p>', unsafe_allow_html=True)
    lh = st.number_input("lh", min_value=0.0, max_value=300.0, step=1.0, key="lh", label_visibility="collapsed")
else:
    lz, lh, nz = None, None, None

# 计算按钮
if st.button("计算抗剪承载力"):
    if model_type == "单钉模型":
        X = np.array([[d, h, Ec, fcu, fsy, fsu]])
        y_pred = single_model.predict(X)[0]
    else:
        X = np.array([[d, h, lz, nz, lh, Ec, fcu, fsy, fsu]])
        y_pred = group_model.predict(X)[0]
    st.success(f"预测抗剪承载力: {y_pred:.2f} kN")
