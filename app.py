import streamlit as st
import joblib
import numpy as np

# 加载模型
single_model = joblib.load("single_model.pkl")
group_model = joblib.load("group_model.pkl")

st.title("焊钉抗剪承载力计算平台")
st.write("输入参数后即可预测抗剪承载力")

model_type = st.radio("请选择模型类型：", ("单钉模型", "群钉模型"))

# 输入参数
d = st.number_input("焊钉直径 d (mm)", min_value=10.0, max_value=30.0, step=0.1)
h = st.number_input("焊钉高度 h (mm)", min_value=50.0, max_value=500.0, step=0.1)
Ec = st.number_input("混凝土弹性模量 Ec (GPa)", min_value=20.0, max_value=60.0, step=100.0)
fcu = st.number_input("混凝土立方体抗压强度 fcu (MPa)", min_value=20.0, max_value=70.0, step=1.0)
fsu = st.number_input("焊钉极限抗拉强度 fsu (MPa)", min_value=200.0, max_value=600.0, step=100.0)
fsy = st.number_input("焊钉屈服强度 fsy (MPa)", min_value=200.0, max_value=700.0, step=100.0)

if model_type == "群钉模型":
    lz = st.number_input("纵向间距 lz (mm)", min_value=40.0, max_value=300.0, step=1.0)
    nz = st.number_input("焊钉层数 nz", min_value=2, max_value=10, step=1)
    lh = st.number_input("横向间距 lh (mm)", min_value=40.0, max_value=300.0, step=1.0)
else:
    lz, lh, nz = None, None, None

if st.button("计算抗剪承载力"):
    if model_type == "单钉模型":
        X = np.array([[d, h, Ec, fcu, fsu, fsy]])
        y_pred = single_model.predict(X)[0]
    else:
        X = np.array([[d, h, lz, nz, lh, Ec, fcu, fsu, fsy]])
        y_pred = group_model.predict(X)[0]
    st.success(f"预测抗剪承载力: {y_pred:.2f} kN")
