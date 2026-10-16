import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# 网页标题配置
st.set_page_config(page_title="垃圾分类智能识别系统", page_icon="🗑️", layout="centered")

st.title("🗑️ 基于 YOLO11 的智能垃圾分类识别系统")
st.write("大三毕业设计在线演示系统")

# 加载模型（优先使用项目目录下的权重文件）
@st.cache_resource
def load_model():
    # 适配线上和本地的权重路径
    model_path = "best.pt" if os.path.exists("best.pt") else "weights/best.pt"
    return YOLO(model_path)

with st.spinner("正在加载 AI 模型，请稍候..."):
    model = load_model()

st.success("模型加载成功！")

# 文件上传组件
uploaded_file = st.file_uploader("请上传一张需要检测的垃圾图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="上传的原始图片", use_column_width=True)
    
    with col2:
        with st.spinner("AI 正在全力识别中..."):
            # 推理
            results = model(np.array(image))
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="AI 识别与分类结果", use_column_width=True)
            
    st.success("识别完成！")