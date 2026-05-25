import streamlit as st

import pandas as pd

import numpy as np

import pickle

import os

from PIL import Image



# --- 1. CẤU HÌNH TRANG ---

st.set_page_config(page_title="Skin Cancer Diagnostic AI", page_icon="🏥", layout="wide")



st.markdown("""

    <style>

    .main {background-color: #f8f9fa;}

    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold;}

    .stButton>button:hover {background-color: #0056b3;}

    .reportview-container .main .block-container{padding-top: 2rem;}

    </style>

    """, unsafe_allow_html=True)



# --- 2. TẢI MODEL VÀ SCALER ---

@st.cache_resource

def load_models():

    try:

        with open('models/preprocessing_tools.pkl', 'rb') as f:

            tools = pickle.load(f)

        with open('models/svm_binary.pkl', 'rb') as f:

            svm_bin = pickle.load(f)

        with open('models/svm_multi.pkl', 'rb') as f:

            svm_mul = pickle.load(f)

            return tools, svm_bin, svm_mul

    except FileNotFoundError:

        st.error("⚠️ Không tìm thấy file model. Đảm bảo đã chạy file Jupyter Notebook để tạo thư mục 'models'.")

        st.stop()



tools, svm_binary, svm_multi = load_models()

scaler = tools['scaler']



# --- 3. BẢNG MÃ HÓA (LABEL ENCODING MAPPING) ---

dict_sex = {'Nữ (Female)': 0, 'Nam (Male)': 1, 'Không rõ (Unknown)': 2}

dict_dx_type = {'Confocal': 0, 'Consensus': 1, 'Follow up': 2, 'Histo (Giải phẫu bệnh)': 3}

dict_loc = {

    'Bụng (Abdomen)': 0, 'Đầu chi (Acral)': 1, 'Lưng (Back)': 2, 'Ngực (Chest)': 3, 

    'Tai (Ear)': 4, 'Mặt (Face)': 5, 'Bàn chân (Foot)': 6, 'Vùng kín (Genital)': 7, 

    'Bàn tay (Hand)': 8, 'Chi dưới (Lower extremity)': 9, 'Cổ (Neck)': 10, 

    'Da đầu (Scalp)': 11, 'Thân mình (Trunk)': 12, 'Không rõ (Unknown)': 13, 

    'Chi trên (Upper extremity)': 14

}



# --- 4. GIAO DIỆN CHÍNH ---

st.title("🏥 Hệ Thống Hỗ Trợ Chẩn Đoán Ung Thư Da (AI)")

st.write("Nhập thông tin lâm sàng và tải lên hình ảnh tổn thương da để mô hình AI phân tích.")

st.divider()



# Chia màn hình làm 2 cột: Trái (Form nhập) | Phải (Ảnh & Kết quả)

col_left, col_right = st.columns([1.1, 1])



with col_left:

    st.subheader("📋 Thông tin Lâm sàng")

    

    # --- Features (Dữ liệu thực sự đưa vào model) ---

    age = st.slider("Độ tuổi bệnh nhân", min_value=0, max_value=100, value=45, step=5)

    

    c1, c2 = st.columns(2)

    with c1:

        sex = st.selectbox("Giới tính", options=list(dict_sex.keys()))

        dx_type = st.selectbox("Phương pháp xét nghiệm", options=list(dict_dx_type.keys()), index=3)

    with c2:

        loc = st.selectbox("Vị trí tổn thương", options=list(dict_loc.keys()), index=2)

        model_choice = st.radio("Chọn mô hình dự đoán:", ["Phân loại 2 Lớp (Lành/Ác)", "Phân loại 7 Lớp (Chi tiết)"])



    st.markdown("<br>", unsafe_allow_html=True)

    submit_btn = st.button("🚀 PHÂN TÍCH DỮ LIỆU")



with col_right:

    st.subheader("🖼️ Hình ảnh Tổn thương")

    

    # Khu vực tải ảnh

    uploaded_file = st.file_uploader("Tải lên ảnh chụp da (Tùy chọn)", type=['jpg', 'jpeg', 'png'])

    

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption=f"Ảnh tải lên: {uploaded_file.name}", use_container_width=True)

    else:

        st.info("💡 Chưa có ảnh nào được tải lên. Kéo thả file ảnh hoặc click vào khu vực phía trên.")



    # --- XỬ LÝ DỰ ĐOÁN KHI BẤM NÚT ---

    if submit_btn:

        st.divider()

        st.subheader("📊 Kết quả Chẩn đoán từ AI")

        

        with st.spinner("Đang chạy mô hình Support Vector Machine..."):

            # 1. Trích xuất giá trị số từ chữ người dùng chọn

            val_sex = dict_sex[sex]

            val_loc = dict_loc[loc]

            val_dx_type = dict_dx_type[dx_type]

            

            # 2. Xếp đúng thứ tự các cột như lúc train

            input_features = np.array([[age, val_sex, val_loc, val_dx_type]])

            

            # 3. Chuẩn hóa dữ liệu

            input_scaled = scaler.transform(input_features)

            

            # 4. Dự đoán và xuất kết quả

            if "2 Lớp" in model_choice:

                pred = svm_binary.predict(input_scaled)[0]

                prob = np.max(svm_binary.predict_proba(input_scaled)) * 100

                

                if pred == 1:

                    st.error(f"🚨 Cảnh báo: Nguy cơ **ÁC TÍNH (Malignant)**")

                else:

                    st.success(f"✅ Kết luận: **LÀNH TÍNH (Benign)**")

                    

            else:

                pred = svm_multi.predict(input_scaled)[0]

                prob = np.max(svm_multi.predict_proba(input_scaled)) * 100

                

                dict_multi = {

                    0: 'Actinic keratoses (AKIEC)', 1: 'Basal cell carcinoma (BCC)', 

                    2: 'Benign keratosis (BKL)', 3: 'Dermatofibroma (DF)', 

                    4: 'Melanoma (MEL)', 5: 'Melanocytic nevi (NV)', 6: 'Vascular lesions (VASC)'

                }

                

                if pred in [1, 4]:

                    st.error(f"🩺 Chẩn đoán: **{dict_multi[pred]}**")

                else:

                    st.warning(f"🩺 Chẩn đoán: **{dict_multi[pred]}**")



            # Thanh tiến trình hiển thị độ tin cậy

            st.write(f"**Độ tin cậy của mô hình:** {prob:.2f}%")

            st.progress(int(prob))