# 🏥 Hệ Thống Hỗ Trợ Chẩn Đoán Ung Thư Da (Skin Cancer Diagnostic AI)

Ứng dụng Web tương tác được xây dựng bằng **Streamlit** kết hợp với mô hình học máy **Support Vector Machine (SVM)** để hỗ trợ các bác sĩ và chuyên gia y tế trong việc chẩn đoán sơ bộ các tổn thương da, phân loại lành tính/ác tính và nhận diện chi tiết 7 loại bệnh da liễu từ tập dữ liệu nổi tiếng **HAM10000**.

---

## 🚀 Tính Năng Chính
- **Chẩn đoán song song 2 chế độ**:
  - **Phân loại 2 Lớp (Binary)**: Xác định nhanh tổn thương là Lành tính (Benign) hay Ác tính (Malignant).
  - **Phân loại 7 Lớp (Multi-class)**: Chỉ ra cụ thể loại bệnh tổn thương da (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
- **Nhập liệu lâm sàng trực quan**: Giao diện cho phép chọn độ tuổi, giới tính, vị trí tổn thương và phương pháp xét nghiệm (Tiêu chuẩn vàng Histo, v.v.).
- **Tải lên hình ảnh (Image Upload)**: Hỗ trợ kéo thả hoặc tải lên ảnh nội soi da (Dermoscopy) để lưu trữ vào hồ sơ bệnh án hiển thị trên giao diện.
- **Đánh giá độ tin cậy**: Hiển thị phần trăm độ tự tin (Probability Score) kèm thanh tiến trình trực quan cho từng ca chẩn đoán.

---

## 📂 Cấu Trúc Thư Mục Dự Án
```text
/project_root
├── app.py                      # File chạy chính giao diện Streamlit
├── requirements.txt            # Danh sách thư viện phụ thuộc của dự án
├── README.md                   # Tài liệu hướng dẫn dự án
├── eda_and_preprocessing.ipynb # Notebook phân tích dữ liệu & tiền xử lý
├── model.ipynb                 # Notebook huấn luyện & đánh giá mô hình SVM
├── data/
│   └── processed_data.csv      # Dữ liệu sạch sau tiền xử lý
└── models/
    ├── preprocessing_tools.pkl # Bộ công cụ chuẩn hóa dữ liệu (Scaler, Imputer)
    ├── svm_binary.pkl          # Mô hình phân loại nhị phân đã huấn luyện
    └── svm_multi.pkl           # Mô hình phân loại đa lớp đã huấn luyện
