# ☁️ Phân tích và Dự báo Mưa Trái Mùa bằng ANN


## 🎯 Mục tiêu
Dự án này nhằm **dự báo khả năng xảy ra mưa trái mùa** dựa trên các yếu tố thời tiết như nhiệt độ, độ ẩm, sức gió, hướng gió, phần trăm mây, áp suất... bằng mô hình **Mạng nơ-ron nhân tạo ~ Artificial Neural Network (ANN)**.


## 🧠 Phương pháp
- Tiền xử lý dữ liệu: chuẩn hóa, xử lý giá trị thiếu, one-hot encoding.
- Mô hình: Mạng nơ-ron nhiều lớp sử dụng `TensorFlow`/`Keras`.
- Đánh giá mô hình: Accuracy, F1-score, biểu đồ loss/accuracy.


## 📁 Cấu trúc dự án
```
rainfall_forecasting_ann/
├── data/
│   ├── raw/                         # Dữ liệu gốc (từ Kaggle, CSV, v.v.)
│   └──     /                   # Dữ liệu đã tiền xử lý (dùng cho mô hình)
│
├── notebooks/
│   └── eda.ipynb                    # Notebook EDA (phân tích dữ liệu ban đầu)
│   └── ann_model.ipynb             # Notebook xây mô hình ANN
│
├── src/
│   ├── preprocessing.py            # Xử lý dữ liệu: fillna, scale, one-hot,...
│   ├── model.py                    # Định nghĩa và train mô hình ANN
│   └── evaluate.py                 # Tính accuracy, f1-score, confusion matrix
│
├── reports/
│   └── figures/                    # Lưu hình ảnh biểu đồ EDA, loss/accuracy
│   └── results.txt                 # Tổng hợp kết quả mô hình (accuracy, f1, v.v.)
│
├── main.py                         # Chạy toàn bộ pipeline: load -> xử lý -> train -> đánh giá
├── requirements.txt                # Thư viện Python cần cài
├── README.md                       # Giới thiệu dự án, cách chạy
└── .gitignore                      # Bỏ qua các file không cần (DS_Store, __pycache__)
```


## 📊 Dữ liệu
Dữ liệu gồm các thuộc tính:
- Max temperature, Min temperature
- Wind speed, Wind direction
- Humidity, Cloud cover, Pressure
- Rainfall (target)
- Date (ngày)

📌 Mục tiêu mô hình: **Dự đoán có mưa (1) hay không (0)**.

## 🧪 Cách chạy
### 1. Cài đặt thư viện
    pip install -r requirements.txt
    #nếu cần, bổ sung thư viện vào trong requirements.txt

### 2. Chạy dự án


-   Chạy thử:
    python src/preprocessing.py
    python src/evaluate.py
    python src/model.py
    
-   Chạy dự án
python main.py

## ✍️ Tác giả
-   Nhóm:
    Thành viên:
    Trường: Đại học Bách Khoa Đại học Quốc gia Thành phố Hồ Chí Minh