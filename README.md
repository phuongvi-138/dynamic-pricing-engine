# Engine định giá linh hoạt cho hàng tươi sống

Web demo (Streamlit) cho đồ án **Tối ưu giá bán linh hoạt (Dynamic Pricing Engine)**, dữ liệu thật **FreshRetailNet-50K** (Dingdong, 28/3–2/7/2024).

Engine theo khung *dự báo rồi tối ưu* (forecast-then-optimize):

1. **Random Forest** dự báo nhu cầu của từng sản phẩm–cửa hàng ở 6 mức giá (70%, 80%, 85%, 90%, 95%, 100% giá gốc).
2. Với mỗi mức giá, tính lợi nhuận dự kiến = (giá − giá vốn) × lượng bán dự kiến − giá vốn × hàng sắp hết hạn không bán được.
3. Chọn mức giá có lợi nhuận dự kiến cao nhất.

## Web app có gì

| Trang | Nội dung |
|---|---|
| 🎯 Đề xuất giá | Chọn sản phẩm–cửa hàng và ngày, nhập tồn kho, hàng sắp hết hạn, biên lợi nhuận; thử thay đổi bối cảnh (khuyến mãi của sàn, ngày lễ, mưa, nhiệt độ). Mô hình chạy trực tiếp và trả về giá đề xuất |
| 📅 Mô phỏng 22 ngày | Giá và lợi nhuận theo ngày của sản phẩm đã chọn với 3 cách đặt giá: giá thực tế, không khuyến mãi, Engine ML |
| 📊 Kết quả toàn hệ thống | So sánh 3 cách đặt giá trên toàn bộ tập test (RQ2) |
| 🛡️ Độ vững & danh mục | Đánh giá chéo bằng mô hình hồi quy, phân tích độ nhạy, kết quả theo danh mục |
| ℹ️ Phương pháp | Pipeline, nguồn dữ liệu, tài liệu tham khảo |

## Cấu trúc thư mục

```
app.py                  # giao diện Streamlit
engine.py               # lõi engine: nạp mô hình, dự báo ở 6 mức giá, chọn giá
app_bundle/             # mô hình + dữ liệu đã xử lý + bảng/hình kết quả (xuất từ Notebook 2)
notebooks/              # Notebook 1 (dự báo cầu) và Notebook 2 (engine định giá)
tools/build_bundle.py   # mã đóng gói app_bundle và đẩy lên GitHub (Bước 11 của Notebook 2)
requirements.txt
```

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Triển khai miễn phí

**Streamlit Community Cloud:** đăng nhập [share.streamlit.io](https://share.streamlit.io) bằng GitHub → *Create app* → chọn repo này, nhánh `main`, tệp `app.py` → *Deploy*.

## Nguồn dữ liệu

Wang, Y. và cộng sự (2025). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail*. arXiv:2505.16319. Dữ liệu: <https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K>

> Kết quả trong app là **mô phỏng phản thực tế** trên dữ liệu lịch sử, không phải thử nghiệm A/B trên cửa hàng thật.
