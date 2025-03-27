# Thành viên thực hiện

## **Giảng viên hướng dẫn**
- **Th.S Đỗ Trọng Hợp**

## **Nhóm sinh viên thực hiện**
- **Nguyễn Thị Mai Trinh**
- **Nguyễn Thị Huyền Trang**

## **Môn học:**  
- DS201.O11 - Deep Learning trong khoa học dữ liệu

# Sử dụng mô hình chuỗi thời gian và học sâu để dự đoán nhiệt độ
# Sử dụng mô hình chuỗi thời gian và học sâu để dự đoán nhiệt độ

## Giới thiệu

Trước thực trạng biến đổi khí hậu và nhu cầu dự báo thời tiết ngày càng chính xác, dự đoán nhiệt độ trở thành một vấn đề quan trọng. Đồ án này nhằm xây dựng hệ thống dự báo nhiệt độ sử dụng các mô hình chuỗi thời gian truyền thống và mô hình học sâu, áp dụng trên dữ liệu thời tiết thực tế của TP. Hồ Chí Minh giai đoạn 2020–2023.

Chúng tôi tập trung dự đoán nhiệt độ theo khung giờ 3 tiếng một lần, sử dụng và so sánh hiệu suất của năm mô hình: **ARIMA**, **SARIMA**, **LSTM**, **CNN-LSTM**, và **LSTMs đa biến**.

## Các đặc điểm chính

- ✅ **Nguồn dữ liệu thực tế**: Thu thập từ WorldWeatherOnline với khung thời gian 3h/lần, từ 01/2020 đến 04/2023.
- 🔧 **Tiền xử lý dữ liệu**: Làm sạch dữ liệu, chuẩn hóa, loại bỏ nhiễu và chọn lọc đặc trưng.
- 🤖 **Mô hình áp dụng**:
  - ARIMA
  - SARIMA
  - LSTM
  - CNN-LSTM
  - LSTMs đa biến
- 📈 **Đánh giá mô hình**: Sử dụng RMSE và R-squared để đo hiệu suất dự đoán.
- 📊 **Trực quan hóa**: Biểu đồ thể hiện sự chênh lệch giữa giá trị thực và dự đoán.

## Các nhiệm vụ nghiên cứu chính

- Thu thập và xử lý bộ dữ liệu thời tiết TP.HCM theo mốc thời gian 3 giờ/lần.
- Tiền xử lý dữ liệu: làm sạch, đổi định dạng, chuẩn hóa về khoảng [0,1].
- Chọn lọc 5 thuộc tính đầu vào chính: `Hour`, `Temperature`, `Forecast`, `Pressure`, `Gust`.
- Triển khai các mô hình dự báo:
  - **ARIMA**: Hiệu suất kém do không xử lý được yếu tố mùa vụ.
  - **SARIMA**: Dự báo chính xác nhờ khai thác tính chu kỳ của dữ liệu.
  - **LSTM**: Tốt cho dữ liệu chuỗi phức tạp, độ chính xác cao.
  - **CNN-LSTM**: Hiệu suất thấp nếu không chuẩn hóa dữ liệu, nhưng cải thiện rõ rệt nếu dùng MinMaxScaler.
  - **LSTMs đa biến**: Tối ưu nhất nếu chọn đúng tổ hợp biến đầu vào. R-squared đạt đến 0.999.
- So sánh, phân tích ưu – nhược điểm và đánh giá mô hình bằng biểu đồ + bảng tổng hợp.

## Hướng phát triển

- 📌 **Tối ưu mô hình**:
  - Tinh chỉnh siêu tham số (hyperparameter tuning)
  - Sử dụng các kỹ thuật tăng cường đặc trưng
- 📌 **Mở rộng ứng dụng**:
  - Triển khai trên dữ liệu thời tiết thực tế theo thời gian thực
  - Áp dụng cho các vùng địa lý khác
- 📌 **Kết hợp mô hình ngôn ngữ lớn (LLM)**:
  - Phân tích dữ liệu thời tiết dạng văn bản
  - Tăng cường khả năng diễn giải và đề xuất hành động từ dự báo
- 📌 **Xây dựng hệ thống AI hoàn chỉnh**: Giao diện người dùng + hệ thống backend tự động hóa quá trình thu thập, xử lý, dự đoán, hiển thị kết quả.

---

🧠 **Từ khóa**: Chuỗi thời gian, học sâu, dự báo nhiệt độ, ARIMA, SARIMA, LSTM, CNN-LSTM, LSTMs.

📬 **Liên hệ**:  
Nguyễn Thị Huyền Trang (21520488@gm.uit.edu.vn)  
Nguyễn Thị Mai Trinh (21522718@gm.uit.edu.vn)  
Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM
