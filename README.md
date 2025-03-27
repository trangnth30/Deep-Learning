# Thành viên thực hiện

## **Giảng viên hướng dẫn**
- **Th.S Đỗ Trọng Hợp**

## **Nhóm sinh viên thực hiện**
- **Nguyễn Thị Mai Trinh**
- **Nguyễn Thị Huyền Trang**

## **Môn học:**  
- DS104 - Tính toán song song và phân tán

# Xây dựng hệ thống dự đoán giá trị bất động sản

## Giới thiệu

Định giá bất động sản là một bài toán quan trọng và đầy thách thức, đặc biệt trong bối cảnh thị trường có nhiều biến động. Đồ án này tập trung xây dựng một hệ thống dự đoán giá trị bất động sản dựa trên dữ liệu thực tế được thu thập từ website Nhadatvui.vn, kết hợp với các mô hình học máy hiện đại trên nền tảng Spark MLlib.

Dự án triển khai và so sánh nhiều mô hình khác nhau như: **Linear Regression**, **Decision Tree**, **Random Forest**, **Isotonic Regression**, và **Gradient Boosting**, nhằm chọn ra phương pháp tối ưu nhất cho bài toán định giá bất động sản.

## Các đặc điểm chính

- 📥 **Nguồn dữ liệu thực tế**: Web Scraping từ Nhadatvui.vn với 4.000 bài đăng, 33 thuộc tính đa dạng.
- 🧹 **Tiền xử lý dữ liệu chuyên sâu**:
  - Làm sạch dữ liệu, loại bỏ đơn vị và chuẩn hóa về định dạng thống nhất.
  - Xử lý giá trị thiếu bằng các phương pháp thống kê.
  - Tối ưu logic với công thức suy diễn chiều dài: `ChiềuDài = DiệnTích / ChiềuRộng`.
- 🧠 **Trích xuất đặc trưng thông minh**:
  - Bucketizer cho phân nhóm người bán.
  - Phân cấp tỉnh thành theo trình độ phát triển.
  - String Indexing và One-hot Encoding cho biến phân loại.
  - Log-transform cho biến mục tiêu `Tổng Giá`.
- 📊 **Mô hình triển khai**:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Isotonic Regression
  - Gradient Boosted Regression
- 🧪 **Đánh giá mô hình**: Sử dụng RMSE, MAE, R².

## Các nhiệm vụ nghiên cứu chính

- Thu thập dữ liệu thực tế từ sàn giao dịch bất động sản bằng Web Scraping.
- Xử lý và chuẩn hóa bộ dữ liệu để phù hợp với yêu cầu mô hình hóa.
- Áp dụng các kỹ thuật phân tích dữ liệu: loại bỏ ngoại lệ, phân cụm, encoding.
- Triển khai thử nghiệm và đánh giá 5 mô hình học máy trên Spark MLlib.
- So sánh kết quả để tìm ra mô hình hiệu quả nhất.
- Phân tích lý do hiệu suất khác nhau giữa các mô hình.

## Hướng phát triển

- 🔍 **Nâng cấp dữ liệu đầu vào**: Kết hợp thêm dữ liệu vệ tinh, dữ liệu giao thông hoặc tiện ích xung quanh.
- ⚙️ **Cải tiến mô hình**:
  - Sử dụng kỹ thuật ensemble nâng cao.
  - Áp dụng feature selection tự động và kỹ thuật giảm chiều dữ liệu.
- 🧩 **Kết hợp mô hình học sâu**: Đưa vào MLP hoặc mô hình Attention để dự đoán phi tuyến tốt hơn.
- 📈 **Triển khai hệ thống thực tế**:
  - Tạo dashboard định giá nhà.
  - Gợi ý vùng giá hợp lý cho người mua/bán.

---

📬 **Liên hệ**:  
Nguyễn Thị Mai Trinh (21522718@gm.uit.edu.vn)  
Nguyễn Thị Huyền Trang (21520488@gm.uit.edu.vn)  
Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM
