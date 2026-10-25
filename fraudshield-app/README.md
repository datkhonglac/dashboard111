# Fraud Shield – Web app Streamlit

Bàn làm việc chống gian lận thanh toán số · Festival Chuyển đổi số 2026 · Bảng B.
Ứng dụng chạy **đúng lõi mô hình trong hồ sơ** (`fraud_core.py`). Trên Data Pack BTC cho ra
F1 0,923 · Recall 0,938 · 528 cảnh báo · 358 case · ngưỡng 0,7396.

## Ứng dụng có gì

- **Dashboard** (mở link là thấy ngay): tổng giao dịch, giao dịch đáng ngờ, đã xử lý / chưa xử lý, xác nhận gian lận,
  báo động giả, số tiền chờ kết luận; biểu đồ theo tuần, mức, kênh, dấu hiệu, tỉnh; danh sách chưa xử lý (chọn kết
  luận ngay trong bảng) và đã xử lý. Bộ lọc thời gian, mức, kênh ở thanh bên.
- Hàng đợi case, tra cứu giao dịch, mô hình & đánh giá, kiểm tra dữ liệu, tải kết quả.
- "Đã xử lý" = có kết luận: nhãn đã xác minh trong Data Pack hoặc quyết định chuyên viên lưu trong app (SQLite trên
  máy chủ – còn đến khi app khởi động lại).

## Cấu trúc

```
streamlit_app.py        giao diện (tệp chính khi deploy)
fraud_core.py           lõi mô hình: hồ sơ hành vi + Rule + Logistic Regression + Isolation Forest
model/pretrained.json   mô hình học sẵn, dùng khi Data Pack tải lên không có nhãn
data/                   Data Pack BTC (9 tệp CSV) cho nút "Chạy với Data Pack BTC"
.streamlit/config.toml  màu sắc, font, giới hạn tải lên 100 MB  ← BẮT BUỘC phải có trên GitHub
requirements.txt        thư viện cần cài
```

## Đưa lên mạng (Streamlit Community Cloud, miễn phí)

**Bước 1 – Tạo repo GitHub**
1. Vào github.com → **New repository** → tên `fraudshield-app` → chọn **Private** → **Create repository**.
2. Bấm **uploading an existing file**, kéo **toàn bộ nội dung** thư mục này vào (kể cả thư mục `.streamlit`).
   - macOS: Finder ẩn thư mục bắt đầu bằng dấu chấm – bấm `Cmd + Shift + .` để hiện rồi kéo cả `.streamlit`.
3. Bấm **Commit changes**. Kiểm tra trên GitHub thấy đủ `.streamlit/config.toml`, `data/`, `model/`.

**Bước 2 – Deploy**
1. Vào https://share.streamlit.io → **Continue with GitHub**, cho phép Streamlit truy cập repo (kể cả repo private).
2. **Create app** → **Yup, I have an app**.
3. Repository: `<tài-khoản>/fraudshield-app` · Branch: `main` · Main file path: `streamlit_app.py`.
4. App URL: đặt tên dễ nhớ, ví dụ `fraudshield-tendoi` → link sẽ là `https://fraudshield-tendoi.streamlit.app`.
5. **Advanced settings** → Python **3.12** → Save → **Deploy**. Lần đầu mất khoảng 2–5 phút.

**Bước 3 – Mở link cho mọi người**
Repo private thì app mặc định cũng private. Trong app bấm **Share** (góc phải trên) → **Make this app public**
(hoặc Settings → Sharing → *This app is public and searchable*). Mở link bằng tab ẩn danh để kiểm tra.

## Lưu ý

- App không có người dùng 12 giờ sẽ "ngủ"; người mở bấm **Yes, get this app back up!** và chờ khoảng 1 phút.
  Nên tự mở link trước khi gửi giáo viên / giám khảo.
- Sửa code trên GitHub → app tự cập nhật sau ít phút.
- Điền tên đội: sửa dòng `TEAM_NAME = ""` đầu tệp `streamlit_app.py`.
- Nhật ký quyết định chỉ lưu trong phiên đang mở – tải `audit_log.csv` ở tab **Tải kết quả** trước khi đóng.
- Nếu BTC không cho phép công khai Data Pack, xoá thư mục `data/`: app vẫn chạy, người dùng tự tải tệp .zip lên.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
