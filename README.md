# SVD Studio

Ứng dụng web tương tác học cách trích rút câu quan trọng theo `btl.pdf`. Vite và ml-matrix tính SVD trong Web Worker. Chế độ cơ bản chạy trong trình duyệt; Underthesea/POS và Word2Vec dùng dịch vụ Python cục bộ, chỉ nhận yêu cầu localhost. Văn bản không gửi tới dịch vụ bên ngoài.

## Chạy ứng dụng

Yêu cầu Node.js 20.19+ hoặc 22.12+.

```bash
npm install
npm run dev
```

Để dùng tách từ tiếng Việt/POS và Word2Vec, tạo môi trường Python 3.12 trong thư mục dự án:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-nlp.txt
```

Trên Linux/macOS dùng `.venv/bin/python`. Vite tự khởi động dịch vụ khi cần, hỗ trợ cả `npm run dev` và `npm run preview`. Hosting static chỉ dùng được chế độ cơ bản; chế độ NLP cần dịch vụ này.

### Chức năng bổ sung theo Noi_dung.docx

- Cấu hình: chọn Underthesea để tách từ và xem POS; lọc N/V/A, top-m TF hoặc TF-IDF, hoặc nạp kho JSON gồm mảng văn bản để xây từ vựng/IDF ngoài. df của kho ngoài đếm **tài liệu**, IDF = ln(N/df), không smoothing. Token không có trong kho bị bỏ. Kho chỉ tồn tại trong phiên.
- Word2Vec/PhoW2V: mở **Mô hình ngôn ngữ và kho ngữ liệu ngoài**, nạp tệp Word2Vec đã giải nén (`.bin` binary hoặc `.vec`/`.txt` có header `số_từ số_chiều`). Hoặc đặt tệp trong `models/` rồi làm mới và chọn. PhoW2V 100 chiều tạo 100 hàng. Mô hình được lập chỉ mục SQLite trên đĩa để tránh giữ toàn bộ vector trong RAM. Từ tra trực tiếp được ưu tiên; từ ghép OOV lấy trung bình các thành phần có vector. Từ hoàn toàn OOV bỏ khỏi mẫu số; câu toàn OOV là vector 0. Trung bình tính theo **lần xuất hiện**, không theo tập từ duy nhất. UI hiển thị độ phủ, đường tra, vector câu và từng phép tính ô. Các hàng U của biểu diễn nhúng là chiều Eᵢ, không phải từ khóa.
- So sánh: LEAD, Random qua seed 11/29/47/71/101, cộng TF-IDF, TextRank cosine và hai phương pháp SVD. Chung tokenizer, lọc từ vựng, c, lựa chọn L2; baseline dùng TF-IDF, SVD dùng biểu diễn được chọn.  TextRank bỏ cạnh tự nối, dùng max(0, cosine), damping .85, tối đa 200 vòng, ngưỡng L1 1e-9, phân phối đều điểm của nút không có cạnh. Hòa điểm ưu tiên vị trí.
- Đồng chọn câu: checkbox tham chiếu, P/R/F1 tính trên tập chỉ số câu.
- ROUGE: token đơn, bigram và số khớp min(count ứng viên, count tham chiếu), chuỗi LCS và cặp vị trí. Dùng cùng tokenizer cho hai phía, không bỏ stopword; ROUGE-L trên chuỗi nối, không phải ROUGE-Lsum. Giải thích LCS giới hạn 4 triệu cặp token.
- Đánh giá người đọc: đầy đủ, ít dư thừa, mạch lạc theo thang 1–5, người chấm và ghi chú; xuất JSON để lưu kết quả phiên.
- Thực nghiệm: nạp dataset 2–100 bản ghi; thử θ=.7/.8/.9/.95 và γ=.3/.5/.7 trên dev theo macro ROUGE-L F1. Cố định cấu hình thắng (hòa điểm theo thứ tự lưới), sau đó chấm các baseline trên test. ID và văn bản trùng bị từ chối. Giữ cùng bộ dữ liệu và ngân sách 7 cấu hình khi so sánh các biểu diễn. Không dùng tập test để xây kho IDF ngoài. Dữ liệu tải từ nút minh họa chỉ là ví dụ thao tác, không phải benchmark chất lượng.
- Hiệu quả: thời gian tổng tiền xử lý + SVD; thời gian benchmark không gồm NLP/chấm ROUGE; bộ nhớ mảng số ước lượng theo số phần tử × 8 byte; Python RSS đo bằng psutil có cả mô hình dùng chung, không phải peak RAM riêng của từng thuật toán.

Dataset JSON:

```json
[
  {"id":"dev-1","split":"dev","text":"Câu một.\nCâu hai.","reference":"Câu một.","gold":[1]},
  {"id":"test-1","split":"test","text":"Nội dung khác.\nÝ bổ sung.","reference":"Ý bổ sung.","gold":[2]}
]
```

`gold` tùy chọn, đánh số câu từ 1 theo bộ tách câu của ứng dụng. Kết quả JSON chứa cấu hình cố định, từng thử nghiệm dev, ID phân chia, kết quả từng văn bản test và số liệu tổng hợp.

Nguồn mô hình: [PhoW2V chính thức](https://github.com/datquocnguyen/PhoW2V). Chỉ dùng nghiên cứu/giáo dục và không phân phối lại mô hình hoặc phần trích xuất. Thư mục `models/` đã được loại khỏi Git. Có thể chạy `.venv\Scripts\python.exe server/download_phow2v.py` để tải mirror do tác giả cung cấp; cần tối thiểu 4 GiB trống cho tải, giải nén và chỉ mục. Lookup áp dụng quy ước dấu của PhoW2V (òa→oà, óa→oá, …); Underthesea thực hiện tách từ/POS theo [API chính thức](https://undertheseanlp.github.io/underthesea/docs). TextRank cosine là biến thể đồ thị của [TextRank](https://aclanthology.org/W04-3252/).

Mở địa chỉ Vite in ra, mặc định http://localhost:5173. Website cần chạy qua web server, không mở trực tiếp index.html.

```bash
npm test
npm run test:browser
npm run build
npm run preview
```

Bản production nằm trong `dist/`; có thể đưa thư mục này lên dịch vụ hosting static. Font Google có fallback hệ thống nếu không có mạng. Kiểm tra trình duyệt dùng Chrome tại `/opt/google/chrome/chrome`; đặt `CHROME_PATH` nếu máy bạn có vị trí khác.

## Hành trình trên trang

1. Nhập văn bản và chủ đề tùy chọn, cấu hình, chọn phân tích.
2. Theo dõi tách câu, lowercase/NFC, token, stopword và từ vựng.
3. Khám phá bảng TF, trọng số, A sau L2; bấm từng ô để xem phép tính.
4. Xem U, Sigma, V, V-transpose; diễn giải A-transpose A, trị riêng, u = Av/sigma, sai số tái dựng/trực chuẩn. Chọn hướng/token/câu để kiểm tra số Av, A-transpose u, Bv và từng số hạng tái dựng A. Kéo q để xem ma trận xấp xỉ hạng thấp và sai số.
5. Chọn k, xem phổ/năng lượng tích lũy, từ khóa theo |u| và bản đồ câu; bấm cột phổ để đổi k.
6. Xem đóng góp từng chiều, bảng điểm, các vòng chọn câu, sao chép bản tóm tắt, xuất báo cáo JSON, đánh giá ROUGE.

## Quy ước theo PDF

- Ma trận A: token làm hàng, câu làm cột. TF, Binary, Log-TF, TF-IDF `f*ln(n/df)`, Log-entropy `ln(1+f)*g`; không cộng smoothing IDF.
- `g = 1 + sum(p*ln(p))/ln(n)`; `p = f / sum_sentence(f)`.
- Chuẩn hóa L2 trước SVD là tùy chọn, cột 0 giữ nguyên.
- Hiển thị SVD theo rank r (`U: m*r`, `Sigma: r*r`, `V: n*r`); các cột U/V trực chuẩn. Phân rã đầy đủ trong PDF có thêm hướng với giá trị kỳ dị 0.
- k: năng lượng tích lũy, ngưỡng tương đối hoặc cố định; Gong–Liu dùng k = c, giới hạn rank.
- Steinberger–Ježek dùng **căn bậc hai**: `s_j = sqrt(sum(sigma_i^2 * v_ij^2))`.
- Điều chỉnh độ dài `s'_j = s_j / ||a_j||^alpha` dùng chuẩn của chính cột đầu vào SVD. Nó khác chuẩn hóa cột trước SVD.
- Gong–Liu chọn tuần tự theo `|v_ij|`, bỏ câu đã chọn, không dùng alpha. Nếu c > rank, ứng dụng bổ sung theo điểm SVD tổng hợp, đánh dấu rõ đây là dự phòng mở rộng.
- Ma trận toàn 0/từ vựng rỗng: dự phòng theo vị trí. Câu cuối cùng sắp theo thứ tự gốc.

## Giới hạn được công bố

Chế độ tokenizer cơ bản theo khoảng trắng/dấu câu, giữ token nối dấu `_`; với tiếng Việt thường chấm theo âm tiết. Có thể chọn Underthesea để chấm theo từ. Stopword là danh sách nhỏ được công bố trong engine.js. Tối đa 120 câu, 60.000 ký tự, 1.200 token phân biệt sau lọc. Ma trận phân trang 12 hàng/10 cột; số liệu xuất JSON đầy đủ. ROUGE dùng một tham chiếu, không bỏ stopword, nối câu; ROUGE-L không phải ROUGE-Lsum. Các mẫu là dữ liệu minh họa. Mô hình vector lớn cần đủ dung lượng đĩa để nạp/lập chỉ mục, không tự thay thế bằng vector ngẫu nhiên.

## Cấu trúc

- `src/engine.js`: trọng số, SVD, chấm/chọn câu, ROUGE.
- `src/worker.js`: tính toán ngoài luồng UI.
- `src/main.js`: luồng ứng dụng và tương tác.
- `src/views.js`: heatmap ma trận và SVG trực quan.
- `src/styles.css`: giao diện responsive, hỗ trợ giảm chuyển động.
- `tests/engine.test.js`: kiểm tra các bất biến đại số và quy tắc lựa chọn.

Tài liệu thư viện: https://github.com/mljs/matrix ; https://vite.dev/guide/

## Bản làm việc trên ổ D

Dự án đang chạy từ `D:\bt_nlp`. Mô hình, tệp tải và chỉ mục lưu trong `D:\bt_nlp\models`. Mở PowerShell tại thư mục này rồi chạy `npm.cmd run dev -- --host 127.0.0.1`. PhoW2V có thể tải bằng `.venv\Scripts\python.exe server/download_phow2v.py`. Không nạp lại tệp đã đặt trong models qua upload; chọn trực tiếp trong danh sách mô hình để tránh tạo bản sao lớn.

## Tab Thực nghiệm VietNews

Mở `http://localhost:5173/?tab=vietnews`. Trang được tách thành ba tab: Demo SVD, Thực nghiệm VietNews, Đánh giá nâng cao. Trạng thái đầu vào demo được giữ khi đổi tab.

- Thử một bài: chọn từ 10 bài validation (dev) và 20 bài test; tiêu đề và tham chiếu tách riêng khỏi đầu vào. So sánh TF-IDF/SVD với PhoW2V/SVD theo cùng số câu, L2, tokenizer basic giữ token `_`, phương pháp Steinberger–Ježek;  ROUGE chấm trên token gốc có stopword.
- Khối cuối ngắn chỉ được gợi ý là chú thích ảnh; mặc định giữ nguyên, người dùng có thể xem rồi loại. Chỉnh sửa và lựa chọn loại khối được ghi vào JSON xuất. Phần xem dễ đọc ẩn dấu `_` nhưng không đổi token tính toán.
- Chạy nhiều bài: thử riêng 7 cấu hình trên dev cho mỗi biểu diễn rồi cố định cấu hình trước khi chấm test; báo cáo macro ROUGE F1 và kết quả từng bài. Dùng bài gốc giữ khối cuối, không dùng chỉnh sửa của màn thử một bài. JSON có ID tập chia, cấu hình, nguồn, commit và model.
- Mẫu cục bộ tải bằng `.venv\Scripts\python.exe server/fetch_vietnews.py`; nguồn https://github.com/ThanhChinhBK/vietnews . Đây là 10 tên tệp đầu validation và 20 tên đầu test, không phải mẫu ngẫu nhiên hoặc toàn bộ benchmark. Nguồn, commit và SHA-256 từng tệp được lưu trong `public/data/vietnews.json`. Tệp nguồn chỉ được tải về máy cho demo; không có thao tác công bố dữ liệu lên hosting.
