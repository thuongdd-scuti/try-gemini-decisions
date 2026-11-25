# Demo đánh giá Gemini Decisions API

Demo Python 3.10+ không cần cài thư viện. So sánh phân loại **48 tài liệu văn bản tổng hợp**, 12 mẫu mỗi nhóm Invoice, Receipt, Contract, Other. Chọn tài liệu thay cho bộ ảnh Animal/Food/Vehicle để minh họa phân luồng công việc văn phòng. Đây **không phải benchmark ảnh, PDF scan hay OCR**.

## Bài blog và kết quả thực tế

`article.html` là bản bài viết để đăng WordPress. Các link nguồn trong bài trỏ tới file của repo trên nhánh `main`. Ảnh trong bài dùng URL từ thư viện WordPress.

Lần `artifacts/live` đã thử đủ 48 tài liệu mỗi model: Decisions đúng 48/48; baseline đúng 47/47 response hợp lệ và timeout một request. Accuracy trên tổng attempts là 100% và 97,92%. Mean latency hợp lệ tương ứng 477,86 ms và 5.739,74 ms. Không suy ra baseline phân loại sai từ timeout.

`live-repeat3` chỉ đạt 81/144 attempts mỗi model trước HTTP 429: kết quả phụ chưa hoàn tất. Chi phí baseline toàn bộ 48 attempts chưa xác định vì timeout thiếu usage. So cặp cùng 47 tài liệu thành công: Decisions 0,02129 USD/1.000, baseline 0,05018 USD/1.000, giảm khoảng 57,6% theo paid-tier estimate. Phép tính tại `artifacts/blog-paired-cost.csv` và `artifacts/blog-audit.json`.

Log và cấu hình gốc nằm trong `artifacts/live/`, `artifacts/live-repeat3/` và `artifacts/smoke/`.

## Chạy demo

1. Lấy key tại [Google AI Studio](https://aistudio.google.com/api-keys), có quyền truy cập cả hai model.
2. Copy `.env.example` thành `.env`, thay giá trị bằng key trên máy. `.env` được gitignore. Hoặc dùng biến môi trường `GEMINI_API_KEY`. Script cũng nhận `GOOGLE_API_KEY`.
3. Chạy:

```bash
python3 -m unittest discover -s tests -v
# Smoke test: 1 tài liệu, 2 request (không dùng làm benchmark).
python3 benchmark.py run --limit 1 --out artifacts/new-smoke
# Benchmark: 48 tài liệu × 2 model = 96 request.
python3 benchmark.py run --out artifacts/new-run
# Tùy chọn đo độ dao động: 288 request; không coi 144 mẫu là độc lập.
python3 benchmark.py run --repeats 3 --out artifacts/new-repeat3
# Tạo lại CSV/chart từ response đã lưu, không gọi API.
python3 benchmark.py report artifacts/new-run
```

Mỗi thư mục run phải chưa tồn tại để tránh ghi đè dữ liệu. Khi gặp 401/403/404/429, script dừng và giữ bằng chứng một phần; xử lý quyền/quota rồi chạy vào thư mục mới. Không có retry ngầm, không đổi model ngầm. Lệnh chạy thực hiện request có thể tính phí theo tài khoản; chi phí ước tính trong báo cáo không phải hóa đơn Google.

## Thiết kế so sánh

- Decisions: `POST /v1beta/decisions`, model `gemini-decision-1`, câu hỏi `choice` với bốn criteria; đọc `answers.category.choice`, `confidence`, `probabilities`, `usage`.
- Baseline: `POST /v1beta/models/gemini-3.5-flash-lite:generateContent`, cùng nội dung, quy tắc và định nghĩa nhãn. Yêu cầu duy nhất một nhãn, temperature 0, thinking MINIMAL, tối đa 64 output tokens. Có hỗ trợ ảnh nhưng lần này chỉ thử text. Output sai định dạng hoặc bị cắt được tính lỗi; không lấy nhãn bằng cách tìm từ khóa trong lời giải thích.
- Seed cố định 20261120; xáo thứ tự tài liệu, đảo thứ tự hai phương pháp cho từng cặp; chạy tuần tự và nghỉ một giây giữa request. Không warmup. `urllib` dùng kết nối mới; latency gồm HTTP/TLS, mạng, server và đọc JSON, không gồm ghi file hay dựng payload. Không phải thời gian compute riêng của model; không so trực tiếp với latency quảng bá.
- Ground truth và ID không gửi lên model. Có tài liệu Việt/Anh/Nhật/Pháp, hóa đơn đã trả, trả một phần, hợp đồng chưa ký, báo giá, nhắc tới receipt nhưng không phải receipt, và prompt injection. Quy tắc: invoice còn tiền phải trả; receipt xác nhận đã thanh toán/hoàn tiền; contract quy định nghĩa vụ, kể cả draft; báo giá/purchase order/memo là Other.
- Dataset tự soạn, không PII, không đại diện phân bố production. Nhãn do tác giả thiết lập trước khi gọi model; cần người thứ hai rà nhãn trước khi xuất bản. 48 tài liệu rất ngắn và cùng phong cách, dễ có ceiling effect. Nếu cả hai đạt 100%, không kết luận ngang nhau trên mọi tài liệu. Không dùng repeats để làm tăng giả cỡ mẫu accuracy. Cần bộ held-out thực tế và khoảng tin cậy cho kết luận mạnh.

## File kết quả và cách đọc

| File trong thư mục run | Nội dung |
|---|---|
| `manifest.json` | Thời điểm UTC, model, rates, SHA-256 dataset, seed, quy trình |
| `responses.jsonl` | Từng request/response thật, HTTP status, nhãn, confidence, token, latency, lỗi; không lưu key |
| `predictions.csv` | Một dòng mỗi request để đối chiếu nhãn đúng |
| `summary.csv`, `summary.json` | Accuracy, average, P95, error rate, coverage token và chi phí |
| `confusion.json` | Ma trận nhầm lẫn, thêm cột ERROR |
| `benchmark.svg` | Biểu đồ chỉ được sinh từ records của lần chạy |

Accuracy chính = số đúng / tổng request, tính lỗi là sai. Có accuracy trên response hợp lệ để phân biệt chất lượng model và tính ổn định API. Error rate gồm HTTP, network, parse, output sai định dạng. Mean và P95 chính tính trên response hợp lệ; có mean toàn bộ request để thấy tác động lỗi. P95 dùng nearest rank `ceil(0.95 × n)`. Khi run bị dừng, mọi chỉ số chỉ phản ánh phần đã thử: kiểm tra số attempts phải đủ `48 × repeats` **cho từng model** trước khi dùng bảng làm benchmark hoàn chỉnh.

Giá standard paid tier kiểm tra ngày 20/11/2026: Decisions $0.10/1M input và $0 output; Gemini 3.5 Flash-Lite $0.30/1M input, $2.50/1M output gồm thinking. Nguồn: [bảng giá Google](https://aistudio.google.com/docs/pricing), [thông báo Decisions](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/). Không có số liệu chi phí đo được chỉ vì đã biết đơn giá.

`cost = (input_tokens × input_rate + (answer_tokens + thinking_tokens) × output_rate) / 1.000.000`.

Chi phí/1.000 = tổng cost / số attempts × 1.000, chỉ xuất khi mọi request có usage. Usage thiếu trả N/A, không mặc định miễn phí. Free tier có thể làm số tiền thực trả bằng 0; báo cáo dùng paid-tier estimate để so sánh. Nếu có implicit cached tokens, script dùng full input rate nên đây là ước tính bảo thủ. Chưa áp dụng thuế, caching discount, batch, OCR ngoài API hay chi phí nhân công. Không đổi model trong `MODELS` mà quên cập nhật `RATES`.

## Đăng bài trên WordPress

Dán nội dung `article.html` vào block **Custom HTML** và mở Preview để kiểm tra. Các đoạn mã khai báo màu chữ và nền riêng để dễ đọc với CSS của theme.

## Quan điểm của tôi

Theo tôi, Decisions API phù hợp nhất làm tầng phân luồng với tập lựa chọn hữu hạn và tiêu chí rõ: loại tài liệu đầu vào, chọn pipeline xử lý, kiểm tra điều kiện trước khi gọi tool, gắn nhãn nội dung. Tôi ưu tiên thử nó khi một đầu vào cần nhiều kiểm tra độc lập, bởi API cho phép dùng chung input và trả xác suất có cấu trúc. Đây là nhận định về thiết kế API, chưa phải kết luận từ benchmark này. [Tài liệu Decisions](https://aistudio.google.com/docs/decisions-api) mô tả cơ chế và cách dùng confidence.

Tôi sẽ tiếp tục dùng generative AI cho trích xuất trường, giải thích điều khoản, tóm tắt, viết nội dung hoặc câu hỏi mở. Decisions chỉ chọn trong các đáp án đã định nghĩa; confidence cao vẫn có thể sai, nhất là dữ liệu lạ hay prompt injection. Tôi chưa thể khẳng định nó nhanh hơn, rẻ hơn tổng pipeline hoặc chính xác hơn baseline cho công việc của mình trước khi có kết quả thật.

## Cách tôi sẽ áp dụng vào công việc hằng ngày

Tôi sẽ thử phân loại file khách hàng trước khi đọc thủ công: invoice/receipt đi vào extraction để lấy số tiền, ngày, mã tham chiếu; contract chuyển sang generative AI tóm tắt nghĩa vụ; Other, lỗi API hoặc confidence thấp đưa vào hàng đợi rà soát. Sau đó tôi sẽ dùng các kiểm tra độc lập cho tài liệu thiếu thông tin, ngôn ngữ và nhu cầu review để giảm số lần gọi model tạo văn bản không cần thiết. File scan/PDF cần một benchmark multimodal riêng trước khi mở rộng từ demo text này.

Ngưỡng confidence phải được chọn trên tập validation thực tế riêng, rồi đo trên held-out test; không chọn ngưỡng bằng chính 48 mẫu báo cáo. Tôi sẽ theo dõi tỷ lệ xử lý tự động, số lần chuyển nhầm, thời gian sửa và chi phí toàn pipeline, gồm fallback và human review. Tôi kỳ vọng tiết kiệm thời gian phân loại file và chọn prompt; mức tiết kiệm chưa được đo. Không tự động thanh toán hay xác nhận nghĩa vụ hợp đồng từ nhãn của model.

## Kiểm thử

`tests/test_benchmark.py` kiểm tra cân bằng/duy nhất dataset, payload, parser, thinking-token accounting, accuracy khi có lỗi, nearest-rank P95, xử lý thiếu usage, xuất CSV/SVG và HTTP request serialization qua mocked transport. Các fixture **không chứng minh dịch vụ Google hoạt động**. Sandbox không cho mở socket server cục bộ, nên transport được mock. Smoke test có key là bước tiếp theo để kiểm tra schema/model trên dịch vụ thật.
