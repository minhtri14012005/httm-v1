# Tìm người · Local AI

Ứng dụng cục bộ **React + Vite / FastAPI / SQLite**, dùng SCRFD, AdaFace IR-101 WebFace12M và YOLO11n + BoT-SORT. Tái sử dụng baseline và checkpoint hiện có. Không huấn luyện lại, không gửi ảnh/video lên dịch vụ ngoài.

## Chạy trên máy hiện tại

```powershell
cd 'D:\httm v1'
.\start.cmd
```

Mở **http://127.0.0.1:8000**. Chờ AI chuyển sang sẵn sàng. `Ctrl+C` dừng server. Venv, checkpoint, Node portable và frontend build đã được chuẩn bị. `start.cmd -Build` build lại giao diện sau khi sửa frontend.

Luồng sử dụng: **Thư viện → Thêm người và ảnh → Bật danh sách cần tìm → Video Upload hoặc Live Camera → Kết quả**. Một ảnh rõ, đúng một mặt là đủ để đăng ký; có thể thêm/xóa nhiều ảnh và sửa/xóa người. Ảnh quá nhỏ/mờ hoặc nhiều mặt bị từ chối. Embedding được lưu lại, không tính lại mỗi lần mở app.

Video được giải mã trực tiếp bằng PyAV, chạy nhanh nhất theo tài nguyên thực có, không chờ phát realtime. Có tiến độ, dừng an toàn, video chú thích không âm thanh và frame minh chứng. Các mốc là thời gian từ đầu phiên: quan sát đầu, xác nhận, quan sát cuối, mất dấu. Video chú thích chỉ chứa các frame được phân tích; không phải bản sao đủ mọi frame của video gốc.

Live chỉ có **Webcam Laptop** và **Camo Camera** trên máy backend. Dò tên DirectShow mỗi lần mở, không cố định index. Chuyển nguồn dừng phiên cũ, giải phóng camera và tạo tracking mới. Nếu tên không rõ ràng, nhập tên DirectShow chính xác trong Cấu hình. Việc có thiết bị Camo không chứng minh iPhone đang truyền hình hoặc đang dùng camera sau; chọn camera sau trong Camo Studio. Máy không phải Windows hiện chưa có adapter tự dò camera.

## Thiết bị, bằng chứng và giới hạn

- Mặc định `auto`: kiểm tra và chạy thật cả ba model, ưu tiên CUDA, chuyển cả pipeline về CPU nếu GPU không tương thích. Có chế độ `cpu`, `cuda`; `cuda` ép thiết bị thì báo lỗi thay vì đổi âm thầm. Thiếu VRAM trong `auto` sẽ khởi tạo lại trên CPU và kết thúc các ràng buộc tracking cũ. Chỉ một phiên video/live được chạy cùng lúc; đổi thư viện/watchlist/cấu hình khi đã dừng phiên.
- SCRFD CUDA có kiểm tra profile: tính toán ảnh chạy CUDA, một số phép shape int64 chạy CPU. YOLO và AdaFace được kiểm tra forward trên thiết bị thực. Preprocessing/alignment/NMS và tracker dùng CPU. Không giả định AMD/Intel hỗ trợ CUDA.
- Danh tính cần nhiều quan sát khuôn mặt, score và margin so với người đứng thứ hai. Ngưỡng mặc định là **giá trị thử nghiệm chưa hiệu chuẩn**, không phải độ chính xác đã xác nhận. Có thể chỉnh trong Cấu hình hoặc `config.json`; không thay ngưỡng chỉ để làm đẹp kết quả một video.
- Khi mặt khuất, danh tính được duy trì trên track đã xác nhận và hiển thị trạng thái tracking. Không suy ra danh tính mới bằng cơ thể; không chuyển danh tính tự động sang track ID mới. BoT-SORT hiện **không bật appearance Re-ID**. Liên kết face/body mơ hồ bị bỏ qua; detector có thể bỏ sót mặt nhỏ hoặc người bị che. Track vẫn có nguy cơ đổi nhầm ID khi người cắt ngang; cần dữ liệu có nhãn để đánh giá.
- Mất dấu không có nghĩa người chắc chắn rời khỏi phòng. Các mốc có độ phân giải phụ thuộc tần suất phân tích. Chưa đo FPS/độ chính xác hoặc xác minh nhận diện ở 3–8 m.

`data/` chứa SQLite, ảnh, video và minh chứng; `models/` chứa pretrained; đều được gitignore. Ứng dụng chỉ bind loopback và chưa có đăng nhập; không mở public qua mạng khi chưa bổ sung kiểm soát truy cập. Các báo cáo baseline cũ giữ nguyên để tham khảo, không đại diện cho dependency app hiện tại.

## Cài trên máy Windows khác

Dùng Python **3.12 x64**. Không copy `.venv`; sao chép source và tùy chọn `models/` có manifest. Chọn một profile trên venv mới:

```powershell
# Máy chỉ CPU / AMD / Intel (không cài runtime GPU)
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Profile cpu

# Hoặc NVIDIA: kiểm tra driver trên máy đó trước, không cần tự cài CUDA Toolkit
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Profile nvidia
.\start.cmd
```

Setup chỉ cài vào `.venv`, tải checkpoint chính thức và Node 22 portable vào `.tools` nếu thiếu Node. Không sửa Python global, driver hoặc PATH hệ thống. Profile CPU dùng torch/torchvision CPU + `onnxruntime`; NVIDIA dùng torch/torchvision cu121 + `onnxruntime-gpu`. Chỉ một OpenCV (`opencv-python`) và một ONNX Runtime trong venv. Script từ chối trộn profile có sẵn; không tự xóa môi trường. Yêu cầu Internet khi setup, không cần Internet khi inference. Xem `requirements/app.txt`, `cpu.txt`, `nvidia.txt`, `constraints.txt`; lock thực tế máy này: `requirements/lock-app-nvidia-windows-py312.txt` (không dùng cho CPU).

Kiểm tra riêng mỗi máy, không suy diễn từ RTX 4050 của máy phát triển:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe face_baseline.py --mode cpu smoke --image samples/img2.jpeg
# Chỉ khi có NVIDIA tương thích:
.\.venv\Scripts\python.exe face_baseline.py --mode cuda smoke --image samples/img2.jpeg
```

## Kiểm thử đã thực hiện

Build React thành công, HTTP/UI assets và API trả thành công; **11 tests đạt**. Integration chạy model thật trên ảnh mẫu upstream, video fixture lặp ảnh mẫu: CRUD, nhiều ảnh/embedding, watchlist, upload, encode/decode video chú thích, minh chứng, WebSocket, hủy tác vụ. Đã chạy model trên CPU/CUDA; giả lập OOM rồi chạy lại model thật trên CPU. Camera được liệt kê thực tế: ACER HD User Facing và Camo; chuyển nguồn/mất kết nối mới kiểm tra bằng mock, **chưa mở camera thật để kiểm chứng**. Công cụ điều khiển trình duyệt bị lỗi trusted RPC dependency, nên chưa QA bằng thao tác trình duyệt/screenshot.

Ảnh/video cá nhân đính kèm đọc được; chỉ kiểm tra metadata và giải mã video H.264 720×1280, khoảng 57 giây. Không dùng khuôn mặt đính kèm để trực tiếp xác định/theo dõi người trong video. Không có kết luận nhận diện thực tế hoặc ground truth từ các tệp đó. Video fixture không phải bằng chứng cho chuyển động, che khuất hoặc đám đông. Chưa thử máy vật lý CPU-only; profile CPU cần smoke test trên máy đích.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests dùng `results/app_test_<id>/` riêng, không đưa dữ liệu thử vào thư viện ứng dụng. Các lỗi startup/model hiển thị trong giao diện và terminal. Khi GPU ép `cuda` lỗi, đổi `config.json` về `cpu` rồi khởi động lại. Frontend dev: `npm run dev` trong `frontend` với backend ở port 8000; proxy Vite cho API và WebSocket.

## Nguồn và giấy phép

[AdaFace/CVLFace](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m) giữ revision `54f602a0737bd1ee4a4e7e9fd089a485f397fefd`, safetensors, RGB112 chuẩn hóa `[-1,1]`, L2 512 chiều. MIT áp dụng cho mã CVLFace; giấy phép trọng số còn phụ thuộc dữ liệu WebFace, chưa xác minh được nguyên văn điều khoản do website lỗi TLS. [InsightFace pretrained](https://github.com/deepinsight/insightface/blob/v0.7/python-package/README.md) giới hạn nghiên cứu phi thương mại; mã SCRFD kèm license tại `baseline/`. [YOLO11n](https://github.com/ultralytics/assets/releases/tag/v8.3.0) và [Ultralytics/BoT-SORT](https://github.com/ultralytics/ultralytics) theo AGPL-3.0 hoặc giấy phép Enterprise phù hợp. Không coi giấy phép mã là quyền sử dụng thương mại tất cả trọng số. URL/SHA256 lưu tại `models/manifest.json`; checkpoint được kiểm tra hash trước khi nạp.
