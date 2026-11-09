# Báo cáo kiểm tra môi trường AI

Ngày kiểm tra: 08/11/2026, múi giờ Asia/Bangkok. Thư mục: `D:\httm v1`.

Phạm vi: máy Windows đang chạy Codex; không đại diện cho máy khác. Chỉ tạo báo cáo này, không cài/nâng cấp thư viện, không tạo môi trường ảo, không tải model/dataset, không sửa source hoặc cấu hình. Các phép thử dùng tensor/model tổng hợp trong bộ nhớ, không benchmark hoặc huấn luyện.

Quy ước: **Đã kiểm tra** là kết quả lệnh thực tế; **Chưa xác minh** là chưa đủ bằng chứng; **Đề xuất/suy luận** không phải kết quả đo pipeline.

## 1. Phần cứng thực tế

| Thành phần | Kết quả đã kiểm tra |
|---|---|
| CPU | Intel Core i5-13500HX, 14 nhân vật lý, 20 luồng |
| RAM vật lý Windows báo cáo | 16.886.128.640 byte, khoảng 15,73 GiB; khả dụng khoảng 2,80 GiB tại thời điểm lấy mẫu |
| GPU NVIDIA | NVIDIA GeForce RTX 4050 Laptop GPU, 1 GPU CUDA |
| VRAM | NVIDIA-SMI: 6.141 MiB; PyTorch: 6.438.780.928 byte tổng, 5.326.766.080 byte trống tại lúc kiểm tra |
| Driver NVIDIA | 617.42, WDDM |
| CUDA capability của driver | NVIDIA-SMI ghi `CUDA UMD Version: 13.4` |
| Ổ C | Tổng khoảng 275,59 GiB; trống 85,82 GiB |
| Ổ D | Tổng khoảng 200,19 GiB; trống 146,99 GiB |

Bằng chứng: tên CPU từ registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor`; số nhân từ Win32 API `GetLogicalProcessorInformationEx`; 20 luồng từ .NET và 20 mục registry CPU. RAM từ `GlobalMemoryStatusEx`; dung lượng đĩa từ `shutil.disk_usage`; GPU từ `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` và PyTorch. Bộ nhớ trống là ảnh chụp thời điểm, có thể thay đổi.

`Get-CimInstance` cho Processor, ComputerSystem, OperatingSystem, VideoController, LogicalDisk đều trả `Access denied`, HRESULT `0x80041003`. Đã dùng các nguồn chỉ đọc khác ở trên; chưa liệt kê được đầy đủ GPU khác qua WMI.

## 2. Phần mềm và môi trường Python

| Thành phần | Kết quả |
|---|---|
| Hệ điều hành | `Windows-11-10.0.26200-SP0`, AMD64; chưa xác minh edition |
| Python | 3.12.10, 64 bit; `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe` |
| pip | 25.0.1; `python -m pip --version` thành công |
| Môi trường đang dùng | Python global; `sys.prefix == sys.base_prefix`, không có VIRTUAL_ENV/CONDA/CUDA/PYTHON env var được truy vấn |
| torch | 2.5.1+cu121, import thành công |
| torchvision | 0.20.1+cu121, import thành công |
| onnxruntime-gpu | 1.19.2; import `onnxruntime` thành công, phiên bản 1.19.2 |
| onnxruntime bản CPU riêng | Chưa cài theo distribution metadata; CPU provider vẫn có trong bản GPU |
| ONNX | 1.22.0; tạo model tổng hợp thành công |
| OpenCV | `opencv-python` 4.9.0.80 và `opencv-python-headless` 4.11.0.86 cùng tồn tại; `cv2.__version__` thực tế là 4.9.0 |
| NumPy | 1.26.4, import thành công |
| Pillow | 12.3.0, import thành công |
| FastAPI | 0.110.0, import thành công |
| PyAV | 17.1.0, import và thử video trong bộ nhớ thành công |
| Ultralytics | 8.4.143 theo metadata; không import để tránh tự tạo settings; chưa chạy model |
| Git | 2.53.0.windows.2 |
| Node.js / npm | Không tìm thấy lệnh trên PATH; không thấy ở đường dẫn cài phổ biến/registry đã kiểm tra; chưa xác minh bản portable ngoài phạm vi |
| FFmpeg CLI | Không tìm thấy lệnh trên PATH; chưa xác minh bản portable ngoài phạm vi |
| Camo Studio | Registry ghi phiên bản 2.9.3.26054; có `C:\Program Files (x86)\Camo Studio` |

Thư mục dự án hiện tại không có source, video mẫu, `pyvenv.cfg`, requirements, pyproject hoặc package.json trong kết quả liệt kê/tìm kiếm. Chưa tìm thấy môi trường tại các vị trí thông dụng `.virtualenvs`, `miniconda3`, `anaconda3`; không thể kết luận toàn máy không có môi trường khác. Liệt kê thư mục gốc `C:\Users\Admin` bị từ chối quyền. `py -0p` trả `No installed Pythons found!` dù gọi `python` trực tiếp thành công: cơ chế dò Python của launcher chưa phản ánh interpreter đang dùng; chưa xác định nguyên nhân chính xác.

### Phụ thuộc lỗi hoặc có nguy cơ chồng lấn

`python -m pip check` báo:

```text
insightface 2.0 requires onnxruntime, which is not installed.
streamlit 1.63.0 has requirement python-multipart<1,>=0.0.10, but you have python-multipart 0.0.9.
streamlit 1.63.0 has requirement starlette<2,>=0.46.0, but you have starlette 0.36.3.
streamlit 1.63.0 has requirement uvicorn<1,>=0.30.0, but you have uvicorn 0.28.0.
```

Lỗi InsightFace là kiểm tra tên distribution: `onnxruntime-gpu` đã cung cấp module và chạy thử được, nhưng không thỏa tên dependency `onnxruntime` mà metadata yêu cầu. Chưa import/chạy InsightFace nên chưa xác nhận tương thích thực tế. Không suy ra cần cài đồng thời hai bản ONNX Runtime. Hai distribution OpenCV dùng chung namespace `cv2` là nguy cơ chồng file; phiên bản thực tế khác bản headless được ghi nhận. Không tự sửa các vấn đề này.

## 3. GPU, CUDA và khả năng chạy AI

### Phân biệt ba tầng CUDA

- **Driver capability:** NVIDIA-SMI hoạt động và báo CUDA UMD 13.4; đây không phải bằng chứng Toolkit 13.4 được cài.
- **CUDA Toolkit:** `nvcc --version` không chạy được vì không tìm thấy lệnh; không thấy thư mục CUDA Toolkit ở vị trí tiêu chuẩn và mục cài đặt tương ứng trong truy vấn. Chưa xác minh có Toolkit ngoài PATH/đường dẫn khác. Không cần kết luận Toolkit hiện diện để giải thích việc PyTorch chạy được.
- **Runtime của PyTorch:** `torch.version.cuda == '12.1'`, torch wheel `+cu121`; đã chạy thật trên GPU với driver hiện tại.

### PyTorch — đã kiểm tra thành công

`torch.cuda.is_available()` trả `True`; `torch.cuda.device_count()` trả `1`. Thiết bị báo RTX 4050 Laptop GPU, compute capability 8.9, 20 multiprocessor. Tensor `[1., 2., 3.]` được tạo trên CUDA; tính tổng bình phương, đồng bộ CUDA và đọc kết quả về CPU được `14.0`, không có lỗi. Không chạy SCRFD/AdaFace/Person Detection hoặc đo tốc độ.

### ONNX Runtime — có phép thử thực thi, không chỉ liệt kê provider

`get_available_providers()` trả:

```text
TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
```

Tạo model ONNX `Add` nhỏ trong RAM, opset 13, IR 9. Input `[1,2,3] + [1,2,3]`:

- CPU session thực thi thành công, trả `[2,4,6]`.
- CUDA session thực thi thành công, trả `[2,4,6]`; đã đặt `session.disable_cpu_ep_fallback=1`. Session báo CUDA và CPU providers; CPU provider vẫn xuất hiện trong danh sách session không đồng nghĩa đã fallback phép Add. Kiểm tra này xác nhận chạy được graph nhỏ với cấu hình cấm CPU fallback, không chứng minh mọi toán tử/model đều tương thích.
- TensorRT chỉ xuất hiện trong danh sách, chưa tạo session hoặc chạy thử TensorRT.

CUDA ONNX được thử sau khi import PyTorch trong cùng process. Vì vậy **chưa xác minh** ONNX CUDA hoạt động độc lập khi không import PyTorch trước; thứ tự nạp DLL/runtime cần được kiểm tra trong backend thực tế. Chưa đo hay xác nhận placement từng node bằng profiling; chưa chạy model pretrained.

## 4. Camera và video

**Đã kiểm tra:** OpenCV build có FFmpeg (avcodec 58.134.100, avformat 58.76.100), DirectShow và Media Foundation; registry backend xác nhận FFMPEG/MSMF/DSHOW sẵn dùng. GStreamer và Intel MFX không sẵn dùng trong build này. FFmpeg tích hợp OpenCV khác FFmpeg executable trên PATH.

PyAV đã encode một frame tổng hợp 32×32 bằng MJPEG vào AVI trong RAM, rồi mở và decode lại được frame 32×32. Không tạo file video. Đây là phép thử codec/container nhỏ; không bảo đảm mọi video upload, codec điện thoại hoặc video lỗi đều đọc được. OpenCV chưa được thử đọc file video thực tế do không có mẫu và chỉ được tạo báo cáo; khả năng video của OpenCV mới xác nhận ở mức build/backend.

**Camo:** đã có phần mềm và process `CamoService` PID 4776 tại thời điểm kiểm tra. `Get-PnpDevice -PresentOnly` bị `Access denied` qua WMI. Chưa xác nhận Camo virtual camera được enumerated, có stream hoặc iPhone 13 camera sau đang kết nối. Không mở camera bằng `VideoCapture`, không quét index để tránh chiếm camera/gián đoạn ứng dụng. Việc có service không chứng minh đang có hình.

## 5. Thành phần đã sẵn sàng và mức độ triển khai

**Đã kiểm tra:** Python, pip, Git, torch/torchvision, CUDA tensor, ONNX CPU và CUDA graph nhỏ, NumPy/Pillow/FastAPI/PyAV/OpenCV. GPU NVIDIA thực tế dùng được qua hai framework trong phạm vi thử nghiệm nêu trên. CPU ONNX cũng đã thực thi thành công.

**Suy luận kỹ thuật:** CPU 14 nhân/20 luồng, RAM khoảng 16 GB và GPU khoảng 6 GB phù hợp để bắt đầu prototype pipeline nhiều khuôn mặt, với ngân sách bộ nhớ được quản lý. RAM trống chỉ khoảng 2,8 GiB tại lúc kiểm tra là hạn chế đáng chú ý khi nạp đồng thời IR-101 và nhiều detector. Chưa đo đỉnh RAM/VRAM, FPS, latency hoặc độ chính xác, nên chưa kết luận đáp ứng realtime hay quy mô đám đông cụ thể. Nhận diện góc nghiêng/che khuất và giữ đúng danh tính qua tracking cần đánh giá bằng dữ liệu thử thực tế.

**Chưa sẵn sàng đầy đủ:** frontend React/Vite chưa có Node/npm khả dụng; chưa có source, model pretrained hay video mẫu tại dự án để kiểm thử pipeline; chưa xác minh Camo stream; môi trường global có xung đột phụ thuộc. BoT-SORT, SCRFD và AdaFace IR-101 WebFace12M chưa được chạy/kiểm tra tương thích hoặc xác minh artifact. Không tải các thành phần này trong đợt kiểm tra.

Có thể bắt đầu bước chuẩn bị prototype AI tối thiểu dựa trên phần cứng/framework đã kiểm tra; chưa coi đây là môi trường tích hợp hoàn chỉnh. CPU fallback khả thi về phần mềm qua ONNX CPU đã thử và PyTorch có API CPU, nhưng cả pipeline trên CPU và tốc độ chấp nhận được chưa được kiểm chứng.

## 6. Đề xuất môi trường Python

Đây là đề xuất dựa trên các phiên bản vừa kiểm tra tại máy, không phải xác nhận một bộ dependency mới đã được kiểm thử hoặc khuyến nghị phiên bản mới nhất.

- Giai đoạn triển khai sau: tạo môi trường riêng, dùng Python 64 bit. Python 3.12.10 hiện có đã chạy được các framework trên; nếu repository pretrained có yêu cầu khác, chọn phiên bản Python sau khi đối chiếu và thử tương thích, không tự thay global.
- Dùng torch 2.5.1/torchvision 0.20.1 làm mốc đã import thành công; cấu hình NVIDIA hiện tại có wheel cu121 đã chạy được. Không áp đặt wheel GPU này cho máy CPU hoặc GPU hãng khác.
- ONNX Runtime GPU 1.19.2 là mốc graph nhỏ đã chạy trên máy này; cần xác minh DLL, CUDA/cuDNN và model thật trong môi trường riêng. CPU dùng distribution CPU phù hợp. Chỉ chọn một distribution cung cấp `onnxruntime` trong mỗi môi trường.
- Chỉ chọn một distribution OpenCV. Với backend không cần GUI, có thể chọn headless sau khi kiểm tra các module cần dùng. Không giữ hai gói đang chồng namespace.
- Tách dependency backend khỏi Streamlit không dùng đến; giải quyết các ràng buộc FastAPI/Starlette/Uvicorn/python-multipart bằng resolver và `pip check` trong môi trường riêng. Không sửa môi trường global như một phần audit.
- Chuẩn bị nhóm dependency common, CPU, NVIDIA CUDA; khóa phiên bản riêng sau khi chạy smoke test trên từng loại máy. Không buộc máy CPU cài torch CUDA, ONNX GPU hoặc TensorRT. Cơ chế cài lựa chọn wheel cần tường minh, không tự tải dependency khi server khởi động.
- Node/npm và FFmpeg CLI là bước chuẩn bị riêng nếu cần frontend và xử lý video bằng executable; PyAV/OpenCV hiện đã có decoder, không suy ra bắt buộc phải có FFmpeg CLI cho mọi đường xử lý.

## 7. Kiến trúc tương thích nhiều phần cứng — yêu cầu cho giai đoạn sau

Chưa triển khai bất kỳ source code nào. Thiết kế dưới đây là đề xuất và tiêu chí cần kiểm chứng.

1. **Cấu hình thiết bị:** mặc định `auto`; hỗ trợ `cpu` và backend GPU được triển khai rõ ràng, ví dụ `cuda`. Không hardcode RTX 4050, tên GPU, VRAM, device index hoặc đường dẫn CUDA. Chọn thiết bị từ danh sách được dò hoặc selector cấu hình, kiểm tra selector hợp lệ; không cố định GPU số 0 như giả định kiến trúc.
2. **Khởi động và kiểm tra:** dò runtime/framework và thiết bị nhìn thấy trong process; ghi phiên bản driver/runtime nếu lấy được. Có GPU trong hệ thống hoặc provider trong danh sách chưa đủ. Thực hiện phép thử nhỏ và tạo session/nạp từng model bằng backend được chọn. Kiểm tra preprocessing, toán tử, dtype, memory và inference thật cho Face Detection, Face Recognition, Person Detection trước khi báo backend sẵn sàng.
3. **Lựa chọn nhất quán:** một bộ quản lý thiết bị đưa ra kế hoạch runtime cho toàn pipeline, truyền cấu hình vào mọi adapter model. Trong `auto`, ưu tiên NVIDIA CUDA khi toàn bộ đường chạy dự kiến đã qua kiểm tra; nếu thất bại, chuyển toàn bộ các model inference về CPU theo chính sách rõ ràng và báo lý do. Không để từng module tự chọn GPU/CPU âm thầm. Tracker và preprocessing có thể có phần CPU theo thiết kế, cần ghi rõ phân công và chi phí copy.
4. **Chế độ ép backend:** `cpu` không cần stack GPU; backend GPU được yêu cầu rõ nhưng lỗi phải báo chi tiết, không âm thầm tuyên bố đã tăng tốc. Nếu cho phép fallback ở chế độ ép backend, cần tùy chọn tường minh và trạng thái thực tế hiển thị cho người dùng. Startup smoke test không thay thế xử lý OOM/lỗi runtime trong khi chạy.
5. **AMD/Intel:** chỉ nghiên cứu adapter/runtime tương thích và có wheel hỗ trợ đúng OS/model sau này; không coi GPU AMD/Intel là CUDA, không báo tăng tốc trước khi model thực thi thành công. Audit này chưa kiểm tra runtime AMD/Intel hoặc DirectML/OpenVINO.
6. **Nguồn hình ảnh:** camera là nguồn cấu hình được phát hiện bằng capability hệ điều hành, hỗ trợ selector thiết bị/URI tùy backend, không hardcode index hoặc yêu cầu iPhone/Camo. Kiểm tra mở/đọc camera khi người dùng bắt đầu phiên dùng camera, xử lý thiết bị bận/mất kết nối. Video Upload có luồng xử lý độc lập, không mở camera khi startup hoặc khi chỉ xử lý file.
7. **Máy yếu:** profile cấu hình độ phân giải, batch size và chu kỳ inference có giới hạn bộ nhớ; ghi lại profile thực dùng. Giảm tần suất phải đi kèm tracking và trạng thái nhận diện rõ ràng; không âm thầm tắt nhận diện, không coi track dự đoán là lần xác nhận danh tính mới. Xác minh chất lượng theo từng profile, không hứa FPS/độ chính xác khi chưa đo.
8. **Khả năng quan sát:** báo backend mong muốn/thực tế cho từng model, lỗi nạp model/provider, fallback, RAM/VRAM khi có và nguồn camera/video đang dùng. Mỗi máy mới phải chạy audit/smoke test riêng; không tái sử dụng kết luận phần cứng của máy hiện tại.

## 8. Bước triển khai tiếp theo

Các bước sau chỉ là kế hoạch, chưa được thực hiện:

1. Chuẩn bị môi trường riêng theo nhóm CPU hoặc NVIDIA; làm sạch dependency ở môi trường mới và chạy `pip check`.
2. Chọn artifact pretrained SCRFD, AdaFace IR-101 WebFace12M và person detector; kiểm tra nguồn, license, preprocessing/alignment và định dạng runtime trước khi sử dụng.
3. Thử từng model trên CPU/CUDA và kiểm tra ONNX CUDA trong process sạch; xác nhận session thực thi và phát hiện CPU fallback ngoài ý muốn bằng công cụ kiểm tra placement phù hợp.
4. Xác minh camera được cấu hình trong phiên kiểm tra có chủ đích; thử video upload riêng với codec/container thực tế, bao gồm dữ liệu điện thoại và file lỗi.
5. Ghép pipeline nhận diện và tracking, đăng ký một hoặc nhiều ảnh; đánh giá góc nghiêng, chuyển động, che khuất, nhầm danh tính và mất/khôi phục track. Đo FPS/latency, bộ nhớ và chất lượng trên các profile CPU/GPU trước khi kết luận năng lực.
6. Chuẩn bị Node/npm cho frontend và kiểm thử triển khai trên ít nhất một máy CPU khác. Không suy diễn cấu hình hoặc kết quả của máy đó từ audit này.

## 9. Giới hạn và bằng chứng kiểm tra

Các lệnh/chức năng chính đã dùng: `Get-Command`, `py -0p`, `python -B -`, `python -m pip --version`, `python -m pip check`, `nvidia-smi`, `nvcc --version`, truy vấn registry phần mềm/CPU, `Get-Process`, `Get-PnpDevice`, Win32 API lấy topology/RAM và Python metadata/import. Không mở camera, không thử TensorRT, không tải hoặc chạy model nhận diện, không benchmark nặng. Không dùng nguồn Internet để suy diễn thông số thay cho kết quả tại máy.

Không tìm thấy lệnh trên PATH không chứng minh phần mềm hoàn toàn không tồn tại ở mọi thư mục. WMI và một phần thư mục người dùng bị giới hạn quyền; không thay quyền hoặc cấu hình để khắc phục. Các giới hạn này được giữ nguyên và báo cáo thay vì tự sửa.
