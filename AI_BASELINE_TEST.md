# Báo cáo baseline nhận diện khuôn mặt

Ngày: 08/11/2026, Asia/Bangkok. Đã đọc toàn bộ `ENVIRONMENT_AUDIT.md` trước khi thực hiện. Báo cáo audit được giữ nguyên như ảnh chụp của giai đoạn trước.

## 1. Kết quả bàn giao

Đã tạo môi trường riêng `.venv` Python 3.12.10 x64, `include-system-site-packages = false`. Không cài vào Python global; không đổi driver hoặc cài CUDA Toolkit. Đã triển khai và chạy baseline:

**Ảnh → SCRFD → 5-point similarity alignment → AdaFace IR-101 WebFace12M → L2 embedding → cosine similarity.**

CLI `face_baseline.py` có `smoke`, `enroll`, `match`, mặc định `--mode auto`, hỗ trợ `cpu` và `cuda`. Hướng dẫn đầy đủ tại [README.md](README.md). Không có React, FastAPI, camera, upload video, person detection hoặc tracking. Không huấn luyện, không tải dataset, không chạy benchmark kéo dài.

## 2. Model/checkpoint và giấy phép

### SCRFD

- Detector thực tế: **SCRFD-500MF có 5 keypoint**, file `det_500m.onnx` lấy từ archive `buffalo_sc.zip` của [InsightFace release v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7). Lưu tại `models/scrfd_500m_kps.onnx`.
- [Model zoo v0.7](https://github.com/deepinsight/insightface/blob/v0.7/python-package/README.md) ghi rõ buffalo_sc dùng SCRFD-500MF. Không nhầm với person detector hoặc recognition MobileFaceNet trong cùng gói. Chỉ giải nén/sử dụng detector; archive tải 14.969.382 byte, không giải nén model recognition không cần thiết.
- Model ONNX đã tạo session và inference thành công trên CPU/CUDA, có 5 landmark và decode/NMS từ code SCRFD chính thức. Dùng biến thể nhỏ để kiểm tra tính đúng đắn, chưa so sánh chất lượng với SCRFD-2.5G/10G.
- [README InsightFace](https://github.com/deepinsight/insightface/blob/master/README.md) giới hạn pretrained models cho nghiên cứu phi thương mại. License mã không thay thế license trọng số. Thư mục source SCRFD có Apache 2.0, dù README chung ghi code MIT; đã giữ Apache và attribution trong `baseline/LICENSE_SCRFD.txt`.

### AdaFace

- Nguồn: [minchul/cvlface_adaface_ir101_webface12m](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m).
- Revision cố định: `54f602a0737bd1ee4a4e7e9fd089a485f397fefd`.
- Artifact thực tải: **model.safetensors**, 260.980.552 byte, lưu tại `models/adaface_ir101_webface12m.safetensors`. Đây là bản CVLFace IR-101 WebFace12M, không phải ví dụ AdaFace IR-50 MS1MV2 hoặc checkpoint `.ckpt` cũ.
- Đọc `wrapper.py`, `config.json`, `pretrained_model/model.yaml`, `models/iresnet/model.py`, lớp IResNet/BaseModel và model card trước khi tải. Wrapper chính thức dùng Transformers và còn nạp `pretrained_model/model.pt`; baseline tránh tải checkpoint trùng và dependency huấn luyện bằng cách dựng đúng backbone từ source, nạp safetensors trực tiếp. Bỏ prefix `model.net.` rồi `load_state_dict(..., strict=True)` thành công; không bỏ qua missing/unexpected keys.
- Mã IR-101 upstream ánh xạ tới `Backbone(..., num_layers=100, mode='ir')`; giữ nguyên cách đặt tên/kiến trúc chính thức, không đổi thành backbone khác dựa trên tên gọi 101. Bỏ duy nhất import fvcore và đoạn profiling standalone.
- [CVLFace LICENSE](https://github.com/mk-minchul/CVLface/blob/main/LICENSE) là MIT, lưu nguyên văn. Model card yêu cầu tuân thủ license dữ liệu huấn luyện; metadata không có giấy phép trọng số độc lập. Website WebFace được tham chiếu gặp lỗi TLS hostname, chưa lấy được toàn văn điều khoản từ đó; không vô hiệu kiểm tra chứng chỉ, không khẳng định trọng số được phép thương mại. Đây là vấn đề license còn cần xác minh trước sử dụng ngoài thử nghiệm.

Chi tiết attribution và phạm vi giấy phép tại [THIRD_PARTY.md](THIRD_PARTY.md); tài liệu đã đọc trong `sources/`.

### Hash thực tế

| Artifact | SHA256 |
|---|---|
| AdaFace safetensors | `2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56` |
| SCRFD ONNX | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| buffalo_sc.zip | `57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72` |

Manifest `models/manifest.json` lưu URL, revision, kích thước và hash, bao gồm ảnh mẫu. Downloader và loader kiểm tra hash cố định của nội dung đã tải lần đầu từ HTTPS chính thức; không coi đây là chữ ký tác giả. Loader không gọi mạng và không thực thi remote code bằng `trust_remote_code`.

## 3. Tiền xử lý và alignment đã dùng

1. Đọc ảnh bằng Pillow, áp dụng EXIF orientation, chuyển RGB rồi BGR uint8 cho detector.
2. SCRFD: giữ tỷ lệ, resize/pad vào góc trên trái canvas 640×640 mặc định; `blobFromImage` với `swapRB=True`, mean 127.5 và scale `1/128`, đúng source detector. Decode ba stride 8/16/32, NMS 0.4. Detection confidence mặc định 0.5 là ngưỡng detector, không phải ngưỡng danh tính.
3. Lấy 5 landmark theo thứ tự hai mắt, mũi, hai khóe miệng. Fit similarity transform không reflection bằng least squares trên đủ 5 điểm; warp bilinear với padding đen vào ảnh 112×112. Template là tọa độ chính thức trong [CVLFace aligner_helper](https://github.com/mk-minchul/CVLface/blob/main/cvlface/research/recognition/code/run_v1/aligners/retinaface_aligner/aligner_helper.py): `(38.2946,51.6963)`, `(73.5318,51.5014)`, `(56.0252,71.7366)`, `(41.5493,92.3655)`, `(70.7299,92.2041)`.
4. **AdaFace checkpoint này nhận RGB**, chuyển ảnh aligned BGR sang RGB, CHW FP32, `pixel / 127.5 - 1`. Tương đương `ToTensor` + Normalize(mean=0.5,std=0.5) của model card. Không dùng chuyển BGR trong `AdaFace/inference.py` cũ cho checkpoint CVLFace RGB này.
5. `eval`, `inference_mode`, output raw `[1,512]`, kiểm tra finite/nonzero, L2-normalize rồi dot product để có cosine. Chuẩn hóa embedding được áp dụng tường minh, không giả định raw output đã có norm 1.

Ba unit test cho phục hồi transform từ rotation/scale/translation biết trước, thứ tự RGB/chuẩn hóa và từ chối landmark suy biến đều đạt. SCRFD landmarks thay MTCNN/RetinaFace của ví dụ upstream; template và preprocessing được đối chiếu, nhưng chưa đánh giá tác động chất lượng của detector/alignment mới trên tập có nhãn.

## 4. Môi trường/dependency thực tế

| Thành phần | Phiên bản trong `.venv` |
|---|---|
| Python / pip | 3.12.10 x64 / 25.0.1 |
| torch | 2.5.1+cu121 |
| ONNX Runtime GPU | 1.19.2; không cài distribution CPU cùng lúc |
| OpenCV headless | 4.11.0.86; không cài opencv-python |
| NumPy | 1.26.4 |
| Pillow | 11.1.0 |
| safetensors | 0.5.3 |
| sympy | 1.13.1, theo ràng buộc torch |

Toàn bộ dependency thực cài lưu ở `requirements/lock-nvidia-windows-py312.txt`. `python -m pip check` trong venv trả **No broken requirements found**. Không kế thừa xung đột InsightFace/Streamlit/OpenCV của global. Không cần torchvision, Transformers, InsightFace package, mmcv/mmdet hoặc requirements huấn luyện để chạy đường inference đã chọn.

Tách `requirements/common.txt`, `cpu.txt`, `nvidia.txt`; wheel torch CPU/CUDA được chọn bằng index chính thức riêng theo hướng dẫn. Máy CPU không cần cài wheel CUDA hoặc ORT GPU. Không tự cài CUDA Toolkit; runtime torch là CUDA 12.1, driver hiện tại 617.42 có capability UMD 13.4 như audit. Không suy ra mọi máy có GPU đều hỗ trợ bộ wheel này.

## 5. Kiểm thử model thật CPU và CUDA

### Bằng chứng thực thi

| Kiểm tra | CPU | NVIDIA CUDA |
|---|---|---|
| Nạp SCRFD ONNX | Đạt | Đạt |
| Inference SCRFD thật + profiling | 123 CPU kernel events | 91 CUDA kernel events, 12 CPU events phụ trợ |
| Nạp AdaFace safetensors strict | Đạt | Đạt |
| Inference AdaFace thật | torch CPU, embedding 512 | torch CUDA được runtime chọn, embedding 512 |
| Phát hiện/căn chỉnh/embedding 3 ảnh mẫu | Mỗi ảnh 1 mặt | Mỗi ảnh 1 mặt |
| Ảnh ghép cùng ảnh mẫu hai lần | 2 mặt | 2 mặt |
| Ảnh đen tổng hợp | 0 mặt | 0 mặt |

Profiling SCRFD CUDA ghi 12 CPU events: Slice×2, Gather×4, Unsqueeze×4, Concat×2, với input int64 nhỏ dùng xử lý shape. Phần tính toán ảnh đã có CUDA events thực thi; không tuyên bố toàn bộ graph hoặc preprocessing nằm trên GPU. Mã kiểm tra profile, từ chối CPU compute ngoài danh sách shape cho phép hoặc input không phải int64. CPU preprocessing/alignment/NMS là chủ đích của baseline.

Việc provider có trong danh sách không được dùng làm bằng chứng duy nhất. Có kiểm tra session giữ CUDA, inference thật, profile node, nạp trọng số và forward AdaFace trên thiết bị thực tế. Profile JSON giữ trong `results/profiles/`; kết quả tổng hợp trong `results/integration.json`, `results/smoke_cpu.json`, `results/smoke_cuda.json`.

### Điểm tương đồng quan sát được

Đăng ký img1, so với ba ảnh mẫu; mỗi cột dùng embedding đăng ký tạo trên chính backend đó trong integration:

| Cặp | CPU cosine | CUDA cosine |
|---|---:|---:|
| img1 → img1 | 1.000000 | 1.000000 |
| img1 → img2 | 0.740476 | 0.740506 |
| img1 → img3 | -0.080455 | -0.080498 |

Cosine giữa embedding img1 của CPU và CUDA trong lượt integration: **0.999999285**. Đây là kiểm tra tính gần nhau về số học trên một mẫu, không phải tỷ lệ chính xác. Khác biệt nhỏ giữa CPU/CUDA và các lượt chạy có thể đến từ tính toán dấu phẩy động, detector/landmark/warp; chưa điều tra riêng sai số từng tầng hoặc áp chế độ deterministic.

CLI cũng đã đăng ký chỉ một ảnh img1 bằng `auto` (chọn CUDA), ghi `results/reference.json`, rồi so img2 bằng CPU và CUDA: lần lượt **0.740701079** và **0.740638852**. Đây là tham chiếu CUDA dùng chung nên không nhất thiết bằng hai cột integration. `identity_threshold` là `null`; không gán kết luận cùng/khác danh tính.

Ba ảnh mẫu nhỏ lấy từ kho AdaFace để thử phần mềm, chưa có nhãn danh tính được xác minh cho bài toán này. Không có ảnh đăng ký/kiểm thử thực tế của người dùng; không tuyên bố đã kiểm chứng nhận diện người mục tiêu ngoài thực tế hoặc độ chính xác sinh trắc học.

### Auto và fallback

- `auto` thực tế đã chọn CUDA sau khi cả hai model qua smoke test.
- Giả lập `torch.cuda.is_available() == False`: `auto` ghi lý do `PyTorch CUDA unavailable`, nạp lại cả SCRFD/AdaFace trên CPU và trích xuất được mặt từ img1. Đây là **failure injection**, không phải máy thật không có GPU.
- `cpu` chạy model thật trong venv có torch CUDA, không gọi CUDA cho inference. Chưa kiểm tra trên một máy vật lý CPU-only hoặc venv chỉ chứa CPU wheels; công thức cài CPU là hướng dẫn cần chạy smoke lại ở máy đích.
- `cuda` không fallback âm thầm khi bị ép; lỗi được in và exit code khác 0. Startup chọn thống nhất thiết bị cho hai model; lỗi phát sinh sau startup hiện làm CLI dừng, có thể chạy lại CPU. Chưa xây cơ chế phục hồi phiên dài.

## 6. Lỗi phát sinh và xử lý

| Vấn đề | Bằng chứng / xử lý |
|---|---|
| Terminal sandbox chặn socket | `WinError 10013`; thực hiện các lệnh mạng được cấp quyền để đọc nguồn/tải dependency và model đã được yêu cầu. Không thay cấu hình mạng hệ thống. |
| Encoding stdout khi đọc README có emoji | `UnicodeEncodeError` cp1252; đặt UTF-8 cho stdout trong lệnh đọc tài liệu, không thay encoding hệ thống. |
| URL LICENSE InsightFace ở root/python-package | HTTP 404; đọc LICENSE thực có trong `detection/scrfd`, lưu Apache 2.0 và ghi rõ khác biệt với README chung. |
| Website WebFace | TLS hostname mismatch; giữ kiểm tra TLS, đánh dấu nguyên văn điều khoản chưa xác minh. |
| Cấm mọi CPU node của SCRFD CUDA | ONNX Runtime báo graph có node được gán CPU EP nhưng CPU fallback bị tắt. Đã kiểm tra profile model thật và đổi sang chính sách cho phép các toán tử shape int64 tường minh, không cho phép CPU compute ảnh âm thầm. |
| Cleanup profile trên Windows | Trong nhánh session lỗi, TemporaryDirectory xóa file profile đang mở gây `WinError 32`, che lỗi ORT ban đầu. Chuyển profile sang `results/profiles/<unique>/` để giữ bằng chứng và tránh cleanup che lỗi. Không xóa thư mục tạm cũ ngoài dự án. |

Không có lỗi nạp checkpoint strict hoặc lỗi inference cuối cùng trên CPU/CUDA. Không cài gói để chữa bằng thử ngẫu nhiên hoặc thay đổi driver/Toolkit. Các lần đọc mạng thất bại được ghi nhận, không coi là không tồn tại model.

## 7. Khả năng đã và chưa kiểm chứng

**Đã kiểm chứng:** môi trường độc lập; pip check sạch; đúng checkpoint IR-101 WebFace12M theo revision/config; preprocessing RGB và similarity alignment có kiểm tra số học; model thật chạy CPU/CUDA; có ảnh thì qua chuỗi detection/alignment/embedding; CLI đăng ký một ảnh và tính score cho ảnh khác; nhiều detection trong ảnh ghép; ảnh trống; auto ưu tiên CUDA và CPU fallback qua failure injection. Inference không tự tải tài sản.

**Chưa kiểm chứng:** ảnh thật người dùng và danh tính ground truth; ngưỡng nhận diện/FAR/FRR; nhiều người khác nhau trong đám đông; mặt nhỏ, nghiêng, mờ, chuyển động hoặc che khuất; tốc độ realtime/FPS/latency và đỉnh RAM/VRAM; SCRFD lớn hơn; AMD/Intel acceleration, TensorRT; máy vật lý CPU-only, Windows khác, Linux hoặc nhiều GPU; camera, video, tracking. Ảnh ghép hai bản cùng ảnh không đại diện độ khó của cảnh đông người.

Các file JSON chứa embedding là dữ liệu đầu ra thử nghiệm, không phải cơ sở dữ liệu danh tính hoàn chỉnh. Một ảnh đủ cho thao tác đăng ký; không suy ra một ảnh đủ để đạt chất lượng thực tế trong mọi điều kiện. Baseline hiện chưa có đăng ký nhiều ảnh hoặc cập nhật template.

## 8. Bước tiếp theo trong phạm vi AI

1. Dùng ảnh đăng ký và ảnh kiểm thử thực tế được cung cấp phù hợp, có ground truth; chạy CLI để xem detection/landmark/score.
2. Xây tập đánh giá riêng gồm cùng người/khác người và các điều kiện cần hỗ trợ, rồi mới chọn threshold. Không dùng ba ảnh mẫu này để hiệu chỉnh và tuyên bố chất lượng.
3. Chạy profile CPU-only và kiểm tra từng máy đích bằng smoke test; chỉ báo backend sau khi model thực thi thành công. Nếu cần chọn wheel mới cho GPU/OS khác, kiểm tra lại tương thích và `pip check` riêng.
4. Hoàn tất xác minh điều khoản pretrained WebFace/AdaFace và quyền sử dụng trước khi chuyển mục đích ngoài thử nghiệm nghiên cứu. Không coi MIT là quyền thương mại của toàn bộ tài sản.
