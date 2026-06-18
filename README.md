# Recursive Reasoning Models

Chào mừng Reviewer đến với kho dự án **Recursive Reasoning Models**! Hệ thống đã được thiết lập sẵn sàng cho việc training và inference các mô hình đệ quy qua 7 cấu hình chuẩn.

## 1. Môi trường và Dữ liệu
Đảm bảo bạn đã sinh dữ liệu (Phase 2) với `data/metadata.json` đã có sẵn. 

**Đăng nhập Weights & Biases (W&B):**
Hệ thống sử dụng W&B làm công cụ logging (Tier 1) bắt buộc. Bạn cần đăng nhập trước khi chạy:
- **Trên Windows (Local):**
  Mở Terminal/PowerShell và gõ:
  ```bash
  wandb login
  ```
  Sau đó dán API key của bạn vào.
- **Trên Linux (Kaggle/Colab):**
  Thêm đoạn code sau vào một cell hoặc chạy trên terminal trước khi gọi script:
  ```bash
  !pip install wandb
  !wandb login <YOUR_API_KEY>
  ```

## 2. Training
Để tiến hành huấn luyện, sử dụng script `train.py`. Tham số quan trọng nhất là cấu hình yaml.

**Câu lệnh chuẩn:**
```bash
python scripts/train.py --config_path config/base_loop.yaml
```

**Testing nhanh:**
Nếu bạn chỉ muốn test luồng code mà không muốn chờ chạy hết epoch, có thể dùng các cờ sau:
- `--test_batches X`: Chạy nhanh mô hình trên X batches thay vì toàn bộ dataset. Rất hữu ích để check lỗi hoặc dry-run.
- `--run_name "Tên_Run"`: Đặt tên tuỳ chỉnh cho run trên WandB để dễ phân biệt (ví dụ: `--run_name "BaseLoop_Test1"`). Nếu không có cờ này, tên run sẽ được tự động tạo.
- `--batch_size X`: Ghi đè cấu hình `batch_size` trực tiếp từ Terminal mà không cần sửa file YAML.
- `--num_epochs X`: Ghi đè số lượng epochs trực tiếp từ Terminal (VD: `--num_epochs 2`).

Ví dụ test nhanh 5 batches với số epoch nhỏ (ví dụ 2) và gắn tên run:
```bash
python scripts/train.py --config_path config/test_arch/base_loop.yaml --test_batches 5 --num_epochs 2 --run_name "DryRun_BaseLoop"
```

Các file config hiện có:
- `config/base_loop.yaml` -> Lưu checkpoint tên `BaseLoop_final.pt`
- `config/base_loop_enforced.yaml` -> Lưu checkpoint tên `BaseLoopEnforced_final.pt`
- `config/prelude_coda.yaml` -> Lưu checkpoint tên `PreludeCoda_final.pt`
- `config/prelude_coda_enforced.yaml` -> Lưu checkpoint tên `PreludeCodaEnforced_final.pt`
- `config/prelude_coda_mismatch.yaml` -> Lưu checkpoint tên `PreludeCodaMismatch_final.pt`
- `config/prelude_coda_mismatch_enforced.yaml` -> Lưu checkpoint tên `PreludeCodaMismatchEnforced_final.pt`
- `config/trm.yaml` -> Lưu checkpoint tên `TRM_final.pt`

*(Lưu ý: Nếu dùng `--test_batches`, tên checkpoint sẽ tự động cộng thêm suffix, ví dụ: `BaseLoop_5batches.pt`)*

## 3. Inference & Visualization (Tier 2 Plots)
Để chạy test, sinh biểu đồ Extrapolation / Interpolation:

**Câu lệnh chuẩn:**
```bash
python scripts/inference.py --config_path config/base_loop.yaml --checkpoint_path outputs/checkpoints/BaseLoop_final.pt
```

**Các tham số tùy chỉnh:**
- `--test_batches <int>`: Khai báo số lượng batch validation muốn chạy test (Mặc định chạy hết toàn bộ tập Val).
- `--num_loops <int>`: Khai báo số vòng lặp khi nội suy / ngoại suy. (Mặc định lấy từ `max_train_loops` của config).
- `--print_samples <int>`: Số lượng samples in ra console trong lúc inference (Mặc định: 5). Phải nhỏ hơn hoặc bằng `batch_size`.

Ví dụ:
```bash
python scripts/inference.py --config_path config/base_loop.yaml --checkpoint_path outputs/checkpoints/BaseLoop_final.pt --test_batches 5 --num_loops 10
```

Hình ảnh sẽ tự động được sinh và lưu tại thư mục tương ứng với Model, ví dụ: `outputs/BaseLoop_5batches_10loops/` hoặc `outputs/BaseLoop/` nếu chạy toàn bộ.
