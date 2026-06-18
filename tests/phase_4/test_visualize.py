import torch
import os
import sys
import shutil

# Thêm project root vào sys.path để Python không nhầm thư mục 'src' với built-in module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.plotter import generate_tier_2_plots

def test_visualization():
    # 1. Tạo Dummy Data giả lập kết quả từ Phase 3
    max_train_loops = 6
    total_loops = 10
    
    acc_data = {}
    ent_data = {}
    local_dist_data = {}
    global_dist_data = {}
    fixed_point_data = {}
    
    for hop_id in range(2, 11):
        # Sinh tensor [total_loops] chứa các giá trị random
        acc_data[hop_id] = torch.rand(total_loops)
        ent_data[hop_id] = torch.rand(total_loops) * 2.0
        
        local_dist_data[hop_id] = torch.rand(total_loops)
        global_dist_data[hop_id] = torch.rand(total_loops)
        
        # Computed loops tịnh tiến theo độ khó
        fixed_point_data[hop_id] = float(min(hop_id, 10))

    metrics_data = {
        "accuracy": acc_data,
        "entropy": ent_data,
        "distance": {
            "local_dist": local_dist_data,
            "global_dist": global_dist_data
        },
        "fixed_point": fixed_point_data
    }
    
    # 2. Setup thư mục lưu ảnh
    save_dir = os.path.join(os.path.dirname(__file__), "test_plots_output")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Thử save Tensor file .pt (như logic của inference.py)
    tensor_path = os.path.join(save_dir, "dummy_metrics.pt")
    torch.save(metrics_data, tensor_path)
    
    # 4. Generate plots
    generate_tier_2_plots(metrics_data, max_train_loops, save_dir)
    
    # 5. Assertions
    assert os.path.exists(tensor_path), "Lỗi: Tensor .pt file không được lưu."
    assert os.path.exists(os.path.join(save_dir, "plot1_accuracy.png")), "Lỗi: Plot 1 chưa sinh ra."
    assert os.path.exists(os.path.join(save_dir, "plot2_entropy.png")), "Lỗi: Plot 2 chưa sinh ra."
    assert os.path.exists(os.path.join(save_dir, "plot3_local_distance.png")), "Lỗi: Plot 3 chưa sinh ra."
    assert os.path.exists(os.path.join(save_dir, "plot4_global_distance.png")), "Lỗi: Plot 4 chưa sinh ra."
    assert os.path.exists(os.path.join(save_dir, "plot5_computed_loops.png")), "Lỗi: Plot 5 chưa sinh ra."
    
    print("Test Visualization: Đã sinh thành công 5 PNGs và 1 file .pt!")

if __name__ == "__main__":
    import traceback
    log_path = os.path.join(os.path.dirname(__file__), "test_visualize_result.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- Visualization Test Log ---\n")
        try:
            test_visualization()
            f.write("[OK] Visualization test passed. 5 PNGs and .pt file created successfully.\n")
            print(f"All visualize tests passed! Logs saved to {log_path}")
        except Exception as e:
            f.write("[FAILED] Visualization test error:\n")
            f.write(traceback.format_exc())
            print(f"Tests failed! Check {log_path} for details.")
