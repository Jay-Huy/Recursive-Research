try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import os
from typing import Dict, Any

def plot_two_subplots(data_dict, title, ylabel, max_train_loops, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Interpolation: Hops 2-6
    for hop_id in range(2, 7):
        if hop_id in data_dict:
            y_vals = data_dict[hop_id]
            if hasattr(y_vals, "cpu"):
                y_vals = y_vals.cpu().numpy()
            x_vals = range(1, len(y_vals) + 1)
            ax1.plot(x_vals, y_vals, marker='o', markersize=3, label=f"Hop {hop_id}")
    
    ax1.axvline(x=max_train_loops, color='r', linestyle='--', label=f'max_train_loops ({max_train_loops})')
    ax1.set_title(f"{title} (Interpolation Hops 2-6)")
    ax1.set_xlabel("Loops")
    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    ax1.legend()
    
    # Extrapolation: Hops 7-10
    for hop_id in range(7, 11):
        if hop_id in data_dict:
            y_vals = data_dict[hop_id]
            if hasattr(y_vals, "cpu"):
                y_vals = y_vals.cpu().numpy()
            x_vals = range(1, len(y_vals) + 1)
            ax2.plot(x_vals, y_vals, marker='o', markersize=3, label=f"Hop {hop_id}")
            
    ax2.axvline(x=max_train_loops, color='r', linestyle='--', label=f'max_train_loops ({max_train_loops})')
    ax2.set_title(f"{title} (Extrapolation Hops 7-10)")
    ax2.set_xlabel("Loops")
    ax2.set_ylabel(ylabel)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_computed_loops(computed_loops_dict, save_path):
    hops = sorted(computed_loops_dict.keys())
    loops = [computed_loops_dict[h] for h in hops]
    
    plt.figure(figsize=(7, 5))
    plt.plot(hops, loops, marker='s', linewidth=2, label="Computed Loops")
    
    # Vẽ đường lý tưởng y = x
    if len(hops) > 0:
        min_hop, max_hop = min(hops), max(hops)
        plt.plot([min_hop, max_hop], [min_hop, max_hop], linestyle='--', color='gray', label="y = x")
    
    plt.title("Plot 5: Average Computed Loop vs Hops")
    plt.xlabel("Hops")
    plt.ylabel("Computed Loops")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_tier_2_plots(metrics_data: Dict[str, Any], max_train_loops: int, save_dir: str):
    if plt is None:
        print("Warning: matplotlib is not available. Skipping Tier 2 visualizations.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    if "accuracy" in metrics_data:
        plot_two_subplots(metrics_data["accuracy"], "Plot 1: Accuracy vs Loops", "Accuracy", max_train_loops, 
                          os.path.join(save_dir, "plot1_accuracy.png"))
                          
    if "entropy" in metrics_data:
        plot_two_subplots(metrics_data["entropy"], "Plot 2: Entropy vs Loops", "Entropy", max_train_loops, 
                          os.path.join(save_dir, "plot2_entropy.png"))
                          
    if "distance" in metrics_data:
        if "local_dist" in metrics_data["distance"]:
            plot_two_subplots(metrics_data["distance"]["local_dist"], "Plot 3: Local Distance vs Loops", "Frobenius Norm", max_train_loops, 
                              os.path.join(save_dir, "plot3_local_distance.png"))
        if "global_dist" in metrics_data["distance"]:
            plot_two_subplots(metrics_data["distance"]["global_dist"], "Plot 4: Global Distance vs Loops", "Frobenius Norm", max_train_loops, 
                              os.path.join(save_dir, "plot4_global_distance.png"))
                              
    if "fixed_point" in metrics_data:
        plot_computed_loops(metrics_data["fixed_point"], os.path.join(save_dir, "plot5_computed_loops.png"))
