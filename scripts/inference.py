import os
import sys
import json
import yaml
import argparse
import torch

# Đảm bảo import từ src root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
import src.models.base_loop
import src.models.prelude_coda
import src.models.trm
import src.data.dataset

from src.core.metrics import AccuracyMetric, EntropyMetric, DistanceMetric, FixedPointMetric
from src.utils.plotter import generate_tier_2_plots

def main():
    parser = argparse.ArgumentParser(description="Inference and Tier 2 Visualization")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--num_loops", type=int, default=None, help="Number of loops to run (Extrapolation test). If not set, uses max_train_loops.")
    parser.add_argument("--test_batches", type=int, default=None, help="Number of batches to run for quick testing.")
    parser.add_argument("--print_samples", type=int, default=5, help="Number of samples to print to console.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for PyTorch initialization")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data_config", {})
    data_dir = data_config.get("data_dir", "data/multi_hop")
    
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing {metadata_path}. Please run data generator first.")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 1. Khởi tạo Dataset & DataLoader (Dùng tập Validation)
    dataset_type = data_config.get("dataset_type", "SymbolicReasoningDataset")
    batch_size = data_config.get("batch_size", 128)
    
    assert args.print_samples <= batch_size, f"print_samples ({args.print_samples}) cannot be greater than batch_size ({batch_size})"
    
    val_dataset = DATASET_REGISTRY.build(dataset_type, data_dir=data_dir, split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Khởi tạo Model
    model_config = config.get("model_config", {})
    model_type = model_config.pop("model_type")
    
    if model_type == "TRM":
        model_config["vocab_size"] = metadata["vocab_size"]
        max_train_loops = model_config.get("max_train_loops", 6)
        model = MODEL_REGISTRY.build(model_type, config=model_config, vocab_size=metadata["vocab_size"], max_train_loops=max_train_loops)
    else:
        model_config["vocab_size"] = metadata["vocab_size"]
        if model_type in ["BaseLoop", "AdapterLoop"]:
            from src.models.configs import BaseLoopConfig
            cfg_obj = BaseLoopConfig(**model_config)
        elif model_type == "PreludeCoda":
            from src.models.configs import PreludeCodaConfig
            cfg_obj = PreludeCodaConfig(**model_config)
        model = MODEL_REGISTRY.build(model_type, config=cfg_obj)

    # Load checkpoint
    # if os.path.exists(args.checkpoint_path):
    #     model.load_state_dict(torch.load(args.checkpoint_path))
    #     print(f"Successfully loaded checkpoint: {args.checkpoint_path}")
    # else:
    #     raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    model.eval()
    
    # Số loop thực thi
    exec_num_loops = args.num_loops if args.num_loops is not None else config["model_config"].get("max_train_loops", 6)

    # 3. Khởi tạo Metrics
    acc_metric = AccuracyMetric()
    ent_metric = EntropyMetric()
    dist_metric = DistanceMetric()
    fp_metric = FixedPointMetric(epsilon=1e-4)

    # 4. Inference
    print("=" * 50)
    print(f"Running inference with {exec_num_loops} loops...")
    print("=" * 50)
    
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x

    samples_printed = 0
    
    pbar = tqdm(val_loader, desc="Inference")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Truyền num_loops vào forward của model
            outputs = model(inputs, num_loops=exec_num_loops)
            
            # Cập nhật metrics
            acc_metric.update(outputs, targets["target"])
            ent_metric.update(outputs, targets["target"])
            dist_metric.update(outputs, targets["target"])
            fp_metric.update(outputs, targets["target"])
            
            # In ra test samples (console)
            if samples_printed < args.print_samples:
                input_ids = inputs["input_ids"]
                predictions = outputs.predictions
                logits = outputs.logits
                for i in range(input_ids.shape[0]):
                    if samples_printed >= args.print_samples:
                        break
                    print(f"\n--- Sample {samples_printed + 1} ---")
                    # Lấy logits vòng lặp cuối cùng ([-1])
                    print(f"Logits shape (last loop): {logits[i, -1].shape}") 
                    print(f"Input: {val_dataset.decode_output(input_ids[i])}")
                    # Đối với prediction (mảng chứa dự đoán ở mọi vòng lặp), in ra dự đoán của vòng lặp cuối ([-1]) và token cuối ([-1])
                    pred_token = predictions[i, -1, -1].item()
                    target_token = targets["target"][i].item()
                    
                    # Convert id to string manually for target/prediction since decode_output processes whole tensor
                    def decode_token(tok):
                        if tok < val_dataset.rel_offset: return f"e{tok}"
                        return f"r{tok - val_dataset.rel_offset}"
                        
                    print(f"Target: {decode_token(target_token)} | Prediction: {decode_token(pred_token)}")
                    samples_printed += 1
                    
            if args.test_batches is not None and batch_idx + 1 >= args.test_batches:
                break

    # 5. Lưu Metrics và Plots
    metrics_data = {
        "accuracy": acc_metric.compute(),
        "entropy": ent_metric.compute(),
        "distance": dist_metric.compute(),
        "fixed_point": fp_metric.compute()
    }
    
    folder_name = model_type
    if args.test_batches is not None:
        folder_name += f"_{args.test_batches}batches"
    if args.num_loops is not None:
        folder_name += f"_{args.num_loops}loops"
        
    save_plot_dir = os.path.join("outputs", folder_name)
    os.makedirs(save_plot_dir, exist_ok=True)
    generate_tier_2_plots(metrics_data, config["model_config"].get("max_train_loops", 6), save_dir=save_plot_dir)
    print("\n" + "=" * 50)
    print(f"Inference completed. Tier 2 visualizations saved to {save_plot_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
