import os
import json
import yaml
import argparse
import sys
import torch
from torch.optim import AdamW

# Đảm bảo import từ src root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Đảm bảo các registry được register trước khi build
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
import src.models.base_loop
import src.models.prelude_coda
import src.models.trm
import src.data.dataset

from src.core.metrics import AccuracyMetric, EntropyMetric, MetricCollection
from src.core.engine import TrainingOrchestrator
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Train Recursive Reasoning Models")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--test_batches", type=int, default=None, help="Number of batches to run for testing.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for WandB run")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs from config")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.num_epochs is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["num_epochs"] = args.num_epochs

    data_config = config.get("data_config", {})
    data_dir = data_config.get("data_dir", "data/multi_hop")
    batch_size = args.batch_size if args.batch_size is not None else data_config.get("batch_size", 128)
    
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing {metadata_path}. Please run data generator first.")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 1. Dataset & DataLoader (train.jsonl and val.jsonl)
    dataset_type = data_config.get("dataset_type", "SymbolicReasoningDataset")
    train_dataset = DATASET_REGISTRY.build(dataset_type, data_dir=data_dir, split="train")
    val_dataset = DATASET_REGISTRY.build(dataset_type, data_dir=data_dir, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model initialization
    model_config = config.get("model_config", {})
    model_type = model_config.pop("model_type")
    
    # Bổ sung thông tin từ data cho model
    if model_type == "TRM":
        # Đối với TRM
        model_config["vocab_size"] = metadata["vocab_size"]
        max_train_loops = model_config.get("max_train_loops", 6)
        model = MODEL_REGISTRY.build(model_type, config=model_config, vocab_size=metadata["vocab_size"], max_train_loops=max_train_loops)
    else:
        # Đối với BaseLoop, PreludeCoda (Dùng Pydantic config class)
        model_config["vocab_size"] = metadata["vocab_size"]
        if model_type == "BaseLoop":
            from src.models.configs import BaseLoopConfig
            cfg_obj = BaseLoopConfig(**model_config)
        elif model_type == "PreludeCoda":
            from src.models.configs import PreludeCodaConfig
            cfg_obj = PreludeCodaConfig(**model_config)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        model = MODEL_REGISTRY.build(model_type, config=cfg_obj)

    # 3. Optimizer
    training_config = config.get("training", {})
    lr = float(training_config.get("lr", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr)

    # 3. Khởi tạo Metrics cho quá trình Train
    train_metrics = MetricCollection({
        "accuracy": AccuracyMetric(),
        "entropy": EntropyMetric()
    })

    # Khởi tạo orchestrator
    orchestrator = TrainingOrchestrator(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        metrics=train_metrics,
        test_batches=args.test_batches,
        run_name=args.run_name
    )

    # 4. Train
    orchestrator.train_fn()

    # Lưu model checkpoint sau khi train xong
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    if args.test_batches is not None:
        save_path = f"outputs/checkpoints/{model_type}_{args.test_batches}batches.pt"
    else:
        save_path = f"outputs/checkpoints/{model_type}_final.pt"
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
