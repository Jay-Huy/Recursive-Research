import wandb
from typing import Dict, Any
import torch

class WandbLogger:
    def __init__(self, project_name: str, name: str = None, config: Dict[str, Any] = None):
        self.run = wandb.init(project=project_name, name=name, config=config)
        
    def log_metrics(self, metrics_data: Dict[str, Any], split: str = "train", step: int = None, loss: float = None):
        """
        metrics_data: Bọc kết quả từ các metric classes.
        Ví dụ: {"accuracy": {1: Tensor([L1, ..., Lmax]), 2: ...}, "entropy": ...}
        split: "train" hoặc "val"
        step: Epoch hoặc global step hiện tại
        loss: Giá trị loss trung bình của epoch
        """
        log_dict = {}
        if loss is not None:
            log_dict[f"{split}/avg_loss"] = loss
        
        acc_data = metrics_data.get("accuracy", {})
        ent_data = metrics_data.get("entropy", {})
        
        global_accs = []
        global_ents = []
        
        # Bóc tách L_max của từng Hop cho Accuracy
        for hop_id, acc_tensor in acc_data.items():
            lmax_acc = acc_tensor[-1].item()
            global_accs.append(lmax_acc)
            log_dict[f"{split}/acc_hop_{hop_id}"] = lmax_acc
            
        # Bóc tách L_max của từng Hop cho Entropy
        for hop_id, ent_tensor in ent_data.items():
            lmax_ent = ent_tensor[-1].item()
            global_ents.append(lmax_ent)
            log_dict[f"{split}/entropy_hop_{hop_id}"] = lmax_ent
            
        # Tính Average xuyên suốt các Hops tại L_max (Nhóm Global)
        if len(global_accs) > 0:
            log_dict[f"{split}/avg_acc"] = sum(global_accs) / len(global_accs)
        if len(global_ents) > 0:
            log_dict[f"{split}/avg_entropy"] = sum(global_ents) / len(global_ents)
            
        # Gửi toàn bộ lên WandB
        wandb.log(log_dict, step=step)
        
    def finish(self):
        wandb.finish()
