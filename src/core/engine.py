import torch
import yaml
from typing import Dict, Optional
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import wandb
from src.utils.logger import WandbLogger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class TrainingOrchestrator:
    def __init__(self, config: dict, model: torch.nn.Module, train_loader, val_loader, optimizer, metrics, test_batches: Optional[int] = None, run_name: str = None):
        self.config = config
        self.test_batches = test_batches
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.metrics = metrics
        self.run_name = run_name
        
        # Parse configs
        self.device = torch.device(self.config.get('training', {}).get('device', 'cpu'))
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Log Information
        total_params = sum(p.numel() for p in self.model.parameters())
        train_samples = len(self.train_loader.dataset) if self.train_loader else 0
        val_samples = len(self.val_loader.dataset) if self.val_loader else 0
        print("=" * 50)
        print("Model and Data Info:")
        print(f"  Model Parameters: {total_params:,}")
        print(f"  Train Samples: {train_samples:,}")
        print(f"  Validation Samples: {val_samples:,}")
        print("=" * 50)
        
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 10)
        self.eval_epoch = self.config.get('training', {}).get('eval_epoch', 1)
        self.use_clip = self.config.get('training', {}).get('use_clip', False)
        self.clip_val = self.config.get('training', {}).get('clip_val', 1.0)
        self.use_warmup = self.config.get('training', {}).get('use_warmup', False)
        
        self.early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 0)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Setup Scheduler
        if self.use_warmup:
            warmup_epochs = self.config.get('training', {}).get('warmup_epochs', 5)
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=self.num_epochs - warmup_epochs)
            self.scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            self.scheduler = None

        project_name = self.config.get('training', {}).get('wandb_project', 'recursive-reasoning')
        
        # Determine the name for WandB run
        actual_run_name = self.run_name or self.config.get('training', {}).get('run_name') or self.config.get('model_config', {}).get('model_type', 'run')
        if self.test_batches is not None:
            actual_run_name += f"_{self.test_batches}batches"
            
        self.logger = WandbLogger(project_name=project_name, name=actual_run_name, config=config)

    def _extract_global_lmax(self, metrics_data):
        acc_data = metrics_data.get("accuracy", {})
        lmax_accs = [v[-1].item() for v in acc_data.values()]
        avg_acc = sum(lmax_accs)/len(lmax_accs) if lmax_accs else 0.0
        
        ent_data = metrics_data.get("entropy", {})
        lmax_ents = [v[-1].item() for v in ent_data.values()]
        avg_ent = sum(lmax_ents)/len(lmax_ents) if lmax_ents else 0.0
        
        return avg_acc, avg_ent

    def process_data(self, is_train: bool = True):
        if self.metrics is not None:
            self.metrics.reset()
            
        if is_train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        
        desc = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # move dicts to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets_tensor = targets["target"].to(self.device) if isinstance(targets, dict) else targets.to(self.device)
            
            if is_train:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Tính toán loss (Mô hình chỉ dự đoán token cuối cùng ở vòng lặp cuối cùng)
                loss = self.criterion(outputs.logits[:, -1, -1, :], targets_tensor)
                
                loss.backward()
                if self.clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                    
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.logits[:, -1, -1, :], targets_tensor)
                    
            if self.metrics is not None:
                self.metrics.update(outputs, targets_tensor)
            
            total_loss += loss.item()
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if self.test_batches is not None and batch_idx + 1 >= self.test_batches:
                break
            
        avg_loss = total_loss / (batch_idx + 1)
        
        metrics_res = {}
        if self.metrics is not None:
            metrics_res = self.metrics.compute()
            
        return avg_loss, metrics_res

    def train_fn(self):
        import time
        print("Starting training...")
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_metrics = self.process_data(is_train=True)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            train_acc, train_ent = self._extract_global_lmax(train_metrics)
            print(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train Ent: {train_ent:.4f}")
            
            if self.logger is not None:
                self.logger.log_metrics(train_metrics, split="train", step=epoch, loss=train_loss)
            
            if epoch % self.eval_epoch == 0:
                val_loss, val_metrics = self.process_data(is_train=False)
                val_acc, val_ent = self._extract_global_lmax(val_metrics)
                print(f"Epoch {epoch}/{self.num_epochs} - Val Loss:   {val_loss:.4f} - Val Acc:   {val_acc:.4f} - Val Ent:   {val_ent:.4f}")
                
                if self.logger is not None:
                    self.logger.log_metrics(val_metrics, split="val", step=epoch, loss=val_loss)
                
                if self.early_stopping_patience > 0:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        # TODO: Save checkpoint
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print("Early stopping triggered!")
                            break
                            
        end_time = time.time()
        elapsed = end_time - start_time
        mins = int(elapsed // 60)
        secs = elapsed % 60
        print(f"Training finished in {mins}m {secs:.2f}s.")
        
        if hasattr(self.metrics, 'visualize'):
            import os
            os.makedirs("outputs/plots/", exist_ok=True)
            print("Generating visualizations...")
            self.metrics.visualize(save_dir="outputs/plots/")
            
        if self.logger is not None:
            self.logger.finish()
