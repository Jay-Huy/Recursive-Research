import torch
import torch.nn.functional as F
from typing import Dict, Any
from collections import defaultdict
from src.core.base import BaseMetric, ReasoningOutput

class AccuracyMetric(BaseMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_sums = defaultdict(lambda: None)
        self.counts = defaultdict(int)

    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        # Mảng predictions có shape [batch, max_loops, seq_len]
        # Trong tác vụ SymbolicReasoning, token mục tiêu (target) được dự đoán ở cuối chuỗi
        last_token_preds = output.predictions[:, :, -1]
        
        # Mở rộng targets ra shape [batch, max_loops] để so sánh cho toàn bộ vòng lặp
        targets_expanded = targets.unsqueeze(1).expand_as(last_token_preds)
        
        corrects = (last_token_preds == targets_expanded).float()
        
        hops = output.hops
        
        for i in range(len(hops)):
            hop = hops[i].item()
            if self.correct_sums[hop] is None:
                self.correct_sums[hop] = torch.zeros_like(corrects[i])
            self.correct_sums[hop] += corrects[i]
            self.counts[hop] += 1

    def compute(self) -> Dict[int, torch.Tensor]:
        result = {}
        for hop, correct_sum in self.correct_sums.items():
            result[hop] = correct_sum / self.counts[hop]
        return result


class EntropyMetric(BaseMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.entropy_sums = defaultdict(lambda: None)
        self.counts = defaultdict(int)

    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        # logits: [batch, max_loops, seq_len, vocab_size]
        probs = F.softmax(output.logits, dim=-1)
        
        # Tính entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # [batch, max_loops, seq_len]
        
        # Trung bình entropy trên toàn bộ sequence
        avg_entropy = entropy.mean(dim=-1) # [batch, max_loops]
        
        hops = output.hops
        for i in range(len(hops)):
            hop = hops[i].item()
            if self.entropy_sums[hop] is None:
                self.entropy_sums[hop] = torch.zeros_like(avg_entropy[i])
            self.entropy_sums[hop] += avg_entropy[i]
            self.counts[hop] += 1

    def compute(self) -> Dict[int, torch.Tensor]:
        result = {}
        for hop, ent_sum in self.entropy_sums.items():
            result[hop] = ent_sum / self.counts[hop]
        return result


class DistanceMetric(BaseMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.local_dist_sums = defaultdict(lambda: None)
        self.global_dist_sums = defaultdict(lambda: None)
        self.counts = defaultdict(int)

    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        # X: [batch, max_loops, seq_len, hidden_size]
        X = output.last_hidden_states_loops
        batch_size, max_loops, seq_len, hidden_size = X.shape
        
        local_dist = torch.zeros((batch_size, max_loops), device=X.device)
        for t in range(1, max_loops):
            diff = X[:, t] - X[:, t-1]
            local_dist[:, t] = torch.norm(diff.reshape(batch_size, -1), p='fro', dim=1)
        # Padding vị trí t=0 (do không có t-1)
        local_dist[:, 0] = -100.0
        
        global_dist = torch.zeros((batch_size, max_loops), device=X.device)
        X_Lmax = X[:, -1]
        for t in range(max_loops):
            diff = X_Lmax - X[:, t]
            global_dist[:, t] = torch.norm(diff.reshape(batch_size, -1), p='fro', dim=1)
            
        hops = output.hops
        for i in range(len(hops)):
            hop = hops[i].item()
            if self.local_dist_sums[hop] is None:
                self.local_dist_sums[hop] = torch.zeros_like(local_dist[i])
                self.global_dist_sums[hop] = torch.zeros_like(global_dist[i])
            self.local_dist_sums[hop] += local_dist[i]
            self.global_dist_sums[hop] += global_dist[i]
            self.counts[hop] += 1

    def compute(self) -> Dict[str, Dict[int, torch.Tensor]]:
        res_local = {}
        res_global = {}
        for hop in self.counts:
            res_local[hop] = self.local_dist_sums[hop] / self.counts[hop]
            res_global[hop] = self.global_dist_sums[hop] / self.counts[hop]
            # Khẳng định cứng lại giá trị padding trong trường hợp phép chia ảnh hưởng
            res_local[hop][0] = -100.0
        return {"local_dist": res_local, "global_dist": res_global}


class FixedPointMetric(BaseMetric):
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.computed_loops_sum = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        X = output.last_hidden_states_loops
        batch_size, max_loops = X.shape[0], X.shape[1]
        hops = output.hops
        
        X_Lmax = X[:, -1]
        
        for i in range(batch_size):
            hop = hops[i].item()
            
            # Khởi tạo mặc định là chạm max_loops
            computed_loop = max_loops
            
            # Bắt đầu quét từ vòng lặp index 1 (tức là t=2 ngoài thực tế)
            for t in range(1, max_loops):
                diff_local = X[i, t] - X[i, t-1]
                diff_global_t = X[i, t] - X_Lmax[i]
                diff_global_t_prev = X[i, t-1] - X_Lmax[i]
                
                norm_local = torch.norm(diff_local)
                norm_g_t = torch.norm(diff_global_t)
                norm_g_prev = torch.norm(diff_global_t_prev)
                
                if norm_local < self.epsilon and norm_g_t < self.epsilon and norm_g_prev < self.epsilon:
                    # Index `t` ứng với vòng lặp thứ `t+1`
                    computed_loop = t + 1
                    break
            
            self.computed_loops_sum[hop] += computed_loop
            self.counts[hop] += 1

    def compute(self) -> Dict[int, float]:
        result = {}
        for hop in self.counts:
            result[hop] = self.computed_loops_sum[hop] / self.counts[hop]
        return result

class MetricCollection:
    def __init__(self, metrics_dict: Dict[str, BaseMetric]):
        self.metrics = metrics_dict

    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        for metric in self.metrics.values():
            metric.update(output, targets)

    def compute(self) -> Dict[str, Any]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
