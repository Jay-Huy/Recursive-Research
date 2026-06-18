import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from src.core.base import ReasoningOutput
from src.core.metrics import AccuracyMetric, EntropyMetric, DistanceMetric, FixedPointMetric

def test_accuracy_metric():
    metric = AccuracyMetric()
    batch = 2
    max_loops = 5
    seq_len = 3
    vocab_size = 10
    
    logits = torch.randn(batch, max_loops, seq_len, vocab_size)
    predictions = torch.randint(0, vocab_size, (batch, max_loops, seq_len))
    hidden = torch.randn(batch, max_loops, seq_len, 32)
    hops = torch.tensor([1, 2])
    
    targets = predictions[:, -1, -1].clone() 
    
    output = ReasoningOutput(logits=logits, predictions=predictions, last_hidden_states_loops=hidden, hops=hops)
    metric.update(output, targets)
    
    res = metric.compute()
    
    assert isinstance(res, dict)
    assert 1 in res
    assert 2 in res
    assert res[1].shape == (max_loops,)
    assert res[2].shape == (max_loops,)
    # Loop cuối phải bằng 1 vì ta chủ đích gán targets = prediction ở loop cuối
    assert res[1][-1].item() == 1.0

def test_entropy_metric():
    metric = EntropyMetric()
    batch = 2
    max_loops = 5
    seq_len = 3
    vocab_size = 10
    
    logits = torch.randn(batch, max_loops, seq_len, vocab_size)
    predictions = torch.randint(0, vocab_size, (batch, max_loops, seq_len))
    hidden = torch.randn(batch, max_loops, seq_len, 32)
    hops = torch.tensor([1, 1])
    
    output = ReasoningOutput(logits=logits, predictions=predictions, last_hidden_states_loops=hidden, hops=hops)
    metric.update(output, torch.zeros(batch))
    
    res = metric.compute()
    
    assert 1 in res
    assert res[1].shape == (max_loops,)
    # Entropy >= 0
    assert torch.all(res[1] >= 0)

def test_distance_metric():
    metric = DistanceMetric()
    batch = 2
    max_loops = 5
    seq_len = 3
    hidden_size = 16
    
    logits = torch.randn(batch, max_loops, seq_len, 10)
    predictions = torch.randint(0, 10, (batch, max_loops, seq_len))
    hidden = torch.randn(batch, max_loops, seq_len, hidden_size)
    hops = torch.tensor([2, 3])
    targets = torch.tensor([0, 1])
    
    output = ReasoningOutput(logits=logits, predictions=predictions, last_hidden_states_loops=hidden, hops=hops)
    metric.update(output, targets)
    res = metric.compute()
    
    assert "local_dist" in res
    assert "global_dist" in res
    
    # check padding local_dist[hop_id][0] == -100.0
    assert res["local_dist"][2][0].item() == -100.0
    assert res["local_dist"][3][0].item() == -100.0

def test_fixed_point_metric():
    metric = FixedPointMetric(epsilon=0.1)
    batch = 1
    max_loops = 4
    seq_len = 2
    hidden_size = 4
    
    hidden = torch.zeros(batch, max_loops, seq_len, hidden_size)
    # L0 khác biệt
    hidden[0, 0] = torch.ones(seq_len, hidden_size) 
    # L1, L2, L3 giống nhau hoàn toàn
    hidden[0, 1] = torch.zeros(seq_len, hidden_size)
    hidden[0, 2] = torch.zeros(seq_len, hidden_size)
    hidden[0, 3] = torch.zeros(seq_len, hidden_size)
    
    hops = torch.tensor([5])
    output = ReasoningOutput(logits=torch.randn(1,4,2,10), predictions=torch.zeros(1,4,2), last_hidden_states_loops=hidden, hops=hops)
    
    metric.update(output, torch.tensor([0]))
    res = metric.compute()
    
    assert 5 in res
    # Vòng lặp thứ 2 (index 1) đã đạt fixed point do X_1 == X_0 (sai) -> X_1 == X_max và X_1 == X_0?
    # Ở t=1: diff_local = X_1 - X_0 = 0 - 1 = -1 (Norm > epsilon). -> Không pass.
    # Ở t=2: diff_local = X_2 - X_1 = 0 - 0 = 0 (Norm < epsilon).
    #        diff_g_t = X_2 - X_3 = 0.
    #        diff_g_prev = X_1 - X_3 = 0.
    # Nên computed loop sẽ chốt ở t=2 (Tức là vòng lặp số 3)
    assert res[5] == 3.0 

if __name__ == "__main__":
    import os
    import traceback
    
    log_path = os.path.join(os.path.dirname(__file__), "test_metrics_result.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- Metrics Core Test Log ---\n")
        try:
            test_accuracy_metric()
            f.write("[OK] test_accuracy_metric() passed.\n")
            
            test_entropy_metric()
            f.write("[OK] test_entropy_metric() passed.\n")
            
            test_distance_metric()
            f.write("[OK] test_distance_metric() passed.\n")
            
            test_fixed_point_metric()
            f.write("[OK] test_fixed_point_metric() passed.\n")
            
            f.write("\nSUCCESS: All metric tests passed perfectly!\n")
            print(f"All metric tests passed! Logs saved to {log_path}")
        except Exception as e:
            f.write("\nFAILED: Error occurred during testing:\n")
            f.write(traceback.format_exc())
            print(f"Tests failed! Please check {log_path} for details.")

