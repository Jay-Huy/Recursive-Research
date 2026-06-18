import os
import sys

# Add root directory to path to allow importing src modules (insert at 0 to avoid stdlib shadowing)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import SymbolicReasoningDataset

def validate_split(ds, name):
    hop_counts = {}
    for item in ds.data:
        hops = item['hops']
        hop_counts[hops] = hop_counts.get(hops, 0) + 1
    
    print(f"--- {name} Split ---")
    print(f"Total size: {len(ds)}")
    print("Hop distribution:")
    for h in sorted(hop_counts.keys()):
        print(f"  Hop {h}: {hop_counts[h]} samples")

def test_dataset():
    data_dir = 'data'
    try:
        train_ds = SymbolicReasoningDataset(data_dir, 'train')
        val_ds = SymbolicReasoningDataset(data_dir, 'val')
        test_ds = SymbolicReasoningDataset(data_dir, 'test')
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    validate_split(train_ds, "Train")
    validate_split(val_ds, "Val")
    validate_split(test_ds, "Test")
    
    print("\n--- Testing encode/decode on a sample ---")
    # Take a random 3-hop sample from Val
    sample_idx = next(i for i, item in enumerate(val_ds.data) if item['hops'] == 3)
    inputs, targets = val_ds[sample_idx]
    
    print(f"Raw Input Tensor: {inputs['input_ids']}")
    print(f"Decoded Input String: '{val_ds.decode_output(inputs['input_ids'])}'")
    print(f"Target Tensor: {targets['target']}")
    print(f"Decoded Target String: '{val_ds.decode_output(targets['target'].unsqueeze(0))}'")

if __name__ == '__main__':
    test_dataset()
