import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from src.core.base import ReasoningDataset
from src.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("SymbolicReasoningDataset")
class SymbolicReasoningDataset(ReasoningDataset):
    def __init__(self, data_dir: str, split: str = 'train', max_seq_len: int = 15):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            
        self.vocab_size = self.metadata['vocab_size']
        self.pad_token = self.metadata['pad_token']
        self.rel_offset = self.metadata['rel_offset']
        
        self.data = self.parse_raw_data(os.path.join(data_dir, f"{split}.jsonl"))

    def parse_raw_data(self, raw_path: str):
        data = []
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def encode_input(self, seq: list) -> torch.Tensor:
        # padding
        padded = seq + [self.pad_token] * (self.max_seq_len - len(seq))
        return torch.tensor(padded, dtype=torch.long)

    def decode_output(self, tensor: torch.Tensor) -> str:
        # convert tensor back to human readable string
        ids = tensor.tolist()
        tokens = []
        for i in ids:
            if i == self.pad_token:
                continue
            elif i < self.rel_offset:
                tokens.append(f"e{i}")
            else:
                tokens.append(f"r{i - self.rel_offset}")
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = self.encode_input(item['input_ids'])
        target_tensor = torch.tensor(item['target'], dtype=torch.long)
        
        # We wrap input_ids inside a dict for the engine's generic handling
        inputs = {
            "input_ids": input_tensor,
            "hops": item['hops']
        }
        
        # Model target can be wrapped in a dict or plain tensor, we use dict here for generic metrics handling
        targets = {
            "target": target_tensor
        }
        
        return inputs, targets

# Utility function to test loader
def get_dataloader(data_dir, split, batch_size=32):
    ds = SymbolicReasoningDataset(data_dir, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'))
