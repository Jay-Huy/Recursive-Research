# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ReasoningOutput:
    logits: torch.Tensor  # shape: [batch, max_loops, sequence_length, vocab_size]
    predictions: torch.Tensor  # shape: [batch, max_loops, sequence_length]
    last_hidden_states_loops: torch.Tensor  # shape: [batch, max_loops, sequence_length, hidden_size]
    hops: torch.Tensor  # shape: [batch]


class ReasoningDataset(ABC, Dataset):
    @abstractmethod
    def parse_raw_data(self, raw_path: str):
        pass

    @abstractmethod
    def encode_input(self, data: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def decode_output(self, tensor: torch.Tensor) -> Any:
        pass

class ReasoningModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> ReasoningOutput:
        pass

    @abstractmethod
    def compute_loss(self, outputs: ReasoningOutput, targets: torch.Tensor) -> torch.Tensor:
        pass

class BaseMetric(ABC):
    @abstractmethod
    def update(self, output: ReasoningOutput, targets: torch.Tensor):
        pass

    @abstractmethod
    def compute(self) -> Dict[int, torch.Tensor]:
        pass

    @abstractmethod
    def reset(self):
        pass
