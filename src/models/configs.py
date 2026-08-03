import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseLoopConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    scale_mlp: int
    n_layers: int
    max_train_loops: int
    arch_case: int
    tie_embeddings: bool

    def __post_init__(self):
        self.effective_expected_depth = self.n_layers * self.max_train_loops

    @classmethod
    def from_yaml(cls, path: str, vocab_size: int):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(vocab_size=vocab_size, **data)

@dataclass
class PreludeCodaConfig:
    vocab_size: int
    d_model: int
    d_loop: Optional[int]
    n_heads: int
    scale_mlp: int
    n_prelude: int
    n_loop: int
    n_coda: int
    max_train_loops: int
    enforced: bool
    tie_embeddings: bool

    def __post_init__(self):
        self.effective_expected_depth = self.n_prelude + self.n_coda + self.n_loop * self.max_train_loops

    @classmethod
    def from_yaml(cls, path: str, vocab_size: int):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(vocab_size=vocab_size, **data)
