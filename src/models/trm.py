import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.trm_blocks import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    trunc_normal_init_
)
from src.core.base import ReasoningOutput
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("TRM")
class TRMStrippedModel(nn.Module):
    """
    Bản Stripped của TRM (TinyRecursiveModel):
    - Đã dời toàn bộ block sang `trm_blocks.py` để hoạt động độc lập, không phụ thuộc folder TinyRecursiveModels.
    - Loại bỏ hoàn toàn puzzle_identifiers, puzzle_emb (theo yêu cầu).
    - Loại bỏ hoàn toàn ACT (Halting).
    - Sử dụng max_train_loops như là H_cycles trong logic chạy (1 macroscopic loop có độ sâu 42 layers).
    - Thực thi backprop truncation: max_train_loops - 1 bước đầu tiên chạy dưới torch.no_grad(), chỉ bật grad ở bước cuối cùng.
    """
    def __init__(self, config: dict, vocab_size: int, max_train_loops: int):
        super().__init__()
        
        # Clean config for Pydantic
        config["vocab_size"] = vocab_size
        config["batch_size"] = 1 # Dummy
        config["seq_len"] = 2048 # Dummy max seq len cho positional encodings
        
        self.trm_config = TinyRecursiveReasoningModel_ACTV1Config(**config)
        self.max_train_loops = max_train_loops
        
        self.forward_dtype = getattr(torch, self.trm_config.forward_dtype)
        self.embed_scale = math.sqrt(self.trm_config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Embeddings & Heads
        self.embed_tokens = CastedEmbedding(self.trm_config.vocab_size, self.trm_config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.trm_config.hidden_size, self.trm_config.vocab_size, bias=False)

        # Positional Embeddings
        if self.trm_config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.trm_config.hidden_size // self.trm_config.num_heads,
                max_position_embeddings=self.trm_config.seq_len,
                base=self.trm_config.rope_theta
            )

        # L_level: Cốt lõi của TRM
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.trm_config) for _ in range(self.trm_config.L_layers)]
        )

        # Initial states (Registers) H và L
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.trm_config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.trm_config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

    def _input_embeddings(self, input_ids: torch.Tensor):
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    def forward(self, inputs: dict, num_loops: int = None):
        assert "hops" in inputs, "Missing 'hops' in inputs. This is critical for visualization!"
        input_ids = inputs["input_ids"]
        hops = inputs["hops"]
        
        bsz, seqlen = input_ids.shape
        input_embeddings = self._input_embeddings(input_ids)

        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            seq_info = dict(cos_sin=(cos[:seqlen, :], sin[:seqlen, :]))
        else:
            seq_info = dict(cos_sin=None)

        # Khởi tạo states H và L cho toàn batch (đã bỏ puzzle_emb_len)
        z_H = self.H_init.view(1, 1, -1).expand(bsz, seqlen, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(bsz, seqlen, -1).clone()

        last_hidden_states_loops = []

        # Lấy số loop từ arg hoặc từ config
        num_loops = num_loops if num_loops is not None else self.max_train_loops

        # Cơ chế Backprop Truncation của TRM (chỉ bật grad ở vòng lặp cuối cùng)
        for loop_idx in range(num_loops):
            
            # Ta chỉ bật gradient ở H_cycle cuối cùng của MỖI loop_idx.
            for _H_step in range(self.trm_config.H_cycles):
                is_last_H_step = (_H_step == self.trm_config.H_cycles - 1)
                
                with torch.set_grad_enabled(is_last_H_step):
                    for _L_step in range(self.trm_config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)
            # Trích xuất hidden state của vòng lặp hiện tại
            last_hidden_states_loops.append(z_H.clone())

        hidden_states_history = torch.stack(last_hidden_states_loops, dim=1)
        
        # Logits từ TẤT CẢ vòng lặp
        logits = self.lm_head(hidden_states_history) # [batch, max_loops, seq_len, vocab_size]
        predictions = logits.argmax(dim=-1)

        return ReasoningOutput(
            logits=logits,
            predictions=predictions,
            last_hidden_states_loops=hidden_states_history,
            hops=hops
        )
