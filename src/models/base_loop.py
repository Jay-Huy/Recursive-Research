import math
import torch
import torch.nn as nn
from src.models.blocks import RMSNorm, TransformerPreNormBlock, SandwichBlock, precompute_freqs_cis
from src.models.configs import BaseLoopConfig
from src.core.base import ReasoningOutput
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("BaseLoop")
class SimpleLoopModel(nn.Module):
    """
    Nhóm 1: Mô hình lặp cơ bản (BaseLoopModel) 
    (Tương ứng Condition 1 & 2 trong Architecture Matrix)
    Không có Prelude/Coda layers. Loop được bao bọc trực tiếp sau Embedding và trước LM Head.
    """
    def __init__(self, config: BaseLoopConfig):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        BlockClass = SandwichBlock # if config.enforced else TransformerPreNormBlock
        if config.enforced:
            self.adapter = nn.Linear(config.d_model * 2, config.d_model, bias=False)
        else:
            self.adapter = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.loop_blocks = nn.ModuleList([
            BlockClass(config.d_model, config.n_heads, config.d_model * config.scale_mlp)
            for _ in range(config.n_layers)
        ])
            
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Reference Huginn-0125: Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
            
        for name, module in self.named_modules():
            self._init_weights(module, name)

    # def _init_weights(self, module, name=""):
    #     std = math.sqrt(2 / (5 * self.config.d_model))
    #     out_proj_std = std / math.sqrt(2 * self.config.effective_expected_depth)
        
    #     if isinstance(module, RMSNorm):
    #         torch.nn.init.ones_(module.weight)
    #     elif isinstance(module, nn.Linear):
    #         if "wo" in name or "down_proj" in name:
    #             torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=out_proj_std, a=-3*out_proj_std, b=3*out_proj_std)
    #         else:
    #             torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def _init_weights(self, module, name=""):
        std = math.sqrt(2 / (5 * self.config.d_model))
        out_proj_std = std / math.sqrt(2 * self.config.effective_expected_depth)
        golden_std = std / 2.0
        
        if isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            if "adapter" in name:
                w_shape = module.weight.shape
                W = torch.empty(w_shape)
                torch.nn.init.trunc_normal_(W, mean=0.0, std=golden_std, a=-3*golden_std, b=3*golden_std)
                if w_shape[0] == w_shape[1]:
                    W = (W + W.transpose(0, 1)) / 2.0
                module.weight.data.copy_(W)
            elif "wo" in name or "down_proj" in name:
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=out_proj_std, a=-3*out_proj_std, b=3*out_proj_std)
            else:
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def initialize_state(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Reference Huginn-0125: Khởi tạo x_0 sử dụng state_init = "like-init"
        """
        x = torch.randn_like(input_embeds)
        std = math.sqrt(2 / (5 * self.config.d_model))
        if std > 0:
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            x = x * math.sqrt(self.config.d_model) # embed_scale
        else:
            x.zero_()
        return x

    def forward(self, inputs: dict, num_loops: int = None):
        assert "hops" in inputs, "Missing 'hops' in inputs. This is critical for visualization!"
        input_ids = inputs["input_ids"]
        hops = inputs["hops"]
        bsz, seqlen = input_ids.shape
        freqs_cis = precompute_freqs_cis(self.config.d_model // self.config.n_heads, seqlen).to(input_ids.device)
        
        # 1. Đầu vào (Tokens -> Embeddings)
        input_embeds = self.wte(input_ids)
        input_embeds = input_embeds * math.sqrt(self.config.d_model) # embed_scale
        
        # 2. Khởi tạo trạng thái vòng lặp đầu tiên
        if self.config.enforced:
            # Enforced logic: state_init bằng noise
            x = self.initialize_state(input_embeds)
        else:
            # Base logic: Bắt đầu thẳng từ embeddings
            x = input_embeds

        last_hidden_states_loops = []
        
        # Lấy số loop từ arg hoặc từ config
        num_loops = num_loops if num_loops is not None else self.config.max_train_loops
        
        # 3. Chạy vòng lặp
        for loop_idx in range(num_loops):
            # 3a. Input Injection
            if self.config.enforced:
                # Reference Huginn-0125: injection_type = "linear" (Concat -> Linear)
                x = self.adapter(torch.cat([x, input_embeds], dim=-1))
            else:
                x = self.adapter(x)
            
            # 3b. Execute loop blocks
            for block in self.loop_blocks:
                x = block(x, freqs_cis)
                
            # 3c. Extract hidden state (sử dụng clone để tránh in-place issue)
            last_hidden_states_loops.append(x.clone())
            
        # 4. Gom output (Shape: [batch, max_loops, seq_len, d_model])
        hidden_states_history = torch.stack(last_hidden_states_loops, dim=1)
        
        # 5. Output via LM Head cho TẤT CẢ các vòng lặp
        final_x = self.ln_f(hidden_states_history) # [batch, max_loops, seq_len, d_model]
        logits = self.lm_head(final_x) # [batch, max_loops, seq_len, vocab_size]
        
        # Chuẩn hóa format output
        predictions = logits.argmax(dim=-1) # [batch, max_loops, seq_len]
        return ReasoningOutput(
            logits=logits,
            predictions=predictions,
            last_hidden_states_loops=hidden_states_history,
            hops=hops
        )

@MODEL_REGISTRY.register("AdapterLoop")
class AdapterLoopModel(nn.Module):
    """
    Nhóm 1: Mô hình lặp cơ bản (BaseLoopModel) 
    (Tương ứng Condition 1 & 2 trong Architecture Matrix)
    Không có Prelude/Coda layers. Loop được bao bọc trực tiếp sau Embedding và trước LM Head.
    """
    def __init__(self, config: BaseLoopConfig):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        BlockClass = SandwichBlock
        self.adapter = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.loop_blocks = nn.ModuleList([
            BlockClass(config.d_model, config.n_heads, config.d_model * config.scale_mlp)
            for _ in range(config.n_layers)
        ])
            
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Reference Huginn-0125: Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
            
        for name, module in self.named_modules():
            self._init_weights(module, name)

    def _init_weights(self, module, name=""):
        std = math.sqrt(2 / (5 * self.config.d_model))
        out_proj_std = std / math.sqrt(2 * self.config.effective_expected_depth)
        
        if isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            if "wo" in name or "down_proj" in name:
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=out_proj_std, a=-3*out_proj_std, b=3*out_proj_std)
            else:
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    # def _init_weights_experimental(self, module, name=""):
    #     std = math.sqrt(2 / (5 * self.config.d_model))
    #     out_proj_std = std / math.sqrt(2 * self.config.effective_expected_depth)
    #     golden_std = std / 2.0
        
    #     if isinstance(module, RMSNorm):
    #         torch.nn.init.ones_(module.weight)
    #     elif isinstance(module, nn.Linear):
    #         if "adapter" in name:
    #             w_shape = module.weight.shape
    #             W = torch.empty(w_shape)
    #             torch.nn.init.trunc_normal_(W, mean=0.0, std=golden_std, a=-3*golden_std, b=3*golden_std)
    #             if w_shape[0] == w_shape[1]:
    #                 W = (W + W.transpose(0, 1)) / 2.0
    #             module.weight.data.copy_(W)
    #         elif "wo" in name or "down_proj" in name:
    #             torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=out_proj_std, a=-3*out_proj_std, b=3*out_proj_std)
    #         else:
    #             torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def initialize_state(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Reference Huginn-0125: Khởi tạo x_0 sử dụng state_init = "like-init"
        """
        x = torch.randn_like(input_embeds)
        std = math.sqrt(2 / (5 * self.config.d_model))
        if std > 0:
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            x = x * math.sqrt(self.config.d_model) # embed_scale
        else:
            x.zero_()
        return x

    def forward(self, inputs: dict, num_loops: int = None):
        assert "hops" in inputs, "Missing 'hops' in inputs. This is critical for visualization!"
        input_ids = inputs["input_ids"]
        hops = inputs["hops"]
        bsz, seqlen = input_ids.shape
        freqs_cis = precompute_freqs_cis(self.config.d_model // self.config.n_heads, seqlen).to(input_ids.device)
        
        # 1. Đầu vào (Tokens -> Embeddings)
        input_embeds = self.wte(input_ids)
        input_embeds = input_embeds * math.sqrt(self.config.d_model) # embed_scale
        
        # 2. Khởi tạo trạng thái vòng lặp đầu tiên (dùng noise như bản Enforced để so sánh công bằng)
        x = input_embeds

        last_hidden_states_loops = []
        
        # Lấy số loop từ arg hoặc từ config
        num_loops = num_loops if num_loops is not None else self.config.max_train_loops
        
        # 3. Chạy vòng lặp
        for loop_idx in range(num_loops):
            
            x = self.adapter(x)
            
            # 3b. Execute loop blocks
            for block in self.loop_blocks:
                x = block(x, freqs_cis)
                
            # 3c. Extract hidden state (sử dụng clone để tránh in-place issue)
            last_hidden_states_loops.append(x.clone())
            
        # 4. Gom output (Shape: [batch, max_loops, seq_len, d_model])
        hidden_states_history = torch.stack(last_hidden_states_loops, dim=1)
        
        # 5. Output via LM Head cho TẤT CẢ các vòng lặp
        final_x = self.ln_f(hidden_states_history) # [batch, max_loops, seq_len, d_model]
        logits = self.lm_head(final_x) # [batch, max_loops, seq_len, vocab_size]
        
        # Chuẩn hóa format output
        predictions = logits.argmax(dim=-1) # [batch, max_loops, seq_len]
        return ReasoningOutput(
            logits=logits,
            predictions=predictions,
            last_hidden_states_loops=hidden_states_history,
            hops=hops
        )
