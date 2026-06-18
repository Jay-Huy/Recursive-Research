import math
import torch
import torch.nn as nn
from src.models.blocks import RMSNorm, TransformerPreNormBlock, SandwichBlock, precompute_freqs_cis
from src.models.configs import PreludeCodaConfig
from src.core.base import ReasoningOutput
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("PreludeCoda")
class PreludeCodaLoopModel(nn.Module):
    """
    Nhóm 2: Mô hình lặp có Prelude và Coda layers (Tương ứng các Condition trong Architecture Matrix)
    Bao gồm logic hỗ trợ d_loop khác d_model thông qua phép chiếu Project in/out.
    """
    def __init__(self, config: PreludeCodaConfig):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        # 1. Prelude
        BlockClass = SandwichBlock if config.enforced else TransformerPreNormBlock
        self.prelude_blocks = nn.ModuleList([
            BlockClass(config.d_model, config.n_heads, config.d_model * config.scale_mlp)
            for _ in range(config.n_prelude)
        ])
        
        # 2. Logic d_loop (nếu khác d_model)
        self.has_d_loop = config.d_loop is not None and config.d_loop != config.d_model
        d_loop_dim = config.d_loop if self.has_d_loop else config.d_model
        
        if self.has_d_loop:
            self.project_in = nn.Linear(config.d_model, d_loop_dim, bias=False)
            self.project_out = nn.Linear(d_loop_dim, config.d_model, bias=False)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
            
        # 3. Recurrent Loop blocks
        self.adapter = nn.Linear(d_loop_dim * 2, d_loop_dim, bias=False) if config.enforced else nn.Identity()
        self.loop_blocks = nn.ModuleList([
            BlockClass(d_loop_dim, config.n_heads, d_loop_dim * config.scale_mlp) # Chú ý truyền d_loop_dim
            for _ in range(config.n_loop)
        ])
            
        # 4. Coda
        self.coda_blocks = nn.ModuleList([
            BlockClass(config.d_model, config.n_heads, config.d_model * config.scale_mlp)
            for _ in range(config.n_coda)
        ])
            
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Reference Huginn-0125: Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Scale std theo d_model hoặc d_loop tùy vào module input features
            std = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(self.config.d_model)
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def initialize_state(self, input_embeds: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Reference Huginn-0125: Khởi tạo x_0 sử dụng state_init = "like-init"
        Khởi tạo bằng Gaussian Noise std = 1/sqrt(dim)
        """
        x = torch.randn_like(input_embeds)
        std = 1.0 / math.sqrt(dim)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return x

    def forward(self, inputs: dict, num_loops: int = None):
        assert "hops" in inputs, "Missing 'hops' in inputs. This is critical for visualization!"
        input_ids = inputs["input_ids"]
        hops = inputs["hops"]
        bsz, seqlen = input_ids.shape
        freqs_cis = precompute_freqs_cis(self.config.d_model // self.config.n_heads, seqlen).to(input_ids.device)
        
        # Nếu d_loop_dim khác d_model, ta cần freqs_cis riêng cho loop block do d_loop_dim thay đổi
        d_loop_dim = self.config.d_loop if self.has_d_loop else self.config.d_model
        if self.has_d_loop:
            freqs_cis_loop = precompute_freqs_cis(d_loop_dim // self.config.n_heads, seqlen).to(input_ids.device)
        else:
            freqs_cis_loop = freqs_cis

        # 1. Embeddings
        x = self.wte(input_ids)
        
        # 2. Prelude
        for block in self.prelude_blocks:
            x = block(x, freqs_cis)
            
        # 3. Anchor Injection
        # Reference Huginn-0125: input_embeds lấy từ sau khi đi qua Prelude
        input_embeds = x 
        
        # 4. Project vào không gian loop (nếu có d_loop)
        x = self.project_in(x)
        input_embeds = self.project_in(input_embeds)
        
        # 5. Khởi tạo trạng thái vòng lặp
        if self.config.enforced:
            x = self.initialize_state(input_embeds, d_loop_dim)
            
        last_hidden_states_loops = []
        
        # Lấy số loop từ arg hoặc từ config
        num_loops = num_loops if num_loops is not None else self.config.max_train_loops
        
        # 6. Chạy vòng lặp
        for loop_idx in range(num_loops):
            # 6a. Input Injection
            if self.config.enforced:
                # Reference Huginn-0125: linear adapter injection
                x = self.adapter(torch.cat([x, input_embeds], dim=-1))
            
            # 6b. Execute loop blocks
            for block in self.loop_blocks:
                x = block(x, freqs_cis_loop)
                
            # 6c. Trích xuất hidden state
            # Lưu ý: Yêu cầu chuẩn hóa shape là d_model, nên nếu d_loop != d_model,
            # ta buộc phải de-project trước khi ném vào last_hidden_states_loops.
            last_hidden_states_loops.append(x)
            
        # 7. Gom output của các loop (Shape: [batch, max_loops, seq_len, d_loop])
        hidden_states_history = torch.stack(last_hidden_states_loops, dim=1)
        
        # Để pass qua Coda blocks, reshape history thành [batch * max_loops, seq_len, d_loop]
        hidden_states_flat = hidden_states_history.view(bsz * num_loops, seqlen, d_loop_dim)
        
        # Up-projection if needed before Coda
        if self.config.d_model != d_loop_dim:
            hidden_states_flat = self.up_proj(hidden_states_flat)
            
        # 8. Coda Blocks (chạy trên tất cả các bước lặp để vẽ đồ thị accuracy theo loop)
        x_coda = hidden_states_flat
        for block in self.coda_blocks:
            x_coda = block(x_coda, freqs_cis)
            
        # 9. LM Head
        final_x = self.ln_f(x_coda)
        logits_flat = self.lm_head(final_x)
        
        # Reshape lại thành [batch, max_loops, seq_len, vocab_size]
        logits = logits_flat.view(bsz, num_loops, seqlen, self.config.vocab_size)
        predictions = logits.argmax(dim=-1)
        
        return ReasoningOutput(
            logits=logits,
            predictions=predictions,
            last_hidden_states_loops=hidden_states_history,
            hops=hops
        )
