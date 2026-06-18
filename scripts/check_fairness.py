import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from code.models.configs import BaseLoopConfig, PreludeCodaConfig
from code.models.base_loop import SimpleLoopModel
from code.models.prelude_coda import PreludeCodaLoopModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model(name, config_path, vocab_size, is_base=False):
    print(f"\n{'='*50}")
    print(f"Analyzing {name}")
    print(f"{'='*50}")
    
    if is_base:
        config = BaseLoopConfig.from_yaml(config_path, vocab_size=vocab_size)
        model = SimpleLoopModel(config)
        # N Layers
        n_layers = config.n_layers
        print(f"N Layers: {n_layers} (Loop Only)")
        
        total_params = count_parameters(model)
        loop_params = count_parameters(model.loop_blocks)
        if config.enforced:
            loop_params += count_parameters(model.adapter)
    else:
        config = PreludeCodaConfig.from_yaml(config_path, vocab_size=vocab_size)
        model = PreludeCodaLoopModel(config)
        
        n_prelude = config.n_prelude
        n_loop = config.n_loop
        n_coda = config.n_coda
        print(f"N Layers: {n_prelude} Prelude, {n_loop} Loop, {n_coda} Coda")
        
        total_params = count_parameters(model)
        loop_params = count_parameters(model.loop_blocks)
        if config.enforced:
            loop_params += count_parameters(model.adapter)
            
    print(f"Param size (Total): {total_params:,}")
    print(f"Param size (Loop section): {loop_params:,}")
    
    # Compute Budget: C = P_total + P_loop * (L_max - 1)
    l_max = config.max_train_loops
    compute_budget = total_params + loop_params * (l_max - 1)
    print(f"Compute Budget (FLOP proxy): {compute_budget:,}")
    print(f"Target Compute Depth Formula: P_total + P_loop * ({l_max} - 1)")
    
    print("\n--- Architecture Details ---")
    if is_base:
        print(f"Loop Blocks: {len(model.loop_blocks)}")
        print(f"Sample Loop Block:\n{model.loop_blocks[0]}")
    else:
        print(f"Prelude Blocks: {len(model.prelude_blocks)}")
        print(f"Sample Prelude Block:\n{model.prelude_blocks[0]}")
        print(f"Loop Blocks: {len(model.loop_blocks)}")
        print(f"Sample Loop Block:\n{model.loop_blocks[0]}")

if __name__ == "__main__":
    vocab_size = 111 # Từ data/metadata.json
    
    base_path = os.path.join(project_root, "config", "arch", "base_loop.yaml")
    uniform_path = os.path.join(project_root, "config", "arch", "prelude_coda.yaml")
    mismatch_path = os.path.join(project_root, "config", "arch", "prelude_coda_mismatch.yaml")
    
    analyze_model("BaseLoop (Model 1)", base_path, vocab_size, is_base=True)
    analyze_model("PreludeCoda Uniform (Model 2)", uniform_path, vocab_size, is_base=False)
    analyze_model("PreludeCoda Mismatched (Model 3)", mismatch_path, vocab_size, is_base=False)
