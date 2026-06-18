import os
import sys
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.models.trm import TRMStrippedModel

def main():
    # Load TRM config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "TinyRecursiveModels", "config", "arch", "trm.yaml")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        trm_config = yaml.safe_load(f)
        
    vocab_size = 111
    max_train_loops = trm_config.get("halt_max_steps", 16) # Dùng halt_max_steps như max_train_loops
    
    # Init Model
    model = TRMStrippedModel(trm_config, vocab_size, max_train_loops)
    
    # Calculate params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Extract components
    h_cycles = model.trm_config.H_cycles
    l_cycles = model.trm_config.L_cycles
    l_layers = model.trm_config.L_layers
    
    # Print report
    print("="*50)
    print("TRM Stripped Model Report")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    print(f"H_cycles: {h_cycles}")
    print(f"L_cycles: {l_cycles}")
    print(f"L_layers: {l_layers}")
    print(f"Max Train Loops (L_max): {max_train_loops}")
    
    # Tính toán total loops 
    depth_per_loop = h_cycles * (l_cycles + 1) * l_layers
    total_depth = depth_per_loop * max_train_loops
    
    print(f"Depth per Macroscopic Loop (H_cycles * (L_cycles + 1)): {depth_per_loop} layers")
    print(f"Total Compute Depth: {total_depth} layers")
    print("="*50)

if __name__ == "__main__":
    main()
