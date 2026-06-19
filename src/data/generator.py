import os
import json
import random
import argparse

def build_graph(num_entities, num_relations):
    edges = []
    graph_dict = {i: [] for i in range(num_entities)}
    for e in range(num_entities):
        for r in range(num_relations):
            e_next = random.randint(0, num_entities - 1)
            edges.append((e, r, e_next))
            graph_dict[e].append((r, e_next))
    return edges, graph_dict

def generate_simple_paths(graph_dict, num_entities, hops, max_samples):
    paths = set()
    attempts = 0
    max_attempts = max_samples * 50
    
    while len(paths) < max_samples and attempts < max_attempts:
        attempts += 1
        current = random.randint(0, num_entities - 1)
        path_edges = []
        visited = {current}
        
        found = True
        for _ in range(hops):
            options = graph_dict[current].copy()
            random.shuffle(options)
            
            step_found = False
            for r, e_next in options:
                if e_next not in visited:
                    path_edges.append((current, r, e_next))
                    visited.add(e_next)
                    current = e_next
                    step_found = True
                    break
                    
            if not step_found:
                found = False
                break
                
        if found:
            # path is a tuple of edges: ((e1, r1, e2), (e2, r2, e3))
            paths.add(tuple(path_edges))
            
    return list(paths)

def format_sample(path, rel_offset):
    # input: e_start, r1, r2... r_k
    # target: e_target
    e_start = path[0][0]
    input_ids = [e_start]
    for edge in path:
        input_ids.append(edge[1] + rel_offset)
    e_target = path[-1][2]
    return {
        "input_ids": input_ids,
        "target": e_target,
        "hops": len(path)
    }

def format_1hop_sample(edge, rel_offset):
    return {
        "input_ids": [edge[0], edge[1] + rel_offset],
        "target": edge[2],
        "hops": 1
    }

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_entities', type=int, default=100)
    parser.add_argument('--num_relations', type=int, default=10)
    parser.add_argument('--train_samples_per_hop', type=int, default=2000)
    parser.add_argument('--eval_samples_per_hop', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    num_entities = args.num_entities
    num_relations = args.num_relations
    rel_offset = num_entities
    pad_token = num_entities + num_relations
    vocab_size = pad_token + 1
    
    # Generate full graph
    edges, graph_dict = build_graph(num_entities, num_relations)
    random.shuffle(edges)
    
    # Split C_ID (80%) and C_OOD (20%)
    split_idx = int(0.8 * len(edges))
    c_id_edges = edges[:split_idx]
    c_ood_edges = edges[split_idx:]
    
    # Build dicts for path generation
    c_id_dict = {i: [] for i in range(num_entities)}
    for e, r, nxt in c_id_edges:
        c_id_dict[e].append((r, nxt))
        
    c_ood_dict = {i: [] for i in range(num_entities)}
    for e, r, nxt in c_ood_edges:
        c_ood_dict[e].append((r, nxt))
        
    print(f"Graph built. C_ID edges: {len(c_id_edges)}, C_OOD edges: {len(c_ood_edges)}")
    
    train_data = []
    val_data = []
    test_data = []
    
    # 1. Train set gets ALL atomic facts (1-hop)
    for edge in edges:
        train_data.append(format_1hop_sample(edge, rel_offset))
        
    # 2. Multi-hop for Train and Val (from C_ID)
    # Val gets 2 to 10 hops (200 samples/hop)
    # Train gets 2 to 6 hops (2000 samples/hop)
    for hop in range(2, 11):
        print(f"Generating C_ID paths for hop {hop}...")
        
        need_train = (hop <= 6)
        train_target = args.train_samples_per_hop if need_train else 0
        val_target = args.eval_samples_per_hop
        total_needed = train_target + val_target
        
        paths = generate_simple_paths(c_id_dict, num_entities, hop, total_needed)
        random.shuffle(paths)
        
        # Split into train and val
        if len(paths) >= total_needed:
            train_paths = paths[:train_target]
            val_paths = paths[train_target:train_target+val_target]
        else:
            # Not enough paths, split proportionally
            val_count = max(1, int(len(paths) * (val_target / total_needed)))
            train_count = len(paths) - val_count
            train_paths = paths[:train_count]
            val_paths = paths[train_count:]
        
        for p in train_paths:
            train_data.append(format_sample(p, rel_offset))
        for p in val_paths:
            val_data.append(format_sample(p, rel_offset))
            
    # 3. Multi-hop for Test (from C_OOD) -> 2 to 10 hops
    for hop in range(2, 11):
        print(f"Generating C_OOD paths for hop {hop}...")
        paths = generate_simple_paths(c_ood_dict, num_entities, hop, args.eval_samples_per_hop)
        for p in paths:
            test_data.append(format_sample(p, rel_offset))
            
    # Shuffle Train and Val so hops are mixed
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    save_jsonl(train_data, os.path.join(args.out_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(args.out_dir, 'val.jsonl'))
    save_jsonl(test_data, os.path.join(args.out_dir, 'test.jsonl'))
    
    # Save metadata
    metadata = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "rel_offset": rel_offset,
        "pad_token": pad_token,
        "vocab_size": vocab_size,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data)
    }
    with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Dataset generated! Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == '__main__':
    main()
