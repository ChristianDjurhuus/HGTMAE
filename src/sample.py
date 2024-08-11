import dgl
import torch
import argparse
from utils import sample_subgraph
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='HGMAE')
    parser.add_argument('--num_batches', type=int, default=3,
                        help='Number of batches to sample')
    parser.add_argument('--sample_depth', type=int, default=10,
                        help='Number of layers to sample')
    parser.add_argument('--num_neighbors_sampled', type=int, default=256,
                        help='Number of neighbors to sample')
    parser.add_argument('--graph_path', type=str, default='toy_graphs/imdb.bin',)
    return parser.parse_args()


def main():
    args = get_args()
    HG = dgl.load_graphs(args.graph_path)[0][0]
    ntypes = HG.ntypes
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    init_samples = {ntype: 1000 for ntype in HG.ntypes}
        
    sample_weights = {ntype: torch.ones(HG.number_of_nodes(ntype)) for ntype in HG.ntypes}

    sample_weights = {key: torch.Tensor(value) for key, value in sample_weights.items()}
    OS = []
    for _ in range(args.num_batches):
        batch = {}
        for ntype in HG.ntypes:
            sampled_nodes = torch.multinomial(sample_weights[ntype], num_samples=init_samples[ntype], replacement=False)
            batch[ntype] = sampled_nodes
            if ntype in ['person', 'household', 'workplace']:
                # Decrease sample weights for sampled nodes
                sample_weights[ntype][sampled_nodes] *= 0.5
        OS.append(batch)

    batches = [sample_subgraph(OS[i], HG, args.sample_depth, args.num_neighbors_sampled, save=True, batch_name=f"batch_{i}", start_time=timestamp) for i in range(args.num_batches)]
    number_unique_sampled_nodes = {key: len(torch.unique(torch.cat([d[key] for d in OS], dim = 0))) for key in OS[0].keys()}
    del OS # Free up memory
    del HG # Free up memory
    print("Graph samples created.")
    print("-"*100)
    print("Sample statistics")
    print(f"Number of bathces: {args.num_batches}")
    unique_nodes = {ntype:[] for ntype in ntypes}
    for idx, (subgraph, _, _, _) in enumerate(batches):
        for ntype in subgraph.ntypes:
            num_nodes = subgraph.number_of_nodes(ntype)
            unique_nodes[ntype].append(subgraph.nodes(ntype))
            print(f"Batch no. {idx+1}: Number of {ntype} nodes: {num_nodes}")
    for ntype in ntypes:
        print(f'Number of unique {ntype} nodes: {number_unique_sampled_nodes[ntype]}')
    print("-"*100)

if __name__ == '__main__':
    main()
