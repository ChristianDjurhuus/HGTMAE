import numpy as np
import torch
import torch.nn.functional as F
import dgl
import pandas as pd
import h5py
import os

from dgl import to_homogeneous
from node2vec import Node2vecModel


##################
# Loss functions #
##################

def meta_edge_reconstruction_loss(recon_adj, true_adj, gamma=3.0):
    recon_adj = F.normalize(recon_adj, p=2, dim=-1)
    true_adj = F.normalize(true_adj, p=2, dim=-1)
    cosine_similarity = torch.sum(recon_adj * true_adj, dim=-1)
    loss = torch.mean((1 - cosine_similarity) ** gamma)
    return loss

def target_attribute_restoration_loss(node_attributes, restored_node_attributes, masked_node_indices, node_type_list, gamma=1.0):
    node_type_losses = torch.zeros(len(node_type_list))
    for idx, node_type in enumerate(node_type_list):
        cos_sim = F.cosine_similarity(node_attributes[node_type][masked_node_indices[node_type]], restored_node_attributes[node_type][masked_node_indices[node_type]], dim=1)
        losses = (1 - cos_sim).pow(gamma)
        losses = losses.mean()
        node_type_losses[idx] = losses
    return node_type_losses.mean()

def PFP_reconstruction_loss(predicted_positions, positions, gamma=2.0):
    pred_pos = torch.concat(list(predicted_positions.values()))
    pred_pos = F.normalize(pred_pos, p=2, dim=1)
    positions = F.normalize(positions, p=2, dim=1)
    cosine_similarity = torch.sum(pred_pos * positions, dim=1)
    loss = torch.mean((1 - cosine_similarity) ** gamma)
    return loss


    
##################
#   Model Misc   #
##################

def TAR_masking(features, mask_rate, m_mask_token, leave_unchanged, replace_rate, training=True):
    num_nodes = len(features)
    permutation = torch.randperm(num_nodes) #Set device

    # Random masking
    num_mask_nodes = int(mask_rate * num_nodes) #|~V|
    mask_nodes = permutation[: num_mask_nodes] #The nodes that we are to meddle with
    keep_nodes = permutation[num_mask_nodes:] # The nodes that we won't

    if training:
        permutation_mask = torch.randperm(num_mask_nodes) #Set device
        num_leave_nodes = int(leave_unchanged * num_mask_nodes) #The number of nodes we leave be
        num_noise_nodes = int(replace_rate * num_mask_nodes) #The number of nodes to apply noise to
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes #Number of nodes to be masked aside from left nodes and noise nodes
        token_nodes = mask_nodes[permutation_mask[:num_real_mask_nodes]] #These are the nodes that we will mask with tokens

        noise_nodes = mask_nodes[permutation_mask[-num_noise_nodes:]] #These are the noise nodes
        noise_to_be_chosen = torch.randperm(num_nodes)[:num_noise_nodes] #Set device - These nodes are the nodes that we will pick

        out_features = features.clone()
        out_features[token_nodes] = 0.0 #Set mask-nodes to zero
        out_features[token_nodes] += m_mask_token #Add our mask
        if num_noise_nodes > 0:
            out_features[noise_nodes] = features[noise_to_be_chosen] #Apply noise hehe

        return out_features, (mask_nodes, keep_nodes)
    else:
        return features, (mask_nodes, keep_nodes)
    
def dynamic_mask_rate(MIN_p_a, MAX_p_a, max_epoch, epochs_to_converge):
    """
    This mask rate is meant for Target Attribute Restoration
    """

    # Define points to interpolate between
    x1 = 0
    x2 = epochs_to_converge
    y1 = MIN_p_a
    y2 = MAX_p_a

    # Loop through each epoch to determine a masking rate at that epoch
    mask_rates = torch.zeros(max_epoch + epochs_to_converge)
    for epoch in range(max_epoch + epochs_to_converge):
        if epoch <= epochs_to_converge:
            # Linear interpolation until epochs_to_converge
            mask_rates[epoch] = y1 + (epoch - x1) * ((y2 - y1) / (x2 - x1))
        else:
            # After epochs_to_converge, maintain the maximum value
            mask_rates[epoch] = MAX_p_a

    return mask_rates

class EarlyStopping:
    """
    EarlyStopping utility to halt training when validation loss does not improve sufficiently.
    IMPORTANT: Higher validation loss indicates worse performance. Thus, an increase is considered negative.
    """

    def __init__(self, patience=10, min_delta=0.001, warmup_epochs=25, cumulative_patience=5, improvement_threshold=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.cumulative_patience = cumulative_patience
        self.improvement_threshold = improvement_threshold
        self.cumulative_counter = 0
        self.cumulative_improvement = 0

    def __call__(self, val_score):
        self.epoch += 1

        # During warm-up, do not apply early stopping
        if self.epoch <= self.warmup_epochs:
            self.best_score = max(self.best_score, val_score)
            return False

        # Calculate the improvement
        improvement = self.best_score - val_score

        if improvement > self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.cumulative_counter = 0
            self.cumulative_improvement = 0
            return False
        else:
            self.counter += 1
            self.cumulative_improvement += improvement

        # Check if cumulative improvement over a few epochs meets the threshold
        if self.cumulative_counter < self.cumulative_patience:
            self.cumulative_counter += 1
            if self.cumulative_improvement > self.improvement_threshold:
                self.cumulative_counter = 0
                self.cumulative_improvement = 0
                self.counter = 0
                return False

        if self.counter >= self.patience:
            return True
        return False

def save_model(model, runname, epoch, args, all_losses_MER=None, all_losses_TAR=None, all_losses_PFP=None, all_losses_total=None,
all_losses_MER_val=None, all_losses_TAR_val=None, all_losses_PFP_val=None, all_losses_total_val=None):
    path_to_model = 'models/'+runname
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
        with open(f'{path_to_model}config.txt', 'w') as f:
            for arg, value in sorted(vars(args).items()):
                f.write(f'Argument: {arg}, Value: {value}\n')

    # Save model checkpoint
    torch.save(model.state_dict(), f'{path_to_model}checkpoint_{epoch}.pth')

    # Save learning curves
    torch.save(all_losses_MER, f'{path_to_model}train_MER_losses.pt')
    torch.save(all_losses_TAR, f'{path_to_model}train_TAR_losses.pt')
    torch.save(all_losses_PFP, f'{path_to_model}train_PFP_losses.pt')
    torch.save(all_losses_total, f'{path_to_model}train_total_losses.pt')

    torch.save(all_losses_MER_val, f'{path_to_model}val_MER_losses.pt')
    torch.save(all_losses_TAR_val, f'{path_to_model}val_TAR_losses.pt')
    torch.save(all_losses_PFP_val, f'{path_to_model}val_PFP_losses.pt')
    torch.save(all_losses_total_val, f'{path_to_model}val_total_losses.pt')

############
# Sampling #
############


def add_nodes_to_tensor_set(tensor_set, new_nodes):
    return torch.unique(torch.cat((tensor_set, new_nodes), dim=0))

def add_in_budget(B, sampled_nodes, HG, NS, ntype):
    for s_type, e_type, t_type in HG.canonical_etypes:
        
        if t_type != ntype:
            continue
        
        edgelist = torch.vstack((HG.edges(etype=e_type))).T
        target_nodes = edgelist[:, 1].unsqueeze(0)
        expanded_sample_nodes = sampled_nodes.unsqueeze(1)
        mask = (target_nodes == expanded_sample_nodes).any(dim=0)
        filtered_edges = edgelist[mask, :]
        predecessors = filtered_edges[:, 0]
        targets = filtered_edges[:, 1]

        if predecessors.shape[0] == 0:
            continue
        
        target_idx, target_degrees = targets.unique(return_counts=True)
        _, positions = (targets.unsqueeze(1) == target_idx).max(dim=1)
        D_t = 1 / target_degrees[positions]
        if s_type in NS.keys():
            NS_temp = NS[s_type]
            pred_mask = (predecessors[:, None] != NS_temp).all(dim=1)
            indices = predecessors[pred_mask]
            D_t =  D_t[pred_mask]
        else:
            indices = predecessors
        B[s_type].index_add_(0, indices, D_t)
    return None

def sample_subgraph(OS, HG, L, n, save=False, batch_name=None, start_time=None, training=True):
    """
    Heterogeneous graph mini-batch sampling
    Reference: https://arxiv.org/abs/2003.01332

    Input:
        - OS (Output set of samples nodes)
        - HG (DGL heterogenous graph object)
        - L  (Sampling depth)
        - n  (Number of neighbors to sample)
    Output:
        - triple (Sample graph, normalised features, sample adjacency matrix)

    """
    B       = {ntype: torch.zeros(HG.number_of_nodes(ntype))  for ntype in HG.ntypes}

    if training:
        for ntype in OS.keys():
            add_in_budget(B, OS[ntype], HG, OS, ntype)
    else:
        add_in_budget(B, OS['person'], HG, OS, 'person')

    for l in range(1, L):
        for src_ntype in B.keys():
            # Vectorized probability computation
            src_ntype_budget = torch.pow(B[src_ntype], 2)
            total_budget = torch.pow(torch.norm(B[src_ntype], p=2), 2)
            probs = (src_ntype_budget / total_budget).clamp(min=0)

            # Skip if no sampled nodes
            if probs.sum() == 0:
                continue

            # Batch sampling
            if n>len(probs):  #if we have a nodetype with very few nodes, we just sample them all :D
                sampled_nodes = torch.multinomial(probs, num_samples=len(probs), replacement=False)
            else:
                sampled_nodes = torch.multinomial(probs, num_samples=n, replacement=False)

            # Adding sampled nodes to visited set
            if src_ntype in OS.keys():
                OS[src_ntype] = add_nodes_to_tensor_set(OS[src_ntype], sampled_nodes)
            else:
                OS[src_ntype] = sampled_nodes

            add_in_budget(B, sampled_nodes, HG, OS, src_ntype)
            B[src_ntype].index_fill(0, sampled_nodes, 0.0)

    # Construct subgraphs based on samples
    sub_HG = HG.subgraph(OS)
    features = {node_type: normalize_features(sub_HG.nodes[node_type].data['feat']) for node_type in sub_HG.ntypes}
    
    for ntype in sub_HG.ntypes:
        add = 0.0 #We need to differentiate between people and other node types
        if ntype == 'person':
            add = 1
        sub_HG.nodes[ntype].data['ntype'] = torch.zeros(sub_HG.nodes[ntype].data['feat'].shape[0]) + add

    adjacency_matrix, _ = aggregate_adj_matrices(sub_HG, sub_HG.ntypes)

    homo_HG = to_homogeneous(sub_HG)
    # DGL node2vec
    node2vec = Node2vecModel(homo_HG, 128, 6, device='cpu')
    node2vec.train(epochs=50, learning_rate=0.1, batch_size=homo_HG.number_of_nodes())
    positions = node2vec.embedding()[:-1].detach().cpu()

    if save:
        print("\tSaving batch sample...")
        if not os.path.exists(f"toy_graphs/batches/{start_time}"):
            os.makedirs(f"toy_graphs/batches/{start_time}")
            
        torch.save(
            obj={"sub_HG": sub_HG, "features": features, "adjacency_matrix": adjacency_matrix, "positions": positions},
            f=f"toy_graphs/batches/{start_time}/{batch_name}.pt"
        )
        print("\tBatch saved.")
    return (sub_HG, features, adjacency_matrix, positions)

def aggregate_adj_matrices(HG, node_types):
    num_nodes = {ntype: HG.number_of_nodes(ntype) for ntype in node_types}
    total_nodes = HG.number_of_nodes()
    indices = [[], []] # Collect row and column indices for non-zero entries in adjacency matrix

    # Compute offsets for each nodetype
    offsets = {ntype: sum(num_nodes[nt] for nt in node_types[:i]) for i, ntype in enumerate(node_types)}
    
    for src_ntype, edge_type, dst_ntype in HG.canonical_etypes:
        src, dst = HG.edges(etype=edge_type)
        src_offset = offsets[src_ntype]
        dst_offset = offsets[dst_ntype]

        # Adjust source and destination indices based on offsets respectively
        src_idx = src + src_offset
        dst_idx = dst + dst_offset

        # Add edges to indices, Note: following code assumes undirected graph
        indices[0].extend(src_idx.tolist())
        indices[1].extend(dst_idx.tolist())
        indices[0].extend(dst_idx.tolist())
        indices[1].extend(src_idx.tolist())

    # Create sparce adjacency matrix (edge list)
    indices_tensor = torch.LongTensor(indices)
    values = torch.ones(indices_tensor.shape[1])
    agg_adj_matrix = torch.sparse.FloatTensor(indices_tensor, values, torch.Size([total_nodes, total_nodes]))
    return agg_adj_matrix, offsets

def normalize_features(feature_matrix):
    feature_matrix = feature_matrix.type(torch.FloatTensor)  
    X_mean = feature_matrix.mean(axis=0, keepdim=True)
    X_std  = feature_matrix.std(axis=0, keepdim=True)
    return (feature_matrix-X_mean) / (X_std + 1e-06) # Avoid division by zero in case of 0 variance.