import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DropEdge

from HGT import HGT
from HGT_TEMPORAL import HGT_TEMPORAL
from utils import TAR_masking

import scipy.sparse as sp

class HGMAE(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim, n_heads, n_out_heads, node_type_list, edge_type_list, edge_type_mapping, 
    leave_unchanged, replace_rate, drop_rate=0.5, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    temporal = False):
        super(HGMAE, self).__init__()
        ########################
        #TAR-specific properties
        self.m_mask_token = {}
        self.dm_mask_token = {}
        for node_type in node_type_list:
            self.m_mask_token[node_type] = nn.Parameter(torch.zeros(1, in_dim[node_type], device=device)) #Set device
            self.dm_mask_token[node_type] = nn.Parameter(torch.zeros(1, out_dim[node_type], device=device)) #Set device
        self.leave_unchanged = leave_unchanged
        self.replace_rate = replace_rate
        assert self.leave_unchanged + self.replace_rate < 1 #Sum cannot exceed 1..
        ########################
        self.node_type_list = node_type_list

        # HAN encoder and decoder
        if not temporal:
            self.GNN_encoder = HGT(
                n_layers,
                in_dim,
                hidden_dim,
                out_dim, 
                n_heads,
                node_type_list,
                edge_type_list, 
                edge_type_mapping
            )
        else:
            self.GNN_encoder = HGT_TEMPORAL(
                n_layers,
                in_dim,
                hidden_dim,
                out_dim, 
                n_heads,
                node_type_list,
                edge_type_list, 
                edge_type_mapping
            )
        
        if not temporal:
            self.GNN_decoder = HGT(
                #n_layers,
                1, # GITHUB source code from HGMAE use only single layer in decoder, seems suprising but let us test is out
                out_dim,
                hidden_dim,
                in_dim, 
                n_out_heads,
                node_type_list,
                edge_type_list, 
                edge_type_mapping
            )
        else:
            self.GNN_decoder = HGT_TEMPORAL(
                #n_layers,
                1, # GITHUB source code from HGMAE use only single layer in decoder, seems suprising but let us test is out
                out_dim,
                hidden_dim,
                in_dim, 
                n_out_heads,
                node_type_list,
                edge_type_list, 
                edge_type_mapping,
                encoder=False
            )

        self.dropEdge = DropEdge(p=drop_rate)
        self.projections = nn.ModuleDict({ntype: nn.Linear(in_dim[ntype], 64).to(device) for ntype in node_type_list}) #TODO: output_dim for projection should be an input variable

        self.MLP_decoder = nn.ModuleDict({ntype: nn.Sequential(
            nn.Linear(out_dim[ntype], out_dim[ntype]),
            nn.PReLU(),
            nn.Linear(out_dim[ntype], out_dim[ntype]),
            nn.PReLU(),
            nn.Linear(out_dim[ntype], 128)#in_dim[ntype])
            ) for ntype in node_type_list        
        })

        self.enc_to_dec_tar = nn.ModuleDict()
        self.enc_to_dec_mer = nn.ModuleDict()
        for ntype in node_type_list:
            self.enc_to_dec_tar[ntype] = nn.Linear(out_dim[ntype], out_dim[ntype], bias=False)
            self.enc_to_dec_mer[ntype] = nn.Linear(out_dim[ntype], out_dim[ntype], bias=False)

    def forward(self, G, H, TAR_mask_rate):

        # Mask graph (remove meta-edges)
        if self.training:
            G_masked = self.dropEdge(G)
        else:
            G_masked = G

        # Encode
        H1 = self.GNN_encoder(G_masked, H)

        for ntype in H1: # HGMAE paper has linear layer between encoder and decoder
            H1[ntype] = self.enc_to_dec_mer[ntype](H1[ntype])

        # Decode
        H2 = self.GNN_decoder(G_masked, H1)

        # Reconstruct adjacency matrices
        projected_embeddings = []

        ##################################################
        #TAR feature matrix generation dict for node-types
        masked_attribute_matrices = {}
        masked_node_indices = {}
        ##################################################
        for node_type, embeddings in H2.items():

            ###############################
            # TAR feature matrix generation
            masked_attribute_matrix, (masked_node_idx, _) = TAR_masking(H[node_type], TAR_mask_rate, self.m_mask_token[node_type], self.leave_unchanged, self.replace_rate, training=self.training) #TODO: Skal måske bare rykkes i sit eget loop, men tænkte bare lige at smide den her ;)
            masked_attribute_matrices[node_type] = masked_attribute_matrix
            masked_node_indices[node_type] = masked_node_idx
            ###############################
            
            # check if embeddings contain nan
            if torch.isnan(embeddings).any():
                raise ValueError("Embeddings contain NaN")
            projection_layer = self.projections[node_type]
            projected_embeddings.append(projection_layer(embeddings))
        H_emb = torch.cat(projected_embeddings, dim=0)
        #recon_adj = F.sigmoid(torch.mm(H_emb, H_emb.t())) # Paper use sigmoid
        recon_adj = torch.mm(H_emb, H_emb.t())
        del H_emb

        #######
        # TAR #
        #######

        # Apply attribute masking separately to get H_3 embedding
        H3 = self.GNN_encoder(G, masked_attribute_matrices)  # Use masked_attribute_matrix for H_3
        
        for ntype in H3: # HGMAE paper has linear layer between encoder and decoder
            H3[ntype] = self.enc_to_dec_tar[ntype](H3[ntype])

        # For TAR, we apply another mask to the already masked nodes of the attribute matrix in H_3.
        H3_tilde = {}
        
        for node_type, embeddings in H3.items():
            H3_tilde[node_type] = embeddings.clone()
            if self.training:
                H3_tilde[node_type][masked_node_indices[node_type]]  = 0
                H3_tilde[node_type][masked_node_indices[node_type]] += self.dm_mask_token[node_type]

        # Decode H_3 embeddings to obtain Z
        Z = self.GNN_decoder(G, H3_tilde)

        #######
        # PFP #
        #######

        P_tilde = {ntype: self.MLP_decoder[ntype](H3[ntype]) for ntype in self.node_type_list}

        return recon_adj, Z, masked_node_indices, P_tilde

    def preprocess_attention(self, HG, norm=True):
        node_types = HG.ntypes
        num_nodes = sum(HG.number_of_nodes(ntype) for ntype in node_types)
        num_heads = next(iter(HG.edata['att'].values())).shape[1] # Assuming all types have equal number of attention heads

        all_head_A = [sp.lil_matrix((num_nodes, num_nodes)) for _ in range(num_heads)]

        offsets = {ntype: sum(HG.number_of_nodes(nt) for nt in node_types[:i]) for i, ntype in enumerate(node_types)}
    
        for (src_ntype, edge_type, dst_ntype), att in HG.edata['att'].items():
            src, dst = HG.edges(etype=edge_type)
            src = src.detach().cpu().numpy()
            dst = dst.detach().cpu().numpy()
            src_offset = offsets[src_ntype]
            dst_offset = offsets[dst_ntype]

            for j in range(num_heads):
                att_matrix = att[:, j].detach().cpu().numpy()
                all_head_A[j][dst + dst_offset, src + src_offset] = att_matrix
                #all_head_A[j][src + src_offset, dst + dst_offset] = att_matrix # TODO: Not sure directed is a given here

        if norm:
            for j in range(num_heads):
                #all_head_A[j] = normalize(all_head_A[j], norm='l1', axis=1).tocsr() # This ensures that we normalize over neighbors
                coo_matrix = all_head_A[j].tocoo()
                indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
                values = torch.FloatTensor(coo_matrix.data)
                shape = coo_matrix.shape
                sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
                sparse_tensor = torch.sparse.softmax(sparse_tensor, dim=1)
                sparse_tensor_cpu = sparse_tensor.coalesce().cpu()

                # Convert to SciPy COO matrix
                indices = sparse_tensor_cpu.indices().numpy()
                values = sparse_tensor_cpu.values().numpy()
                shape = sparse_tensor_cpu.shape

                coo_matrix = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
                all_head_A[j] = coo_matrix

        return all_head_A, offsets


    def get_embeddings(self, G, H, return_att=False):
        # Encode the graph to get embeddings
        if return_att:
            H_emb, G = self.GNN_encoder(G, H, return_att)
            all_att, offsets = self.preprocess_attention(G)
            return H_emb, all_att, offsets
        else:
            H_emb = self.GNN_encoder(G, H)
            return H_emb