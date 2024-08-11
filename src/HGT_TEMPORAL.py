import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, node_type_list, 
                 edge_type_list, edge_type_mapping, last_residual = True, 
                 last_norm = True, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 encoder=True):
        super(HGTLayer, self).__init__()
        self.encoder = encoder
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.n_heads        = n_heads
        self.d_over_h       = {ntype: out_dim[ntype] // n_heads for ntype in node_type_list}
        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.num_edge_types = len(edge_type_list)
        self.num_node_types = len(node_type_list)
        self.edge_type_mapping = edge_type_mapping
        self.residual = last_residual

        self.QueryVectors   = nn.ModuleDict()
        self.KeyVectors     = nn.ModuleDict()
        self.MLinears       = nn.ModuleDict()
        self.ALinears       = nn.ModuleDict()

        for ntype in in_dim.keys():
            self.QueryVectors[ntype] = nn.Linear(in_features=in_dim[ntype], out_features=out_dim[ntype], bias=False).to(device)
            self.KeyVectors[ntype]   = nn.Linear(in_features=in_dim[ntype], out_features=out_dim[ntype], bias=False).to(device)
            self.MLinears[ntype]     = nn.Linear(in_features=in_dim[ntype], out_features=out_dim[ntype], bias=False).to(device)
            self.ALinears[ntype]     = nn.Linear(in_features=out_dim[ntype], out_features=out_dim[ntype], bias=False).to(device)
        
        self.W_ATT          = nn.ParameterDict()
        self.W_MSG          = nn.ParameterDict()
        self.prior_scale    = nn.ParameterDict()

        for edge_type, (src_type, dst_type) in edge_type_mapping.items():
            self.W_ATT[edge_type] = nn.Parameter(torch.Tensor(self.d_over_h[src_type], self.d_over_h[dst_type])).to(device)
            self.W_MSG[edge_type] = nn.Parameter(torch.Tensor(self.d_over_h[src_type], self.d_over_h[dst_type])).to(device)
            self.prior_scale[edge_type] = nn.Parameter(torch.Tensor(1))

            # Initialize the weights
            nn.init.xavier_uniform_(self.W_ATT[edge_type])
            nn.init.xavier_uniform_(self.W_MSG[edge_type])
            nn.init.constant_(self.prior_scale[edge_type], 1)

        # Activation function
        self.elu        = nn.ELU()
        self.sigmoid    = nn.Sigmoid()

        # Linear layer, batch norm and layer norm for each node type
        self.projection = nn.ModuleDict()
        self.LayerNorms = nn.ModuleDict()
        self.residual_alpha = nn.ParameterDict()
        if self.encoder:
            self.RTE = nn.ModuleDict()
        for ntype in node_type_list:
            self.projection[ntype] = nn.Linear(in_dim[ntype], out_dim[ntype]).to(device)
            self.LayerNorms[ntype] = nn.LayerNorm(out_dim[ntype]).to(device)
            if self.residual:
                self.residual_alpha[ntype] = nn.Parameter(torch.ones(1))
            if self.encoder:
                self.RTE[ntype] = RelativeTemporalEncoding(out_dim[ntype], device=device)

    def compute_edge_data(self, G):
        for s_type, e_type, t_type in G.canonical_etypes:
            G.apply_edges(self.apply_edge_attention, etype=(s_type, e_type, t_type))

    def apply_edge_attention(self, edges):
        s_type, e_type, t_type = edges.canonical_etype
        if self.encoder:
            delta_T = 27759 - edges.data['time']
            H_src = self.RTE[s_type](edges.src['h'], delta_T)
        else:
            H_src = edges.src['h']
        H_dst = edges.dst['h']
        
        K = self.KeyVectors[s_type](H_src).view(-1, self.n_heads, self.d_over_h[s_type])
        Q = self.QueryVectors[t_type](H_dst).view(-1, self.n_heads, self.d_over_h[t_type])
        W_att = self.W_ATT[e_type]

        K_W = torch.einsum('bhd,de->bhe', K, W_att)
        att = torch.einsum('bhd,bhd->bh', Q, K_W) * self.prior_scale[e_type] / math.sqrt(self.d_over_h[t_type])

        M = self.MLinears[s_type](H_src).view(-1, self.n_heads, self.d_over_h[s_type])
        W_msg = self.W_MSG[e_type]
        msg = torch.einsum('bhd,de->bhe', M, W_msg)

        # Store computed attention and message in edge data
        return {'att': att, 'msg': msg}

    # Message function using DGL's built-in functions
    def custom_message_func(self, edges):
        return {'msg': edges.data['msg'], 'att': edges.data['att']}

    # Reduce function to aggregate messages at the target nodes
    def custom_reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        agg_msg = torch.sum(att.unsqueeze(-1) * nodes.mailbox['msg'], dim=1)
        return {'aggregated_msg': agg_msg}

    def update(self, aggregated_msgs, ntype, original_features):
        # Apply a linear transformation to the aggregated messages
        transformed_msgs = self.ALinears[ntype](self.elu(aggregated_msgs.view(aggregated_msgs.size(0), -1)))
        if self.residual:
            alpha = self.sigmoid(self.residual_alpha[ntype])
            updated_features = alpha * transformed_msgs + (1-alpha) * original_features
        else:
            updated_features = transformed_msgs

        # Apply layer norm
        updated_features = self.LayerNorms[ntype](updated_features)
        
        return updated_features

    def forward(self, G, H, return_att = False):
        for ntype in H:
            G.nodes[ntype].data['h'] = H[ntype]
        # Compute and set edge data for all edges
        self.compute_edge_data(G)

        edge_func_dict = {
            etype: (self.custom_message_func, self.custom_reduce_func)
            for etype in G.etypes
        }

        # Execute message passing using DGL's multi_update_all
        G.multi_update_all(edge_func_dict, cross_reducer='mean')
        # Create a new dictionary for updated node features
        H_updated = {}
        for ntype in G.ntypes:
            # Instead of modifying H[ntype], create a new tensor for the updated features
            H_updated[ntype] = self.update(G.nodes[ntype].data['aggregated_msg'], ntype, G.nodes[ntype].data['h'])

        # Return the updated features without having modified the input H in-place
        if return_att:
            return H_updated, G
        return H_updated

class RelativeTemporalEncoding(nn.Module):
    """
    Relative temporal encoding class.
    """
    def __init__(self, n_hid, max_len=27759, #2015-12-31
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(RelativeTemporalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000) / n_hid))
        emb = nn.Embedding(max_len, n_hid).to(device)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        if n_hid%2==0: #if n_hid is even:
            emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        else:
            emb.weight.data[:, 1::2] = (torch.cos(position * div_term) / math.sqrt(n_hid))[:,:-1]
        emb.requires_grad_ = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid).to(device)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class HGT_TEMPORAL(nn.Module):
    def __init__(self,
                n_layers, 
                in_dim,
                hidden_dim, 
                out_dim,
                n_heads, 
                node_type_list, 
                edge_type_list,
                edge_type_mapping,
                encoding = False,
                residual = True,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                encoder=True):
        super(HGT_TEMPORAL, self).__init__()

        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(HGTLayer(in_dim, out_dim, n_heads, node_type_list, edge_type_list, edge_type_mapping, last_residual = False, encoder=encoder))
        #self.layers.append(HGTLayer(in_dim, hidden_dim, n_heads, node_type_list, edge_type_list, edge_type_mapping))
        else: 
            for _ in range(n_layers-1):
                self.layers.append(HGTLayer(hidden_dim, hidden_dim, n_heads, node_type_list, edge_type_list, edge_type_mapping, encoder=encoder))
            # Last layer
            self.layers.append(HGTLayer(hidden_dim, out_dim, n_heads, node_type_list, edge_type_list, edge_type_mapping, encoder=encoder))
        self.encoder = encoder
        if self.encoder:
            self.projection = nn.ModuleDict()
            for ntype in node_type_list:
                self.projection[ntype] = nn.Linear(in_dim[ntype], hidden_dim[ntype]).to(device)


    def forward(self, G, H, return_att=False):
        H_new = {}
        if self.encoder:
            for ntype in H:
                H_new[ntype] = self.projection[ntype](H[ntype])
        if return_att:
            for layer in self.layers:
                H_new, G_new = layer(G, H_new, return_att)
            return H_new, G_new
        else:
            for layer in self.layers:
                H_new = layer(G, H_new)
            return H_new