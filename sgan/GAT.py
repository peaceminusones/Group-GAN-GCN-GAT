import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class BatchMultiHeadGraphAttention(nn.Module):
#     def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
#         super(BatchMultiHeadGraphAttention, self).__init__()
#         self.n_head = n_head
#         self.f_in = f_in
#         self.f_out = f_out
#         self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
#         self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#         self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(attn_dropout)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(f_out))
#             nn.init.constant_(self.bias, 0)
#         else:
#             self.register_parameter("bias", None)

#         nn.init.xavier_uniform_(self.w, gain=1.414)
#         nn.init.xavier_uniform_(self.a_src, gain=1.414)
#         nn.init.xavier_uniform_(self.a_dst, gain=1.414)

#     def forward(self, h):
#         bs, n = h.size()[:2]
#         h_prime = torch.matmul(h.unsqueeze(1), self.w)
#         attn_src = torch.matmul(h_prime, self.a_src)
#         attn_dst = torch.matmul(h_prime, self.a_dst)

#         attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
#         attn = self.leaky_relu(attn)
#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.matmul(attn, h_prime)
#         if self.bias is not None:
#             return output + self.bias, attn
#         else:
#             return output, attn

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + " ("
#             + str(self.n_head)
#             + " -> "
#             + str(self.f_in)
#             + " -> "
#             + str(self.f_out)
#             + ")"
#         )


# class GAT(nn.Module):
#     def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
#         super(GAT, self).__init__()
#         self.n_layer = len(n_units) - 1
#         self.dropout = dropout
#         self.layer_stack = nn.ModuleList()

#         for i in range(self.n_layer):
#             f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
#             self.layer_stack.append(
#                 BatchMultiHeadGraphAttention(
#                     n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
#                 )
#             )

#         self.norm_list = [
#             torch.nn.InstanceNorm1d(32).cuda(),
#             torch.nn.InstanceNorm1d(64).cuda(),
#         ]

#     def forward(self, x):
#         bs, n = x.size()[:2]
#         for i, gat_layer in enumerate(self.layer_stack):
#             x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
#             x, attn = gat_layer(x)
#             if i + 1 == self.n_layer:
#                 x = x.squeeze(dim=1)
#             else:
#                 x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
#                 x = F.dropout(x, self.dropout, training=self.training)
#         else:
#             return x


# class GATEncoder(nn.Module):
#     def __init__(self, n_units, n_heads, dropout, alpha):
#         super(GATEncoder, self).__init__()
#         self.gat_net = GAT(n_units, n_heads, dropout, alpha)

#     def forward(self, obs_traj_embedding, seq_start_end):
#         graph_embeded_data = []
#         for start, end in seq_start_end.data:
#             curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
#             curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
#             print("curr_seq_embedding_traj:", curr_seq_embedding_traj.size())
#             print("curr_seq_graph_embedding:", curr_seq_graph_embedding.size())
#             graph_embeded_data.append(curr_seq_graph_embedding)
#         graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
#         return graph_embeded_data


# class BatchMultiHeadGraphAttention(nn.Module):
#     """
#         graph attetion layer(GAL)
#     """
#     def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
#         super(BatchMultiHeadGraphAttention, self).__init__()
#         self.n_head = n_head
#         self.f_in = f_in
#         self.f_out = f_out
#         self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
#         self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#         self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(attn_dropout)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(f_out))
#             nn.init.constant_(self.bias, 0)
#         else:
#             self.register_parameter("bias", None)

#         nn.init.xavier_uniform_(self.w, gain=1.414)
#         nn.init.xavier_uniform_(self.a_src, gain=1.414)
#         nn.init.xavier_uniform_(self.a_dst, gain=1.414)

#     def forward(self, h, adj):
#         bs, n = h.size()[:2]
#         h_prime = torch.matmul(h.unsqueeze(1), self.w)
#         attn_src = torch.matmul(h_prime, self.a_src)
#         attn_dst = torch.matmul(h_prime, self.a_dst)

#         attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
#         attn = self.leaky_relu(attn)
#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         attn = torch.matmul(attn, adj)
#         output = torch.matmul(attn, h_prime)
#         if self.bias is not None:
#             return output + self.bias, attn
#         else:
#             return output, attn

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + " ("
#             + str(self.n_head)
#             + " -> "
#             + str(self.f_in)
#             + " -> "
#             + str(self.f_out)
#             + ")"
#         )


# """
#     modified by zyl 2021/2/6 graph attetion network
# """
# class GAT(nn.Module):
#     def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
#         super(GAT, self).__init__()
#         self.n_layer = len(n_units) - 1
#         self.dropout = dropout
#         self.layer_stack = nn.ModuleList()

#         for i in range(self.n_layer):
#             f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
#             self.layer_stack.append(
#                 BatchMultiHeadGraphAttention(
#                     n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
#                 )
#             )

#         self.norm_list = [
#             torch.nn.InstanceNorm1d(32).cuda(),
#             torch.nn.InstanceNorm1d(64).cuda(),
#         ]

#     def forward(self, x, adj):
#         bs, n = x.size()[:2]
#         for i, gat_layer in enumerate(self.layer_stack):
#             x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
#             x, attn = gat_layer(x, adj)
#             if i + 1 == self.n_layer:
#                 x = x.squeeze(dim=1)
#             else:
#                 x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
#                 x = F.dropout(x, self.dropout, training=self.training)
#         else:
#             return x


# """
#     modified by zyl 2021/2/6 graph attetion network encoder
# """
# class GATEncoder(nn.Module):
#     def __init__(self, n_units, n_heads, dropout, alpha):
#         super(GATEncoder, self).__init__()
#         self.gat_intra = GAT([40,72,16], n_heads, dropout, alpha)
#         self.gat_inter = GAT([16,72,16], n_heads, dropout, alpha)
#         self.out_embedding = nn.Linear(16*2, 24)

#     def normalize(self, adj, dim):
#         N = adj.size()
#         adj2 = torch.sum(adj, dim)       # 对每一行求和
#         norm = adj2.unsqueeze(1).float()         # 扩展张量维度
#         norm = norm.pow(-1)              # 求倒数
#         norm_adj = adj.mul(norm)         # 点乘
#         return norm_adj

#     def forward(self, obs_traj_embedding, seq_start_end, end_pos, end_group):
#         graph_embeded_data = []
#         for start, end in seq_start_end.data:
#             curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
#             h_states = torch.squeeze(obs_traj_embedding, dim=0)
            
#             num_ped = end - start
#             curr_end_group = end_group[start:end]
#             eye_mtx = torch.eye(num_ped, device=end_group.device).bool()
#             A_g = curr_end_group.repeat(1, num_ped)
#             B_g = curr_end_group.transpose(1, 0).repeat(num_ped, 1)
#             M_intra = (A_g == B_g) & (A_g != 0) | eye_mtx
#             A_intra = self.normalize(M_intra, dim=1).cuda()

#             curr_seq_graph_intra = self.gat_intra(curr_seq_embedding_traj, A_intra)
            
#             # print("curr_seq_embedding_traj:", curr_seq_embedding_traj.size())
#             # print("curr_seq_graph_intra:", curr_seq_graph_intra.size())

#             R_intra_unique = torch.unique(M_intra, sorted=False, dim=0)
#             n_group = R_intra_unique.size()[0]
#             R_intra_unique.unsqueeze_(1)
#             R_intra = []
#             for i in range(n_group-1, -1, -1):
#                 R_intra.append(R_intra_unique[i])
#             R_intra = torch.cat(R_intra, dim=0)
#             R_intra = self.normalize(R_intra, dim=1).cuda()
            
#             curr_seq_graph_state_in = torch.matmul(R_intra, curr_seq_graph_intra)

#             M_inter = torch.ones((n_group, n_group), device=end_group.device).bool()
#             A_inter = self.normalize(M_inter, dim=1).cuda()

#             curr_seq_graph_out = self.gat_inter(curr_seq_graph_state_in, A_inter)
#             curr_seq_graph_inter = torch.matmul(R_intra.T, curr_seq_graph_out)

#             curr_gat_state = torch.cat([curr_seq_graph_intra, curr_seq_graph_inter],dim=2)
#             curr_gat_state = torch.squeeze(curr_gat_state, dim=0)
#             curr_gat_state = self.out_embedding(curr_gat_state)
#             curr_gat_state = torch.unsqueeze(curr_gat_state, 0)

#             graph_embeded_data.append(curr_gat_state)
        
#         graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
#         return graph_embeded_data

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  
        self.out_features = out_features   
        self.alpha = alpha 
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.$W$.data, gain=1.414)  
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))  
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) 
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) 
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) 
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] 
        # 对第0个维度复制N遍
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  
        # 对第1个维度复制N遍
        Wh_repeated_alternating = Wh.repeat(N, 1) 
        # 在第1维上做全连接操作，得到了（N * N, 2 * out_features）的矩阵
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()  
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
      
    def forward(self, x, adj):   
        # dropout不改变x的维度
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)   
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))    
        return F.log_softmax(x, dim=1)

class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_intra = GAT(40, 72, 16, dropout, alpha, n_heads)
        self.gat_inter = GAT(16, 72, 16, dropout, alpha, n_heads)
        self.out_embedding = nn.Linear(16*2, 24)

    def normalize(self, adj, dim):
        N = adj.size()
        adj2 = torch.sum(adj, dim)       # 对每一行求和
        norm = adj2.unsqueeze(1).float()         # 扩展张量维度
        norm = norm.pow(-1)              # 求倒数
        norm_adj = adj.mul(norm)         # 点乘
        return norm_adj

    def forward(self, h_states, seq_start_end, end_pos, end_group):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_state = h_states[start:end]
            curr_end_group = end_group[start:end]
            
            num_ped = end - start
            
            eye_mtx = torch.eye(num_ped, device=end_group.device).bool()
            A_g = curr_end_group.repeat(1, num_ped)
            B_g = curr_end_group.transpose(1, 0).repeat(num_ped, 1)
            M_intra = (A_g == B_g) & (A_g != 0) | eye_mtx
            A_intra = self.normalize(M_intra, dim=1).cuda()

            curr_gat_state_intra = self.gat_intra(curr_state, A_intra)

            R_intra_unique = torch.unique(M_intra, sorted=False, dim=0)
            n_group = R_intra_unique.size()[0]
            R_intra_unique.unsqueeze_(1)
            R_intra = []
            for i in range(n_group-1, -1, -1):
                R_intra.append(R_intra_unique[i])
            R_intra = torch.cat(R_intra, dim=0)
            R_intra = self.normalize(R_intra, dim=1).cuda()
            
            curr_gat_group_state_in = torch.matmul(R_intra, curr_gat_state_intra)

            M_inter = torch.ones((n_group, n_group), device=end_group.device).bool()
            A_inter = self.normalize(M_inter, dim=1).cuda()

            curr_gat_group_state_out = self.gat_inter(curr_gat_group_state_in, A_inter)
            curr_gat_state_inter = torch.matmul(R_intra.T, curr_gat_group_state_out)

            curr_gat_state = torch.cat([curr_gat_state_intra, curr_gat_state_inter],dim=2)
            curr_gat_state = self.out_embedding(curr_gat_state)

            graph_embeded_data.append(curr_gat_state)
        
        graph_embeded_data = torch.cat(graph_embeded_data, dim=0)
        return graph_embeded_data