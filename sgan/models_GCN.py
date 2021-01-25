import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    # make_mlp主要是构造多层的全连接网络，并且根据需求决定激活函数的类型，其参数dim_list是全连接网络各层维度的列表
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    # get_noise函数主要是生成特定的噪声
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """
    Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator
    网络结构主要包括一个全连接层和一个LSTM网络
    """

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        最原始的输入是这一批输数据所有人的观测数据中的相对位置变化坐标，即当前帧相对于上一帧每个人的坐标变化，
        其经过一个2*16的全连接层，全连接层的输入的shape:[obs_len*batch,2]，输出：[obs_len*batch,16]

        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        # 其经过一个2*16的全连接层，全连接层的输入的shape:[obs_len*batch,2],输出：[obs_len*batch,16]
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        # 经过维度变换变成3维的以符合LSTM网络中输入input的格式要求
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        # 初始化h_0, c_0
        state_tuple = self.init_hidden(batch)
        # LSTM，LSTM的输入input的shape为[seq_len,batch,input_size], 然后再把h_0和c_0输入LSTM
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        # 输出隐藏状态h_t记为final_h
        final_h = state[0]

        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""

    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.gcn_h = 72
        self.gcn_o = 8

        # mlp [2,16]
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        # lstm
        # input_size: 16
        # hidden_size: 32
        # num_layers: 1
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        # mlp [32,2]
        self.hidden2pos = nn.Linear(h_dim, 2)

        self.pool_net = GCNPooling(embedding_dim=self.embedding_dim)

        # mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
        # self.mlp = make_mlp(
        #     mlp_dims,
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout
        # )

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class GCN(nn.Module):
    """GCN module"""

    def __init__(
        self, input_dim=48, hidden_dim=72, out_dim=8, gcn_layers=2
    ):
        super(GCN, self).__init__()
        self.X_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gcn_layers = gcn_layers

        # graph convolution layer
        self.W = torch.nn.ParameterList()
        for i in range(self.gcn_layers):
            if i == 0:
                self.W.append(nn.Parameter(torch.randn(self.X_dim, self.hidden_dim)))
            elif i == self.gcn_layers-1:
                self.W.append(nn.Parameter(torch.randn(self.hidden_dim, self.out_dim)))
            else:
                self.W.append(nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim)))

    def forward(self, A, X):
        next_H = H = X
        for i in range(self.gcn_layers):
            next_H = F.relu(torch.matmul(torch.matmul(A, H), self.W[i]))
            H = next_H

        feat = H
        return feat


class GCNPooling(nn.Module):
    """Pooling module with GCN layer"""

    def __init__(
        self, embedding_dim=16, input_dim=48, hidden_dim=72, out_dim=8, gcn_layers=2, h_dim=32
    ):
        super(GCNPooling, self).__init__()
        self.h_dim = h_dim

        # mlp: 2*16
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        # GCN_intra: 48*72*8
        self.gcn_pooling_net_intra = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            gcn_layers=gcn_layers)
        # GCN_inter: 48*72*8
        self.gcn_pooling_net_inter = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            gcn_layers=gcn_layers)
        
        # mlp: 

    def normalize(self, adj, dim):
        N = adj.size()
        adj2 = torch.sum(adj, dim)       # 对每一行求和
        norm = adj2.unsqueeze(1)         # 扩展张量维度
        norm = norm.repeat(1, N[0],  1)  # 沿着指定维度重复张量
        norm = norm.permute(2, 0, 1)     # 转置
        norm = norm.pow(-1)              # 求倒数
        norm_adj = adj.mul(norm)         # 点乘
        return norm_adj

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos, end_group):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim) 即encoder的return：final_h
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        - end_group: group labels at the last time step (t_obs); shape: (batch, 1)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start  # num_ped: number of pedestrians in the scene
            # print("num_ped:", num_ped)
            # print("h_states:", h_states.shape)

            # get the modulated adjacency matrix arrays
            # Generate masks from the group labels
            # labels can only be used to distinguish groups at a timestep.
            # var: end_group; def: group labels at the last time step (t_obs); shape: (batch, 1)
            curr_end_group = end_group[start:end]  # clip one onservation-prediction window out of multiple windows.
            # curr_end_group = torch.randint(0, 3, (1, num_ped))
            # curr_end_group = curr_end_group.t()
            # N adjacency matrices for N pedestrians, each corresponds to a star topology graph
            A = np.eye(num_ped)
            B = np.zeros((num_ped, num_ped))
            B[:, 0] = 1
            C = B.T
            adj_all = [np.logical_or(np.logical_or(A, np.roll(B, i, axis=1)), np.roll(C, i, axis=0)) for i in range(num_ped)]
            adj_all = np.array(adj_all)
            adj_all = torch.from_numpy(adj_all).float()
            adj_all = adj_all.cuda()
            # get the coherency mask, dimension: (N, N)
            # coherency mask is shared by all pedestrians in the scene
            eye_mtx = torch.eye(num_ped, device=end_group.device).bool()
            A_g = curr_end_group.repeat(1, num_ped)
            B_g = curr_end_group.transpose(1, 0).repeat(num_ped, 1)
            mask_same = (A_g == B_g) & (A_g != 0) | eye_mtx
            mask_diff = (mask_same == 0) | eye_mtx
            # get the modulated adjacency matrix arrays, each has dimension: [N, N, N]
            adj_same = adj_all*mask_same.float()  # intra group
            adj_same = self.normalize(adj_same, dim=2).cuda()
            adj_diff = adj_all * mask_diff.float()  # inter group
            adj_diff = self.normalize(adj_diff, dim=2).cuda()

            # h_states == final_h (即这里h_states就是LSTM的输出)
            # h_states([1,batch,32])  ->  cur_hidden([N,32])
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            # print("curr_hidden: ", curr_hidden.shape)

            # Repeat -> H1, H2, H1, H2
            # curr_hidden([N,32])  ->  curr_hidden_1([N*N,32])
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = end_pos[start:end]
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            # curr_rel_pos: [N*N,2]
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            # self.spatial_embedding(mlp): 2*16
            # curr_rel_embedding: [N*N,16]
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            # mlp_h_inpur: [N*N,48]
            gcn_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            # gcn_h_input_review: [N,N,48]
            gcn_h_input_review = gcn_h_input.view(num_ped, num_ped, -1)

            curr_gcn_pool = []
            for i in range(num_ped):
                adj_same_i = adj_same[i]
                adj_diff_i = adj_diff[i]

                gcn_h_input_review_i = gcn_h_input_review[i]
                # print(adj_same_i.shape)
                # print(gcn_h_input_review_i.shape)
                curr_gcn_pool_same = self.gcn_pooling_net_intra(adj_same_i, gcn_h_input_review_i)  # [N,8]
                curr_gcn_pool_diff = self.gcn_pooling_net_inter(adj_diff_i, gcn_h_input_review_i)  # [N,8]
                # curr_gcn_pool_i: [N,16]
                curr_gcn_pool_i = torch.cat([curr_gcn_pool_same, curr_gcn_pool_diff], dim=1)




                curr_gcn_pool.append(curr_gcn_pool_i)

            curr_gcn_pool = torch.cat(curr_gcn_pool, dim=0).reshape(num_ped, num_ped, -1)
            # curr_gcn_pool: [N,N,16]
            curr_gcn_pool = curr_gcn_pool.max(1)[0]  # [N,N,16] -->[N,16]
            pool_h.append(curr_gcn_pool)

        # pool_h: [batch,16]: a pooled tensor Pi for each person
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h
