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

    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 2*16
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        # input_size: 16
        # hidden_size: 32
        # num_layers: 1
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
        # Encode observed Trajectory (batch即batch_size个sequence序列中的总人数
        batch = obs_traj.size(1)
        '''经过一个2*16的全连接层，全连接层的输入的shape:[obs_len*batch,2]，输出：[obs_len*batch,16]'''
        # shape:
        # "obs_traj":                          [obs_len,batch,2]
        # "obs_traj.contiguous().view(-1, 2)": [obs_len*batch,2]
        # "obs_traj_embedding":                [obs_len*batch,16]
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        # 经过维度变换变成3维的以符合LSTM网络中输入input的格式要求
        # "obs_traj_embedding":                [obs_len,batch,16]
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        # lstm模块初始化h_0, c_0
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
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        # mlp [2,16]
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        # lstm
        # input_size: 16
        # hidden_size: 32
        # num_layers: 1
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        # mlp [32,2]
        self.hidden2pos = nn.Linear(h_dim, 2)

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

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
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
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


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""

    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim  # 16

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]  # mlp_pre_pool_dims: [48,512,8]

        # mlp: 2*16
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        # mlp: 48*512*8
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

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

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim) 即encoder的return：final_h
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            # print("num_ped:", num_ped)
            # print("h_states:", h_states.shape)

            # h_states == final_h (即这里h_states就是LSTM的输出)
            # h_states([1,batch,32])  ->  cur_hidden([N,32])
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            # print("curr_hidden: ", curr_hidden.shape)

            # Repeat -> H1, H2, H1, H2
            # curr_hidden([N,32])  ->  curr_hidden_1([N*N,32])
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # print("curr_hidden_1: ", curr_hidden_1.shape)

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
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            # curr_pool_h: [N*N,8]
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            # curr_pool_h: [N,8]
            # print(curr_pool_h.view(num_ped, num_ped, -1)[0])
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]  # [N,N,8] -->[n,8]
            # print(curr_pool_h)
            # print("curr_pool_h:", curr_pool_h.shape)
            pool_h.append(curr_pool_h)

        # pool_h: [batch,8]: a pooled tensor Pi for each person
        pool_h = torch.cat(pool_h, dim=0)
        # print("pool_h:", pool_h.shape)
        return pool_h


class GCN(nn.Module):
    """GCN module"""

    def __init__(self, input_dim=48, hidden_dim=72, out_dim=8, gcn_layers=2):
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


class GCNModule(nn.Module):
    """group information aggregation with GCN layer"""

    def __init__(
        self, input_dim=40, hidden_dim=72, out_dim=16, gcn_layers=2, final_dim=24
    ):
        super(GCNModule, self).__init__()

        # GCN_intra: 40*72*16
        self.gcn_intra = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            gcn_layers=gcn_layers)
        # GCN_inter: 16*72*16
        self.gcn_inter = GCN(
            input_dim=16,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            gcn_layers=gcn_layers)

        # mlp:16*8
        self.out_embedding = nn.Linear(out_dim*2, final_dim)

    # def normalize(self, adj, dim):
    #     N = adj.size()
    #     adj2 = torch.sum(adj, dim)       # 对每一行求和
    #     norm = adj2.unsqueeze(1)         # 扩展张量维度
    #     norm = norm.repeat(1, N[0],  1)  # 沿着指定维度重复张量
    #     norm = norm.permute(2, 0, 1)     # 转置
    #     norm = norm.pow(-1)              # 求倒数
    #     norm_adj = adj.mul(norm)         # 点乘
    #     return norm_adj
    def normalize(self, adj, dim):
        N = adj.size()
        adj2 = torch.sum(adj, dim)       # 对每一行求和
        norm = adj2.unsqueeze(1).float()         # 扩展张量维度
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
        - h_states: Tensor of shape (batch, h_dim) 即encoder+pooling net的return
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        - end_group: group labels at the last time step (t_obs); shape: (batch, 1)
        Output:
        - gcn_aggre: Tensor of shape (batch, bottleneck_dim)
        """
        gcn_aggre = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start  # num_ped: number of pedestrians in the scene
            # curr_state: [N,40]
            curr_state = h_states[start:end]

            # get the modulated adjacency matrix arrays
            # Generate masks from the group labels
            # labels can only be used to distinguish groups at a timestep.
            # var: end_group; def: group labels at the last time step (t_obs); shape: (batch, 1)
            # clip one onservation-prediction window out of multiple windows.
            curr_end_group = end_group[start:end]
            # get the coherency adjacency, dimension: (N, N)
            # coherency mask is shared by all pedestrians in the scene
            eye_mtx = torch.eye(num_ped, device=end_group.device).bool()
            A_g = curr_end_group.repeat(1, num_ped)
            B_g = curr_end_group.transpose(1, 0).repeat(num_ped, 1)
            # M_intra: [N,N]
            M_intra = (A_g == B_g) & (A_g != 0) | eye_mtx
            # get the modulated normalized adjacency matrix arrays
            # normalized M_intra: [N,N]
            A_intra = self.normalize(M_intra, dim=1).cuda()

            """gcn_intra"""
            # curr_gcn_state_intra: [N,16] (GCN:[40,72,16])
            curr_gcn_state_intra = self.gcn_intra(A_intra, curr_state)

            """GPool =================================================================="""
            # M_intra: [N,N]
            # R_intra_unique: [M,N]
            R_intra_unique = torch.unique(M_intra, sorted=False, dim=0)
            # group 的数量
            n_group = R_intra_unique.size()[0]
            R_intra_unique.unsqueeze_(1)  # 增加一维
            # 从下到上翻转R_intra_unique
            R_intra = []
            for i in range(n_group-1, -1, -1):
                R_intra.append(R_intra_unique[i])
            R_intra = torch.cat(R_intra, dim=0)
            # 归一化
            R_intra = self.normalize(R_intra, dim=1).cuda()
            # 提取群组部分 [M,N]*[N,16]
            # curr_gcn_group_state: [M,16]
            curr_gcn_group_state_in = torch.matmul(R_intra, curr_gcn_state_intra)
            """=========================================================================="""

            """gcn_inter"""
            # M_inter: [M,M]
            M_inter = torch.ones((n_group, n_group), device=end_group.device).bool()
            # normalize
            A_inter = self.normalize(M_inter, dim=1).cuda()
            # M_inter_norm: [M,M]
            # curr_gcn_group_state_in: [M,16]  (GCN:[16,72,16])
            # curr_gcn_group_state_out: [M,16]
            curr_gcn_group_state_out = self.gcn_inter(A_inter, curr_gcn_group_state_in)

            """GUnpool================================================================="""
            # [N,M]*[M,16]
            # curr_gcn_state_inter: [N,16]
            curr_gcn_state_inter = torch.matmul(R_intra.T, curr_gcn_group_state_out)
            """========================================================================="""

            # curr_gcn_state: [N,32]
            curr_gcn_state = torch.cat([curr_gcn_state_intra, curr_gcn_state_inter], dim=1)

            # curr_gcn_state: [N,24]
            curr_gcn_state = self.out_embedding(curr_gcn_state)

            gcn_aggre.append(curr_gcn_state)

        # gcn_aggre: [batch,24]:
        gcn_aggre = torch.cat(gcn_aggre, dim=0)
        return gcn_aggre


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim is None:
            self.noise_dim = None
        elif self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.gcn_module = GCNModule(
            input_dim=input_dim,
            hidden_dim=72,
            out_dim=16,
            gcn_layers=2,
            final_dim=decoder_h_dim - self.noise_first_dim
        )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    # modified by zyl 2021/1/12
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_traj_g, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)

            # Construct input hidden states for decoder
            # final_encoder_h: [batch, 32]
            # pool_h: [batch, 8]
            # mlp_decoder_context_input: [batch, 40]
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            # noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
            end_pos = obs_traj[-1, :, :]
            # modified by zyl 2021/1/12 9:56
            end_group = obs_traj_g[-1, :, :]
            noise_input = self.gcn_module(mlp_decoder_context_input, seq_start_end, end_pos, end_group)
        else:
            noise_input = mlp_decoder_context_input

        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        # self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])
        scores = self.real_classifier(classifier_input)
        return scores
