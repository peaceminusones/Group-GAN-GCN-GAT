import torch
import torch.nn as nn


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
