# 数据处理部分
'''
该文件主要对数据集进行了一定的处理，其重点在于TrajectoryDataset类的实现，
该类继承至torch.utils.data中的Dataset类，其主要完成的工作就是准备数据集。

其主要对原始的数据集进行预处理，原始的数据集共有4列，分为为frame id,ped id,x,y,我们要对这些数据进行处理，生成我们想要的数据。
'''

import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    '''
    seq_collate将batch_size的数据重新打包，将这些数据打包成我们要需要的数据格式，以便送入网络进行训练。
    也就是说，传入seq_collate的是batch_size个数据，每个数据对应一个序列。
    注意：这里的数据并不是TrajectoryDataset中准备的全部数据，而是仅仅batch_size个数据，seq_collate将这batch_size数据合并打包组成一个mini-batch。
          这个函数内部只是对TrajectoryDataset类中准备的数据再次进行了一定的加工处理，例如将维度进行了交换，
          [N,2,seq_len]→[seq_len,N,2]，其主要是为了和LSTM网络的输入格式保持一致。
    '''
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    判断轨迹是否线性。该函数的大致意思通过对预测轨迹进行最小二乘拟合，当拟合的残差大于一定阈值，认为轨迹不线性。
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
                 min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
                   我们可以修改TrajectoryDataset中的参数min_ped来控制一个序列中完整出现的人数。
                   当你想考虑较多人之间的交互，可以改大min_ped值，该值默认为1。
        - delim: Delimiter in the dataset files

        其主要对每个序列sequence进行处理，每个sequence的长度为seq_len=obs_len+pred_len,
        其主要是取出完整出现在这个序列seq_len个帧中的人的数据，
        并且每个序列中的完整出现的人的数量必须要大于其参数min_ped，程序默认是1。

        eg:
            因为完整出现在这个序列的人才一个，没有办法找到人与人之间的交互，对行人轨迹预测意义不大。
            而如果完整出现在个序列中的人数为3，那么这3个人的数据我们将都会保存，因为这里面可以考虑到人与人之间的交互关系。
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end)
                              for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        '''
        __init__函数最终得到下列数据
        num_ped                                            数据集当中一共有多少满足的人
        self.obs_traj       #shape[num_ped,2,obs_len]      这num_ped个人在obs_len个坐标数据
        self.pred_traj      #shape[num_ped,2,pred_len]     pred_traj都不是预测轨迹，而是预测轨迹的真值
        self.obs_traj_rel   #shape[num_ped,2,obs_len]      每一帧相对于上一帧的位置变化
        self.pred_traj_rel  #shape[num_ped,2,pred_len]
        self.loss_mask      #shape[num_ped,seq_len]
        self.non_linear_ped #shape[num_ped]                表示这个人的轨迹是否线性，其是通过调用trajectories.py文件中的poly_fit函数返回是否线性的标志位
        self.seq_start_end                                 self.seq_start_end其是一个元组列表，其长度表示一共有多少满足条件的序列
                                                           eg: 举个例子，假设在所给数据集中一共有5个序列满足完整出现的人数大于min_ped，
                                                               且这5个序列分别有2,3,2,4,3个人完整出现，那么self.seq_start_end的长度为5，
                                                               self.seq_start_end等于[(0,2),(2,5),(5,7),(7,11),(11,14)]，也就是说num_ped=14,
                                                               self.seq_start_end的主要作用是为了以后一个一个序列的分析的方便，即由要分析的序列，
                                                               即可根据它的值得到对应在这个序列中有哪几个人以及这几个人的所有相关数据。
        '''
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
