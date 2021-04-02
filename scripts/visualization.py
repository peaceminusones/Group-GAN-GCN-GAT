import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import animation

import sys
sys.path.append(".")
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

total_aa,total_bb=[],[]

xdata0, ydata0 = [], []
xdata1, ydata1 = [], []
xdata2, ydata2 = [], []
xdata3, ydata3 = [], []
# xdata4, ydata4 = [], []

xdatap0, ydatap0 = [np.zeros(20) for row in range(20)], [np.zeros(20) for row in range(20)]
xdatap1, ydatap1 = [np.zeros(20) for row in range(20)], [np.zeros(20) for row in range(20)]
xdatap2, ydatap2 = [np.zeros(20) for row in range(20)], [np.zeros(20) for row in range(20)]
xdatap3, ydatap3 = [np.zeros(20) for row in range(20)], [np.zeros(20) for row in range(20)]
# xdatap4, ydatap4 = [np.zeros(20) for row in range(20)], [np.zeros(20) for row in range(20)]

aa, bb = [], []

def ploot():

    # ground truth
    for t in range(20):  # 预测的时间长度T
        xdata0.append(total_aa[0][:,0][t])
        ydata0.append(total_aa[0][:,1][t])

        xdata1.append(total_aa[1][:,0][t])
        ydata1.append(total_aa[1][:,1][t])

        xdata2.append(total_aa[2][:,0][t])
        ydata2.append(total_aa[2][:,1][t])

        xdata3.append(total_aa[3][:,0][t])
        ydata3.append(total_aa[3][:,1][t])

        # xdata4.append(total_aa[4][:,0][t])
        # ydata4.append(total_aa[4][:,1][t])
    
    plt.plot(xdata0, ydata0, 'y--', linewidth=3)
    plt.plot(xdata1, ydata1, 'g--', linewidth=3)
    plt.plot(xdata2, ydata2, 'r--', linewidth=3)
    plt.plot(xdata3, ydata3, 'c--', linewidth=3)
    # plt.plot(xdata4, ydata4, 'y--', linewidth=3)

    # predicted traj
    for num in range(20):  # 10个轨迹
        for t in range(20):  # 20帧
            xdatap0[num][t] = total_bb[0][num][:,0][t]
            ydatap0[num][t] = total_bb[0][num][:,1][t]
            
            xdatap1[num][t] = total_bb[1][num][:,0][t]
            ydatap1[num][t] = total_bb[1][num][:,1][t]
            
            xdatap2[num][t] = total_bb[2][num][:,0][t]
            ydatap2[num][t] = total_bb[2][num][:,1][t]

            xdatap3[num][t] = total_bb[3][num][:,0][t]
            ydatap3[num][t] = total_bb[3][num][:,1][t]
            
            # xdatap4[num][t] = total_bb[4][num][:,0][t]
            # ydatap4[num][t] = total_bb[4][num][:,1][t]

        plt.plot(xdatap0[num], ydatap0[num], 'y:')
        plt.plot(xdatap1[num], ydatap1[num], 'g:')
        plt.plot(xdatap2[num], ydatap2[num], 'r:')
        plt.plot(xdatap3[num], ydatap3[num], 'c:')
        # plt.plot(xdatap4[num], ydatap4[num], 'y:')
    
    plt.show()
    # plt.close()


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    
    n_units = (
        [40]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [40]
    )
    
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        n_units=n_units,
        n_heads=args.n_heads,
        dropout1=args.dropout1,
        alpha=args.alpha).cuda()
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate(args, loader, generator, num_samples):
    with torch.no_grad():
        for idf, batch in enumerate(loader):
            if idf<=3:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_traj_rel_v, pred_traj_rel_v, obs_traj_g, pred_traj_g,
                non_linear_ped, loss_mask, seq_start_end) = batch

                for idi, (start, end) in enumerate(seq_start_end):
                    if(idi<=3):
                        for i in range(start, end): # num_samples
                            # ground truth
                            gt=(pred_traj_gt[:,i,:].data).cpu()
                            # observed traj
                            input_a=(obs_traj[:,i,:].data).cpu()
                            # observed + ground truth
                            aa=np.concatenate((input_a, gt),axis=0)

                            # 10 个 predicted traj
                            out_a=[]
                            for num in range(20):
                                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_traj_g)
                                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                                # predicted traj
                                out_a.append((pred_traj_fake[:,i,:].data).cpu())
                            # 10 个 observed + predicted traj
                            bb=[]
                            for num in range(20):
                                bb.append(np.concatenate((input_a, out_a[num]),axis=0))

                            global x0, y0, x1, y1, total_aa, total_bb
                            # 多来几个人
                            total_aa.append(aa)
                            total_bb.append(bb)


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        evaluate(_args, loader, generator, args.num_samples)
        ploot()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)