import sys
sys.path.append(".")
from sgan.utils import relative_to_abs, get_dset_path
from sgan.losses import displacement_error, final_displacement_error
from sgan.models import TrajectoryGenerator
from sgan.data.loader import data_loader
import argparse
import os
import torch

from attrdict import AttrDict


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])

    n_units = (
        [40]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [40]
    )
    # n_heads = [int(x) for x in args.heads.strip().split(",")]

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


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            # modified by zyl 2020/12/14 (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_traj_rel_v, pred_traj_rel_v, obs_traj_g, pred_traj_g,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                # modified by zyl 2020/12/14 pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_traj_g)
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_traj_g)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


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
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))
        
        # print parameters
        for k, v in checkpoint['args'].items():
            print(k, v)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# Dataset: zara1, Pred Len: 12, ADE: 0.42, FDE: 0.84
# dataset_name zara1
# delim
# loader_num_workers 4
# obs_len 8
# pred_len 12
# skip 1
# batch_size 64
# num_iterations 3628
# num_epochs 200
# embedding_dim 16
# num_layers 1
# dropout 0
# batch_norm 0
# mlp_dim 64
# encoder_h_dim_g 32
# decoder_h_dim_g 32
# noise_dim (8,)
# noise_type gaussian
# noise_mix_type global
# clipping_threshold_g 2.0
# g_learning_rate 0.0001
# g_steps 1
# pooling_type gcn
# pool_every_timestep 0
# bottleneck_dim 8
# neighborhood_size 2.0
# grid_size 8
# d_type global
# encoder_h_dim_d 48
# d_learning_rate 0.001
# d_steps 2
# clipping_threshold_d 0
# l2_loss_weight 1
# best_k 1
# output_dir D:\pedestrian analysis\group detection and prediction\GCN\sgan-master
# print_every 100
# checkpoint_every 300
# checkpoint_name checkpoint
# checkpoint_start_from None
# restore_from_checkpoint 1
# num_samples_check 5000
# use_gpu 1
# timing 0
# gpu_num 0