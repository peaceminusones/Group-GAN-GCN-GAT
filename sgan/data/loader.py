# 数据加载部分

from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset, seq_collate
from sgan.data.trajectories_GCN import TrajectoryDataset, seq_collate


def data_loader(args, path):
    # 创建一个Dataset对象，Dataset是一个代表着数据集的抽象类，
    # 所有关于数据集的类都可以定义成其子类，只需要重写__gititem__函数和__len__函数即可。
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    # 创建一个DataLoader对象，由于定义好的数据集不能都存放在内存中，否则内存会爆表，所以需要定义一个迭代器，
    # 每一步生成一个batch,这就是DataLoader的作用，其能够为我们自动生成一个多线程的迭代器，只要传入几个参数即可，
    # 例如batchsize的大小，数据是否打乱等。
    loader = DataLoader(
        dset,                                    # 传入的数据集，即TrajectoryDataset准备好的数据集
        batch_size=args.batch_size,              # 每个batch中有多少样本，默认64
        shuffle=True,                            # 是否将数据打乱
        num_workers=args.loader_num_workers,     # 处理数据加载的进程数
        collate_fn=seq_collate)                  # 将一个列表中的样本组成一个mini-batch的函数

    return dset, loader
