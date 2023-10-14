import torch
from torch.utils import data
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler

from .general_eval import MVSDataset as EvalDataset
from .dtu_cl import MVSDataset as DtuCLDataset


def get_loader(args, listfile, mode="train"):
    if args.dataset_name == "dtu_cl":
        dataset = DtuCLDataset(args, listfile, mode)
    elif args.dataset_name == "general_eval":
        dataset = EvalDataset(args, listfile, mode)
    else:
        raise NotImplementedError("Don't support dataset: {}".format(args.dataset_name))

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = RandomSampler(dataset) if (mode == "train") else SequentialSampler(dataset)

    data_loader = data.DataLoader(dataset, args.batch_size, sampler=sampler, num_workers=4, drop_last=(mode == "train"), pin_memory=True)

    return data_loader, sampler
