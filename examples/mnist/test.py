from importlib.resources import path
from tkinter.messagebox import NO
from numpy import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import multiprocessing
import bagua.torch_api as bagua
import sys
import argparse
import logging
sys.path.append("../../tests/torch_api")
from test_low_precision_decentralized import LowPrecDecentralizedAlgor, _init_torch_env
from main import train, test, Net

def run_torch_model(
    rank, args, hierarchical, communication_interval
):
    if rank == 0:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    # logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    nprocs = args.nprocs
    _init_torch_env(rank, nprocs, args.backend)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if rank == 0:
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )

    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1, num_replicas=nprocs, rank=rank
    )
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": train_kwargs["batch_size"],
            "shuffle": False,
        }
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    model = LowPrecDecentralizedAlgor(model, optimizer, hierarchical, communication_interval, compressor=args.compressor)

    for epoch in range(1, args.epochs+1):
        # print("rank %d, epoch_index: %d" % (rank, epoch))
        train(args, model, train_loader, optimizer, epoch, rank=rank)
        test(model, test_loader, rank=rank)

def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example in Python")
    parser.add_argument(
        "--nprocs",
        type=int,
        default=4,
        metavar="N",
        help="number of process (default: 4)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="communication backends (default: nccl)",
    )    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--compressor",
        type=str,
        default="float16",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async",
    )

    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )

    args = parser.parse_args()

    print("---", args)

    # nprocs = torch.cuda.device_count()
    nprocs = args.nprocs
    hierarchical = True
    communication_interval = 1

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    # logging.getLogger().setLevel(logging.INFO)

    logging.info("----test main---------")
    print("nproc: ", nprocs)

    mp = multiprocessing.get_context("spawn")
    processes = []
    for i in range(nprocs):
        p = mp.Process(
            target=run_torch_model,
            args=(
                i,
                args,
                hierarchical,
                communication_interval,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=60)

if __name__ == "__main__":
    main()
