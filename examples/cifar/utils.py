from __future__ import print_function
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
import bagua.torch_api as bagua


def get_optimizer_algorithm(model, args, optimizer = None):
    if optimizer is None:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    if args.algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif args.algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif args.algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif args.algorithm == "qadam":
        from bagua.torch_api.algorithms import q_adam
        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100, rank=bagua.get_rank()
        )
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    elif args.algorithm == "qgadam":
        from bagua.torch_api.algorithms import q_adam
        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = q_adam.QGAdamAlgorithm(optimizer)
    elif args.algorithm == "floatgrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.Float16GradAlgorithm()
    elif args.algorithm == "qgadam_low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized
        optimizer = decentralized.QGAdamOptimizer(
            model.parameters(), lr=args.lr
        )
        algorithm = decentralized.QGAdamLowPrecisionDecentralizedAlgorithm(optimizer)
    elif args.algorithm == "qadam_low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized
        from bagua.torch_api.algorithms import q_adam
        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = decentralized.QGAdamLowPrecisionDecentralizedAlgorithm(optimizer)
    elif args.algorithm == "gradient_allreduce_sketch":
        from bagua.torch_api.algorithms import gradient_allreduce
        algorithm = gradient_allreduce.GradientAllReduceSketchAlgorithm(optimizer)
    elif args.algorithm == "sparsepy":
        import sys
        sys.path.append("../mnist/sparsepy")
        import sparsepy
        algorithm = sparsepy.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-allgather":
        import sys
        sys.path.append("../mnist/sparsepy")
        import sparsepy_allgather
        algorithm = sparsepy_allgather.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-allgather-full":
        import sys
        sys.path.append("../mnist/sparsepy")
        import sparsepy_allgather_full
        algorithm = sparsepy_allgather_full.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-test":
        import sys
        sys.path.append("../mnist/sparsepy")
        import sparsepy
        sys.path.append("../mnist")
        from gradient_allreduce import SimpleOptimizer
        optimizer = SimpleOptimizer(model.parameters(), lr=args.lr)
        algorithm = sparsepy.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-simple":
        import sys
        sys.path.append("../mnist/sparsepy")
        import sparsepy_simple
        algorithm = sparsepy_simple.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "test":
        import sys
        sys.path.append("../mnist")
        from gradient_allreduce import GradientAllReduceAlgorithm
        algorithm = GradientAllReduceAlgorithm()
    elif args.algorithm == "test_simple":
        import sys
        sys.path.append("../mnist")
        from gradient_allreduce import GradientAllReduceAlgorithm, SimpleOptimizer
        optimizer = SimpleOptimizer(model.parameters(), lr=args.lr)
        algorithm = GradientAllReduceAlgorithm()
    else:
        raise NotImplementedError
    return (optimizer, algorithm)

def get_cifar(data_dir, train_kwargs, test_kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if bagua.get_local_rank() == 0:
        trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )

    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )

    # train_kwargs.update(
    #     {
    #         "sampler": train_sampler,
    #     }
    # )
    train_kwargs["sampler"] = train_sampler

    trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    return (trainloader, testloader)
