from __future__ import print_function
import argparse
from typing import Tuple
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
from datetime import datetime
import os
import utils
from resnet import *
from vgg import *

def train(args, model, train_loader, optimizer, epoch, criterion=None, rank=0):
    model.train()
    train_loss = 0
    start_epoch = datetime.now()
    ranks = bagua.get_world_size()
    # logging.info("-----------train, rank{}, train_loader length: {}".format(rank, len(train_loader)))
    # print("rank: %d, epoch: %d, all datasize: %d, args.batch_size: %d." % (rank, epoch, len(train_loader), args.batch_size))
    for batch_idx, (data, target) in enumerate(train_loader):
        # logging.info("-----------train, rank{}, train_data length: {}".format(rank, len(data)))
        start_batch = datetime.now()
        # print("rank: %d, epoch: %d, batch index: %d, datasize: %d, args.batch_size: %d" % (rank, epoch, batch_idx, len(data), args.batch_size))
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if criterion:
            loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target)
        # print("----pre backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        # print("----pre backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        loss.backward()
        # print("----post backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        # print("----post backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        train_loss += loss.item()
        # print("----post optimizer.step batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        # print("----post optimizer.step batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        end_batch = datetime.now()
        time_delta = (end_batch - start_batch).seconds
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Rank: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time: {}s.".format(
                    rank,
                    epoch,
                    batch_idx * len(data) * ranks,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    time_delta,
                )
            )
    end_epoch = datetime.now()
    time_delta = (end_epoch - start_epoch).seconds
    logging.info(
        "Train Rank: {} Epoch: {} [{}/{}]\tLoss: {:.6f} time: {}s.".format(
            rank,
            epoch,
            batch_idx * len(data) * ranks,
            len(train_loader.dataset),
            train_loss,
            time_delta,
        )
    )

def test(model, test_loader, criterion=None, rank=0):
    model.eval()
    test_loss = 0
    correct = 0
    # total = 0
    start_epoch = datetime.now()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            if criterion:
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target, reduction="sum")
            # test_loss += F.nll_loss(
            #     output, target, reduction="sum"
            # ).item()  # sum up batch loss
            test_loss += loss.item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # total += target.size(0)
            # _, pred = output.max(1)
            # correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    # test_loss /= total
    end_epoch = datetime.now()
    time_delta = (end_epoch - start_epoch).seconds
    logging.info(
        "\nTest Rank: {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) time: {}s.\n".format(
            rank,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            time_delta,
        )
    )

def main():
    # Training settings
    start = datetime.now()
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ResNet18",
        help="model name",
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
        default=256,
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
        default=0.001,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gradient_allreduce",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async",
    )
    parser.add_argument(
        "--async-sync-interval",
        default=500,
        type=int,
        help="Model synchronization interval(ms) for async algorithm",
    )
    parser.add_argument(
        "--set-deterministic",
        action="store_true",
        default=False,
        help="set deterministic or not",
    )
    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )

    args = parser.parse_args()
    print("set_deterministic:", args.set_deterministic)
    if args.set_deterministic:
        print("set_deterministic: True")
        np.random.seed(666)
        random.seed(666)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666 + int(bagua.get_rank()))
        torch.set_printoptions(precision=10)

    torch.cuda.set_device(bagua.get_local_rank())
    print("current rank: ", bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.INFO)
    logging.info("----main start time: {} (rank {}).".format(start.strftime("%Y-%m-%d %H:%M:%S.%f"), bagua.get_rank()))

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    train_kwargs.update(
        {
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    train_loader, test_loader = utils.get_cifar("../data", train_kwargs, test_kwargs)

    #net = ResNet18()
    if (args.model_name == 'ResNet18'):
        net = ResNet18()
    elif (args.model_name == 'ResNet50'):
        net = ResNet50()
    elif (args.model_name == 'VGG19'):
        net = VGG('VGG19')
    elif (args.model_name == 'VGG16'):
        net = VGG('VGG16')
    elif (args.model_name == 'VGG13'):
        net = VGG('VGG13')
    elif (args.model_name == 'VGG11'):
        net = VGG('VGG11')

    model = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer, algorithm = utils.get_optimizer_algorithm(model, args)

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        train(args, model, train_loader, optimizer, epoch, criterion=criterion, rank=bagua.get_rank())

        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        test(model, test_loader, criterion=criterion, rank=bagua.get_rank())
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = datetime.now()
    logging.info("----test main end time: {} (rank {}).".format(end.strftime("%Y-%m-%d %H:%M:%S.%f"), bagua.get_rank()))
    time_delta = (end - start).seconds
    logging.info("----running time: {}s (rank {}).".format(time_delta, bagua.get_rank()))


if __name__ == "__main__":
    main()
