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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print("--------Net forward begin.")
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        print("--------Net forward end.")
        return output


def train(args, model, train_loader, optimizer, epoch, rank=0):
    model.train()
    ranks = bagua.get_world_size()
    # logging.info("-----------train, rank{}, train_loader length: {}".format(rank, len(train_loader)))
    # print("rank: %d, epoch: %d, all datasize: %d, args.batch_size: %d." % (rank, epoch, len(train_loader), args.batch_size))
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("rank: %d, epoch: %d, batch index: %d, datasize: %d, args.batch_size: %d" % (rank, epoch, batch_idx, len(data), args.batch_size))
        # logging.info("-----------train, rank{}, train_data length: {}".format(rank, len(data)))
        data, target = data.cuda(), target.cuda()
        # if isinstance(optimizer, list):
        #     for opt in optimizer:
        #         opt.zero_grad()
        # else:
        #     optimizer.zero_grad()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # if args.algorithm == "sketch":
        #     print("----pre backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][-1].grad[0:10]))
        #     for tensor in optimizer.param_groups[0]["params"]:
        #         if tensor.is_bagua_tensor():
        #             print("----pre backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, tensor.newgrad))
        # elif "sketch" in args.algorithm:
        #     print("----pre backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        #     print("----pre backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        # else:
        #     print("----pre backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        loss.backward()
        # size_grad = 0
        # size_grad_nonzero = 0
        # if args.algorithm == "sparse":
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             if param.requires_grad:
        #                 size_grad += param.grad.numel()
        #                 size_grad_nonzero += param.grad.count_nonzero().item()
        # logging.info("-----------after backward000: {}/{}".format(size_grad_nonzero, size_grad))
        # if args.algorithm == "sparse":
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             # logging.info("-------hasattr(param, sketch_grad): {}".format(hasattr(param, "sketch_grad")))
        #             # logging.info("-------hasattr(param, grad): {}".format(hasattr(param, "grad")))
        #             if hasattr(param, "grad"):
        #                 assert hasattr(param, "sketch_grad")
        #                 param.grad.set_(param.sketch_grad)
        #                 # logging.info("-----------why: {}".format(param.grad.count_nonzero().item()))
        #                 #param.grad.zero_()
        # size_grad = 0
        # size_grad_nonzero = 0
        # if args.algorithm == "sparse":
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             if param.requires_grad:
        #                 size_grad += param.grad.numel()
        #                 size_grad_nonzero += param.grad.count_nonzero().item()
        # logging.info("-----------after backward: {}/{}".format(size_grad_nonzero, size_grad))
        # if args.algorithm == "sketch":
        #     print("----post backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][-1].grad[0:10]))
        #     for tensor in optimizer.param_groups[0]["params"]:
        #         if tensor.is_bagua_tensor():
        #             print("----post backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, tensor.newgrad))
        # elif "sketch" in args.algorithm:
        #     print("----post backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        #     print("----post backward batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        # else:
        #     print("----post backward batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        # if args.fuse_optimizer:
        #     if isinstance(optimizer, list):
        #         for opt in optimizer:
        #             opt.fuse_step()
        #     else:
        #         optimizer.fuse_step()
        # else:
        #     if isinstance(optimizer, list):
        #         for opt in optimizer:
        #             opt.step()
        #     else:
        #         optimizer.step()
        optimizer.step()
        # size_grad = 0
        # size_grad_nonzero = 0
        # if args.algorithm == "sparse":
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             if param.requires_grad:
        #                 size_grad += param.grad.numel()
        #                 size_grad_nonzero += param.grad.count_nonzero().item()
        # logging.info("-----------after step: {}/{}".format(size_grad_nonzero, size_grad))
        # optimizer.step()
        # if args.algorithm == "sketch":
        #     print("----post optimizer.step batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][-1].grad[0:10]))
        #     for tensor in optimizer.param_groups[0]["params"]:
        #         if tensor.is_bagua_tensor():
        #             print("----post optimizer.step batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, tensor.newgrad))
        # elif "sketch" in args.algorithm:
        #     print("----post optimizer.step batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        #     print("----post optimizer.step batch_idx {} in cuda:{}: newgrad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].newgrad))
        # else:
        #     print("----post optimizer.step batch_idx {} in cuda:{}: grad---{}.".format(batch_idx, rank, optimizer.param_groups[0]["params"][0].grad[0:10]))
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Rank: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    rank,
                    epoch,
                    batch_idx * len(data) * ranks,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader, rank=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest Rank: {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            rank,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    start = datetime.now()
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
    logging.getLogger().setLevel(logging.INFO)
    logging.info("----main start time: {} (rank {}).".format(start.strftime("%Y-%m-%d %H:%M:%S.%f"), bagua.get_rank()))

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if bagua.get_local_rank() == 0:
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
        dataset1, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer1 = optim.Adadelta(model.parameters(), lr=args.lr+1)

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
            model.parameters(), lr=args.lr, warmup_steps= 10
        )
        algorithm = decentralized.QGAdamLowPrecisionDecentralizedAlgorithm(optimizer)
    elif args.algorithm == "gradient_allreduce_sketch":
        from bagua.torch_api.algorithms import gradient_allreduce
        algorithm = gradient_allreduce.GradientAllReduceSketchAlgorithm(optimizer)
    elif args.algorithm == "sketch":
        from bagua.torch_api.algorithms import sketch
        algorithm = sketch.SketchAlgorithm(optimizer)
    elif args.algorithm == "sketch-max":
        from sketch import SketchAlgorithm
        # NOTE: may switch to Adadelta optimizer and try it out.
        for m in model.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.do_sketching = False
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = SketchAlgorithm(optimizer, c=100, r=10, k=100, lr=0.001)
    elif args.algorithm == "qsparse":
        # import qsparselocal
        # # from bagua.torch_api.algorithms import qsparselocal
        # optimizer = qsparselocal.QSparseLocalOptimizer(
        #     model.parameters(), lr=args.lr, k=10
        # )
        # algorithm = qsparselocal.QSparseLocalAlgorithm(optimizer)
        import qsparselocal
        learning_rate = args.lr
        #Gap between synchronization rounds
        gap = 1

        optimizer = qsparselocal.QSparseLocalOptimizer(
            model.parameters(), lr=learning_rate, schedule = gap +1
        )
        algorithm = qsparselocal.QSparseLocalAlgorithm(optimizer)
    elif args.algorithm == "signum":
        import signum
        signum_optimizer = signum.SignumOptimizer(model.parameters(), lr=args.lr)
        algorithm = signum.SignumAlgorithm(signum_optimizer, False)
    elif args.algorithm == "sparse-sketch":
        import sparse
        # optimizer = sparse.SketchedSGD(optim.SGD(model.parameters(), lr=args.lr))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = sparse.SketchedModel(model)
        algorithm = sparse.SparseAlgorithm(optimizer=optimizer, c=1000, r=10, k=1000)
        # from bagua.torch_api.algorithms import gradient_allreduce
        # algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "marina":
        from marina import MarinaAlgorithm, MarinaOptimizer

        optimizer = MarinaOptimizer(
            model.parameters(), lr=args.lr
        )
        algorithm = MarinaAlgorithm(optimizer)
    elif args.algorithm == "test":
        from gradient_allreduce import GradientAllReduceAlgorithm
        algorithm = GradientAllReduceAlgorithm()
    elif args.algorithm == "gradient_allreduce_sgd":
        from bagua.torch_api.algorithms import gradient_allreduce
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "test_simple":
        from gradient_allreduce import GradientAllReduceAlgorithm, SimpleOptimizer
        optimizer = SimpleOptimizer(model.parameters(), lr=args.lr)
        algorithm = GradientAllReduceAlgorithm()
    elif args.algorithm == "sparsepy-simple":
        from sparsepy import sparsepy_simple
        algorithm = sparsepy_simple.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy":
        from sparsepy import sparsepy
        algorithm = sparsepy.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-allgather":
        from sparsepy import sparsepy_allgather
        algorithm = sparsepy_allgather.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparsepy-allgather-full":
        from sparsepy import sparsepy_allgather_full
        algorithm = sparsepy_allgather_full.SparsepyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse":
        from bagua.torch_api.algorithms import sparse
        algorithm = sparse.SparseAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-test":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparseAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-test-inplace":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparseInplaceAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-py":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparsePyAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-py2":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparsePy2Algorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-test-independ":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparseIndependAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-py-independ":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparsePyIndependAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-py-cuda":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparsePyCudaAlgorithm(optimizer=optimizer)
    elif args.algorithm == "sparse-py-rust":
        from sparsepy import sparse_test
        algorithm = sparse_test.SparsePyRustAlgorithm(optimizer=optimizer)
    else:
        raise NotImplementedError

    model = model.with_bagua(
        [optimizer], # [optimizer, optimizer1],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)
        # optimizer1 = bagua.contrib.fuse_optimizer(optimizer1)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        # train(args, model, train_loader, [optimizer, optimizer1], epoch, rank=bagua.get_rank())
        train(args, model, train_loader, optimizer, epoch, rank=bagua.get_rank())

        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        test(model, test_loader, rank=bagua.get_rank())
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = datetime.now()
    logging.info("----test main end time: {} (rank {}).".format(end.strftime("%Y-%m-%d %H:%M:%S.%f"), bagua.get_rank()))
    time_delta = (end - start).seconds
    logging.info("----running time: {}s (rank {}).".format(time_delta, bagua.get_rank()))


if __name__ == "__main__":
    main()
