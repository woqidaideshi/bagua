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
import time

def train(args, model, train_loader, optimizer, epoch, criterion=None, rank=0):
    model.train()
    train_loss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if criterion:
            loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target)
        loss.backward()
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        train_loss += loss.item()
        # if batch_idx % args.log_interval == 0:
        #     logging.info(
        #         "Train Rank: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}.".format(
        #             rank,
        #             epoch,
        #             batch_idx * len(data),
        #             len(train_loader.dataset),
        #             100.0 * batch_idx / len(train_loader),
        #             loss.item()
        #         )
        #     )
    end = time.time()
    # time_delta = end - start
    # logging.info(
    #     "Train Rank: {} Epoch: {} [{}/{}]\tLoss: {:.6f} time: {}s.".format(
    #         rank,
    #         epoch,
    #         batch_idx * len(data),
    #         len(train_loader.dataset),
    #         train_loss,
    #         time_delta,
    #     )
    # )
    return (start, end)

def test(model, test_loader, criterion=None, rank=0):
    model.eval()
    test_loss = 0
    correct = 0
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

    # logging.info(
    #     "\nTest Rank: {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%).\n".format(
    #         rank,
    #         test_loss/len(test_loader.dataset),
    #         correct,
    #         len(test_loader.dataset),
    #         100.0 * correct / len(test_loader.dataset)
    #     )
    # )
    end = time.time()
    return (test_loss, correct, end)

def get_current_time_filename():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_current_time() :
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

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
        default=256,
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

    model = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer, algorithm = utils.get_optimizer_algorithm(model, args)

    log_name = "cifar_" + args.algorithm + "_lr" + str(args.lr) + "_gpu" + str(bagua.get_world_size()) + "_rank" + str(bagua.get_rank()) + "_" + get_current_time_filename() + ".txt"
    outdir = os.path.join("./log", "sparsepy", "cifar", args.algorithm)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_file = os.path.join(outdir, log_name)
    
    writer = open(log_file, "w")

    for arg in vars(args):
        writer.write("[params] " + str(arg) + " = " + str(getattr(args, arg)) + "\n")

    train_tuple_num = len(train_loader.dataset)
    val_tuple_num = len(test_loader.dataset)
    writer.write('[Computed] train_tuple_num = %d, val_tuple_num = %d' % (train_tuple_num, val_tuple_num))
    writer.write('\n')

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    writer.write('[%s] Start iteration' % get_current_time())
    writer.write('\n')

    avg_exec_t = 0.0
    avg_grad_t = 0.0
    avg_loss_t = 0.0

    first_exec_t = 0.0
    first_grad_t = 0.0
    first_loss_t = 0.0

    second_exec_t = 0.0
    second_grad_t = 0.0
    second_loss_t = 0.0

    max_accuracy = 0.0

    epochs = int(args.epochs)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, epochs + 1):
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        (start_time, grad_end_t) = train(args, model, train_loader, optimizer, epoch, criterion=criterion, rank=bagua.get_rank())

        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        (epoch_loss, epoch_acc, loss_end_t) = test(model, test_loader, criterion=criterion, rank=bagua.get_rank())
        scheduler.step()

        exec_t = loss_end_t - start_time
        grad_t = grad_end_t - start_time
        loss_t = exec_t - grad_t

        avg_exec_t += exec_t
        avg_grad_t += grad_t
        avg_loss_t += loss_t

        if epoch == 1:
            first_exec_t = exec_t
            first_grad_t = grad_t
            first_loss_t = loss_t
        elif epoch == 2:
            second_exec_t = exec_t
            second_grad_t = grad_t
            second_loss_t = loss_t

        accuracy = epoch_acc / val_tuple_num * 100

        writer.write('[%s] [Epoch %2d] Loss = %.2f, acc = %.2f, exec_t = %.2fs, train_t = %.2fs, test_t = %.2fs' % 
            (get_current_time(), epoch, epoch_loss, accuracy, round(exec_t, 2),
			round(grad_t, 2), round(loss_t, 2)))
        writer.write('\n')
        writer.flush()

        if accuracy > max_accuracy:
            max_accuracy = accuracy

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    writer.write('[%s] [Finish] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
            (get_current_time(), avg_exec_t / epochs,
            avg_grad_t / epochs, avg_loss_t / epochs))
    writer.write('\n')

    if epochs > 2:
        avg_exec_t -= first_exec_t
        avg_grad_t -= first_grad_t
        avg_loss_t -= first_loss_t

        writer.write('[%s] [-first] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
                (get_current_time(), avg_exec_t / (epochs - 1),
				avg_grad_t / (epochs - 1), avg_loss_t / (epochs - 1)))
        writer.write('\n')
		
        avg_exec_t -= second_exec_t
        avg_grad_t -= second_grad_t
        avg_loss_t -= second_loss_t

        writer.write('[%s] [-1 & 2] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
                (get_current_time(), avg_exec_t / (epochs - 2),
                avg_grad_t / (epochs - 2), avg_loss_t / (epochs - 2)))
        writer.write('\n')

        writer.write('[%s] [MaxAcc] max_accuracy = %.2f' % 
				(get_current_time(), max_accuracy))
        writer.write('\n')

    end = datetime.now()
    logging.info("----test main end time: {} (rank {}).".format(end.strftime("%Y-%m-%d %H:%M:%S.%f"), bagua.get_rank()))
    time_delta = (end - start).seconds
    logging.info("----running time: {}s (rank {}).".format(time_delta, bagua.get_rank()))


if __name__ == "__main__":
    main()
