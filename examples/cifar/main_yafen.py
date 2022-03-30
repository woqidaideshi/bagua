'''Train CIFAR10 with PyTorch.'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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

import os
import argparse
import sys
import time
from bagua.torch_api import algorithms
from resnet import *
from vgg import *
import utils
# Training
def train(epoch, net, trainloader, optimizer, criterion):
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #print(targets.tolist())

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, net, testloader, criterion):
    #global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    return (acc, test_loss)

def main_worker(args):
    log_file = args.log_file
    writer = open(log_file, 'w')

    for k in vars(args):
        writer.write("[params] " + str(k) + " = " + str(getattr(args, k)) + '\n')

    writer.flush()

    writer.write('[%s] Start iteration' % get_current_time())
    writer.write('\n')

    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch

    data_dir = args.data_dir
    download = args.download
    num_workers = args.num_workers
    epochs = args.epochs
    lr = args.lr
    saving = args.saving
    model_name = args.model_name
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size

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
    bagua.init_process_group()

    # Data
    print('==> Preparing data..')
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
            root=data_dir, train=True, download=download, transform=transform_train
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=transform_train
        )

    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform_test)


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # Model
    writer.write('==> Building model..\n')

    #net = ResNet18()
    if (model_name == 'ResNet18'):
        net = ResNet18()
    elif (model_name == 'ResNet50'):
        net = ResNet50()
    elif (model_name == 'VGG19'):
        net = VGG('VGG19')
    elif (model_name == 'VGG16'):
        net = VGG('VGG16')

    model = net.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr,
    #                     momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=lr,
    #                     weight_decay=5e-4)

    optimizer, algorithm = utils.get_optimizer_algorithm(model, args)

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

    print('[%s] Start training' % get_current_time())

    for epoch in range(start_epoch, start_epoch + epochs):
        start = time.time()
        train(epoch, model, trainloader, optimizer, criterion)
        grad_end = time.time()
        (acc, test_loss) = test(epoch, model, testloader, criterion)
        loss_end = time.time()
        
        exec_t = loss_end - start
        grad_t = grad_end - start
        loss_t = exec_t - grad_t

      
        if saving == True and acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        scheduler.step()

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

        writer.write('[%s] [Iter %2d] Loss = %.2f, acc = %.2f, exec_t = %.2fs, grad_t = %.2fs, loss_t = %.2fs' % 
            (get_current_time(), epoch, test_loss, acc, round(exec_t, 2),
			round(grad_t, 2), round(loss_t, 2)))
        writer.write('\n')
        writer.flush()

        if acc > max_accuracy:
            max_accuracy = acc


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

def get_current_time() :
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_current_time_filename():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    log_base_dir = '../log/'
    data_dir = '../data/'
    log_dir = 'train_log_cifar_bench_50_adam'

    model_name = 'ResNet18'
    #model_name = 'VGG19'

    data_name = 'cifar10'

    parser = argparse.ArgumentParser(description="PyTorch Cifar Example")
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
        default=50,
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
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--lr-decay",
        default=0.95,
        type=float,
        help="decay of learning rate",
    )
    args = parser.parse_args()
    args.model_name = model_name
    args.resume = False
    args.data_dir = data_dir
    args.download = True
    args.saving = False

    log_txt =  data_name + '_lr' + str(args.lr) + '_' + get_current_time_filename() + '.txt'
    outdir = os.path.join(log_base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(args.batch_size), args.algorithm)
    log_file = os.path.join(outdir, log_txt)
    args.log_file = log_file
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    main_worker(args)

if __name__ == '__main__':
    main()
