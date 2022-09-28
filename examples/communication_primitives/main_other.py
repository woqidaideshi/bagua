from __future__ import print_function
import argparse
from tkinter.messagebox import NO
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua
import time
import os

def init_env():
    torch.set_printoptions(precision=20)

    assert bagua.get_world_size() >= 1, "world size must be at least 1"

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()
    return comm

def gather():
    comm = init_env()
    # send, recv
    if bagua.get_rank() == 0:
        send_tensor = torch.ones(4, dtype=torch.float32).cuda() / 2
        recv_tensor = torch.zeros(4*bagua.get_world_size(), dtype=torch.float32).cuda()
        bagua.gather(send_tensor, recv_tensor, dst=0, comm=comm)
        print("gather, rank: {}, send_tensor: {}.".format(bagua.get_rank(), send_tensor))
        print("gather rank: {}, recv_tensor: {}.".format(bagua.get_rank(), recv_tensor))
    else:
        send_tensor = torch.ones(4, dtype=torch.float32).cuda() * bagua.get_rank()
        recv_tensor = torch.zeros(4*bagua.get_world_size(), dtype=torch.float32).cuda()
        bagua.gather(send_tensor, recv_tensor, dst=0, comm=comm)
        print("gather, rank: {}, send_tensor: {}.".format(bagua.get_rank(), send_tensor))
        print("gather rank: {}, recv_tensor: {}.".format(bagua.get_rank(), recv_tensor))

def gather_inplace():
    comm = init_env()
    if bagua.get_rank() == 0:
        recv_tensor = torch.ones(4*bagua.get_world_size(), dtype=torch.float32).cuda() / 2
        bagua.gather_inplace(recv_tensor, 4, dst=0, comm=comm)
        print("gather_inplace, rank: {}, recv_tensor: {}.".format(bagua.get_rank(), recv_tensor))
    else:
        send_tensor = torch.ones(4, dtype=torch.float32).cuda() * bagua.get_rank()
        bagua.gather_inplace(send_tensor, 4, dst=0, comm=comm)
        print("gather_inplace, rank: {}, send_tensor: {}.".format(bagua.get_rank(), send_tensor))

    # # broadcast
    # if bagua.get_rank() == 0:
    #     tensor = torch.rand(4, dtype=torch.float32).cuda()
    #     dist.broadcast(tensor, 0)
    #     bagua.broadcast(tensor, 0, comm=comm)
    # else:
    #     recv_tensor = torch.zeros(4, dtype=torch.float32).cuda()
    #     recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
    #     dist.broadcast(recv_tensor, 0)
    #     bagua.broadcast(recv_tensor_bagua, 0, comm=comm)
    #     assert torch.equal(
    #         recv_tensor, recv_tensor_bagua
    #     ), "recv_tensor:{a}, recv_tensor_bagua:{b}".format(
    #         a=recv_tensor, b=recv_tensor_bagua
    #     )
def allgather():
    comm = init_env()
    send_tensor = torch.rand(4, dtype=torch.float32).cuda()
    recv_tensor_bagua = torch.zeros(
        4 * bagua.get_world_size(), dtype=torch.float32
    ).cuda()
    print("rank: {}, recv tensor id: {}.".format(bagua.get_rank(), id(recv_tensor_bagua)))
    bagua.allgather(send_tensor, recv_tensor_bagua, comm=comm)
    print("rank: {}, send tensor: {}, recv tensor id: {}, recv tensor: {}".format(bagua.get_rank(), send_tensor, id(recv_tensor_bagua), recv_tensor_bagua))
    send_tensor.zero_()
    print("rank: {}, send tensor: {}, recv tensor id: {}, recv tensor: {}".format(bagua.get_rank(), send_tensor, id(recv_tensor_bagua), recv_tensor_bagua))

def allgather_inplace():
    comm = init_env()
    send_tensor = torch.rand(4, dtype=torch.float32).cuda()
    recv_tensor_bagua = torch.zeros(
        4 * bagua.get_world_size(), dtype=torch.float32
    ).cuda()
    print("rank: {}, recv tensor id: {}.".format(bagua.get_rank(), id(recv_tensor_bagua)))
    bagua.allgather_inplace(send_tensor, recv_tensor_bagua, comm=comm)
    print("rank: {}, send tensor: {}, recv tensor id: {}, recv tensor: {}".format(bagua.get_rank(), send_tensor, id(recv_tensor_bagua), recv_tensor_bagua))

def allgather_torch(): # error
    comm = init_env()
    rank = bagua.get_rank()
    send_tensor = torch.rand(rank+1, dtype=torch.float32).cuda()
    recv_tensors = [
        torch.zeros(4, dtype=torch.float32).cuda()
        for i in range(bagua.get_world_size())
    ]
    dist.all_gather(recv_tensors, send_tensor)
    print("rank: {}, send tensor: {}, recv tensor: {}.".format(rank, send_tensor, recv_tensors))

def send_recv():
    torch.set_printoptions(precision=20)

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    size = 1000000
    print("rank: ", bagua.get_rank())
    for index in range(1, 10000):
        print("index: ", index)
        send_tensor = torch.rand(size, dtype=torch.float32).cuda()
        # recv_tensor = torch.zeros(size, dtype=torch.float32).cuda()
        recv_tensor_bagua = torch.zeros(size, dtype=torch.float32).cuda()
        # send, recv
        if bagua.get_rank() == 0:
            # dist.send(send_tensor, 1)
            bagua.send(send_tensor, 1, comm=comm)
            print("----------{}".format(send_tensor.is_bagua_tensor()))
            # logging.info("recv_tensor == recv_tensor_bagua is {}".format(torch.equal(recv_tensor, recv_tensor_bagua)))
            # send_tensor.drop_bagua_tensor()
            send_tensor = None
        elif bagua.get_rank() == 1:
            # dist.recv(recv_tensor, 0)
            bagua.recv(recv_tensor_bagua, 0, comm=comm)
            print("----------{}".format(recv_tensor_bagua.is_bagua_tensor()))
            # print("recv_tensor == recv_tensor_bagua is {}".format(torch.equal(recv_tensor, recv_tensor_bagua)))
            # recv_tensor_bagua.drop_bagua_tensor()
            recv_tensor_bagua = None

def allgather_test(datasize, epochs, div):
    comm = init_env()
    outdir = os.path.join("./log", "allgather_comm")
    ranks = bagua.get_world_size()
    rank = bagua.get_rank()
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    log_name = "allgather_gpu" + str(ranks) + "_rank" + str(rank) + "_datasize" + str(datasize) + "_div" + str(div) + "_epochs" + str(epochs) + ".txt"
    log_file = os.path.join(outdir, log_name)
    writer = open(log_file, "w")
    rst = ""
    for i in range(0, 5):
        send_value_tensor = torch.randn(datasize, dtype=torch.float32).cuda()
        send_index_tensor = torch.randint_like(send_value_tensor, 0, datasize, dtype=torch.int64).cuda()
        recv_value_tensor = torch.randn(datasize * ranks, dtype=torch.float32).cuda()
        recv_index_tensor = torch.randint_like(recv_value_tensor, 0, datasize, dtype=torch.int64).cuda()
        duration = 0
        for index in range(epochs):
            torch.cuda.synchronize()
            start = time.time()
            bagua.allgather(send_index_tensor, recv_index_tensor, comm=comm)
            bagua.allgather(send_value_tensor, recv_value_tensor, comm=comm)
            torch.cuda.synchronize()
            end = time.time()
            duration_tmp = end - start
            duration += duration_tmp
            writer.write("allgather on ranks={}, rank={}, step={}, datasize={}, div={}, epochs={}, time={}.\n".format(ranks, rank, index, datasize, div, epochs, duration_tmp))
        rst += "allgather on ranks={}, rank={}, datasize={}, div={}, epochs={}, time={}, average time={}.\n".format(ranks, rank, datasize, div, epochs, duration, duration/epochs)
        datasize //= div
        writer.write('\n\n')
        writer.flush()

    writer.write('\n')
    writer.write(rst)
    writer.flush()

def send_recv_test(datasize, epochs, div):
    comm = init_env()
    outdir = os.path.join("./log", "sendrecv_comm")
    ranks = bagua.get_world_size()
    if ranks == 0:
        print("world size must be at least 2")
        return
    elif ranks == 2:
        neighbours = 1
    else:
        neighbours = 2
    rank = bagua.get_rank()
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    log_name = "sendrecv_gpu" + str(ranks) + "_rank" + str(rank) + "_datasize" + str(datasize) + "_div" + str(div) + "_epochs" + str(epochs) + ".txt"
    log_file = os.path.join(outdir, log_name)
    writer = open(log_file, "w")
    rst = ""
    for i in range(0, 5):
        send_value_tensors = [torch.randn(datasize, dtype=torch.float32).cuda() for x in range(0, neighbours)]
        # send_index_tensor = torch.randint_like(send_value_tensor, 0, datasize, dtype=torch.int64).cuda()
        recv_value_tensors = [torch.randn(datasize, dtype=torch.float32).cuda() for x in range(0, neighbours)]
        # recv_index_tensor = torch.randint_like(recv_value_tensor, 0, datasize, dtype=torch.int64).cuda()
        duration = 0
        for index in range(epochs):
            torch.cuda.synchronize()
            start = time.time()
            for neighbour in set([(rank - 1) % ranks, (rank + 1) % ranks]):
                if rank > neighbour:
                    bagua.recv(recv_value_tensors[-1], neighbour)
                    bagua.send(send_value_tensors[-1], neighbour)
                else:
                    bagua.send(send_value_tensors[0], neighbour)
                    bagua.recv(recv_value_tensors[0], neighbour)
            torch.cuda.synchronize()
            end = time.time()
            duration_tmp = end - start
            duration += duration_tmp
            writer.write("sendrecv on ranks={}, rank={}, step={}, datasize={}, div={}, epochs={}, time={}.\n".format(ranks, rank, index, datasize, div, epochs, duration_tmp))
        rst += "sendrecv on ranks={}, rank={}, datasize={}, div={}, epochs={}, time={}, average time={}.\n".format(ranks, rank, datasize, div, epochs, duration, duration/epochs)
        datasize //= div
        writer.write('\n\n')
        writer.flush()

    writer.write('\n\n')
    writer.write(rst)
    writer.flush()

if __name__ == "__main__":
    # gather()
    # gather_inplace()
    # allgather()
    # allgather_torch() # error
    # send_recv()
    # allgather_test(25557032, 100, 10)
    # allgather_test(1199800, 1000, 10)
    # send_recv_test(25557032, 100, 10)
    send_recv_test(1199800, 1000, 10)