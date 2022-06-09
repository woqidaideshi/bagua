from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua


def init_env():
    torch.set_printoptions(precision=20)

    assert bagua.get_world_size() >= 2, "world size must be at least 2"

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
        print("gather", bagua.get_rank(), send_tensor)
        print("gather", bagua.get_rank(), recv_tensor)
    else:
        send_tensor = torch.ones(4, dtype=torch.float32).cuda() * bagua.get_rank()
        recv_tensor = torch.zeros(4*bagua.get_world_size(), dtype=torch.float32).cuda()
        bagua.gather(send_tensor, recv_tensor, dst=0, comm=comm)
        print("gather", bagua.get_rank(), send_tensor)
        print("gather", bagua.get_rank(), recv_tensor)

def gather_inplace():
    comm = init_env()
    if bagua.get_rank() == 0:
        recv_tensor = torch.ones(4*bagua.get_world_size(), dtype=torch.float32).cuda() / 2
        bagua.gather_inplace(recv_tensor, 4, dst=0, comm=comm)
        print("gather_inplace", bagua.get_rank(), recv_tensor)
    else:
        send_tensor = torch.ones(4, dtype=torch.float32).cuda() * bagua.get_rank()
        bagua.gather_inplace(send_tensor, 4, dst=0, comm=comm)
        print("gather_inplace", bagua.get_rank(), send_tensor)

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

if __name__ == "__main__":
    # gather()
    gather_inplace()