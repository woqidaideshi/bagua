from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua


def main():
    torch.set_printoptions(precision=20)
    parser = argparse.ArgumentParser(description="Communication Primitives Example")
    parser.parse_args()

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    size = 1000000
    print("rank: ", bagua.get_rank())
    send_tensor = torch.rand(size, dtype=torch.float32).cuda()
    recv_tensor = torch.zeros(size, dtype=torch.float32).cuda()
    recv_tensor_bagua = torch.zeros(size, dtype=torch.float32).cuda()
    for index in range(1, 100000):
        print("index: ", index)
        # send, recv
        if bagua.get_rank() == 0:
            dist.send(send_tensor, 1)
            bagua.send(send_tensor, 1, comm=comm)
        elif bagua.get_rank() == 1:
            dist.recv(recv_tensor, 0)
            bagua.recv(recv_tensor_bagua, 0, comm=comm)
            assert torch.equal(
                recv_tensor, recv_tensor_bagua
            ), "recv_tensor:{a}, recv_tensor_bagua:{b}".format(
                a=recv_tensor, b=recv_tensor_bagua
            )

if __name__ == "__main__":
    main()
