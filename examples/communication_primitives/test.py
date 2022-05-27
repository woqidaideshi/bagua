from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua
import time

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
def test():
    torch.set_printoptions(precision=20)
    parser = argparse.ArgumentParser(description="Communication Primitives Example")
    parser.parse_args()

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    rank = bagua.get_local_rank()
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    print("rank: ", bagua.get_rank())
    s = torch.ones(size).cuda()
    r = torch.zeros(size).cuda()
    nb = list(range(bagua.get_world_size()))
    time_bagua = 0
    time_torch = 0
    time_torch_i = 0
    for index in range(count):
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                bagua.send(s, n)
                bagua.recv(r, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, recv_tensor_bagua:{a}, send_tensor_bagua:{b}".format(
                    n, a=r, b=s
                )
            else:
                bagua.recv(r, n)
                bagua.send(s, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, recv_tensor_bagua:{a}, send_tensor_bagua:{b}".format(
                    n, a=r, b=s
                )
        end = time.time()
        duration = end - start
        time_bagua += duration
        print("bagua rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.send(s, n)
                torch.distributed.recv(r, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, recv_tensor:{a}, send_tensor:{b}".format(
                    n, a=r, b=s
                )
            else:
                torch.distributed.recv(r, n)
                torch.distributed.send(s, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, recv_tensor:{a}, send_tensor:{b}".format(
                    n, a=r, b=s
                )
        end = time.time()
        duration = end - start
        time_torch += duration
        print("torch rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.isend(s, n)
                torch.distributed.irecv(r, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, irecv_tensor:{a}, isend_tensor:{b}".format(
                    n, a=r, b=s
                )
            else:
                torch.distributed.irecv(r, n)
                torch.distributed.isend(s, n)
                assert torch.equal(
                    r, s
                ), "rank: {}, irecv_tensor:{a}, isend_tensor:{b}".format(
                    n, a=r, b=s
                )
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))


def test_synchronize():
    torch.set_printoptions(precision=20)
    parser = argparse.ArgumentParser(description="Communication Primitives Example")
    parser.parse_args()

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    rank = bagua.get_local_rank()
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    print("rank: ", bagua.get_rank())
    s = torch.ones(size).cuda()
    r = torch.zeros(size).cuda()
    nb = list(range(bagua.get_world_size()))
    time_bagua = 0
    time_torch = 0
    time_torch_i = 0
    for index in range(count):
        torch.cuda.synchronize()
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                bagua.send(s, n)
                bagua.recv(r, n)
            else:
                bagua.recv(r, n)
                bagua.send(s, n)
        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        time_bagua += duration
        print("bagua rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        torch.cuda.synchronize()
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.send(s, n)
                torch.distributed.recv(r, n)
            else:
                torch.distributed.recv(r, n)
                torch.distributed.send(s, n)
        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        time_torch += duration
        print("torch rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        torch.cuda.synchronize()
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.isend(s, n)
                torch.distributed.irecv(r, n)
            else:
                torch.distributed.irecv(r, n)
                torch.distributed.isend(s, n)
        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))
    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))

def test_error():
    torch.set_printoptions(precision=20)
    parser = argparse.ArgumentParser(description="Communication Primitives Example")
    parser.parse_args()

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    rank = bagua.get_local_rank()
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    print("rank: ", bagua.get_rank())
    s = torch.ones(size).cuda()
    r = torch.zeros(size).cuda()
    nb = list(range(bagua.get_world_size()))
    time_bagua = 0
    time_torch = 0
    time_torch_i = 0
    for index in range(count):
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                bagua.send(s, n)
                bagua.recv(r, n)
            else:
                bagua.recv(r, n)
                bagua.send(s, n)
        end = time.time()
        duration = end - start
        time_bagua += duration
        print("bagua rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.send(s, n)
                torch.distributed.recv(r, n)
            else:
                torch.distributed.recv(r, n)
                torch.distributed.send(s, n)
        end = time.time()
        duration = end - start
        time_torch += duration
        print("torch rank: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                torch.distributed.isend(s, n)
                torch.distributed.irecv(r, n)
            else:
                torch.distributed.irecv(r, n)
                torch.distributed.isend(s, n)
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))
    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))

if __name__ == "__main__":
    # main()
    count = 100000
    size = 1000000
    print("-----------------test--------")
    test()
    # print("-----------------test_synchronize--------")
    # test_synchronize()
    # print("-----------------test_error--------")
    # test_error()