from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua
import time
import sys

def main():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))
    
    torch.set_printoptions(precision=20)

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
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)
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

def test_iwait():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

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
                req = torch.distributed.isend(s, n)
                req.wait()
                req = torch.distributed.irecv(r, n)
                req.wait()
                assert torch.equal(
                    r, s
                ), "rank: {}, irecv_tensor:{a}, isend_tensor:{b}".format(
                    n, a=r, b=s
                )
            else:
                req = torch.distributed.irecv(r, n)
                req.wait()
                req = torch.distributed.isend(s, n)
                req.wait()
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

def test_iwaitlist():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

    assert bagua.get_world_size() >= 1, "world size must be at least 2"

    rank = bagua.get_local_rank()
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

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

        req_list = []
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                req = torch.distributed.isend(s, n)
                req_list.append(req)
                req = torch.distributed.irecv(r, n)
                req_list.append(req)
                assert torch.equal(
                    r, s
                ), "rank: {}, irecv_tensor:{a}, isend_tensor:{b}".format(
                    n, a=r, b=s
                )
            else:
                req = torch.distributed.irecv(r, n)
                req_list.append(req)
                req = torch.distributed.isend(s, n)
                req_list.append(req)
                assert torch.equal(
                    r, s
                ), "rank: {}, irecv_tensor:{a}, isend_tensor:{b}".format(
                    n, a=r, b=s
                )
        for req in req_list:
            req.wait()
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))

    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))

def test_synchronize():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

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

def test_synchronize_iwait():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

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
                req = torch.distributed.isend(s, n)
                req.wait()
                req = torch.distributed.irecv(r, n)
                req.wait()
            else:
                req = torch.distributed.irecv(r, n)
                req.wait()
                req = torch.distributed.isend(s, n)
                req.wait()
        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))
    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))

def test_synchronize_iwaitlist():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

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

        req_list = []
        torch.cuda.synchronize()
        start = time.time()
        for n in nb:
            if n == rank:
                continue
            if n < rank:
                req = torch.distributed.isend(s, n)
                req_list.append(req)
                req = torch.distributed.irecv(r, n)
                req_list.append(req)
            else:
                req = torch.distributed.irecv(r, n)
                req_list.append(req)
                req = torch.distributed.isend(s, n)
                req_list.append(req)
        for req in req_list:
            req.wait()
        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        time_torch_i += duration
        print("torch rank isend/irecv: {}({}), time: {}.".format(bagua.get_rank(), index, duration))
    print("\nrank:{}, communicate for {} times, size: {}.".format(rank, count, size))
    print("rank: {}, bagua: {}, torch: {}, torch_i: {}".format(rank, time_bagua, time_torch, time_torch_i))

def test_error():
    print("rank: {} in func: {}.".format(bagua.get_rank(), sys._getframe().f_code.co_name))

    torch.set_printoptions(precision=20)

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
                req = torch.distributed.isend(s, n)
                req.wait()
                req = torch.distributed.irecv(r, n)
                req.wait()
            else:
                req = torch.distributed.irecv(r, n)
                req.wait()
                req = torch.distributed.isend(s, n)
                req.wait()
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
    import argparse
    parser = argparse.ArgumentParser(description="send/recv time")
    parser.add_argument("--func",
                        default="test",
                        help="chose a func",
                        type=str,
                        )
    args = parser.parse_args()
    func_list = {
        "test": test,
        "test_iwait": test_iwait,
        "test_iwaitlist": test_iwaitlist,
        "test_synchronize": test_synchronize,
        "test_synchronize_iwait": test_synchronize_iwait,
        "test_synchronize_iwaitlist": test_synchronize_iwaitlist
    }

    # func_list = {
    #     0: test,
    #     1: test_iwait,
    #     2: test_iwaitlist,
    #     3: test_synchronize,
    #     4: test_synchronize_iwait,
    #     5: test_synchronize_iwaitlist
    # }
    
    chose = args.func.strip()
    
    if chose in func_list.keys():
        print("--------------------{}.".format(chose))
        func = func_list[chose]()
    else:
        print("-------unrecognized arguments for --func: ", chose)

    # print("-----------------test--------")
    # test()
    # print("-----------------test-iwait--------")
    # test_iwait()
    # print("-----------------test-iwaitlist--------")
    # test_iwaitlist()
    # print("-----------------test_synchronize--------")
    # test_synchronize()
    # print("-----------------test_synchronize-iwait--------")
    # test_synchronize_iwait()
    # print("-----------------test_synchronize-iwaitlist--------")
    # test_synchronize_iwaitlist()
    # print("-----------------test_error--------")
    # test_error()
