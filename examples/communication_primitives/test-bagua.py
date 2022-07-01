from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import bagua.torch_api as bagua
import time

def copy2gpu():
    count = 10000
    size = 1000000
    torch.set_printoptions(precision=20)

    rank = "cuda:0"
    rank_other = "cuda:1"

    # time_time = 0
    # time_event = 0
    # time_time_gpu = 0
    # time_event_gpu = 0
    time4cpu = {0: 0, 1: 0}
    time4gpu = {0: 0, 1: 0}
    for index in range(count):
        tensor = torch.rand(size)
        tensor_cuda = torch.rand(size, device=rank)
        for (tensor_from, rank_to, time_record) in zip([tensor, tensor_cuda], [rank, rank_other], [time4cpu, time4gpu]):
        # for (tensor_from, rank_to, time_record) in zip([tensor, tensor_cuda], [rank, rank_other], [[time_time, time_event], [time_time_gpu, time_event_gpu]]):
            torch.cuda.synchronize()
            start = time.time()
            tensor_from.to(device=rank_to)
            torch.cuda.synchronize()
            end=time.time()
            duration = end - start
            time_record[0] += duration
            print("x.to func(time.time), copy to rank: {}, duration: {}.".format(rank_to, duration))
            
            tensor_from = torch.rand(size)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tensor_from.to(device=rank_to)
            end.record()
            torch.cuda.synchronize()
            duration = start.elapsed_time(end)/1000
            time_record[1] += duration
            print("x.to func(torch.cuda.Event), copy to rank: {}, duration: {}.".format(rank_to, duration))
        
    print("x.to func(time.time), rank: {} copy to rank: {}, time: {}, mean time: {}.".format("cpu", rank, time4cpu[0], time4cpu[0]/count))
    print("x.to func(torch.cuda.Event), rank: {} copy to rank: {}, time: {}, mean time: {}.".format("cpu", rank, time4cpu[1], time4cpu[1]/count))
    print("x.to func(time.time.gpu), rank: {} copy to rank: {}, time: {}, mean time: {}.".format(rank, rank_other, time4gpu[0], time4gpu[0]/count))
    print("x.to func(torch.cuda.Event.gpu), rank: {} copy to rank: {}, time: {}, mean time: {}.".format(rank, rank_other, time4gpu[1], time4gpu[1]/count))

def test():
    # print("tensor1: ", tensor1)
    tensor = torch.ones(10)
    print("device1: ", tensor.device)
    tensor = tensor.cuda()
    print("device2: ", tensor.device)
    tensor = tensor.to(device="cuda:1")
    print("device3: ", tensor.device)

if __name__ == "__main__":
    # test()
    copy2gpu()
