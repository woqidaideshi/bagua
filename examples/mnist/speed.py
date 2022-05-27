#!/usr/bin/env python3
from pickletools import optimize
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
import bagua.torch_api as bagua
from bagua.torch_api.algorithms import gradient_allreduce
import logging
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from typing import List
import torch
import sys
from torch.optim import Optimizer
import time

USE_RELAY = True
OVERWRITE = False
DEBUG = False


class RelayAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        comm_mode: str = "bagua", # bagua, torch, torch_i
    ):
        """
        Implementation of the
        `Decentralized SGD <https://tutorials.baguasys.com/algorithms/decentralized>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        super(RelayAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()
        self.m_recv = {}
        self.c_recv = {}
        self.ones = torch.ones(1, dtype=torch.float32).cuda()
        self.c_temp = torch.zeros(1, dtype=torch.float32).cuda()
        self.n = torch.zeros(1, dtype=torch.float32).cuda()
        self.x_buffered = 0
        self.optimizer = optimizer
        self.param_size = 0
        for layer in optimizer.param_groups[0]['params']:
            self.param_size += layer.numel()
        # get current rank
        self.rank = bagua.get_local_rank()

        # create neighbour list
        neighbours = [(self.rank - 1) // 2, 2 * self.rank + 1, 2 * self.rank + 2]
        self.neighbours_filtered = []
        for nb in neighbours:
            if nb >= 0 and nb < bagua.get_world_size():
                self.neighbours_filtered.append(nb)
                self.m_recv[nb] = torch.zeros(self.param_size, dtype=torch.float32).cuda()
                self.c_recv[nb] = torch.ones(1, dtype=torch.float32).cuda()
        self.m_send = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.c_send = torch.ones(1, dtype=torch.float32).cuda()
        print("rank: ", bagua.get_rank())
        self.s = torch.rand(1000000).cuda()
        self.r = torch.zeros(1000000).cuda()
        self.step = 0
        self.comm_mode = comm_mode
    
    def speed(self):
        nb = list(range(bagua.get_world_size()))
        if self.comm_mode == "bagua":
            torch.cuda.synchronize()
            start = time.time()
            for n in nb:
                if n == bagua.get_local_rank():
                    continue
                if n < bagua.get_local_rank():
                    bagua.send(self.s, n)
                    bagua.recv(self.r, n)
                else:
                    bagua.recv(self.r, n)
                    bagua.send(self.s, n)
            torch.cuda.synchronize()
            end = time.time()
            # if bagua.get_local_rank() == 1: print("Bagua: {}".format(end-start))
            print("bagua rank: {}({}), time: {}.".format(bagua.get_rank(), self.step, end-start))
        elif self.comm_mode == "torch":
            torch.cuda.synchronize()
            start = time.time()
            for n in nb:
                if n == bagua.get_local_rank():
                    continue
                if n < bagua.get_local_rank():
                    torch.distributed.send(self.s, n)
                    torch.distributed.recv(self.r, n)
                else:
                    torch.distributed.recv(self.r, n)
                    torch.distributed.send(self.s, n)
            torch.cuda.synchronize()
            end = time.time()
            # if bagua.get_local_rank() == 1: print("PYT: {}".format(end-start))
            print("torch rank: {}({}), time: {}.".format(bagua.get_rank(), self.step, end-start))
        else:
            torch.cuda.synchronize()
            start = time.time()
            for n in nb:
                if n == bagua.get_local_rank():
                    continue
                if n < bagua.get_local_rank():
                    torch.distributed.isend(self.s, n)
                    torch.distributed.irecv(self.r, n)
                else:
                    torch.distributed.irecv(self.r, n)
                    torch.distributed.isend(self.s, n)
            torch.cuda.synchronize()
            end = time.time()
            # if bagua.get_local_rank() == 1: print("PYT: {}".format(end-start))
            print("torch_i rank: {}({}), time: {}.".format(bagua.get_rank(), self.step, end-start))

        # torch.cuda.synchronize()
        # start = time.time()
        # for n in nb:
        #     if n == bagua.get_local_rank():
        #         continue
        #     if n < bagua.get_local_rank():
        #         torch.distributed.send(self.t, n)
        #         torch.distributed.recv(self.t, n)
        #     else:
        #         torch.distributed.recv(self.t, n)
        #         torch.distributed.send(self.t, n)
        # torch.cuda.synchronize()
        # end = time.time()
        # # if bagua.get_local_rank() == 1: print("PYT: {}".format(end-start))
        # print("torch rank: {}({}), time: {}.".format(bagua.get_rank(), self.step, end-start))
        self.step += 1

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=do_flatten, name=str(0))

        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            return

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            return

        return hook
    
    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            self.speed()
            return

            def pack(tensors):
                """Packs a list of tensors into one buffer for sending to other workers"""
                buffer = torch.cat([t.view(-1) for t in tensors])  # copies
                shapes = [tensor.shape for tensor in tensors]
                return buffer, shapes

            def unpack(buffer, shapes):
                """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
                idx = 0
                entries = []
                for tensor_shape in shapes:
                    end = idx + tensor_shape.numel()
                    entries.append(buffer[idx:end].view(size=tensor_shape))
                    idx = end

                return entries
            
            def sum_wo(dict, wo_key):
                """Sums up values of a given dictionary, excluding the values of wo_key."""
                # if wo_key in dict:
                #     return sum(dict.values()) - dict[wo_key]
                # return sum(dict.values())
                return sum(value for (key, value) in dict.items() if key != wo_key)

            # init X_i^(t + 1/2)
            x_i = [layer for layer in optimizer.param_groups[0]['params']]
            x_i_buffered, shapes = pack(x_i)
            if DEBUG: orig = torch.clone(x_i_buffered)
            self.x_buffered = torch.clone(x_i_buffered)

            print('Bytes: {}'.format(sys.getsizeof(x_i_buffered.storage())))

            def send_messages(neighbour):
                # send messages
                self.m_send.copy_(sum_wo(self.m_recv, neighbour) + x_i_buffered)
                bagua.send(self.m_send, neighbour)

                # send corresponding counters
                self.c_send.copy_(sum_wo((self.c_recv), neighbour) + self.ones)
                if DEBUG: print('Sending c_t={} from {} to {}'.format(self.c_send, self.rank, neighbour))
                bagua.send(self.c_send, neighbour)
            
            def recv_messages(neighbour):
                # # recieve messages
                bagua.recv(self.m_recv[neighbour], neighbour)
                bagua.recv(self.c_temp, neighbour)
                self.c_recv[neighbour] = self.c_temp.clone().detach()

            # iterate over neighbours
            for neighbour in self.neighbours_filtered:
                # Deadlock avoidance
                if neighbour < self.rank:
                    send_messages(neighbour)
                    recv_messages(neighbour)
                else:
                    recv_messages(neighbour)
                    send_messages(neighbour)

            # update n and x_i
            self.n = 1 + sum(self.c_recv.values())
            if DEBUG: print('rank: {} -> n={}'.format(self.rank, self.n))
            self.x_buffered.add_(sum(self.m_recv.values())).div_(self.n)
            # self.x_buffered = 1. / self.n * (self.x_buffered + sum(self.m_recv.values()))

            # unpack x_buffered
            x_i_2 = unpack(self.x_buffered, shapes)

            # overwrite current weights
            for idx, layer in enumerate(optimizer.param_groups[0]['params']):
                # layer.data = x_i_2[idx]
                layer.data.copy_(x_i_2[idx])
            
            # report (convergence) behaviour of X_i
            if DEBUG: print('rank: {} -> absolute diff after sync: {}'.format(self.rank, sum(torch.abs(self.x_buffered - orig))))

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        bucket._peer_weight = weight_tensor.ensure_bagua_tensor("peer_weight")

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()


class RelayAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        comm_mode: str = "bagua", # bagua, torch, torch_i
    ):
        """
        Create an instance of the
        `Decentralized SGD <https://tutorials.baguasys.com/algorithms/decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.optimizer = optimizer
        self.comm_mode = comm_mode

    def reify(self, process_group: BaguaProcessGroup) -> RelayAlgorithmImpl:
        return RelayAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            comm_mode=self.comm_mode
        )

def main():
    #torch.manual_seed(42)
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()
    logging.getLogger().setLevel(logging.INFO)

    model = torch.nn.Sequential(torch.nn.Linear(1000, 1000),torch.nn.Linear(1000, 1)).cuda()

    comm_mode = "torch_i"
    if USE_RELAY:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = RelayAlgorithm(optimizer=optimizer, comm_mode=comm_mode)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()


    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=True,
    )

    model.train()
    X = torch.randn(1000, 1000).cuda()
    y = torch.zeros(1000, 1).cuda()

    torch.cuda.synchronize()
    epochs_start = time.time()
    epochs = 10000
    for epoch in range(1, 10000):
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        if bagua.get_local_rank() == 0: logging.info(f"it {epoch}, loss: {loss.item():.6f}")
    torch.cuda.synchronize()
    epochs_end = time.time()
    print("rank: {}, communication mode: {}, time for {} epochs: {}".format(bagua.get_local_rank(), comm_mode, epochs, epochs_end-epochs_start))


if __name__ == "__main__":
    main()
