#!/usr/bin/env python3
from cmath import log
import logging
from operator import index
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from typing import List
import torch
import bagua.torch_api as bagua
from torch.optim import Optimizer
import sys

__all__ = [
    "SparsepyAlgorithm"
]

def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # ret = torch.zeros(k, dtype=vec.dtype).cuda()

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    # ret.copy_(vec[topkIndices])
    return topkIndices, vec[topkIndices]

class SparsepyAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topK: int = 0,
    ):
        """
        Implementation of the `Sparsepy` algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
            topK:.
        """
        super(SparsepyAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()
        self.optimizer = optimizer
        self.rank = bagua.get_local_rank()
        self.topK = topK
        self.param_size = 0
        # self.tensors = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                self.param_size += param.numel()
                # self.tensors.append(param)
        self.works = bagua.get_world_size()
        if self.topK == 0:
            self.topK = self.param_size // 100
        elif self.topK > self.param_size:
            self.topK = self.param_size
        # self.tensor_send = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.recv_messages = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        self.recv_indexes = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        self.send_messages = torch.zeros(self.topK, dtype=torch.float32).cuda()
        self.send_indexes = torch.zeros(self.topK, dtype=torch.int64).cuda()
        self.tensors_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        logging.info("---------param_size: {}, topK: {}".format(self.param_size, self.topK))

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = []
        for name, param in parameters:
            param.bagua_ensure_grad()
            self.tensors.append(param.grad)
        return []

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        return []

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
            if not self._should_communicate(bagua_ddp):
                return
            def pack():
                """Packs a list of tensors into one buffer for sending to other workers"""
                buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
                _, indexes = torch.topk(buffer**2, self.topK)
                self.send_indexes.copy_(indexes)
                self.send_messages.copy_(buffer[indexes])

            def unpack():
                """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
                self.tensors_buffer.zero_()
                for rank in range(self.works):
                    start = rank * self.topK
                    end = start + self.topK
                    self.tensors_buffer[self.recv_indexes[start:end]] += (self.recv_messages[start:end])
                self.tensors_buffer.div_(self.works)
                unpack2tensors()
            
            def unpack2tensors():
                size = 0
                for tensor in self.tensors:
                    shape = tensor.shape
                    count = tensor.numel()
                    tensor.copy_(self.tensors_buffer[size:count+size].reshape(shape))
                    size += count
            def test():
                nonzero = 0
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        nonzero += param.grad.count_nonzero().item()
                nonzero1 = 0
                for tensor in self.tensors:
                    nonzero1 += tensor.count_nonzero().item()
                logging.info("-----rank: {}, grad nonzero size: {}, self.tensors nonzero size: {}.".format(self.rank, nonzero, nonzero1))
                
            pack()
            bagua.allgather(self.send_indexes, self.recv_indexes)
            bagua.allgather(self.send_messages, self.recv_messages)
            torch.cuda.synchronize()
            unpack()
            test()

        return hook
    
    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            return

        return hook

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        torch.cuda.synchronize()
        bucket.clear_ops()


class SparsepyAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topK: int = 0,
    ):
        """
        Create an instance of the Sparsepy algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = optimizer
        self.topK = topK

    def reify(self, process_group: BaguaProcessGroup) -> SparsepyAlgorithmImpl:
        return SparsepyAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )
