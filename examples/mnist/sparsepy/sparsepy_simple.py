#!/usr/bin/env python3
from asyncio.log import logger
import logging
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
            self.topK = self.param_size // self.works
        # self.tensor_send = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        if self.rank == 0:
            if self.topK * self.works < self.param_size:
                self.tensor_unpack = torch.zeros(self.param_size, dtype=torch.float32).cuda()
                self.param_size = self.topK * self.works
                self.tensor_comm = torch.zeros(self.param_size, dtype=torch.float32).cuda()
            else:
                self.tensor_comm = torch.zeros(self.param_size, dtype=torch.float32).cuda()
                self.tensor_unpack = self.tensor_comm
        else:
            self.tensor_comm = torch.zeros(self.topK, dtype=torch.float32).cuda()
            self.tensor_unpack = torch.zeros(self.param_size, dtype=torch.float32).cuda()

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
                if self.rank == 0:
                    self.tensor_comm.copy_(buffer[0:self.param_size])
                else:
                    start = self.rank * self.topK
                    self.tensor_comm.copy_(buffer[start:start+self.topK])

            def unpack():
                """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
                if self.rank == 0 and self.tensor_comm.numel() < self.tensor_unpack.numel():
                    buffer = torch.cat([t.view(-1) for t in self.tensors])
                    self.tensor_unpack[self.param_size:] = buffer[self.param_size:]
                    self.tensor_unpack[:self.param_size] = self.tensor_comm
                size = 0
                for tensor in self.tensors:
                    shape = tensor.shape
                    count = tensor.numel()
                    tensor.copy_(self.tensor_unpack[size:count+size].reshape(shape))
                    size += count
            pack()
            bagua.gather_inplace(self.tensor_comm, self.topK, dst=0)
            if self.rank == 0:
                unpack()
            bagua.broadcast(self.tensor_unpack, 0)
            if self.rank != 0:
                unpack()
            # logging.info("----------------rank: {} hook end.".format(self.rank))
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
