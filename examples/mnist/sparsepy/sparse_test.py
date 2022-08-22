#!/usr/bin/env python3
from ast import operator
from cmath import log
import logging
from operator import index
from bagua.torch_api import tensor
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
import numpy as np

torch.set_printoptions(threshold=np.inf)

__all__ = [
    "SparseAlgorithm"
]

def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # ret = torch.zeros(k, dtype=vec.dtype).cuda()

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    # ret.copy_(vec[topkIndices])
    return topkIndices, vec[topkIndices]

class SparseAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topK: int = 0,
    ):
        """
        Implementation of the `Sparse` algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
            topK:.
        """
        super(SparseAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()
        self.optimizer = optimizer
        self.rank = bagua.get_rank()
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
        # self.send_indexes.tensors_other = torch.cat([t.view(-1) for t in self.tensors])
        name, param = parameters[-1]
        param.index_tensor = torch.zeros(self.topK, dtype=torch.int64).cuda()
        # param.index_tensor = torch.zeros(self.topK, dtype=torch.float32).cuda()
        
        self.index_tensor = param.ensure_bagua_tensor(
            name,
            bagua_ddp.bagua_module_name,
            getter_closure=lambda param: param.index_tensor,
            setter_closure=lambda param, t: setattr(param, "index_tensor", t),
        )
        self._communication_tensor_names = set((name,))
        logging.info("-------------init tensors.")
        return [self.index_tensor]

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        # bagua_buckets = []
        # for idx, bucket in enumerate(tensors):
        #     bagua_bucket = BaguaBucket(
        #         bucket, flatten=do_flatten, name=str(idx)
        #     )  # TODO: check duplicated names
        #     bagua_buckets.append(bagua_bucket)
        #     # bagua_bucket_other = []
        #     # for bucket_per in bucket:
        #     #     if hasattr(bucket_per, "other"):
        #     #         bagua_bucket_other.append(bucket_per.other)
        #     # if bagua_bucket_other:
        #     #     bagua_bucket.other = BaguaBucket(
        #     #         bagua_bucket_other, flatten=do_flatten, name=str(idx)
        #     #     )
        # return bagua_buckets
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=do_flatten, name=str(0))
        logging.info("-------------tensors_to_buckets.")
        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            pass
            # if parameter_name in self._communication_tensor_names:
            # #     assert (
            # #         parameter.bagua_backend_tensor().data_ptr()
            # #         == parameter.data_ptr()
            # #     ), "bagua backend tensor data_ptr should match parameter grad"
            # #     parameter.bagua_mark_communication_ready()
            #     logging.info("---------------parameter_name: {}.".format(parameter_name))
            #     parameter.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            self.index_tensor.bagua_mark_communication_ready()
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            return
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
                nonzero = 0
                nonzero1 = 0
                for tensor in self.tensors:
                    shape = tensor.shape
                    count = tensor.numel()
                    tmp_buffer = self.tensors_buffer[size:count+size]
                    tmp_tensor = tensor.view(-1)
                    tmp_tensor[tmp_buffer.nonzero()] = 0
                    tmp_tensor.add_(tmp_buffer)
                    nonzero += tmp_buffer.count_nonzero().item()
                    nonzero1 += tmp_tensor.count_nonzero().item()
                    size += count
                # logging.info("-----rank: {}, tensors_buffer nonzero size: {}, tmp_tensor nonzero size: {}.".format(self.rank, nonzero, nonzero1))


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
            # test()

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
        bucket._other_tensor = self.tensors_buffer.ensure_bagua_tensor(
            "other_tensor", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
        logging.info("-------------init_operations.")
        bucket.clear_ops()
        def set_index(*args):
            if hasattr(bucket, "_other_tensor"):
                buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
                _, indexes = torch.topk(buffer**2, self.topK)
                index_tensor = bucket.tensors[0]
                index_tensor.bagua_getter_closure().copy_(indexes)
                count_origin = 0
                count_origin_nonzero = 0
                for tensor in self.tensors:
                    count_origin += tensor.numel()
                    count_origin_nonzero += tensor.count_nonzero().item()
                bucket._other_tensor.bagua_getter_closure().copy_(buffer)
                print("----SparseAlgorithmImpl set_index rank: {}, step: {}, index_size: {}, count: {}, count_nonzero: {}, count_other: {}, count_other_nonzero: {}, count_origin: {}, count_origin_nonzero: {}, other: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, self.topK, indexes.numel(), indexes.count_nonzero().item(), buffer.numel(), buffer.count_nonzero().item(), count_origin, count_origin_nonzero, bucket._other_tensor.bagua_getter_closure().count_nonzero().item()))
                # print("----SparseAlgorithmImpl set_index rank: {}, step: {}, gradient: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, buffer))
                print("----SparseAlgorithmImpl set_index rank: {}, step: {}, value: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, buffer[indexes]))
                print("----SparseAlgorithmImpl set_index rank: {}, step: {}, index: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, indexes))

        def log_func(*args):
            count = 0
            count_nonzero = 0
            count_other = 0
            count_other_nonzero = 0
            count_origin = 0
            count_origin_nonzero = 0
            for tensor in self.tensors:
                count_origin += tensor.numel()
                count_origin_nonzero += tensor.count_nonzero().item()
            count += bucket.tensors[0].bagua_getter_closure().numel()
            count_nonzero += bucket.tensors[0].bagua_getter_closure().count_nonzero().item()
            if hasattr(bucket, "_other_tensor"):
                count_other += bucket._other_tensor.bagua_getter_closure().numel()
                count_other_nonzero += bucket._other_tensor.bagua_getter_closure().count_nonzero().item()
            # print("----SparseAlgorithmImpl log_func rank: {}, step: {}, index_size: {}, count: {}, count_nonzero: {}, count_other: {}, count_other_nonzero: {}, count_origin: {}, count_origin_nonzero: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, self.topK, count, count_nonzero, count_other, count_other_nonzero, count_origin, count_origin_nonzero))
            # print("----SparseAlgorithmImpl log_func rank: {}, step: {}, index: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, bucket.tensors[0].bagua_getter_closure()))
            buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            # real_grad = buffer.clone().detach()
            real_grad = torch.zeros_like(buffer)
            _, _indexes = torch.topk(buffer**2, self.topK)
            real_grad[_indexes]= buffer[_indexes] # += values
            print("--------------rank: {}, step: {}, bucker.index_tensor == _indexes = {},\
               other_tensor == buffer_gradient: {}, other_tensor == real_grad: {}, other_tensor.nonzeor: {}, \
               buffer_gradient nonzero: {}!!!".format(self.rank, bagua_ddp.bagua_train_step_counter, 
               torch.equal(bucket.tensors[0].bagua_getter_closure(), _indexes),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), buffer),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), real_grad),
               bucket._other_tensor.bagua_getter_closure().count_nonzero().item(),
               buffer.count_nonzero().item()))
            print("----SparseAlgorithmImpl log_func rank: {}, step: {}, after value: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, bucket._other_tensor.bagua_getter_closure()))


        bucket.append_python_op(set_index, group=self.process_group)
        bucket.append_python_op(log_func, group=self.process_group)
        bucket.append_centralized_sparse_synchronous_op(
            other_tensor=bucket._other_tensor,
            hierarchical=False,
            group=self.process_group,
        )
        bucket.append_python_op(log_func, group=self.process_group)

class SparseAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topK: int = 0,
    ):
        """
        Create an instance of the Sparse algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = optimizer
        self.topK = topK

    def reify(self, process_group: BaguaProcessGroup) -> SparseAlgorithmImpl:
        return SparseAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )
