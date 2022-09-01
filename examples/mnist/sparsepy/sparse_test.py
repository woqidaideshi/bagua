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

class SparsePyIndependAlgorithmImpl(AlgorithmImpl):
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
        super(SparsePyIndependAlgorithmImpl, self).__init__(process_group)
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
            self.percent = 100
            self.topK = self.param_size // self.percent
        elif self.topK > self.param_size:
            self.topK = self.param_size
            self.percent = 1
        else:
            self.percent = self.param_size // self.topK
        logging.info("---------param_size: {}, topK: {}".format(self.param_size, self.topK))

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters:
            param.topK = param.numel() // self.percent
            if param.topK < 8:
                if param.numel() < 12:
                    param.topK = param.numel()
                elif param.numel() > 64:
                    param.topK = param.numel() // 8
                else:
                    param.topK = param.numel() // 2
            logging.info("---param name: {}, topK: {}".format(name, param.topK))

            param.index_tensor = torch.zeros(param.topK, dtype=torch.int64).cuda()
            param = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.index_tensor,
                setter_closure=lambda param, t: setattr(param, "index_tensor", t),
            )
            tensors.append(param)
            param.tensors_value = torch.zeros(param.numel(), dtype=torch.float32).cuda()

            # param.recv_messages = torch.zeros(param.topK*self.works, dtype=torch.float32).cuda()
            # param.recv_indexes = torch.zeros(param.topK*self.works, dtype=torch.int64).cuda()
            # param.send_messages = torch.zeros(param.topK, dtype=torch.float32).cuda()
            # param.send_indexes = torch.zeros(param.topK, dtype=torch.int64).cuda()
            # param.tensors_buffer = torch.zeros(param.numel(), dtype=torch.float32).cuda()

        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        print("-----------------init_tensors len(parameters): {}".format(len(parameters)))
        print("-----------------init_tensors len(tensors): {}".format(len(tensors)))
        return tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        count = 0
        print("------rank: {}, tensors_to_buckets len(bucket): {} before".format(bagua.get_rank(), len(tensors)))
        for idx, bucket in enumerate(tensors):
            for tensor in bucket:
                bagua_bucket = BaguaBucket([tensor], flatten=do_flatten, name=str(count))
                count += 1
                bagua_buckets.append(bagua_bucket)

        print("------rank: {}, tensors_to_buckets len(bucket): {} after".format(bagua.get_rank(), len(bagua_buckets)))
        return bagua_buckets

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == parameter.index_tensor.data_ptr()
                ), "bagua backend tensor data_ptr should match parameter index_tensor"
                parameter.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def compare_tmp():
            #     for group in self.optimizer.param_groups:
            #         for param in group["params"]:
            #             if param.is_bagua_tensor():
            #                 buffer = param.grad_clone.view(-1)
            #                 _, indexes = torch.topk(buffer**2, param.topK)
            #                 param.send_indexes.copy_(indexes)
            #                 param.send_messages.copy_(buffer[indexes])
            #                 bagua.allgather(param.send_indexes, param.recv_indexes)
            #                 bagua.allgather(param.send_messages, param.recv_messages)
            #                 torch.cuda.synchronize()
            #                 param.tensors_buffer.zero_()
            #                 for rank in range(self.works):
            #                     start = rank * param.topK
            #                     end = start + param.topK
            #                     param.tensors_buffer[param.recv_indexes[start:end]] += param.recv_messages[start:end]
            #                 param.tensors_buffer.div_(self.works)
            #                 buffer[param.tensors_buffer.nonzero()] = 0.0
            #                 buffer.add_(param.tensors_buffer)
            #                 print("----SparsePy2AlgorithmImpl init_post_backward_hook rank: {}, step: {}, grad_clone == grad: {}, grad nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(param.grad_clone, param.grad), param.grad.count_nonzero().item()))

            # compare_tmp()
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
        num_ele = bucket.tensors[0].topK
        bucket._recv_value = torch.zeros(num_ele*self.works, dtype=torch.float32).cuda().ensure_bagua_tensor(
            "recv_value", bagua_ddp.bagua_module_name
        )
        bucket._recv_index = torch.zeros(num_ele*self.works, dtype=torch.int64).cuda().ensure_bagua_tensor(
            "recv_index", bagua_ddp.bagua_module_name
        )
        bucket._send_value = torch.zeros(num_ele, dtype=torch.float32).cuda().ensure_bagua_tensor(
            "send_value", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
        bucket.clear_ops()
        def set_index(*args):
            tensor = bucket.tensors[0]
            buffer = tensor.grad.view(-1)
            _, indexes = torch.topk(buffer**2, tensor.topK)
            tensor.bagua_getter_closure().copy_(indexes)
            bucket._send_value.bagua_getter_closure().copy_(buffer[indexes])

            # tensor.grad_clone = tensor.grad.clone().detach()

        def get_index(*args):
            recv_index = bucket._recv_index.bagua_getter_closure()
            recv_value = bucket._recv_value.bagua_getter_closure()
            num_ele = bucket.tensors[0].topK

            tensor = bucket.tensors[0]
            buffer = tensor.grad.view(-1)
            tensor.tensors_value.zero_()
            start = 0
            for rank in range(self.works):
                tensor.tensors_value[recv_index[start:start+num_ele]] += recv_value[start:start+num_ele]
                start += num_ele
            tensor.tensors_value.div_(self.works)
            buffer[tensor.tensors_value.nonzero()] = 0.0
            buffer.add_(tensor.tensors_value)

        bucket.append_python_op(set_index, group=self.process_group)
        bucket.append_centralized_sparse_py_synchronous_op(
            recv_value=bucket._recv_value,
            recv_index=bucket._recv_index,
            send_value=bucket._send_value,
            hierarchical=False,
            group=self.process_group,
        )
        bucket.append_python_op(get_index, group=self.process_group)

class SparsePy2AlgorithmImpl(AlgorithmImpl):
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
        super(SparsePy2AlgorithmImpl, self).__init__(process_group)
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
            self.percent = 100
            self.topK = self.param_size // self.percent
        elif self.topK > self.param_size:
            self.topK = self.param_size
            self.percent = 1
        else:
            self.percent = self.param_size // self.topK
        # self.recv_messages = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        # self.recv_indexes = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        # self.send_messages = torch.zeros(self.topK, dtype=torch.float32).cuda()
        # self.send_indexes = torch.zeros(self.topK, dtype=torch.int64).cuda()
        # self.tensors_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()

        # self.recv_value = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        # self.recv_index = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        # self.send_value = torch.zeros(self.topK, dtype=torch.float32).cuda()
        # self.tensors_value = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        # self.other_tensor_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        # self.value_tensor_buffer = torch.zeros(self.topK, dtype=torch.float32).cuda()
        logging.info("---------param_size: {}, topK: {}".format(self.param_size, self.topK))

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        self.tensors = []
        # name, param = parameters[-1]
        # param.index_tensor = torch.zeros(self.topK, dtype=torch.int64).cuda()

        # self.index_tensor = param.ensure_bagua_tensor(
        #     name,
        #     bagua_ddp.bagua_module_name,
        #     getter_closure=lambda param: param.index_tensor,
        #     setter_closure=lambda param, t: setattr(param, "index_tensor", t),
        # )
        # self._communication_tensor_names = set((name,))
        # return [self.index_tensor]
        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters:
            param.topK = param.numel() // self.percent
            if param.topK < 8:
                if param.numel() < 12:
                    param.topK = param.numel()
                elif param.numel() > 64:
                    param.topK = param.numel() // 8
                else:
                    param.topK = param.numel() // 2
            logging.info("---param name: {}, topK: {}".format(name, param.topK))

            param.index_tensor = torch.zeros(param.topK, dtype=torch.int64).cuda()
            param = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.index_tensor,
                setter_closure=lambda param, t: setattr(param, "index_tensor", t),
            )
            tensors.append(param)
            param.tensors_value = torch.zeros(param.numel(), dtype=torch.float32).cuda()

            # param.recv_messages = torch.zeros(param.topK*self.works, dtype=torch.float32).cuda()
            # param.recv_indexes = torch.zeros(param.topK*self.works, dtype=torch.int64).cuda()
            # param.send_messages = torch.zeros(param.topK, dtype=torch.float32).cuda()
            # param.send_indexes = torch.zeros(param.topK, dtype=torch.int64).cuda()
            # param.tensors_buffer = torch.zeros(param.numel(), dtype=torch.float32).cuda()

        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        print("-----------------init_tensors len(parameters): {}".format(len(parameters)))
        print("-----------------init_tensors len(tensors): {}".format(len(tensors)))
        return tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        print("------rank: {}, tensors_to_buckets len(bucket): {}".format(bagua.get_rank(), len(tensors)))
        for idx, bucket in enumerate(tensors):
            print("------rank: {}, tensors_to_buckets len(tensors): {}".format(bagua.get_rank(), len(bucket)))
            bagua_bucket = BaguaBucket(
                bucket, flatten=do_flatten, name=str(idx)
            )  # TODO: check duplicated names
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == parameter.index_tensor.data_ptr()
                ), "bagua backend tensor data_ptr should match parameter index_tensor"
                parameter.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def compare_tmp():
            #     for group in self.optimizer.param_groups:
            #         for param in group["params"]:
            #             if param.is_bagua_tensor():
            #                 buffer = param.grad_clone.view(-1)
            #                 _, indexes = torch.topk(buffer**2, param.topK)
            #                 param.send_indexes.copy_(indexes)
            #                 param.send_messages.copy_(buffer[indexes])
            #                 bagua.allgather(param.send_indexes, param.recv_indexes)
            #                 bagua.allgather(param.send_messages, param.recv_messages)
            #                 torch.cuda.synchronize()
            #                 param.tensors_buffer.zero_()
            #                 for rank in range(self.works):
            #                     start = rank * param.topK
            #                     end = start + param.topK
            #                     param.tensors_buffer[param.recv_indexes[start:end]] += param.recv_messages[start:end]
            #                 param.tensors_buffer.div_(self.works)
            #                 buffer[param.tensors_buffer.nonzero()] = 0.0
            #                 buffer.add_(param.tensors_buffer)
            #                 print("----SparsePy2AlgorithmImpl init_post_backward_hook rank: {}, step: {}, grad_clone == grad: {}, grad nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(param.grad_clone, param.grad), param.grad.count_nonzero().item()))

            # compare_tmp()
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
        num_ele = sum(tensor.topK for tensor in bucket.tensors)
        bucket._recv_value = torch.zeros(num_ele*self.works, dtype=torch.float32).cuda().ensure_bagua_tensor(
            "recv_value", bagua_ddp.bagua_module_name
        )
        bucket._recv_index = torch.zeros(num_ele*self.works, dtype=torch.int64).cuda().ensure_bagua_tensor(
            "recv_index", bagua_ddp.bagua_module_name
        )
        bucket._send_value = torch.zeros(num_ele, dtype=torch.float32).cuda().ensure_bagua_tensor(
            "send_value", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
        bucket.clear_ops()
        def set_index(*args):
            start = 0
            for tensor in bucket.tensors:
                buffer = tensor.grad.view(-1)
                _, indexes = torch.topk(buffer**2, tensor.topK)
                tensor.bagua_getter_closure().copy_(indexes)
                bucket._send_value.bagua_getter_closure()[start:start+tensor.topK] = buffer[indexes]
                start += tensor.topK

                # tensor.grad_clone = tensor.grad.clone().detach()

        def get_index(*args):
            recv_index = bucket._recv_index.bagua_getter_closure()
            recv_value = bucket._recv_value.bagua_getter_closure()
            num_ele = sum(tensor.topK for tensor in bucket.tensors)
            start = 0
            for tensor in bucket.tensors:
                buffer = tensor.grad.view(-1)
                tensor.tensors_value.zero_()
                for rank in range(self.works):
                    start_index = start + rank * num_ele
                    tensor.tensors_value[recv_index[start_index:start_index+tensor.topK]] += recv_value[start_index:start_index+tensor.topK]
                tensor.tensors_value.div_(self.works)
                buffer[tensor.tensors_value.nonzero()] = 0.0
                buffer.add_(tensor.tensors_value)
                start += tensor.topK

                # index_tensors = []
                # for rank in range(self.works):
                #     start_index = start + rank * num_ele
                #     index_tensors.append(recv_index[start_index:start_index+tensor.topK])
                # index_unique = torch.cat(index_tensors).unique(return_counts=False)
                # buffer[index_unique] = 0.0
                # for rank in range(self.works):
                #     start_index = start + rank * num_ele
                #     buffer[recv_index[start_index:start_index+tensor.topK]] += recv_value[start_index:start_index+tensor.topK]
                # buffer[index_unique] /= self.works
                # start += tensor.topK

        bucket.append_python_op(set_index, group=self.process_group)
        bucket.append_centralized_sparse_py_synchronous_op(
            recv_value=bucket._recv_value,
            recv_index=bucket._recv_index,
            send_value=bucket._send_value,
            hierarchical=False,
            group=self.process_group,
        )
        bucket.append_python_op(get_index, group=self.process_group)

class SparsePyAlgorithmImpl(AlgorithmImpl):
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
        super(SparsePyAlgorithmImpl, self).__init__(process_group)
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
        # self.recv_messages = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        # self.recv_indexes = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        # self.send_messages = torch.zeros(self.topK, dtype=torch.float32).cuda()
        # self.send_indexes = torch.zeros(self.topK, dtype=torch.int64).cuda()
        # self.tensors_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()

        self.recv_value = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        self.recv_index = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        self.send_value = torch.zeros(self.topK, dtype=torch.float32).cuda()
        self.tensors_value = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        # self.other_tensor_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        # self.value_tensor_buffer = torch.zeros(self.topK, dtype=torch.float32).cuda()
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
        name, param = parameters[-1]
        param.index_tensor = torch.zeros(self.topK, dtype=torch.int64).cuda()

        self.index_tensor = param.ensure_bagua_tensor(
            name,
            bagua_ddp.bagua_module_name,
            getter_closure=lambda param: param.index_tensor,
            setter_closure=lambda param, t: setattr(param, "index_tensor", t),
        )
        self._communication_tensor_names = set((name,))
        return [self.index_tensor]

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
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            self.index_tensor.bagua_mark_communication_ready()
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def pack():
            #     """Packs a list of tensors into one buffer for sending to other workers"""
            #     buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            #     _, indexes = torch.topk(buffer**2, self.topK)
            #     self.send_indexes.copy_(indexes)
            #     self.send_messages.copy_(buffer[indexes])

            # def unpack():
            #     """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
            #     self.tensors_buffer.zero_()
            #     for rank in range(self.works):
            #         start = rank * self.topK
            #         end = start + self.topK
            #         self.tensors_buffer[self.recv_indexes[start:end]] += (self.recv_messages[start:end])
            #     self.tensors_buffer.div_(self.works)

            # def test():
            #     print("----SparsePyAlgorithmImpl init_post_backward_hook rank: {}, step: {}, tensors_buffer == other_tensor: {}, tensors_buffer nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(self.tensors_buffer, self.tensors_value), self.tensors_buffer.count_nonzero().item()))

            def unpack2tensors():
                self.tensors_value.zero_()
                for rank in range(self.works):
                    start = rank * self.topK
                    end = start + self.topK
                    self.tensors_value[self.recv_index[start:end]] += (self.recv_value[start:end])
                self.tensors_value.div_(self.works)
                size = 0
                for tensor in self.tensors:
                    count = tensor.numel()
                    tmp_buffer = self.tensors_value[size:count+size]
                    tmp_tensor = tensor.view(-1)
                    tmp_tensor[tmp_buffer.nonzero()] = 0.0
                    tmp_tensor.add_(tmp_buffer)
                    size += count

            # pack()
            # bagua.allgather(self.send_indexes, self.recv_indexes)
            # bagua.allgather(self.send_messages, self.recv_messages)
            # torch.cuda.synchronize()
            # unpack()
            unpack2tensors()
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
        bucket._recv_value = self.recv_value.ensure_bagua_tensor(
            "recv_value", bagua_ddp.bagua_module_name
        )
        bucket._recv_index = self.recv_index.ensure_bagua_tensor(
            "recv_index", bagua_ddp.bagua_module_name
        )
        bucket._send_value = self.send_value.ensure_bagua_tensor(
            "send_value", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
        bucket.clear_ops()
        def set_index(*args):
            if hasattr(bucket, "_send_value"):
                buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
                _, indexes = torch.topk(buffer**2, self.topK)
                index_tensor = bucket.tensors[0]
                index_tensor.bagua_getter_closure().copy_(indexes)
                count_origin = 0
                count_origin_nonzero = 0
                for tensor in self.tensors:
                    count_origin += tensor.numel()
                    count_origin_nonzero += tensor.count_nonzero().item()
                bucket._send_value.bagua_getter_closure().copy_(buffer[indexes])

        bucket.append_python_op(set_index, group=self.process_group)
        bucket.append_centralized_sparse_py_synchronous_op(
            recv_value=bucket._recv_value,
            recv_index=bucket._recv_index,
            send_value=bucket._send_value,
            hierarchical=False,
            group=self.process_group,
        )

class SparseIndependAlgorithmImpl(AlgorithmImpl):
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
        super(SparseIndependAlgorithmImpl, self).__init__(process_group)
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
            self.percent = 100
            self.topK = self.param_size // self.percent
        elif self.topK > self.param_size:
            self.topK = self.param_size
            self.percent = 1
        else:
            self.percent = self.param_size // self.topK
        logging.info("---------param_size: {}, topK: {}".format(self.param_size, self.topK))

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters:
            param.topK = param.numel() // self.percent
            if param.topK < 8:
                if param.numel() < 12:
                    param.topK = param.numel()
                elif param.numel() > 64:
                    param.topK = param.numel() // 8
                else:
                    param.topK = param.numel() // 2
            logging.info("---param name: {}, topK: {}".format(name, param.topK))

            param.index_tensor = torch.zeros(param.topK, dtype=torch.int64).cuda()
            param = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.index_tensor,
                setter_closure=lambda param, t: setattr(param, "index_tensor", t),
            )
            tensors.append(param)

            # param.recv_messages = torch.zeros(param.topK*self.works, dtype=torch.float32).cuda()
            # param.recv_indexes = torch.zeros(param.topK*self.works, dtype=torch.int64).cuda()
            # param.send_messages = torch.zeros(param.topK, dtype=torch.float32).cuda()
            # param.send_indexes = torch.zeros(param.topK, dtype=torch.int64).cuda()
            # param.tensors_buffer = torch.zeros(param.numel(), dtype=torch.float32).cuda()

        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        print("-----------------init_tensors len(parameters): {}".format(len(parameters)))
        print("-----------------init_tensors len(tensors): {}".format(len(tensors)))
        return tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        count = 0
        print("------rank: {}, tensors_to_buckets len(bucket): {} before".format(bagua.get_rank(), len(tensors)))
        for idx, bucket in enumerate(tensors):
            for tensor in bucket:
                bagua_bucket = BaguaBucket([tensor], flatten=do_flatten, name=str(count))
                count += 1
                bagua_buckets.append(bagua_bucket)
        print("------rank: {}, tensors_to_buckets len(bucket): {} after".format(bagua.get_rank(), len(bagua_buckets)))

        return bagua_buckets

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == parameter.index_tensor.data_ptr()
                ), "bagua backend tensor data_ptr should match parameter index_tensor"
                parameter.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def compare_tmp():
            #     for group in self.optimizer.param_groups:
            #         for param in group["params"]:
            #             if param.is_bagua_tensor():
            #                 buffer = param.grad_clone.view(-1)
            #                 _, indexes = torch.topk(buffer**2, param.topK)
            #                 param.send_indexes.copy_(indexes)
            #                 param.send_messages.copy_(buffer[indexes])
            #                 bagua.allgather(param.send_indexes, param.recv_indexes)
            #                 bagua.allgather(param.send_messages, param.recv_messages)
            #                 torch.cuda.synchronize()
            #                 param.tensors_buffer.zero_()
            #                 for rank in range(self.works):
            #                     start = rank * param.topK
            #                     end = start + param.topK
            #                     param.tensors_buffer[param.recv_indexes[start:end]] += param.recv_messages[start:end]
            #                 param.tensors_buffer.div_(self.works)
            #                 buffer[param.tensors_buffer.nonzero()] = 0.0
            #                 buffer.add_(param.tensors_buffer)
            #                 print("----SparsePy2AlgorithmImpl init_post_backward_hook rank: {}, step: {}, grad_clone == grad: {}, grad nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(param.grad_clone, param.grad), param.grad.count_nonzero().item()))

            # compare_tmp()
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
        bucket._other_tensor = torch.zeros(bucket.tensors[0].numel(), dtype=torch.float32).cuda().ensure_bagua_tensor(
            "other_tensor", bagua_ddp.bagua_module_name
        )
        bucket._value_tensor = torch.zeros(bucket.tensors[0].topK, dtype=torch.float32).cuda().ensure_bagua_tensor(
            "value_tensor", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
        bucket.clear_ops()
        def set_index(*args):
            tensor = bucket.tensors[0]
            buffer = tensor.grad.view(-1)
            _, indexes = torch.topk(buffer**2, tensor.topK)
            tensor.bagua_getter_closure().copy_(indexes)
            bucket._other_tensor.bagua_getter_closure().zero_()
            bucket._value_tensor.bagua_getter_closure().copy_(buffer[indexes])

            # tensor.grad_clone = tensor.grad.clone().detach()

        def get_index(*args):
            buffer = bucket.tensors[0].grad.view(-1)
            buffer[bucket._other_tensor.bagua_getter_closure().nonzero()] = 0.0
            buffer.add_(bucket._other_tensor.bagua_getter_closure())

        bucket.append_python_op(set_index, group=self.process_group)
        bucket.append_centralized_sparse_synchronous_op(
            value_tensor=bucket._value_tensor,
            other_tensor=bucket._other_tensor,
            hierarchical=False,
            group=self.process_group,
        )
        bucket.append_python_op(get_index, group=self.process_group)

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
        # self.recv_messages = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        # self.recv_indexes = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        # self.send_messages = torch.zeros(self.topK, dtype=torch.float32).cuda()
        # self.send_indexes = torch.zeros(self.topK, dtype=torch.int64).cuda()
        # self.tensors_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.other_tensor_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.value_tensor_buffer = torch.zeros(self.topK, dtype=torch.float32).cuda()
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
        name, param = parameters[-1]
        param.index_tensor = torch.zeros(self.topK, dtype=torch.int64).cuda()

        self.index_tensor = param.ensure_bagua_tensor(
            name,
            bagua_ddp.bagua_module_name,
            getter_closure=lambda param: param.index_tensor,
            setter_closure=lambda param, t: setattr(param, "index_tensor", t),
        )
        self._communication_tensor_names = set((name,))
        return [self.index_tensor]

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
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            self.index_tensor.bagua_mark_communication_ready()
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def pack():
            #     """Packs a list of tensors into one buffer for sending to other workers"""
            #     buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            #     _, indexes = torch.topk(buffer**2, self.topK)
            #     self.send_indexes.copy_(indexes)
            #     self.send_messages.copy_(buffer[indexes])

            # def unpack():
            #     """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
            #     self.tensors_buffer.zero_()
            #     for rank in range(self.works):
            #         start = rank * self.topK
            #         end = start + self.topK
            #         self.tensors_buffer[self.recv_indexes[start:end]] += (self.recv_messages[start:end])
            #     self.tensors_buffer.div_(self.works)

            # def test():
            #     print("----SparseAlgorithmImpl init_post_backward_hook rank: {}, step: {}, tensors_buffer == other_tensor: {}, tensors_buffer nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(self.tensors_buffer, self.other_tensor_buffer), self.tensors_buffer.count_nonzero().item()))

            def unpack2tensors():
                size = 0
                for tensor in self.tensors:
                    shape = tensor.shape
                    count = tensor.numel()
                    tmp_buffer = self.other_tensor_buffer[size:count+size]
                    tmp_tensor = tensor.view(-1)
                    tmp_tensor[tmp_buffer.nonzero()] = 0.0
                    tmp_tensor.add_(tmp_buffer)
                    size += count

            # pack()
            # bagua.allgather(self.send_indexes, self.recv_indexes)
            # bagua.allgather(self.send_messages, self.recv_messages)
            # torch.cuda.synchronize()
            # unpack()
            # test()
            unpack2tensors()

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
        bucket._other_tensor = self.other_tensor_buffer.ensure_bagua_tensor(
            "other_tensor", bagua_ddp.bagua_module_name
        )
        bucket._value_tensor = self.value_tensor_buffer.ensure_bagua_tensor(
            "value_tensor", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
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
                bucket._other_tensor.bagua_getter_closure().zero_()
                bucket._value_tensor.bagua_getter_closure().copy_(buffer[indexes])

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
            buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            # real_grad = buffer.clone().detach()
            real_grad = torch.zeros_like(buffer)
            _, _indexes = torch.topk(buffer**2, self.topK)
            real_grad[_indexes]= buffer[_indexes] # += values
            print("--------------SparseAlgorithmImpl rank: {}, step: {}, bucker.index_tensor == _indexes = {},\
               other_tensor == buffer_gradient: {}, other_tensor == real_grad: {},  self.other_tensor_buffer == real_grad: {}, other_tensor.nonzeor: {}, \
               buffer_gradient nonzero: {}!!!".format(self.rank, bagua_ddp.bagua_train_step_counter, 
               torch.equal(bucket.tensors[0].bagua_getter_closure(), _indexes),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), buffer),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), real_grad),
               torch.equal(self.other_tensor_buffer, real_grad),
               bucket._other_tensor.bagua_getter_closure().count_nonzero().item(),
               buffer.count_nonzero().item()))


        bucket.append_python_op(set_index, group=self.process_group)
        # bucket.append_python_op(log_func, group=self.process_group)
        bucket.append_centralized_sparse_synchronous_op(
            value_tensor=bucket._value_tensor,
            other_tensor=bucket._other_tensor,
            hierarchical=False,
            group=self.process_group,
        )
        # bucket.append_python_op(log_func, group=self.process_group)

class SparseInplaceAlgorithmImpl(AlgorithmImpl):
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
        super(SparseInplaceAlgorithmImpl, self).__init__(process_group)
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
        # self.recv_messages = torch.zeros(self.topK*self.works, dtype=torch.float32).cuda()
        # self.recv_indexes = torch.zeros(self.topK*self.works, dtype=torch.int64).cuda()
        # self.send_messages = torch.zeros(self.topK, dtype=torch.float32).cuda()
        # self.send_indexes = torch.zeros(self.topK, dtype=torch.int64).cuda()
        # self.tensors_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.other_tensor_buffer = torch.zeros(self.param_size, dtype=torch.float32).cuda()
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
        name, param = parameters[-1]
        param.index_tensor = torch.zeros(self.topK, dtype=torch.int64).cuda()

        self.index_tensor = param.ensure_bagua_tensor(
            name,
            bagua_ddp.bagua_module_name,
            getter_closure=lambda param: param.index_tensor,
            setter_closure=lambda param, t: setattr(param, "index_tensor", t),
        )
        self._communication_tensor_names = set((name,))
        return [self.index_tensor]

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
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            self.index_tensor.bagua_mark_communication_ready()
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            # def pack():
            #     """Packs a list of tensors into one buffer for sending to other workers"""
            #     buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            #     _, indexes = torch.topk(buffer**2, self.topK)
            #     self.send_indexes.copy_(indexes)
            #     self.send_messages.copy_(buffer[indexes])

            # def unpack():
            #     """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
            #     self.tensors_buffer.zero_()
            #     for rank in range(self.works):
            #         start = rank * self.topK
            #         end = start + self.topK
            #         self.tensors_buffer[self.recv_indexes[start:end]] += (self.recv_messages[start:end])
            #     self.tensors_buffer.div_(self.works)

            # def test():
            #     print("----SparseInplaceAlgorithmImpl init_post_backward_hook rank: {}, step: {}, tensors_buffer == other_tensor: {}, tensors_buffer nonzero size: {}.".format(self.rank, bagua_ddp.bagua_train_step_counter, torch.equal(self.tensors_buffer, self.other_tensor_buffer), self.tensors_buffer.count_nonzero().item()))

            def unpack2tensors():
                size = 0
                for tensor in self.tensors:
                    shape = tensor.shape
                    count = tensor.numel()
                    tmp_buffer = self.other_tensor_buffer[size:count+size]
                    tmp_tensor = tensor.view(-1)
                    tmp_tensor[tmp_buffer.nonzero()] = 0.0
                    tmp_tensor.add_(tmp_buffer)
                    size += count

            # pack()
            # bagua.allgather(self.send_indexes, self.recv_indexes)
            # bagua.allgather(self.send_messages, self.recv_messages)
            # torch.cuda.synchronize()
            # unpack()
            # test()
            unpack2tensors()

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
        bucket._other_tensor = self.other_tensor_buffer.ensure_bagua_tensor(
            "other_tensor", bagua_ddp.bagua_module_name
        )
        torch.cuda.synchronize()
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
            buffer = torch.cat([t.view(-1) for t in self.tensors])  # copies
            # real_grad = buffer.clone().detach()
            real_grad = torch.zeros_like(buffer)
            _, _indexes = torch.topk(buffer**2, self.topK)
            real_grad[_indexes]= buffer[_indexes] # += values
            print("--------------SparseInplaceAlgorithmImpl rank: {}, step: {}, bucker.index_tensor == _indexes = {},\
               other_tensor == buffer_gradient: {}, other_tensor == real_grad: {},  self.other_tensor_buffer == real_grad: {}, other_tensor.nonzeor: {}, \
               buffer_gradient nonzero: {}!!!".format(self.rank, bagua_ddp.bagua_train_step_counter, 
               torch.equal(bucket.tensors[0].bagua_getter_closure(), _indexes),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), buffer),
               torch.equal(bucket._other_tensor.bagua_getter_closure(), real_grad),
               torch.equal(self.other_tensor_buffer, real_grad),
               bucket._other_tensor.bagua_getter_closure().count_nonzero().item(),
               buffer.count_nonzero().item()))


        bucket.append_python_op(set_index, group=self.process_group)
        # bucket.append_python_op(log_func, group=self.process_group)
        bucket.append_centralized_sparse_inplace_synchronous_op(
            other_tensor=bucket._other_tensor,
            hierarchical=False,
            group=self.process_group,
        )
        # bucket.append_python_op(log_func, group=self.process_group)

class SparsePyIndependAlgorithm(Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> SparsePyIndependAlgorithmImpl:
        return SparsePyIndependAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )

class SparsePy2Algorithm(Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> SparsePy2AlgorithmImpl:
        return SparsePy2AlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )

class SparsePyAlgorithm(Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> SparsePyAlgorithmImpl:
        return SparsePyAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )

class SparseIndependAlgorithm(Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> SparseIndependAlgorithmImpl:
        return SparseIndependAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )

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

class SparseInplaceAlgorithm(Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> SparseInplaceAlgorithmImpl:
        return SparseInplaceAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topK=self.topK,
        )
