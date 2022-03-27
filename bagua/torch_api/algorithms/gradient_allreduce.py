#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.tensor import BaguaTensor
from torch.optim.optimizer import Optimizer
import torch
from typing import List, Tuple

class GradientAllReduceAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = False,
        average: bool = True,
    ):
        """
        Implementation of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        super(GradientAllReduceAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )


class GradientAllReduceAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True):
        """
        Create an instance of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> GradientAllReduceAlgorithmImpl:
        return GradientAllReduceAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )

class GradientAllReduceSketchAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        optimizer: Optimizer,
        hierarchical: bool = False,
        average: bool = True,
        size: Tuple[float, float] = (2, 2),
    ):
        super(GradientAllReduceSketchAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average
        self.size = size
        self.optimizer = optimizer

    def init_tensors(
        self, bagua_ddp: BaguaDistributedDataParallel
    ) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            param.newgrad = torch.zeros(self.size, device=param.device)
            registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.newgrad,
                setter_closure=lambda param, t: setattr(param, "newgrad", t),
            )
            tensors.append(registered_tensor)

        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        return tensors

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                parameter.bagua_mark_communication_ready()

        return hook

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        def sketch(*args):
            for tensor in bucket.tensors:
                t = torch.randn(self.size, device=tensor.grad.device)
                tensor.bagua_setter_closure(t)

        bucket.append_python_op(sketch, group=self.process_group)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )


class GradientAllReduceSketchAlgorithm(Algorithm):
    def __init__(self, optimizer: Optimizer, hierarchical: bool = False, average: bool = True):
        """
        Create an instance of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average
        self.optimizer = optimizer

    def reify(self, process_group: BaguaProcessGroup) -> GradientAllReduceSketchAlgorithmImpl:
        return GradientAllReduceSketchAlgorithmImpl(
            process_group,
            self.optimizer,
            hierarchical=self.hierarchical,
            average=self.average,
        )
