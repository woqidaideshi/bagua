#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from typing import List


class ByteGradAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        average: bool = True,
    ):
        """
        Implementation of the
        `ByteGrad <https://tutorials.baguasys.com/algorithms/bytegrad>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        super(ByteGradAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        def loginfo(*args):
            length = 0
            for tensor in bucket.tensors:
                length += tensor.bagua_getter_closure().numel();
            print("----ByteGradAlgorithmImpl loginfo size: {}, len: {}".format(len(bucket.tensors), length))
        bucket.append_python_op(loginfo, group=self.process_group)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            scattergather=True,
            compression="MinMaxUInt8",
            group=self.process_group,
        )

class Float16GradAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        average: bool = True,
    ):
        """
        Implementation of the
        `Float16Grad <https://tutorials.baguasys.com/algorithms/bytegrad>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        super(Float16GradAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        # def to_float16(*args):
        #     for tensor in bucket.tensors:
        #         print(tensor.bagua_getter_closure().dtype)
        #         tensor.bagua_getter_closure().half()
        #         print(tensor.bagua_getter_closure().dtype)
        #         tensor.bagua_setter_closure(tensor.bagua_getter_closure().half())
        #         print(tensor.bagua_getter_closure().dtype)
        # bucket.append_python_op(to_float16, group=self.process_group)
        # bucket.append_centralized_synchronous_op(
        #     hierarchical=self.hierarchical,
        #     average=self.average,
        #     scattergather=True,
        #     group=self.process_group,
        # )

        bucket.append_centralized_test_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            scattergather=True,
            compression="Float2Half",
            group=self.process_group,
        )


class ByteGradAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True, average: bool = True):
        """
        Create an instance of the
        `ByteGrad <https://tutorials.baguasys.com/algorithms/bytegrad>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> ByteGradAlgorithmImpl:
        return ByteGradAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )

class Float16GradAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True, average: bool = True):
        """
        Create an instance of the
        `Float16Grad <https://tutorials.baguasys.com/algorithms/float16grad>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> Float16GradAlgorithmImpl:
        return Float16GradAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )
