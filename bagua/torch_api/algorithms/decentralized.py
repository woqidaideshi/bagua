#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.contrib.fuse.optimizer import is_fused_optimizer
from torch.optim.optimizer import Optimizer
from typing import List, Tuple
import torch
import math


class DecentralizedAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
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
        super(DecentralizedAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        print("----DecentralizedAlgorithmImpl init_tensors({}).".format(bagua_ddp.bagua_train_step_counter - 1))
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        print("----DecentralizedAlgorithmImpl tensors_to_buckets.")
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=do_flatten, name=str(0))

        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            print("----DecentralizedAlgorithmImpl init_forward_pre_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            # print(self.tensors)
            if self._should_communicate(bagua_ddp):
                print("----DecentralizedAlgorithmImpl init_forward_pre_hook_should_communicate({}).".format(bagua_ddp.bagua_train_step_counter - 1))
                for tensor in self.tensors:
                    tensor.bagua_mark_communication_ready()
                # bagua_ddp._bagua_backend.wait_pending_comm_ops()
                # print(self.tensors)

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            print("----DecentralizedAlgorithmImpl init_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            # print(self.tensors)
            return

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            print("----DecentralizedAlgorithmImpl init_post_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            # print(self.tensors)
            if self._should_communicate(bagua_ddp):
                print("----DecentralizedAlgorithmImpl init_post_backward_hook_should_communicate({}).".format(bagua_ddp.bagua_train_step_counter - 1))
                bagua_ddp._bagua_backend.wait_pending_comm_ops()

                torch.cuda.current_stream().record_event(self.cuda_event)
                self.cuda_event.synchronize()
                for bucket in bagua_ddp.bagua_buckets:
                    bucket._decentralized_op.copy_back_peer_weight(
                        bucket.backend_bucket
                    )
                # print(self.tensors)

        return hook

    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed when the
        ``optimizer.step()`` is done.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that gets called after an optimizer's ``step()`` method is called. The function takes the optimizer as its argument.
        """

        def hook(optimizer: torch.optim.Optimizer):
            print("----DecentralizedAlgorithmImpl init_post_optimizer_step_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        bucket._peer_weight = weight_tensor.ensure_bagua_tensor("peer_weight")

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        print("----DecentralizedAlgorithmImpl init_operations({}).".format(bagua_ddp.bagua_train_step_counter - 1))
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()
        def append_python_func(*args):
            print("----DecentralizedAlgorithmImpl append_python_func({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            # print(self.tensors)

        bucket.append_python_op(append_python_func, group=self.process_group)
        decentralized_op = bucket.append_decentralized_synchronous_op(
            peer_weight=bucket._peer_weight,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            group=self.process_group,
        )
        bucket._decentralized_op = decentralized_op


class LowPrecisionDecentralizedAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        communication_interval: int = 1,
    ):
        """
        Implementation of the
        `Low Precision Decentralized SGD <https://tutorials.baguasys.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        super(LowPrecisionDecentralizedAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        print("----LowPrecisionDecentralizedAlgorithmImpl init_tensors({}).".format(bagua_ddp.bagua_train_step_counter - 1))
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        optimizer_param_ids = [
            id(param)
            for optimizer in bagua_ddp.bagua_optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]

        for name, param in parameters:
            if id(param) not in optimizer_param_ids:
                raise RuntimeError(
                    f"Module parameter {name} is not used by your optimizer(s), need to exclude it "
                    "by adding the parameter name to the `List` attribute `_bagua_params_and_buffers_to_ignore` "
                    "of your module."
                )
        return self.tensors

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed before the
        forward process.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes the model's input.
        """

        def hook(input):
            print("-----LowPrecisionDecentralizedAlgorithmImpl init_forward_pre_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            print("----LowPrecisionDecentralizedAlgorithmImpl init_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            print("----LowPrecisionDecentralizedAlgorithmImpl init_post_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            assert not is_fused_optimizer(
                optimizer
            ), "Low decentralized algorithm can not work with fused optimizer at present."
            print("----LowPrecisionDecentralizedAlgorithmImpl init_post_optimizer_step_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            if self._should_communicate(bagua_ddp):
                print("----LowPrecisionDecentralizedAlgorithmImpl init_post_optimizer_step_hook_should_communicate({}).".format(bagua_ddp.bagua_train_step_counter - 1))
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.is_bagua_tensor():
                            param.bagua_mark_communication_ready()

                bagua_ddp._bagua_backend.wait_pending_comm_ops()

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        left_peer_weight_tensor = bucket.flattened_tensor()
        right_peer_weight_tensor = bucket.flattened_tensor()

        bucket._weight = weight_tensor.ensure_bagua_tensor("weight")
        bucket._left_peer_weight = left_peer_weight_tensor.ensure_bagua_tensor(
            "left_peer_weight"
        )
        bucket._right_peer_weight = right_peer_weight_tensor.ensure_bagua_tensor(
            "right_peer_weight"
        )

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        print("----LowPrecisionDecentralizedAlgorithmImpl init_operations({}).".format(bagua_ddp.bagua_train_step_counter - 1))
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()
        def append_python_func(*args):
            print("----LowPrecisionDecentralizedAlgorithmImpl append_python_func({}).".format(bagua_ddp.bagua_train_step_counter - 1))

        bucket.append_python_op(append_python_func, group=self.process_group)
        bucket.append_low_precision_decentralized_synchronous_op(
            weight=bucket._weight,
            left_peer_weight=bucket._left_peer_weight,
            right_peer_weight=bucket._right_peer_weight,
            hierarchical=self.hierarchical,
            compression="MinMaxUInt8",
            group=self.process_group,
        )


class DecentralizedAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
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

    def reify(self, process_group: BaguaProcessGroup) -> DecentralizedAlgorithmImpl:
        return DecentralizedAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )


class LowPrecisionDecentralizedAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True, communication_interval: int = 1):
        """
        Create an instance of the
        `Low Precision Decentralized SGD <https://tutorials.baguasys.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval

    def reify(
        self, process_group: BaguaProcessGroup
    ) -> LowPrecisionDecentralizedAlgorithmImpl:
        return LowPrecisionDecentralizedAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
        )

class QGAdamOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Create a dedicated optimizer used for
        `QAdam <https://tutorials.baguasys.com/algorithms/q-adam>`_ algorithm.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(QGAdamOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QGAdamOptimizer, self).__setstate__(state)

    def _compute_tmp(self):
        state_step = 0
        for group_id, group in enumerate(self.param_groups):
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # state["param"] = torch.zeros_like(
                    #     param, memory_format=torch.preserve_format
                    # )
                state["param"] = torch.clone(param, memory_format=torch.preserve_format).detach()

                state["step"] += 1
                step_id = state["step"]
                state_step = step_id
                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** step_id
                bias_correction2 = 1 - beta2 ** step_id

                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
                step_size = lr / bias_correction1
                update = state["exp_avg"] / denom
                param.data.add_(-step_size * update)
        print("----QGAdamOptimizer _compute step({}) completed.".format(state_step))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        state_step = 0
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]
                update = state["param"] - param.data
                norm = update.norm(p=2)
                update.div_(norm + eps)

                state["exp_avg"].mul_(beta1).add_(
                    update, alpha=1 - beta1
                )
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    update, update, value=1 - beta2,
                )
                # state["param"].sub_(param.data)
                # norm = state["param"].norm(p=2) + eps
                # # state["param"].div_(norm)

                # state["exp_avg"].mul_(beta1).add_(
                #     state["param"], alpha=1 - beta1
                # )
                # state["exp_avg_sq"].mul_(beta2).addcmul_(
                #     state["param"], state["param"], value=1 - beta2,
                # )
                state_step = state["step"]
        print("----QGAdamOptimizer step({}) completed.".format(state_step))

        return loss

class QGAdamLowPrecisionDecentralizedAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        qg_adam_optimizer: QGAdamOptimizer,
        hierarchical: bool = True,
        communication_interval: int = 1,
    ):
        """
        Implementation of the
        `Low Precision Decentralized SGD <https://tutorials.baguasys.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        super(QGAdamLowPrecisionDecentralizedAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = qg_adam_optimizer

    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_tensors({}).".format(bagua_ddp.bagua_train_step_counter - 1))

        parameters = bagua_ddp.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._q_adam_name = name
            param._q_adam_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                    param._q_adam_name,
                    bagua_ddp.bagua_module_name,
                    getter_closure=lambda param: param.data,
                    setter_closure=lambda param, t: setattr(param, "data", t),
                )

                tensor_groups.append(registered_tensor)
        tensor_groups.sort(key=lambda x: x._q_adam_idx)
        optimizer_param_ids = [
            id(param)
            for optimizer in bagua_ddp.bagua_optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]
        print(optimizer_param_ids)

        for name, param in parameters:
            if id(param) not in optimizer_param_ids:
                raise RuntimeError(
                    f"Module parameter {name} is not used by your optimizer(s), need to exclude it "
                    "by adding the parameter name to the `List` attribute `_bagua_params_and_buffers_to_ignore` "
                    "of your module."
                )
        return tensor_groups

        # self.tensors = [
        #     param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
        #     for name, param in parameters.__reversed__()
        # ]
        # optimizer_param_ids = [
        #     id(param)
        #     for optimizer in bagua_ddp.bagua_optimizers
        #     for group in optimizer.param_groups
        #     for param in group["params"]
        # ]

        # for name, param in parameters:
        #     if id(param) not in optimizer_param_ids:
        #         raise RuntimeError(
        #             f"Module parameter {name} is not used by your optimizer(s), need to exclude it "
        #             "by adding the parameter name to the `List` attribute `_bagua_params_and_buffers_to_ignore` "
        #             "of your module."
        #         )
        # return self.tensors

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed before the
        forward process.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes the model's input.
        """

        def hook(input):
            print("-----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_forward_pre_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_post_backward_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            if self._should_communicate(bagua_ddp):
                print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_post_backward_hook_should_communicate({}).".format(bagua_ddp.bagua_train_step_counter - 1))
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if param.is_bagua_tensor():
                            param.bagua_mark_communication_ready()

                bagua_ddp._bagua_backend.wait_pending_comm_ops()

        return hook

    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            assert not is_fused_optimizer(
                optimizer
            ), "Low decentralized algorithm can not work with fused optimizer at present."
            print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_post_optimizer_step_hook({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            # if self._should_communicate(bagua_ddp):
            #     print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl init_post_optimizer_step_hook_should_communicate({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            #     for group in optimizer.param_groups:
            #         for param in group["params"]:
            #             if param.is_bagua_tensor():
            #                 param.bagua_mark_communication_ready()

            #     bagua_ddp._bagua_backend.wait_pending_comm_ops()

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        left_peer_weight_tensor = bucket.flattened_tensor()
        right_peer_weight_tensor = bucket.flattened_tensor()


        bucket._weight = weight_tensor.ensure_bagua_tensor(
                    "weight",
                    getter_closure=lambda weight_tensor: weight_tensor.data,
                    setter_closure=lambda weight_tensor, t: setattr(weight_tensor, "data", t),
                )
        bucket._left_peer_weight = left_peer_weight_tensor.ensure_bagua_tensor(
                    "left_peer_weight",
                    getter_closure=lambda left_peer_weight_tensor: left_peer_weight_tensor.data,
                    setter_closure=lambda left_peer_weight_tensor, t: setattr(left_peer_weight_tensor, "data", t),
                )
        bucket._right_peer_weight = right_peer_weight_tensor.ensure_bagua_tensor(
                    "right_peer_weight",
                    getter_closure=lambda right_peer_weight_tensor: right_peer_weight_tensor.data,
                    setter_closure=lambda right_peer_weight_tensor, t: setattr(right_peer_weight_tensor, "data", t),
                )
        # bucket._weight = weight_tensor.ensure_bagua_tensor("weight")
        # bucket._left_peer_weight = left_peer_weight_tensor.ensure_bagua_tensor(
        #     "left_peer_weight"
        # )
        # bucket._right_peer_weight = right_peer_weight_tensor.ensure_bagua_tensor(
        #     "right_peer_weight"
        # )

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()
        def append_python_func(*args):
            print("----QGAdamLowPrecisionDecentralizedAlgorithmImpl append_python_func({}).".format(bagua_ddp.bagua_train_step_counter - 1))
            self.optimizer._compute_tmp()

        bucket.append_python_op(append_python_func, group=self.process_group)
        bucket.append_low_precision_decentralized_synchronous_op(
            weight=bucket._weight,
            left_peer_weight=bucket._left_peer_weight,
            right_peer_weight=bucket._right_peer_weight,
            hierarchical=self.hierarchical,
            compression="MinMaxUInt8",
            group=self.process_group,
        )

class QGAdamLowPrecisionDecentralizedAlgorithm(Algorithm):
    def __init__(self, qg_adam_optimizer: QGAdamOptimizer, hierarchical: bool = True, communication_interval: int = 1):
        """
        Create an instance of the
        `Low Precision Decentralized SGD <https://tutorials.baguasys.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = qg_adam_optimizer

    def reify(
        self, process_group: BaguaProcessGroup
    ) -> QGAdamLowPrecisionDecentralizedAlgorithmImpl:
        return QGAdamLowPrecisionDecentralizedAlgorithmImpl(
            process_group,
            qg_adam_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
        )
