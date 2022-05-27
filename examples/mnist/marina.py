from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from torch.optim.optimizer import Optimizer
import torch


class MarinaOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 5e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)

        super(MarinaOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MarinaOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]

            for param in group["params"]:
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0

                    state["grad_k"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["grad_prev"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["grad_difference"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["grad_k"] = param.grad

                state["step"] += 1
                grad = param.grad.detach().clone()
                state["grad_prev"] = grad

                state["grad_k"].add_(state["grad_difference"])
                update = state["grad_k"]

                param.data.add_(-lr * update)

        return loss


class MarinaAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        marina_optimizer: MarinaOptimizer,
        hierarchical: bool = True,
    ):
        super(MarinaAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = marina_optimizer

    @property
    def optimizer_step_id(self):
        param = self.optimizer.param_groups[0]["params"][0]
        return self.optimizer.state[param].get("step", 0)

    def need_reset(self):
        if self.optimizer_step_id == 1:
            return True
        else:
            return False

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel):
        parameters = bagua_ddp.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._marina_name = name
            param._marina_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if self.optimizer_step_id < 1:
                    # register grad
                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._marina_name,
                        bagua_ddp.bagua_module_name,
                        getter_closure=lambda param: param.grad,
                        setter_closure=lambda param, t: setattr(param, "grad", t),
                    )
                else:
                    # register gradient difference
                    def set_grad_diff_fn(param, t):
                        self.optimizer.state[param]["grad_difference"] = t

                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._marina_name,
                        bagua_ddp.bagua_module_name,
                        getter_closure=lambda param: self.optimizer.state[param][
                            "grad_difference"
                        ],
                        setter_closure=set_grad_diff_fn,
                    )

                tensor_groups.append(registered_tensor)
        return tensor_groups

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        if self.optimizer_step_id < 1:
            bucket.append_centralized_synchronous_op(
                hierarchical=False,
                average=True,
                group=self.process_group,
            )
        else:
            #For all tensors in the bucket calculate the difference from the new gradient and the gradient from the previous step
            def calculate_gardient_difference(*args):
                for i in range(len(bucket.tensors)):
                    param = self.optimizer.param_groups[0]["params"][i]
                    grad_prev = self.optimizer.state[param]["grad_prev"]
                    tensor = bucket.tensors[i]
                    new_tensor = (tensor.grad).sub(grad_prev)
                    tensor.bagua_setter_closure(new_tensor)

            bucket.append_python_op(calculate_gardient_difference, group=self.process_group)
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,
                scattergather=True,
                compression="MinMaxUInt8",
                group=self.process_group,
            )


    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_grad_difference(parameter_name, parameter):
            parameter.bagua_mark_communication_ready()

        def hook_grad(parameter_name, parameter):
            parameter.bagua_mark_communication_ready()

        return (
            hook_grad if self.optimizer_step_id < 1 else hook_grad_difference
        )


class MarinaAlgorithm(Algorithm):
    def __init__(self, marina_optimizer: MarinaOptimizer, hierarchical: bool = True):
      
        self.hierarchical = hierarchical
        self.optimizer = marina_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> MarinaAlgorithmImpl:
        return MarinaAlgorithmImpl(
            process_group,
            marina_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )