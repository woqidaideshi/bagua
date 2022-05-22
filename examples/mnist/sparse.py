#!/usr/bin/env python3
from cmath import log
import bagua.torch_api as bagua
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.tensor import BaguaTensor
from torch.optim.optimizer import Optimizer
import torch
from typing import List, Tuple
from csvec import CSVec
import logging

class SketchedSGD(torch.optim.Optimizer):
    """SketchedSGD optimizer

    This is a thin wrapper over optim.SGD. Most of the work to do
    sketching is in SketchedSum. SketchedSum handles the learning rate,
    momentum, and weight decay, so we don't want the user's optim.SGD
    instance to apply them a second time.
    """
    def __init__(self, opt):
        """SketchedSGD Constructor

        Args:
            opt: the optim.SGD instance you were using before applying
                 sketching
            k: how many gradient elements to extract from the sketches
            accumulateError: whether or not to accumulate error in the
                             workers. Currently accumulateError=False
                             works only if using signum
            p1: truncate worker gradients to p1*k before sketching. If
                zero, don't truncate
            p2: the parameter server extracts p2*k heavy hitters from
                the summed sketches, requests p2*k actual gradient values
                from each worker, and then computes the topk of the sum
                of the actual values
        """
        # nesterov not supported
        assert(opt.defaults["nesterov"] == False)
        self.opt = opt
    
    def zero_grad(self):
        """Zero out the gradient"""
        self.opt.zero_grad()

    def step(self):
        """Step the optimizer"""
        # the weight update, including lr, momentum, weight decay,
        # and error accumulation, was calculated by sketchedSum
        # and is in self.opt.param_groups
        size_grad = 0
        size_grad_nonzero = 0
        for group in self.opt.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    size_grad += param.grad.numel()
                    size_grad_nonzero += param.grad.count_nonzero().item()
        # logging.info("-----------before step: {}/{}".format(size_grad_nonzero, size_grad))
        self.opt.step()

    def step_and_update_lr(self):
        self.step()

    def __getattr__(self, name):
        return getattr(self.opt, name)

    def __setattr__(self, name, value):
        if name == "opt":
            self.__dict__["opt"] = value
        else:
            opt = self.__dict__["opt"]
            setattr(opt, name, value)


class SketchedModel:
    # not inheriting from nn.Module to avoid the fact that implementing
    # __getattr__ on a nn.Module is tricky, since self.model = model
    # doesn't actually add "model" to self.__dict__ -- instead, nn.Module
    # creates a key/value pair in some internal dictionary that keeps
    # track of submodules
    def __init__(self, model, sketchBiases=False, sketchParamsLargerThan=0):
        self.model = model
        # sketch everything larger than sketchParamsLargerThan
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan

        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.do_sketching = sketchBiases

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Fix to avoid infinite loop in __getattr__
    # Otherwise RecursionError: maximum recursion depth exceeded while
    # calling a Python object
    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        else:
            return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

class SketchedSum:
    def __init__(self, opt, c, r, k, numBlocks=1):
        self.opt = opt
        self.c = c
        self.r = r
        self.k = k
        self.nranks = bagua.get_world_size()
        self.device = opt.param_groups[0]["params"][-1].device
        print("device", self.device)

        grad_size = 0
        sketch_size = 0
        sketchMask = []
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = p.numel()
                    grad_size += size
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                        sketch_size += size
                    else:
                        sketchMask.append(torch.zeros(size))
        self.grad_size = grad_size
        self.sketch_size = sketch_size
        self.no_sketch_size = grad_size - sketch_size
        # a mask indicating which gradient elements we should sketch
        # and which we should send without compression (e.g. bias terms,
        # maybe early layers, etc.)
        self.sketchMask = torch.cat(sketchMask).bool().to(self.device)
        assert self.sketch_size == self.sketchMask.sum()
        assert self.grad_size == self.sketchMask.numel()

        self.logged_bw_savings = False
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

        self.workerSketch = CSVec(d=sketch_size,
                                  c=self.c, r=self.r,
                                  device=self.device,
                                  numBlocks=numBlocks)
        logging.info("--------------init-self.sketch_size--{}".format(self.sketch_size))

    def _getGradVec(self):
        """Return the gradient flattened to a vector"""
        gradVec = []
        with torch.no_grad():
            # flatten
            for group in self.opt.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        # gradVec.append(p.grad.data.view(-1).float())
                        if p.grad is None:
                            gradVec.append(torch.zeros_like(p.data.view(-1)))
                        else:
                            gradVec.append(p.grad.data.view(-1).float())

            # concat into a single vector
            gradVec = torch.cat(gradVec)

        return gradVec

    def _getLRVec(self):
        """Return a vector of each gradient element's learning rate

        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.opt.param_groups) == 1:
            return self.opt.param_groups[0]["lr"]

        lrVec = []
        for group in self.opt.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.requires_grad:
                    if p.grad is None:
                        lrVec.append(torch.zeros_like(p.data.view(-1)))
                    else:
                        grad = p.grad.data.view(-1)
                        lrVec.append(torch.ones_like(grad) * lr)
        return torch.cat(lrVec)

    def _getParamVec(self):
        """Returns the current model weights as a vector"""
        d = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    d.append(p.data.view(-1).float())
        return torch.cat(d).to(self.device)

    def _setGradVec(self, vec):
        """Set the gradient to vec"""
        # put vec into p.grad.data
        vec = vec.to(self.device)
        startPos = 0
        grad_nonzero_old = 0
        grad_nonzero = 0
        for group in self.opt.param_groups:
            for p in group["params"]:
                # logging.info("------p.requires_grad: {}".format(p.requires_grad))
                if p.requires_grad:
                    # logging.info("11111111111111--{}".format(hasattr(p, "grad")))
                    # if not hasattr(p, "sketch_grad"):
                    #     p.sketch_grad = torch.zeros_like(p.grad)
                    # logging.info("-----------sketch_grad nonzero: {}".format(p.sketch_grad.count_nonzero().item()))
                    shape = p.grad.shape
                    size = p.grad.numel()
                    # grad_nonzero_old += p.grad.count_nonzero().item()
                    # logging.info("---------------------_setGradVec before: {}-{}".format(p.grad.count_nonzero().item(), p.grad.numel()))
                    # p.grad.zero_()
                    # p.grad.add_(vec[startPos:startPos + size].reshape(shape))
                    # p.grad.copy_(vec[startPos:startPos + size].reshape(shape))
                    ver_grad = vec[startPos:startPos + size]
                    grad = p.grad.view(-1)
                    grad[ver_grad.nonzero()] = 0
                    grad.add_(ver_grad)
                    # new_grad = vec[startPos:startPos + size].reshape(shape)
                    # new_grad[new_grad.zero()]
                    startPos += size
                    # logging.info("----------------{}=={}, nonzero: {}".format(new_grad.shape, p.grad.shape, new_grad.nonzero()))
                    # p.grad[new_grad.nonzero()] = 0
                    p.grad.copy_(grad.reshape(shape))
                    # grad_nonzero += p.grad.count_nonzero().item()
                    # logging.info("---------------------_setGradVec after: {}-{}".format(p.grad.count_nonzero().item(), p.grad.numel()))
        # logging.info("------------grad_nonzero_old/grad_nonzero={}/{}, vec={}".format(grad_nonzero_old, grad_nonzero, vec.count_nonzero().item()))
        
        size_grad = 0
        size_grad_nonzero = 0
        for group in self.opt.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    size_grad += param.grad.numel()
                    size_grad_nonzero += param.grad.count_nonzero().item()
        # logging.info("-----------_setGradVec-grad/grad_nonzero={}/{}".format(size_grad, size_grad_nonzero))

    def encode(self):
        gradVec = self._getGradVec().to(self.device)
        if self.opt.defaults["weight_decay"] != 0:
            gradVec.add_(self.opt.defaults["weight_decay"] / self.nranks,
                         self._getParamVec())
        self.u.mul_(self.opt.defaults["momentum"]).add_(gradVec)
        self.v.add_(self.u)
        if self.sketchMask.sum() < self.v.numel():
            v = self.v[self.sketchMask]
        else:
            v = self.v
        self.workerSketch.zero()
        self.workerSketch.accumulateVec(v)
        table = self.workerSketch.table.clone() 

        uncompressed = self.v[~self.sketchMask]
        assert uncompressed.size() == torch.Size([self.grad_size - self.sketch_size])

        encoded = torch.cat([table.view(-1), uncompressed])
        
        if not self.logged_bw_savings:
            print(f"Bandwidth savings: from {len(gradVec)} to {len(encoded)} ({(len(gradVec) / len(encoded)):.2f}x)")
            self.logged_bw_savings = True
        # logging.info("---------------------encode uncompressed: {}/{}".format(uncompressed.count_nonzero().item(), uncompressed.numel()))
        logging.info("------------encode  encoded_nonzero/encoded_size={}/{}".format(encoded.count_nonzero().item(), encoded.numel()))
        return encoded

    def decode(self, encoded):
        table_len = self.r * self.c
        sketch_table = encoded[:table_len].view(self.r, self.c)
        uncompressed = encoded[table_len:]
        # logging.info("---------------------before decode decoded: {}/{}".format(encoded.count_nonzero().item(), encoded.numel()))

        self.workerSketch.zero()
        self.workerSketch.table = sketch_table

        # unsketch payload
        gradient = torch.zeros(self.grad_size, device=self.device)
        unsketched = self.workerSketch.unSketch(k=self.k)
        # logging.info("---------------------middle decode decoded: {}/{}".format(unsketched.count_nonzero().item(), unsketched.numel()))

        gradient[self.sketchMask] = unsketched
        
        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0
        # self.u.copy_(self.v-gradient)

        # deal with non-compressed gradients (bias)
        assert uncompressed.size() == torch.Size([self.grad_size - self.sketch_size])
        # logging.info("---------------------0000decode decoded: {}/{}".format(gradient.count_nonzero().item(), gradient.numel()))

        gradient[~self.sketchMask] = uncompressed
        # logging.info("---------------------1111decode decoded: {}/{}".format(gradient.count_nonzero().item(), gradient.numel()))
        self.v[~self.sketchMask] = 0
        # gradient.add_(self.v)

        gradient.mul_(self._getLRVec())
        # logging.info("---------------------decode decoded: {}/{}".format(gradient.count_nonzero().item(), gradient.numel()))
        self._setGradVec(gradient)
        # self.u[gradient.nonzero()] = 0
        # self.u.copy_(self.v - gradient)
        

        logging.info("---------------------set grad decoded: {}/{}".format(gradient.count_nonzero().item(), gradient.numel()))  

class SparseAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        optimizer: Optimizer,
        hierarchical: bool = False,
        average: bool = True,
        c: int = 100,
        r: int = 10,
        k: int = 100,
        accumulateError: bool = True,
        p1: int = 0,
        p2: int = 0,
        transferHalf: bool = False,
        numBlocks: int = 1,
    ):
        super(SparseAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average
        self.c = c
        self.r = r
        self.optimizer = optimizer
        self.doAccumulateError = accumulateError
        self.p1 = p1
        self.p2 = p2
        self.doTransferHalf = transferHalf
        self.sketchSum = SketchedSum(self.optimizer, self.c, self.r, k, numBlocks=numBlocks)

    def init_tensors(
        self, bagua_ddp: BaguaDistributedDataParallel
    ) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            param.newgrad = torch.zeros(self.c * self.r + self.sketchSum.no_sketch_size, device=param.device)
            param.stepid = 0
            registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.newgrad,
                setter_closure=lambda param, t: setattr(param, "newgrad", t),
            )
            tensors.append(registered_tensor)
            break
        
        self._communication_tensor_names = set((parameters[-1][0],))
        print("----SparseAlgorithmImpl init_tensors batch_idx {} in rank: {}, _communication_tensor_names: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self._communication_tensor_names))
        logging.info("----SparseAlgorithmImpl init_tensors batch_idx {} in rank: {}, _communication_tensor_names: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self._communication_tensor_names))
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        return tensors

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                parameter.bagua_mark_communication_ready()
            print("----SparseAlgorithmImpl init_backward_hook({}) in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
            pass

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes no argument.
        """

        def hook():
            print("----SparseAlgorithmImpl init_post_backward_hook({}) in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            self.optimizer.param_groups[0]["params"][-1].stepid += 1
            pass

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
            pass
            # print("----SparseAlgorithmImpl init_post_optimizer_step_hook({}) in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
            # size_grad = 0
            # size_grad_nonzero = 0
            # for group in self.optimizer.param_groups:
            #     for param in group["params"]:
            #         if param.requires_grad:
            #             size_grad += param.grad.numel()
            #             size_grad_nonzero += param.grad.count_nonzero().item()
            # logging.info("-----------init_post_optimizer_step_hook-grad/grad_nonzero={}/{}".format(size_grad, size_grad_nonzero))
            # for tensor in self.optimizer.param_groups[0]["params"]:
            #     if tensor.is_bagua_tensor():
            #         tensor.bagua_mark_communication_ready()
            # bagua_ddp._bagua_backend.wait_pending_comm_ops()
        return hook

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        print("----SparseAlgorithmImpl tensors_to_buckets batch_idx {} in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
        logging.info("----SparseAlgorithmImpl tensors_to_buckets batch_idx {} in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
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
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        def log(*args):
            print("--SparseAlgorithmImpl--log batch_idx {} in {}: grad---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self.optimizer.param_groups[0]["params"][-1].grad[0:10]))
            for tensor in self.optimizer.param_groups[0]["params"]:
                if tensor.is_bagua_tensor():
                    print("--SparseAlgorithmImpl--log batch_idx {} in {}: newgrad---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, tensor.newgrad))

        def sketch(*args):
            encoded_tensor = self.sketchSum.encode()

            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"
            assert encoded_tensor.numel() == self.c * self.r + self.sketchSum.no_sketch_size
            bucket.tensors[0].bagua_setter_closure(encoded_tensor) 
            # logging.info("======sketch======{} == {} -----".format(encoded_tensor.numel(), self.c * self.r + self.sketchSum.no_sketch_size))

        def unsketch(*args):
            # logging.info("--SparseAlgorithmImpl--decode batch_idx {} in {}: grad---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self.optimizer.param_groups[0]["params"][-1].grad[0:10]))
            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"

            encoded_tensor = bucket.tensors[0].bagua_getter_closure().detach()
            assert encoded_tensor.numel() == self.c * self.r + self.sketchSum.no_sketch_size
            self.sketchSum.decode(encoded_tensor)
            # logging.info("--unsketch--{} == {} -----".format(encoded_tensor.numel(), self.c * self.r + self.sketchSum.no_sketch_size))

        # bucket.append_python_op(log, group=self.process_group)
        bucket.append_python_op(sketch, group=self.process_group)
        # bucket.append_python_op(log, group=self.process_group)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )
        bucket.append_python_op(unsketch, group=self.process_group)


class SparseAlgorithm(Algorithm):
    def __init__(self, optimizer: Optimizer, hierarchical: bool = False, average: bool = True, c=100, r=10, k=100):
        self.hierarchical = hierarchical
        self.average = average
        self.optimizer = optimizer
        self.c = c
        self.r = r
        self.k = k

    def reify(self, process_group: BaguaProcessGroup) -> SparseAlgorithmImpl:
        return SparseAlgorithmImpl(
            process_group,
            self.optimizer,
            hierarchical=self.hierarchical,
            average=self.average,
            c=self.c,
            r=self.r,
            k=self.k
        )
