from contextlib import contextmanager
import copy
import itertools
import os
import inspect
import logging
import warnings
from typing import NamedTuple

import torch

from . import comm
import torch.distributed as dist

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors, _dump_DDP_relevant_env_vars, _DDPUnevenInputsConfig


RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.distributed_c10d import ReduceOp
if torch.distributed.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef
from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather, is_namedtuple
from .parallel_apply import parallel_apply
from torch._utils import _get_device_index, _get_all_device_indices
from ._functions import _get_stream

def cpu_comm_hook(
    process_group: dist.ProcessGroup, bucket: dist._GradBucket
) -> torch.futures.Future:
    """
    Example::
        >>> ddp_model.register_comm_hook(process_group, cpu_comm_hook)
    """
    group_to_use = process_group
    world_size = group_to_use.size()

    cpu_tensor = bucket.get_tensors()[0].cpu().div_(world_size)

    fut = dist.all_reduce(
        cpu_tensor, group=group_to_use, async_op=True
    ).get_future()

    def recover(fut):
        gpu_tensor = bucket.get_tensor()
        gpu_tensor.copy_(fut.value()[0])
        return [gpu_tensor]

    return fut.then(recover)


class ElasticDDP(DistributedDataParallel):
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None,
                 process_group_comp=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False,
                 is_computation_process=None,
                 logic_gpus_num=None,
                 local_rank=None,
                 cheif_rank=None,
                 shm_array=None,
                 shm_grads=None):

        # ElasticDDP only supports Single-Process Single-GPU method
        assert((device_ids is None) or (device_ids.size() == 1))
        # TODO(Mingzhen): ElasticDDP does not suport find_unused_parameters and gradient_as bucket_view now.
        assert(find_unused_parameters == False)
        assert(gradient_as_bucket_view == False)
        if is_computation_process == False:
            self.process_group_comp = None
        else:
            self.process_group_comp = process_group_comp
        
        self.is_computation_process = is_computation_process
        self.logic_gpus_num = logic_gpus_num
        self.local_rank = local_rank
        self.cheif_rank=cheif_rank
        self.shm_array = shm_array
        self.shm_grads = shm_grads

        self.cntMicroBatch = 0

        self._grad_accs = []
        
        self._nondefault_stream = None

        super(ElasticDDP, self).__init__(module)

        # TODO(Mingzhen): If users register their own hooks, there may be some unexpected errors.
        if (self.is_computation_process) and (not self.shm_grads[0][0].is_cuda) and (dist.get_world_size(self.process_group) > 1):
            #pass
            self.register_gradacc_hooks()
            # TODO(Mingzhen) Enable this line in pytorch version >= 1.9.0
            # In version 1.8.0, register_comm_hook is only supported by NCCL.
            # Because that the getFuture() is not implemented in ProcessGroup. 
            # self.register_comm_hook(self.process_group, cpu_comm_hook)

    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """

        def parameters(m, recurse=True):
            def model_parameters(m):
                ps = m._former_parameters.values() \
                    if hasattr(m, "_former_parameters") \
                    else m.parameters(recurse=False)
                for p in ps:
                    yield p

            for m in m.modules() if recurse else [m]:
                for p in model_parameters(m):
                    yield p

        if self.device_ids and len(self.device_ids) > 1:

            warnings.warn(
                "Single-Process Multi-GPU is not the recommended mode for "
                "DDP. In this mode, each DDP instance operates on multiple "
                "devices and creates multiple module replicas within one "
                "process. The overhead of scatter/gather and GIL contention "
                "in every forward pass can slow down training. "
                "Please consider using one DDP instance per device or per "
                "module replica by explicitly setting device_ids or "
                "CUDA_VISIBLE_DEVICES. "
            )

            # only create replicas for single-device CUDA modules
            #
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesced, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids, detach=True)
            self._module_copies[0] = self.module

            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), parameters(module_copy)):
                    # Reducer requires param copies have the same strides across replicas.
                    # Fixes up copy_param strides in case replicate didn't match param strides.
                    if param.layout is torch.strided and param.stride() != copy_param.stride():
                        with torch.no_grad():
                            copy_param.set_(copy_param.clone()
                                                      .as_strided(param.size(), param.stride())
                                                      .copy_(copy_param))
                    copy_param.requires_grad = param.requires_grad

        else:
            self._module_copies = [self.module]

        self.modules_params = [list(parameters(m)) for m in self._module_copies]
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            [(buffer, buffer_name) for buffer_name, buffer in m.named_buffers()]
            for m in self._module_copies
        ]
        self.modules_buffers = [
            [
                buffer
                for (buffer, buffer_name) in module_buffers
                if buffer_name not in self.parameters_to_ignore
            ]
            for module_buffers in named_module_buffers
        ]
        # Build tuple of (module, parameter) for all parameters that require grads.
        if self.device_ids and len(self.device_ids) > 1:
            # Single-process multi-device mode,does not support self.parameters_to_ignore.
            if self.parameters_to_ignore:
                raise ValueError(
                    "Single-Process multi-device mode does not "
                    "support ignoring parameters upfront. Please consider "
                    "using one DDP instance per device."
                )

            modules_and_parameters = [
                [
                    (module, parameter)
                    for module in replica.modules()
                    for parameter in filter(
                        lambda parameter: parameter.requires_grad,
                        parameters(module, recurse=False))
                ] for replica in self._module_copies]
        else:
            modules_and_parameters = [
                [
                    (module, parameter)
                    for module_name, module in replica.named_modules()
                    for parameter in [
                        param
                        # Note that we access module.named_parameters instead of
                        # parameters(module). parameters(module) is only needed in the
                        # single-process multi device case, where it accesses replicated
                        # parameters through _former_parameters.
                        for param_name, param in module.named_parameters(recurse=False)
                        if param.requires_grad
                        and f"{module_name}.{param_name}" not in self.parameters_to_ignore
                    ]
                ]
                for replica in self._module_copies
            ]

        # Build list of parameters.
        parameters = [
            list(parameter for _, parameter in replica)
            for replica in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding):
                return module.sparse
            if isinstance(module, torch.nn.EmbeddingBag):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            list(produces_sparse_gradient(module) for module, _ in replica)
            for replica in modules_and_parameters]

        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        bucket_indices = dist._compute_bucket_assignment_by_size(
            parameters[0],
            [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap],
            expect_sparse_gradient[0])

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.ElasticReducer(
            parameters,
            list(reversed(bucket_indices)),
            self.process_group,
            self.process_group_comp,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            
            self.is_computation_process,
            self.logic_gpus_num,
            self.local_rank,
            self.cheif_rank,
            self.shm_array,
            self.shm_grads[self.local_rank])

        # TODO(Mingzhen) Uncomment this part.
        # Set logging data that can be got during construction time.
        # dist._set_construction_logging_data(
        #     self.reducer,
        #     self.module.__class__.__name__,
        #     [] if self.device_ids is None else self.device_ids,
        #     -1 if self.output_device is None else self.output_device,
        #     self.broadcast_buffers)

        ## TODO(Mingzhen) Ignored
        # passing a handle to torch.nn.SyncBatchNorm layer
        #self._passing_sync_batchnorm_handle(self._module_copies)

    def forward(self, *inputs, **kwargs):

        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            ones = torch.ones(
                1, device=self.device
            )
            work = dist.all_reduce(ones, group=self.process_group, async_op=True)
            self.reducer._set_forward_pass_work_handle(
                work, self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
            )

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if self.reducer._rebuild_buckets():
            logging.info("Reducer buckets have been rebuilt in this iteration.")

        if self.require_forward_param_sync:
            self._sync_params()

        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        output = self.module(*inputs, **kwargs)

        #assert(self.find_unused_parameters == False)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False
        
        if self.shm_grads[0][0].is_cuda:
            ## If gradients are stored in CUDA tensors
            ## Let the gradients point to the shm
            if (self.cntMicroBatch < self.logic_gpus_num - 1):
                it = 0
                #parameters_length = sum(1 for _ in self.module.parameters())
                for param in self.module.parameters():
                    if param.requires_grad:
                        param.grad.data = self.shm_grads[self.cntMicroBatch][it].detach()
                        param.grad.data.zero_()
                        it += 1
                #assert(it == parameters_length)
                torch.cuda.synchronize()
            else:
                it = 0
                #parameters_length = sum(1 for _ in self.module.parameters())
                for param in self.module.parameters():
                    if param.requires_grad:
                        param.grad.data.zero_()
                        it += 1
                #assert(it == parameters_length)
                torch.cuda.synchronize()
        ## If gradients are stored in CPU tensors, refer to _register_hooks()
        else:
            it = 0
            #parameters_length = sum(1 for _ in self.module.parameters())
            for param in self.module.parameters():
                if param.requires_grad:
                    param.grad.data.zero_()
                    it += 1
            #assert(it == parameters_length)
            torch.cuda.synchronize()

        self.count_micro_batch()

        return output

    @contextmanager
    def join(self, divide_by_initial_world_size=True, enable=True):
        pass

    def register_gradacc_hooks(self):
        grad_count = 0
        for p in self.module.parameters():
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_D2H_hook(p, grad_count))
                self._grad_accs.append(grad_acc)
                grad_count += 1
    
    def _make_D2H_hook(self, p, grad_count):
        def hook(*ignore):
            #TODO(Mingzhen) hack
            micro = (self.cntMicroBatch - 1 + self.logic_gpus_num) % self.logic_gpus_num 
            if micro < self.logic_gpus_num - 1:
                #with torch.cuda.stream(self._nondefault_stream):
                shm = self.shm_grads[micro][grad_count]
                shm.copy_(p.grad.data, non_blocking=True)
            #    p.grad.data.zero_()
            #else:
            #    p.grad.data.zero_()

        return hook    


    def commProcess_listening(self):
        self.reducer.commProcess_listening()

    def count_micro_batch(self):
        self.cntMicroBatch += 1
        self.cntMicroBatch = self.cntMicroBatch % self.logic_gpus_num
        self.reducer._count_micro_batch()

    def get_shm_array(self):
        return self.reducer._get_shm_array()

    def print_shm_array(self):
        self.reducer._print_shm_array()

    def register_comm_hook(self, state: object, hook: callable):
        self._check_comm_hook(hook)
        dist._register_comm_hook_4Elastic(self.reducer, state, hook)

    def _register_builtin_comm_hook(self, comm_hook_type):
        dist._register_builtin_comm_hook_4Elastic(self.reducer, comm_hook_type)

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        if (dist.get_world_size(self.process_group_comp) > 1):
            dist._broadcast_coalesced(
                self.process_group_comp, tensors, buffer_size, authoritative_rank
            ) 
    
    def _sync_params_and_buffers(self, authoritative_rank=0):
        if not self.is_computation_process:
            return

        module_states = []
        for name, param in self.module.state_dict().items():
            if name not in self.parameters_to_ignore:
                module_states.append(param)

        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size,
                authoritative_rank)

    def _sync_params(self):
        with torch.no_grad():
            if self.device_ids:
                assert(len(self.device_ids) == 1)

            # module buffer sync
            if self.will_sync_module_buffers():
                # Synchronize buffers across processes.
                # If we are running DDP with the join manager, we have to agree
                # upon a rank to sync module buffers from, since rank 0 may
                # already have been joined and have stale module buffers.
                if self.ddp_uneven_inputs_config.ddp_join_enabled:
                    authoritative_rank = self._find_common_rank(dist.get_rank(), True)
                else:
                    # The process with rank 0 is considered the authoritative copy.
                    authoritative_rank = 0
                self._distributed_broadcast_coalesced(
                    self.modules_buffers[0],
                    self.broadcast_bucket_size,
                    authoritative_rank,
                )
