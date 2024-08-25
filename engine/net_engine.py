import os
from typing import Dict

import torch
import torch.distributed as dist

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.accelerator import get_accelerator

import copy
from typing import Dict
from AveMujicaChk.utils import env_utils
from AveMujicaChk.engine.chk_engine import CheckpointEngine
from AveMujicaChk.common.constants import CheckpointConstant, CheckpointMetaKey
from AveMujicaChk.utils.log import default_logger as log
from AveMujicaChk.utils.net_utils import(
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)

from AveMujicaChk.utils.time_utils import (
    cuda_timer,
    timer
)
from AveMujicaChk.engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)
from AveMujicaChk.engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)
from AveMujicaChk.engine.chk_engine import (
    verify_all_rank_step_consistent,
    check_all_rank_ready,
)
from AveMujicaChk.engine.checkpointer import Checkpointer
from AveMujicaChk.engine.dspeed_engine import DeepSpeedCheckpointEngine

def _local_rank0_log(local_rank, message):
    if local_rank == 0:
        log.info(message)


class DeepSpeedCheckpointNETEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict of 
    `DeepSpeedEngine` into the shared memory and notify the agent
    in main process to asynchronously save the state dict from the shared
    memory into the storage.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.
        dp_size (int): the world size of data parallelism.
        global_shard_num (int): the number of shards across all ranks.
        zero_stage (int): the DeepSpeed ZERO Stage number.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
    """
    def __init__(
        self,
        engine: DeepSpeedEngine,
        checkpoint_dir,
        comm_backend = "",
        save_timeout = CheckpointConstant.SAVE_TIMEOUT,  
    ):
        self.engine = engine
        self.state_dict: Dict[str, object] = {}
        self.paths: Dict[str, str] = {}
        super().__init__(checkpoint_dir, comm_backend, save_timeout)

        self.engine = engine
        self.checkpoint_dir = checkpoint_dir
        self.global_shard_num = 1   # shard of ZeRO

        if self.engine.zero_optimization():
            self.global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )

        zero_stage = self.engine.zero_optimization_stage()
        self._local_rank = env_utils.get_local_rank()

        self.global_rank = self.engine.global_rank
        self.world_size = self.engine.world_size

        self.dp_rank = self.global_rank // self.global_shard_num
        self.dp_num = self.world_size // self.global_shard_num

        if zero_stage < ZeroStageEnum.weights and self._local_rank == 0:
            self.engine.save_non_zero_checkpoint = True

        self.comm_group = None

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

    def _get_group_comm_rank(self):
        """
        Create a communication group for communicating between the same rank in different data parallel (dp) groups.

        Use environment variables to determine rank information.

        return:
           dis rank
        """

        # _group = self.global_rank // (2 * self.global_shard_num)
        if self.dp_num % 2 == 0:
            message = (
                    f"The number of data parallel groups meets the group policy condition"
                    f"Check points are saved using group policy"
                )
            _exchange_rank = self.global_rank + self.global_shard_num * (1 if self.dp_rank % 2 == 0 else -1)
        else: 
            message = (
                    f"The number of data parallel groups does not meet the group policy condition"
                    f"Check points are saved using group policy and ring policy"
                )
            if self.world_size // (2 * self.global_shard_num) - 1 > self.global_rank // (2 * self.global_shard_num):
                _exchange_rank = self.global_rank + self.global_shard_num * (1 if self.dp_rank % 2 == 0 else -1)
            else:
                if self.dp_rank % 2 == 0:
                    if self.dp_rank == self.dp_num - 1:
                        _exchange_rank = self.global_rank - 2 * self.global_shard_num
                    elif self.dp_rank == self.dp_num - 2:
                        _exchange_rank = self.global_rank + self.global_shard_num
                    else:
                        _exchange_rank = self.global_rank + self.global_shard_num
                else:
                    _exchange_rank = self.global_rank + self.global_shard_num
        _local_rank0_log(self._local_rank, message)  
        return _exchange_rank
    
    def _create_communication_group(self):
        _exchange_rank = self._get_group_comm_rank
        ranks = sorted([self.global_rank, _exchange_rank])
        comm_group = dist.new_group(ranks = ranks)

        return comm_group
    
    def exchange_bucket(self, bucket, process_group = None):
        tensor = self.flatten(bucket)

        process_group = self.dp_process_group if process_group is None else process_group

        tensor_to_exchange = tensor

        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)

        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor_to_exchange, group=process_group)

        peer_rank = (rank + 1) % world_size if world_size == 2 else 1  
        opposite_rank_tensor = gather_list[0 if rank == peer_rank else 1]

        return opposite_rank_tensor
    
    def exchange_with_multiple_ranks(
            self,
            small_bucket,
            process_group = None,
            bucket_ranks = None,
    ):
        process_group = self.dp_process_group if process_group is None else process_group
        allreduced = self.exchange_bucket(small_bucket, process_group=process_group)
        for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
            if dist.get_rank(group=process_group) == bucket_rank:
                buf.copy_(synced)

    def exchange_state_dict(self, bucket, numel_per_bucket = 500000000, process_group = None):
        small_bucket = []
        small_bucket_ranks = []
        numel = 0
        exchange_size = []

        
        for i, bucket_elem in enumerate(bucket):
            rank, tensor = bucket_elem
            small_bucket.append(tensor)
            small_bucket_ranks.append(rank)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.exchange_with_multiple_ranks(small_bucket, process_group=process_group,
                                                  bucket_ranks=small_bucket_ranks)
                small_bucket = []
                small_bucket_ranks = []
                numel = 0

        if len(small_bucket) > 0:
            self.exchange_with_multiple_ranks(small_bucket, process_group=process_group,
                                              bucket_ranks=small_bucket_ranks)
            
    def exchange_and_copy(self, small_bucket, process_group=None, bucket_ranks=None):
        process_group = self.dp_process_group if process_group is None else process_group
        if self.engine.optimier.overlap_comm:
            if not get_accelerator().resolves_data_dependency():
                get_accelerator().synchronize()
            self._clear_previous_reduced_grads()
            stream = self.engine.optimier.reduction_stream
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            # 使用 exchange_bucket 函数进行张量交换
            exchanged_tensor = self.exchange_bucket(small_bucket, process_group=process_group)
            
            # 恢复展平的张量为原始形状并将其复制到原始张量中
            for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(exchanged_tensor, small_bucket), bucket_ranks):
                if dist.get_rank(group=process_group) == bucket_rank:
                    buf.copy_(synced)

    def _exchange_state_dict(self, module, state_dict, process_group, prefix=""):
        """
        Recursively exchanges the state_dict between two ranks within the given process group,
        ensuring parameters are exchanged layer-by-layer.

        Args:
            module (torch.nn.Module): The model or submodule to exchange parameters for.
            state_dict (dict): The state_dict to exchange.
            process_group: The process group which contains exactly two ranks.
            prefix (str): Prefix for the current module, used to construct full parameter names.
        """
        assert dist.get_world_size(group=process_group) == 2, "Process group must have exactly 2 ranks."

        current_rank = dist.get_rank(group=process_group)
        peer_rank = 1 - current_rank 

        small_bucket = []
        bucket_ranks = []

        for name, param in module.named_parameters(recurse=False):
            if param is None:
                continue

            key = prefix + name
            tensor = state_dict[key]

            small_bucket.append(tensor)
            bucket_ranks.append(peer_rank)

            if len(small_bucket) > 0:
                self.exchange_and_copy(small_bucket, process_group=process_group, bucket_ranks=bucket_ranks)

                small_bucket = []
                bucket_ranks = []

        for name, child in module.named_children():
            if child is not None:
                self._exchange_state_dict(child, state_dict, process_group, prefix=prefix + name + ".")

        if len(small_bucket) > 0:
            self.exchange_and_copy(small_bucket, process_group=process_group, bucket_ranks=bucket_ranks)
