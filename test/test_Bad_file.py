# import sys
# import os

# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, project_path)

# from dataclasses import dataclass
# from typing import Tuple, Callable, Any, Mapping, List

# from AveMujicaChk.engine.shmengine import SharedMemoryEngine

# import torch

# shm = SharedMemoryEngine(local_rank=0)

# shm.unlink()

import torch

def extract_tensors(state_dict):
    bucket = {}

    def recursive_extract(current_dict, current_bucket):
        for key, value in current_dict.items():
            if torch.is_tensor(value):
                current_bucket[key] = value
            elif isinstance(value, dict):
                current_bucket[key] = {}
                recursive_extract(value, current_bucket[key])

    recursive_extract(state_dict, bucket)
    return bucket

def restore_tensors(state_dict, bucket):
    def recursive_restore(current_dict, current_bucket):
        for key, value in current_bucket.items():
            if isinstance(value, dict):
                recursive_restore(current_dict[key], value)
            else:
                current_dict[key] = value

    recursive_restore(state_dict, bucket)

# 示例用法
state_dict = {
    'layer1': torch.tensor([1.0, 2.0]),
    'layer2': {
        'weights': torch.tensor([3.0, 4.0]),
        'bias': torch.tensor([5.0]),
    },
    'layer3': {
        'nested': {
            'weights': torch.tensor([6.0, 7.0]),
        }
    },
    'state': "zer0",
    'version': "10.01"
}

# 提取 tensors 到 bucket
bucket = extract_tensors(state_dict)

print("Bucket with extracted tensors:")
print(bucket)

# 正确清除 state_dict 中的 tensor
def clear_tensors(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if torch.is_tensor(value):
                obj[key] = None
            elif isinstance(value, dict):
                clear_tensors(value)

clear_tensors(state_dict)

print("\nState dict after removing tensors:")
print(state_dict)

# 恢复 tensors 回到原始 state_dict
restore_tensors(state_dict, bucket)

print("\nState dict after restoring tensors:")
print(state_dict)
