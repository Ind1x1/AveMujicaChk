import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import sys
import traceback
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload

# 设置项目路径（假设您的项目路径为两级以上）
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_path)

from AveMujicaChk.engine.dspeed import DeepSpeedCheckpointer
from AveMujicaChk.utils import env_utils

# 定义一个简单的两层模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 第一层，10个输入，20个输出
        self.relu = nn.ReLU()         # 激活函数
        self.fc2 = nn.Linear(20, 10)  # 第二层，20个输入，10个输出

    def forward(self, x):
        x = self.fc1(x)  # 第一层线性变换
        x = self.relu(x) # ReLU激活函数
        x = self.fc2(x)  # 第二层线性变换
        return x

def setup_distributed():
    """初始化分布式环境和GPU设备。"""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

def main():
    # 初始化分布式环境
    setup_distributed()

    # 配置DeepSpeed
    ds_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-3,
                "warmup_num_steps": 100
            }
        },
        "data_parallel_size": 4,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1
    }

    # 初始化模型、优化器和DeepSpeed引擎
    model = SimpleModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # 获取数据并行进程组
    dp_process_group = model_engine.optimizer.dp_process_group
    
    if dp_process_group is not None:
        dp_world_size = torch.distributed.get_world_size(group=dp_process_group)
        print(f"Data Parallel Process Group World Size: {dp_world_size}")

        dp_rank = torch.distributed.get_rank(group=dp_process_group)
        print(f"Rank in Data Parallel Process Group: {dp_rank}")
    else:
        print("Data Parallel Process Group is None")

    # 创建一些示例数据，并转换为FP16精度
    inputs = torch.randn(8, 10, device='cuda', dtype=torch.half)
    labels = torch.randn(8, 10, device='cuda', dtype=torch.half)

    # 初始化DeepSpeed检查点管理器
    MujicaCheckpointer = DeepSpeedCheckpointer(model_engine, "./outputtest")

    # 进行训练
    for step in range(2):
        outputs = model_engine(inputs)
        loss = nn.MSELoss()(outputs, labels)
        loss = loss.half()
        model_engine.backward(loss)
        model_engine.step()

        optimizer_state_dict = model_engine.optimizer.state_dict()
        print(f"Optimizer state dict:\n{optimizer_state_dict}")

        model_state_dict = model_engine.state_dict()
        print(f"Model state dict:\n{model_state_dict}")
        
        # 可选：保存检查点
        # MujicaCheckpointer.save_checkpoint("./outputtest")
        # model_engine.save_checkpoint("./outputtest")

if __name__ == "__main__":
    main()
