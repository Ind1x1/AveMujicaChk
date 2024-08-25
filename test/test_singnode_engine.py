import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import sys
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
import time
import traceback
import multiprocessing
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_path)
from AveMujicaChk.engine.dspeed import DeepSpeedCheckpointer
from AveMujicaChk.utils import env_utils
# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def setup_distributed():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

# def read():
#     model_state_path = "./outputtest/global_step1/mp_rank_00_model_states.pt"
#     optim_state_paths = [
#         "./outputtest/global_step1/zero_pp_rank_0_mp_rank_00_optim_states.pt",
#         "./outputtest/global_step1/zero_pp_rank_1_mp_rank_00_optim_states.pt",
#         "./outputtest/global_step1/zero_pp_rank_2_mp_rank_00_optim_states.pt",
#         "./outputtest/global_step1/zero_pp_rank_3_mp_rank_00_optim_states.pt"
#     ]
    
#     # 加载模型状态
#     try:
#         with open(model_state_path, 'rb') as f:
#             model_state = torch.load(f)
#             print("Model state loaded successfully.")
#             print(model_state)
#     except Exception as e:
#         print(f"Failed to load model state: {e}")
#         traceback.print_exc()
    
#     # 加载每个优化器状态
#     for idx, path in enumerate(optim_state_paths):
#         try:
#             with open(path, 'rb') as f:
#                 optim_state = torch.load(f)
#                 print(f"Optimizer state {idx} loaded successfully.")
#                 print(optim_state)
#         except Exception as e:
#             print(f"Failed to load optimizer state {idx}: {e}")
#             traceback.print_exc()

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
            "stage": 3,
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
    # dp_process_group = model_engine.optimizer.dp_process_group

    # if dp_process_group is not None:
    # # 打印出进程组的world size（组中有多少个rank）
    #     dp_world_size = torch.distributed.get_world_size(group=dp_process_group)
    #     print(f"Data Parallel Process Group World Size: {dp_world_size}")

    #     # 打印出进程组中每个rank的id
    #     # 注意，这里你需要在每个进程中打印自己的rank，而不是循环dp_world_size次
    #     dp_rank = torch.distributed.get_rank(group=dp_process_group)
    #     print(f"Rank in Data Parallel Process Group: {dp_rank}")
    # else:
    #     print("Data Parallel Process Group is None")
    # 创建一些示例数据，并转换为FP16精度
    inputs = torch.randn(8, 10, device='cuda', dtype=torch.half)  # 推荐的方式
    labels = torch.randn(8, 10, device='cuda', dtype=torch.half)  # 推荐的方式

    MujicaCheckpointer = DeepSpeedCheckpointer(model_engine, "./outputtest") 
    
    for step in range(2):  # 让训练走两步
        # 执行一个训练步骤
        outputs = model_engine(inputs)
        loss = nn.MSELoss()(outputs, labels)
        
        # 确保损失也转换为FP16精度
        loss = loss.half()

        # 执行反向传播
        model_engine.backward(loss)
        model_engine.step()
        state_dict = model_engine.optimizer.state_dict()
        print(f"optimizer --<> \n {state_dict}")
        state_dict = model_engine.state_dict()
        print(f"model --<> \n {state_dict}")
        
        MujicaCheckpointer.save_checkpoint("./outputtest")
        
        # model_engine.save_checkpoint("./outputtest")

    # 执行一个训练步骤
    # outputs = model_engine(inputs)
    # loss = nn.MSELoss()(outputs, labels)
    
    # # 确保损失也转换为FP16精度
    # loss = loss.half()

    # # 执行反向传播
    # model_engine.backward(loss)
    # model_engine.step()

    # MujicaCheckpointer = DeepSpeedCheckpointer(model_engine, "./outputtest") 
    # MujicaCheckpointer.save_checkpoint("./outputtest")

    # MujicaCheckpointer.save_checkpoint("./outputtest")
    # # model_engine.save_checkpoint("./outputtest")
    # torch.distributed.barrier()

    # if torch.distributed.get_rank() == 0:
    #     checkpointlist = model_engine._get_all_ckpt_names("./outputtest","global_step1")
    #     print(checkpointlist)

if __name__ == "__main__":
    #read()
    main()
    