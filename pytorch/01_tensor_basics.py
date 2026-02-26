import torch
import numpy as np
from typing import List, Tuple

def tensor_basics() -> None:
    """
    演示 PyTorch 张量的基础创建、属性查看及常用操作。
    """
    print("--- 1. 创建张量 ---")
    
    # 从嵌套列表创建张量
    data: List[List[int]] = [[1, 2], [3, 4]]
    x_data: torch.Tensor = torch.tensor(data)
    print(f"从列表创建:\n{x_data}")

    # 从 NumPy 数组创建张量
    np_array: np.ndarray = np.array(data)
    x_np: torch.Tensor = torch.from_numpy(np_array)
    print(f"从 NumPy 创建:\n{x_np}")

    # 创建特定形状的随机、全一和全零张量
    shape: Tuple[int, int] = (2, 3)
    rand_tensor: torch.Tensor = torch.rand(shape)
    ones_tensor: torch.Tensor = torch.ones(shape)
    zeros_tensor: torch.Tensor = torch.zeros(shape)

    print(f"随机张量: \n {rand_tensor}")
    print(f"全一张量: \n {ones_tensor}")
    print(f"全零张量: \n {zeros_tensor}")

    print("\n--- 2. 张量属性 ---")
    tensor: torch.Tensor = torch.rand(3, 4)
    print(f"形状 (Shape): {tensor.shape}")
    print(f"数据类型 (Datatype): {tensor.dtype}")
    print(f"存储设备 (Device): {tensor.device}")

    print("\n--- 3. 张量操作 ---")
    # 索引和切片操作
    sample_tensor: torch.Tensor = torch.ones(4, 4)
    print(f"第一行: {sample_tensor[0]}")
    print(f"第一列: {sample_tensor[:, 0]}")
    print(f"最后一列: {sample_tensor[..., -1]}")
    
    # 在指定维度上拼接张量
    t1: torch.Tensor = torch.cat([sample_tensor, sample_tensor, sample_tensor], dim=1)
    print(f"拼接后的形状: {t1.shape}")

    # 算术运算：矩阵乘法与逐元素乘法
    # 矩阵乘法 (@ 运算符)
    y1: torch.Tensor = sample_tensor @ sample_tensor.T
    # 逐元素乘法 (* 运算符)
    z1: torch.Tensor = sample_tensor * sample_tensor
    print(f"矩阵乘法结果:\n{y1}")

    print("\n--- 4. 桥接 NumPy ---")
    # 张量转换为 NumPy 数组
    t: torch.Tensor = torch.ones(5)
    print(f"torch tensor: {t}")
    n: np.ndarray = t.numpy()
    print(f"numpy array: {n}")

    # 修改原张量，转换后的 NumPy 数组也会随之改变（共享内存）
    t.add_(1)
    print(f"修改后 t: {t}")
    print(f"共享内存后的 n: {n}")

if __name__ == "__main__":
    # 检查当前环境 GPU 是否可用
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    tensor_basics()
