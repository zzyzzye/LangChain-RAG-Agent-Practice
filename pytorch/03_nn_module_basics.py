import torch
from torch import nn, optim
from typing import Iterator, Tuple

# 1. 定义一个简单的线性回归模型
# 继承 nn.Module 是构建所有神经网络的起点
class LinearRegressionModel(nn.Module):
    """
    一个简单的线性回归模型，实现 y = wx + b。
    """
    def __init__(self) -> None:
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层：输入维度为 1，输出维度为 1
        self.linear: nn.Linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播路径。
        """
        return self.linear(x)

def nn_demo() -> None:
    """
    演示神经网络模型的初始化、训练循环及预测过程。
    """
    print("--- 1. 模型初始化 ---")
    model: LinearRegressionModel = LinearRegressionModel()
    print(f"模型结构:\n{model}")
    
    # 查看模型初始化后的参数 (权重 w 和 偏置 b)
    for name, param in model.named_parameters():
        print(f"参数: {name}, 值: {param.data}")

    print("\n--- 2. 准备数据 ---")
    # 创建模拟训练数据 (目标函数：y = 2x + 1)
    x_train: torch.Tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y_train: torch.Tensor = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)
    print(f"输入 x:\n{x_train}")
    print(f"目标 y:\n{y_train}")

    print("\n--- 3. 定义损失函数和优化器 ---")
    # 均方误差损失 (Mean Squared Error)，用于回归任务
    criterion: nn.MSELoss = nn.MSELoss()
    # 随机梯度下降优化器 (SGD)，学习率设置为 0.01
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.01)

    print("\n--- 4. 训练循环 ---")
    epochs: int = 100
    for epoch in range(epochs):
        # 1. 前向传播：计算模型预测值
        y_pred: torch.Tensor = model(x_train)

        # 2. 计算损失值
        loss: torch.Tensor = criterion(y_pred, y_train)

        # 3. 反向传播和参数更新
        optimizer.zero_grad()  # 必须清空累积梯度
        loss.backward()        # 计算当前梯度
        optimizer.step()       # 根据梯度更新权重

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("\n--- 5. 训练后的预测 ---")
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        test_val: torch.Tensor = torch.tensor([[5.0]])
        predicted: torch.Tensor = model(test_val)
        print(f"当 x=5 时，预测 y = {predicted.item():.4f} (预期结果接近 11)")

if __name__ == "__main__":
    nn_demo()
