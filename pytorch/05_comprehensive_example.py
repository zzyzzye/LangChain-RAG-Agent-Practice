import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

# 1. 定义多层感知机 (MLP) 模型
class SimpleClassifier(nn.Module):
    """
    一个简单的多层感知机分类器，包含一个隐藏层。
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super(SimpleClassifier, self).__init__()
        # 第一个线性层：输入特征 -> 隐藏层
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        # ReLU 激活函数
        self.relu: nn.ReLU = nn.ReLU()
        # 第二个线性层：隐藏层 -> 输出类别
        self.fc2: nn.Linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播逻辑。
        """
        out: torch.Tensor = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main() -> None:
    """
    执行完整的训练和评估流程。
    """
    # 超参数设置
    input_size: int = 2      # 每个样本有 2 个特征
    hidden_size: int = 5     # 隐藏层包含 5 个神经元
    num_classes: int = 2     # 二分类任务
    learning_rate: float = 0.1
    num_epochs: int = 50

    # 2. 生成模拟分类数据
    # 创建 100 个随机样本
    x: torch.Tensor = torch.randn(100, input_size)
    # 简单的分类逻辑：如果特征之和大于 0，则属于类别 1，否则为 0
    y: torch.Tensor = (x[:, 0] + x[:, 1] > 0).long()

    # 使用 TensorDataset 包装数据，方便 DataLoader 调用
    dataset: TensorDataset = TensorDataset(x, y)
    train_loader: DataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 初始化模型、损失函数和优化器
    model: SimpleClassifier = SimpleClassifier(input_size, hidden_size, num_classes)
    # 交叉熵损失函数，适用于多分类/二分类任务
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    # Adam 优化器，通常比 SGD 收敛更快
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    print("开始训练...")
    model.train() # 设置为训练模式
    for epoch in range(num_epochs):
        epoch_loss: float = 0.0
        for i, (features, labels) in enumerate(train_loader):
            # 前向传播：计算模型输出
            outputs: torch.Tensor = model(features)
            loss: torch.Tensor = criterion(outputs, labels)

            # 反向传播和参数优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数
            
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], 平均 Loss: {epoch_loss/len(train_loader):.4f}')

    # 5. 模型测试与评估
    print("\n训练完成，进行简单预测测试:")
    model.eval() # 设置为评估模式
    with torch.no_grad():
        # 测试两个极端情况：明确正类和明确负类
        test_input: torch.Tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        test_outputs: torch.Tensor = model(test_input)
        # 获取概率最大的类别索引
        _, predicted = torch.max(test_outputs.data, 1)
        
        print(f"测试输入 1: [1.0, 1.0] -> 预测类别: {predicted[0].item()} (期望: 1)")
        print(f"测试输入 2: [-1.0, -1.0] -> 预测类别: {predicted[1].item()} (期望: 0)")

if __name__ == "__main__":
    main()
