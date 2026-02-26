import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# 1. 自定义数据集类
class MyDataset(Dataset):
    """
    自定义数据集，演示 Dataset 的基本结构。
    """
    def __init__(self, size: int = 100) -> None:
        # 模拟生成数据：x 为 0 到 size-1 的数字，y 为 x 的两倍
        self.x: torch.Tensor = torch.arange(size).float().view(-1, 1)
        self.y: torch.Tensor = self.x * 2
        self.size: int = size

    def __len__(self) -> int:
        """
        返回数据集的总样本数。
        """
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引返回一个样本 (x, y)。
        """
        return self.x[idx], self.y[idx]

def dataloader_demo() -> None:
    """
    演示如何使用 DataLoader 进行批量加载和打乱数据。
    """
    print("--- 1. 初始化数据集 ---")
    dataset: MyDataset = MyDataset(size=10)
    print(f"数据集大小: {len(dataset)}")
    
    # 访问单条数据示例
    x0, y0 = dataset[0]
    print(f"第一条样本: x={x0.item()}, y={y0.item()}")

    print("\n--- 2. 使用 DataLoader ---")
    # batch_size: 每批加载的样本数
    # shuffle: 是否在每个 epoch 开始时打乱数据顺序
    dataloader: DataLoader = DataLoader(dataset, batch_size=3, shuffle=True)

    for epoch in range(2):
        print(f"--- Epoch {epoch} ---")
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            print(f"批次 {batch_idx}:")
            # 使用 squeeze() 去除多余维度以便打印
            print(f"  Batch X: {batch_x.squeeze().tolist()}")
            print(f"  Batch Y: {batch_y.squeeze().tolist()}")

if __name__ == "__main__":
    dataloader_demo()
