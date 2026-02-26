import torch

def autograd_demo() -> None:
    """
    演示 PyTorch 自动求导 (Autograd) 的基本机制。
    """
    print("--- 1. 基础自动求导 ---")
    
    # 创建一个张量并设置 requires_grad=True 来追踪其计算历史
    x: torch.Tensor = torch.ones(2, 2, requires_grad=True)
    print(f"张量 x:\n{x}")

    # 进行张量操作，y 将自动拥有 grad_fn 属性
    y: torch.Tensor = x + 2
    print(f"y = x + 2:\n{y}")
    print(f"y 的梯度函数: {y.grad_fn}")

    # 进一步进行非线性操作
    z: torch.Tensor = y * y * 3
    out: torch.Tensor = z.mean()
    print(f"z = y * y * 3:\n{z}")
    print(f"out = z.mean(): {out}")

    # 2. 反向传播计算梯度
    print("\n--- 2. 计算梯度 ---")
    out.backward()
    
    # 输出导数 d(out)/dx
    # 数学推导过程：
    # out = 1/4 * sum(z_i)
    # z_i = 3 * (x_i + 2)^2
    # d(out)/dx_i = 1/4 * 3 * 2 * (x_i + 2) = 1.5 * (1 + 2) = 4.5
    print(f"x 的梯度 d(out)/dx:\n{x.grad}")

    # 3. 停止梯度追踪
    print("\n--- 3. 停止梯度追踪 ---")
    # 在模型评估或推理阶段，通常不需要更新梯度，可以使用 torch.no_grad()
    print(f"x.requires_grad: {x.requires_grad}")
    
    # 默认情况下，操作会保留梯度追踪
    print(f"(x**2).requires_grad: {(x**2).requires_grad}")

    # 在 no_grad 上下文中，所有计算都不会追踪梯度
    with torch.no_grad():
        print(f"在 no_grad 块中 (x**2).requires_grad: {(x**2).requires_grad}")

if __name__ == "__main__":
    autograd_demo()
