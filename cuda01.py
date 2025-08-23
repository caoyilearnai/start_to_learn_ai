import torch

print("PyTorch version:", torch.__version__)

# 测试老特性（如 Variable）
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
print("Gradient enabled:", y.requires_grad)  # 老代码常用

# 测试新特性（torch.compile）
def train_step(x):
    return x @ x.T

compiled_step = torch.compile(train_step)  # 2.0+ 新特性
out = compiled_step(torch.randn(3, 3))
print("torch.compile works!")

# 测试 device_mesh（分布式）
try:
    from torch.distributed.device_mesh import init_device_mesh
    print("🎉 device_mesh available!")
except ImportError:
    print("❌ device_mesh not available")