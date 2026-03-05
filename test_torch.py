import torch
import torch.nn as nn

device = torch.device("mps")
print(f"Testing on device: {device}")

# 定义一个简单的卷积
conv = nn.Conv2d(3, 16, kernel_size=3).to(device)
# 创建一个模拟输入并移至 mps
data = torch.randn(1, 3, 64, 64).to(device)

try:
    output = conv(data)
    print("Success! Output shape:", output.shape)
except Exception as e:
    print("Error detected:", e)