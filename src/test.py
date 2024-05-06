
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义 ARIMA 模型
class ARIMA(nn.Module):
    def __init__(self, p, d, q):
        super(ARIMA, self).__init__()
        self.p = p  # AR 阶数
        self.d = d  # 差分阶数
        self.q = q  # MA 阶数
        

        # 定义模型参数
        self.ar_params = nn.Parameter(torch.randn(p))
        self.ma_params = nn.Parameter(torch.randn(q))
        self.constant = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # 实现 ARIMA 模型的前向传播
        # 先进行差分操作
        for _ in range(self.d):
            x = torch.diff(x, 1, dim=0)

        # 计算 AR 部分的预测值
        ar_component = torch.zeros_like(x)
        for i in range(self.p, x.shape[0]):
            ar_component[i] = torch.dot(self.ar_params, x[i-self.p:i])

        # 计算 MA 部分的预测值
        ma_component = torch.zeros_like(x)
        for i in range(self.q, x.shape[0]):
            ma_component[i] = torch.dot(self.ma_params, x[i-self.q:i])

        # ARIMA 模型的预测值是 AR 部分和 MA 部分的线性组合再加上常数项
        prediction = self.constant + ar_component + ma_component
        return prediction

# 准备数据
data = np.sin(np.linspace(0, 100, 100)) + np.random.normal(0, 0.1, 100)
data = torch.tensor(data, dtype=torch.float32)

# 初始化模型
model = ARIMA(p=3, d=1, q=2)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data)
    print(data.shape, output.shape)
    loss = criterion(output, data[1:])
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 绘制预测结果
plt.plot(data.numpy(), label='Original Data')
plt.plot(output.detach().numpy(), label='Predictions')
plt.legend()
plt.show()