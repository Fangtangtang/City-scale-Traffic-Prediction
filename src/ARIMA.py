from dataloader import DateLoader
from dataloader import format_train_data
data_loader = DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.json")


idx = list(test_data.keys())[0]

train_data = data_loader.load_train(train_path="data/train", idx=[idx])
train_data = train_data[idx]
test_data = test_data[idx]

begin_idx, train_data, train_meta = format_train_data(data=train_data)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ARIMA(nn.Module):
    def __init__(self, p, d, q, i):
        super(ARIMA, self).__init__()
        self.p = p  # AR 阶数
        self.d = d  # 差分阶数
        self.q = q  # MA 阶数
        self.pre_len = max(p, d + q)
        self.i = i

        # 定义模型参数
        self.ar_params = nn.Parameter(torch.randn(p))
        self.ma_params = nn.Parameter(torch.randn(q))
        self.constant = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # 实现 ARIMA 模型的前向传播
        # 先进行差分操作
        dif = torch.clone(x)
        for _ in range(self.d):
            dif = torch.diff(dif, 1, dim=0)

        # 计算 AR 部分的预测值
        ar_component = torch.zeros_like(x)
        for i in range(self.i+self.p, x.shape[0]):
            ar_component[i] = torch.dot(self.ar_params, x[i-self.p:i])

        # 计算 MA 部分的预测值
        ma_component = torch.zeros_like(x)
        for i in range(self.i+self.q, x.shape[0]):
            ma_component[i] = torch.dot(self.ma_params, x[i-self.q:i])

        # ARIMA 模型的预测值是 AR 部分和 MA 部分的线性组合再加上常数项
        prediction = self.constant + ar_component + ma_component
        for i in range(self.i):
            prediction[i] = x[i]
        return prediction

model = ARIMA(p=10, d=1, q=8, i=begin_idx)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    output = model(train_data[:, 0])
    to_loss = output[train_data[:, 1] == 1]
    loss = criterion(train_meta, to_loss)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
    if epoch % 50 == 0:
        train_data[train_data[:, 1] == 0, 0] = output[train_data[:, 1] == 0].detach()
    
plt.plot(train_meta.numpy(), label='Original Data')
plt.plot(to_loss.detach().numpy(), label='Predictions')
plt.legend()
plt.show()