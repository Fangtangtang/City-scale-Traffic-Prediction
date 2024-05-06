from dataloader import DateLoader
from dataloader import format_train_data
data_loader = DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.json")


idx = list(test_data.keys())[0]

train_data = data_loader.load_train(train_path="data/train", idx=[idx])
train_data = train_data[idx]
test_data = test_data[idx]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

begin_idx, train_data, train_meta = format_train_data(data=train_data)

class moving_avg(nn.Module):
    def __init__(self, kernel_size:int, stride:int) -> None:
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.kernel_size-1)//2)
        end = x[:, -1:].repeat(1, (self.kernel_size-1)//2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg_pool(x)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size:int) -> None:
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size=kernel_size, stride=1)
    
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DLinear(nn.Module):
    def __init__(self, kernel_size:int, input_len:int, pred_len:int) -> None:
        super(DLinear, self).__init__()
        self.decomp = series_decomp(kernel_size=kernel_size)
        self.Linear_Season = nn.Linear(input_len, pred_len)
        self.Linear_Trend = nn.Linear(input_len, pred_len)
    
    def forward(self, x):
        # x: [batch, input_len]
        season, trend  = self.decomp(x)
        season_output = self.Linear_Season(season)
        trend_output = self.Linear_Trend(trend)
        x = season_output + trend_output
        return x
input_len = 240
pred_len = 1
kernel_size = 25
model = DLinear(kernel_size=kernel_size, input_len=input_len, pred_len=pred_len)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

input_data = torch.zeros((train_data.shape[0] - begin_idx - input_len - pred_len + 1, input_len))
loss_data = torch.zeros((train_data.shape[0] - begin_idx - input_len - pred_len + 1, pred_len))
loss_mask = torch.zeros((train_data.shape[0] - begin_idx - input_len - pred_len + 1, pred_len))
for i in range(train_data.shape[0] - begin_idx - input_len - pred_len + 1):
    input_data[i] = train_data[i+begin_idx:i+begin_idx+input_len, 0]
    loss_data[i] = train_data[i+begin_idx+input_len:i+begin_idx+input_len+pred_len, 0]
    loss_mask[i] = train_data[i+begin_idx+input_len:i+begin_idx+input_len+pred_len, 1]

epochs = 50000
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    output = model(input_data)
    to_loss = output * loss_mask
    loss = criterion(loss_data, to_loss)
    # print(loss_data[0], to_loss[0])
    # print(criterion(loss_data[0], to_loss[0]))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
    if epoch % 100 == 0:
        for j in range(begin_idx, train_data.shape[0]):
            if train_data[j, 1] == 0:
                temp_sum = torch.zeros((1))
                from_begin = min(0, j-begin_idx-input_len-pred_len)
                from_end = min(0,j-begin_idx-input_len)
                for k in range(from_begin, from_end):
                    temp_sum += output[k, j-k-input_len-begin_idx-1]
                train_data[j, 0] = temp_sum.detach()
        for i in range(train_data.shape[0] - begin_idx - input_len - pred_len + 1):
            input_data[i] = train_data[i:i+input_len, 0]

plt.plot(train_meta.numpy(), label='Original Data')
plt.plot(train_data[train_data[:, 1] == 1, 0].detach().numpy(), label='Predictions')
plt.legend()
plt.show()