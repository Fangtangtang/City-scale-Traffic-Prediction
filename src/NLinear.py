from dataloader import DateLoader
from dataloader import format_train_data
data_loader = DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.json")


idx = list(test_data.keys())
ans = {}

# max_n = 0
# min_n = 100000
# n = []
# for i in idx:
#     train_data = data_loader.load_train(train_path="data/train", idx=[i])
#     min_n = min(min_n, len(train_data[i]))
#     max_n = max(max_n, len(train_data[i]))
#     n.append([i, len(train_data[i])])
# print(min_n, max_n)
# print(sorted(n, key=lambda x:x[1]))
# train_data = data_loader.load_train(train_path="data/train", idx=[idx])
# train_data = train_data[idx]
# test_data = test_data[idx]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NLinear(nn.Module):
    def __init__(self, input_len:int, pred_len:int) -> None:
        super(NLinear, self).__init__()
        self.input_len  = input_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.input_len, self.pred_len)
    
    def forward(self, x):
        # x: [batch, input_len]
        seq_last = x[:, -1:].detach()
        x = x - seq_last
        output = self.Linear(x)
        return output + seq_last

for ii in idx[:]:
    ans[str(ii)] = []
    input_len = 24
    pred_len = 1

    model = NLinear(input_len=input_len, pred_len=pred_len).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    train_data = data_loader.load_train(train_path="data/train", idx=[ii])
    train_data = train_data[ii]
    test_data_ = test_data[ii]
    begin_idx_, train_data_, train_meta_ = format_train_data(data=train_data)

    input_data = torch.zeros((train_data_.shape[0] - begin_idx_ - input_len - pred_len + 1, input_len)).to(device)
    loss_data = torch.zeros((train_data_.shape[0] - begin_idx_ - input_len - pred_len + 1, pred_len)).to(device)
    loss_mask = torch.zeros((train_data_.shape[0] - begin_idx_ - input_len - pred_len + 1, pred_len)).to(device)

    for i in range(train_data_.shape[0] - begin_idx_ - input_len - pred_len + 1):
        input_data[i] = train_data_[i+begin_idx_:i+begin_idx_+input_len, 0]
        loss_data[i] = train_data_[i+begin_idx_+input_len:i+begin_idx_+input_len+pred_len, 0]
        loss_mask[i] = train_data_[i+begin_idx_+input_len:i+begin_idx_+input_len+pred_len, 1]

    epochs = 10000
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        output = model(input_data)
        to_loss = output * loss_mask
        loss = criterion(loss_data, to_loss)
        # print(loss_data[0], to_loss[0])
        # print(criterion(loss_data[0], to_loss[0]))
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        # if epoch % 100 == 0:
        #     for j in range(begin_idx, train_data.shape[0]):
        #         if train_data[j, 1] == 0:
        #             temp_sum = torch.zeros((1))
        #             from_begin = min(0, j-begin_idx-input_len-pred_len)
        #             from_end = min(0,j-begin_idx-input_len)
        #             for k in range(from_begin, from_end):
        #                 temp_sum += output[k, j-k-input_len-begin_idx-1]
        #             train_data[j, 0] = temp_sum.detach()
        #     for i in range(train_data.shape[0] - begin_idx - input_len - pred_len + 1):
        #         input_data[i] = train_data[i:i+input_len, 0]

    # plt.plot(train_meta.numpy(), label='Original Data')
    # plt.plot(train_data[train_data[:, 1] == 1, 0].detach().numpy(), label='Predictions')
    # plt.legend()
    # plt.show()
    for j in range(len(output)):
        i_ = j + begin_idx_ + input_len
        if train_data_[i_][1] == 0:
            train_data_[i_][0] = output[j]
    for k in test_data_:
        ans[str(ii)].append(train_data_[k[0]][0].item())
    with open("ans1.json", "a") as fout:
        fout.write(json.dumps({ii:ans[ii]}) + '\n')
        fout.flush()
