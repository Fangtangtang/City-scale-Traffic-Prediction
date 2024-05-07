from m_src.dataloader import DateLoader
import random
import torch
import matplotlib.pyplot as plt

def get_data_by_days(data:torch.tensor)->torch.tensor:
    daily_data = torch.stack([data[i:i+24].sum() for i in range(0, data.size(0), 24)])
    return daily_data

def show_train_data(sample_size=5,start_time=0,end_time=200) -> None:
    '''
        sample some sensors randomly
        show data of the first 200 timestamps(default)
    '''
    data_loader = DateLoader()
    test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

    idx_list = list(test_data.keys())
    sampled_idx_list = random.sample(idx_list, sample_size)

    train_data = data_loader.load_train(train_path="data/train", idx=sampled_idx_list)

    for idx in sampled_idx_list:
        plt.plot((torch.tensor(train_data[idx])[:, 1])[start_time:end_time].numpy(), label='Original Data {}'.format(idx))

    plt.legend()
    plt.show()

def show_daily_train_data(sample_size=5,start_time=0,end_time=20) -> None:
    '''
        sample some sensors randomly
        show daily data of the first 20 days(default)
    '''
    data_loader = DateLoader()
    test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

    idx_list = list(test_data.keys())
    sampled_idx_list = random.sample(idx_list, sample_size)

    train_data = data_loader.load_train(train_path="data/train", idx=sampled_idx_list)

    for idx in sampled_idx_list:
        plt.plot(get_data_by_days(torch.tensor(train_data[idx])[:, 1])[start_time:end_time].numpy(), label='Original Data {}'.format(idx))

    plt.legend()
    plt.show()

show_daily_train_data(sample_size=10,start_time=0,end_time=20)