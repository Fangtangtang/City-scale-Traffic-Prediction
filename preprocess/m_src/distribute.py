import torch
import argparse
import json
from dataloader import DateLoader

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


data_loader = DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

idx_lists = chunk_list(list(test_data.keys()),256)
print(len(list(test_data.keys())))

for i in range(len(idx_lists)):
    with open(f'data/work_list/large_work_list{i}.json', 'w') as file:
        json.dump(idx_lists[i], file)