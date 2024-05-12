选取了sensor 5
- 5.jsonl：原始数据
- 5_zero.jsonl：用0补位（decompose不了）
- 5_bak.jsonl：用邻近数据直接补位
- 5_avg.jsonl：分解为24*7的组，用均值补全，仍然空缺的用24的组、全局均值补全