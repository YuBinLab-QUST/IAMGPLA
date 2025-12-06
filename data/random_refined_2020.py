
import pickle,os
import pandas as pd



dir="D:\\browserdownload\\PDBbind_v2020_refined\\refined-set"
refined_set_ids=[]
for _, dirnames, _ in os.walk(dir):
    for i in dirnames:
        refined_set_ids.append(i)
print(len(refined_set_ids))


all_pdbids=[]
pdbid = pd.read_csv('C://Users//25319//Desktop//renew//aff_20_8.csv')
for i, row in pdbid.iterrows():
    cid, pKa = row['id'], float(row['affinity'])
    all_pdbids.append(cid)
print(len(all_pdbids))
datas=[]

for ind, id in enumerate(all_pdbids):
    if id in refined_set_ids:
        datas.append(ind)

print(len(datas))


import random

def random_split_by_ratio(main_list, ratios):
    # 复制原列表并打乱，避免修改原始数据
    shuffled_list = main_list.copy()
    random.shuffle(shuffled_list)  # 随机打乱顺序
    
    total_len = len(shuffled_list)
    split_result = []
    start_idx = 0
    
    # 按比例计算每个子列表的长度并切片（最后一个子列表取剩余所有元素）
    for i, ratio in enumerate(ratios[:-1]):
        end_idx = start_idx + int(total_len * ratio)
        split_result.append(shuffled_list[start_idx:end_idx])
        start_idx = end_idx
    
    # 添加最后一个子列表（处理比例总和超出1的情况）
    split_result.append(shuffled_list[start_idx:])
    
    return split_result

# 测试示例
original_list = list(range(100))  # 原始列表（100个元素，可替换为你的数据）
ratios = [0.7, 0.1, 0.2]  # 目标比例
split_lists = random_split_by_ratio(datas, ratios)

print(len(split_lists))
print(len(split_lists[0]))
print(len(split_lists[1]))
print(len(split_lists[2]))

with open("C://Users//25319//Desktop//renew//random_refinedset2020_data.pkl", "wb") as f:
    pickle.dump(split_lists, f) 
print("ok")