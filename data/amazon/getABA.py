import os
import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
from tqdm import tqdm
import collections

random.seed(13)
input_dir = 'original/'
output_dir = './'  # 当前目录
melu_output_dir = '../../../MeLU/dbook'   # ../ 当前目录上一级目录
states = ["user_cold_testing"]

"""
os.path.exists(file_path)
file_path:是一个路径
#使用函数exists()对文件存在与否进行判断，存在为True,不存在为False.
也可以在脚本中，加入if语句判断 是否存在文件，从而进行下一步的操作或者返回信息
"""
'''
if not os.path.exists("{}/meta_training/".format(output_dir)):
    os.mkdir("{}/log/".format(output_dir))
    for state in states:
        os.mkdir("{}/{}/".format(output_dir, state))
        os.mkdir("{}/{}/".format(melu_output_dir, state))
        if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
            os.mkdir("{}/{}/{}".format(output_dir, "log", state))
'''

ui_data = pd.read_csv(input_dir + 'user_item.dat', names=['user', 'item', 'rating', 'timestamp'],
                      sep='\t', engine='python')

ib = pd.read_csv(input_dir+'item_brand.dat', names=['item','brand'], sep=',',engine='python')
ic = pd.read_csv(input_dir+'item_category.dat', names=['item','category'], sep=',',engine='python')

# print("输出用户项目交互文件中存在的下标最大的项目id")
print(max(ui_data.item))
#得到集合a和集合b中都包含的元素
#print(len(set(ui_data.user)))
#print(len(set(ul.user)))
user_list = list(set(ui_data.user))
item_list = list(set(ui_data.item) & (set(ib.item) & set(ic.item)))
# print("用户的总个数与项目的总个数")

print(len(set(ui_data.item)))
#6170 2753
print(len(user_list), len(item_list))

# 用户和项目特征
brand_list = list(set(ib[ib.item.isin(item_list)].brand))
category_list = list(set(ic[ic.item.isin(item_list)].category))

print(len(brand_list), len(category_list))


item_fea_homo = {}
item_fea_hete = {}
businesses_feature_hete = []
businesses_feature_homo = []
for i in tqdm(item_list):
    brand_idx = brand_list.index(list(ib[ib['item'] == i].brand)[0])
    brand = torch.tensor([[brand_idx]]).long()
    category_idx = category_list.index(list(ic[ic['item'] == i].category)[0])
    category = torch.tensor([[category_idx]]).long()
    item_idx = torch.Tensor([[i]]).long()
    # 保存了书对应的出版社
    item_fea_hete[i] = torch.cat((item_idx, brand), 1)
    # 保存了书的出版社及作者
    item_fea_homo[i] = torch.cat((item_idx,brand, category), 1)
    businesses_feature_hete.append(item_fea_hete[i].tolist())
    businesses_feature_homo.append(item_fea_homo[i].tolist())
print(len(item_fea_hete), len(item_fea_homo))

# item_fea_hete_array = item_fea_hete.numpy()


pickle.dump(torch.tensor(businesses_feature_hete),open("amazon_feature_hete.pkl", "wb"))
pickle.dump(torch.tensor(businesses_feature_homo),open("amazon_feature_homo.pkl", "wb"))

def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)
    return dict(re_d)

a_brand =  {k: g["brand"].tolist() for k,g in ib[ib.item.isin(item_list)].groupby("item")}
b_amazon = reverse_dict(a_brand)
print(len(a_brand), len(b_amazon))
a_category =  {k: g["category"].tolist() for k,g in ic[ic.item.isin(item_list)].groupby("item")}
c_amazon = reverse_dict(a_category)
print(len(a_brand), len(b_amazon))
print(len(a_category), len(c_amazon))

a_b_dic = {}

for i in tqdm(item_list):
    brand = a_brand[i]
    brand_set = set()
    a_b_dic[i] = set()
    for b in brand:
        brand_set.update(b_amazon[b])
    if len(brand_set) < 15:
        a_b_dic[i].update(brand_set)
    else:
        a_b_dic[i].update(random.sample(brand_set, 15))
    a_b_dic[i] = list(a_b_dic[i])

with open(output_dir + '/a_b_amazon15.json', 'w', encoding='utf-8') as fp:
    json.dump(a_b_dic, fp, ensure_ascii=False)