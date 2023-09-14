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

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

state = 'meta_training'

support_u_amazon = json.load(open(output_dir+state+'/support_u_amazon.json','r'), object_hook=jsonKeys2int)
query_u_amazon = json.load(open(output_dir+state+'/query_u_amazon.json','r'), object_hook=jsonKeys2int)
support_u_amazon_y = json.load(open(output_dir+state+'/support_u_amazon_y.json','r'), object_hook=jsonKeys2int)
query_u_amazon_y = json.load(open(output_dir+state+'/query_u_amazon_y.json','r'), object_hook=jsonKeys2int)
if support_u_amazon.keys() == query_u_amazon.keys():
    u_id_list = support_u_amazon.keys()
print(len(u_id_list))

train_u_amazon = {}

for idx, u_id in tqdm(enumerate(u_id_list)):
    train_u_amazon[int(u_id)] = []
    train_u_amazon[int(u_id)] += support_u_amazon[u_id]+query_u_amazon[u_id]
print(len(train_u_amazon))
train_u_id_list = list(u_id_list).copy()
print(len(train_u_id_list))
# get mp data
print(state)

att_item_neibors = json.load(open('a_b_amazon15.json', 'r'),object_hook=jsonKeys2int)
rate_item_neibor_support = json.load(open(output_dir+state+'/support_u_rate_amazon15.json', 'r'),
                                           object_hook=jsonKeys2int)
rate_item_neibor_query = json.load(open(output_dir+state+'/query_u_rate_amazon15.json', 'r'),
                                           object_hook=jsonKeys2int)



for idx, u_id in tqdm(enumerate(u_id_list)):
    support_x_app = None
    att_item_tenor = None
    rate_item_tensor = None
    att_item_neibor = []
    rate_item_neibor = []
    user_support_dic = rate_item_neibor_support.get(u_id)
    user_query_dic = rate_item_neibor_query.get(u_id)
    flag = True
    user_idx = torch.Tensor([[u_id]]).long()
    for b_id in support_u_amazon[u_id]:
        tmp_x_converted = torch.cat((item_fea_homo[b_id], user_idx), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
        except:
            support_x_app = tmp_x_converted

        #找到项目的属性邻居
        if flag:
            att_item_tenor = item_fea_homo[b_id]
            rate_item_tensor = item_fea_homo[b_id]
            flag = False
        else:
            att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
        # att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
            rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[b_id]), 0)
        att_items = att_item_neibors.get(b_id)
        item_n = [b_id]+att_items
        att_item_neibor.append(item_n)

        for item in att_items:
            att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[item]), 0)


        rate_items_support = user_support_dic.get(b_id)
        item_rate = [b_id]+rate_items_support
        rate_item_neibor.append(item_rate)
        for item in rate_items_support:
            rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[item]), 0)

    support_y_app = torch.FloatTensor(support_u_amazon_y[u_id])

    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    # pickle.dump(support_rate_app, open("{}/{}/support_x_rate_{}.pkl".format(output_dir, state, idx), "wb"))
    # pickle.dump(support_att_app, open("{}/{}/support_x_att_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))


    query_x_app = None

    for b_id in query_u_amazon[u_id]:
        tmp_x_converted = torch.cat((item_fea_homo[b_id], user_idx), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
        except:
            query_x_app = tmp_x_converted

        att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
        rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[b_id]), 0)
        att_items = att_item_neibors.get(b_id)
        item_n = [b_id] + att_items
        att_item_neibor.append(item_n)
        for item in att_items:
            att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[item]), 0)

        rate_items_query = user_query_dic.get(b_id)
        item_rate = [b_id]+rate_items_query
        rate_item_neibor.append(item_rate)
        for item in rate_items_query:
            rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[item]), 0)


    query_y_app = torch.FloatTensor(query_u_amazon_y[u_id])

    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(rate_item_neibor, open("{}/{}/rate_neibor_item_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(att_item_neibor, open("{}/{}/att_neibor_item_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(rate_item_tensor, open("{}/{}/rate_item_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(att_item_tenor, open("{}/{}/att_item_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))

print(idx)


# state = 'warm_up'
# state = 'user_cold_testing'
# state = 'item_cold_testing'
state = 'user_and_item_cold_testing'
for state in states:
    support_u_amazon = json.load(open(output_dir + state + '/support_u_amazon.json', 'r'), object_hook=jsonKeys2int)
    query_u_amazon = json.load(open(output_dir + state + '/query_u_amazon.json', 'r'), object_hook=jsonKeys2int)
    support_u_amazon_y = json.load(open(output_dir + state + '/support_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)
    query_u_amazon_y = json.load(open(output_dir + state + '/query_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)

    att_item_neibors = json.load(open('a_b_amazon15.json', 'r'), object_hook=jsonKeys2int)
    rate_item_neibor_support = json.load(open(output_dir + state + '/support_u_rate_amazon15.json', 'r'),
                                         object_hook=jsonKeys2int)
    rate_item_neibor_query = json.load(open(output_dir + state + '/query_u_rate_amazon15.json', 'r'),
                                       object_hook=jsonKeys2int)

    if support_u_amazon.keys() == query_u_amazon.keys():
        u_id_list = support_u_amazon.keys()
    print(len(u_id_list))

    cur_train_u_amazon = train_u_amazon.copy()

    print(len(u_id_list))
    for idx, u_id in tqdm(enumerate(u_id_list)):
        if u_id not in cur_train_u_amazon:
            cur_train_u_amazon[u_id] = []
        cur_train_u_amazon[u_id] += support_u_amazon[u_id]

    print(len(cur_train_u_amazon), len(train_u_amazon))
    print(len(set(train_u_id_list) & set(u_id_list)))

    print((len(u_id_list) +  len(train_u_amazon) - len(set(train_u_id_list) & set(u_id_list))) == len(set(cur_train_u_amazon)))
    # get mp data
    print(state)


    for idx, u_id in tqdm(enumerate(u_id_list)):
        support_x_app = None
        att_item_tenor = None
        rate_item_tensor = None
        att_item_neibor = []
        rate_item_neibor = []
        user_support_dic = rate_item_neibor_support.get(u_id)
        user_query_dic = rate_item_neibor_query.get(u_id)
        flag = True

        user_idx = torch.Tensor([[u_id]]).long()
        for b_id in support_u_amazon[u_id]:
            tmp_x_converted = torch.cat((item_fea_homo[b_id], user_idx), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted

            # 找到项目的属性邻居
            if flag:
                att_item_tenor = item_fea_homo[b_id]
                rate_item_tensor = item_fea_homo[b_id]
                flag = False
            else:
                att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
                # att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
                rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[b_id]), 0)
            att_items = att_item_neibors.get(b_id)
            item_n = [b_id] + att_items
            att_item_neibor.append(item_n)

            for item in att_items:
                att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[item]), 0)

            rate_items_support = user_support_dic.get(b_id)
            item_rate = [b_id] + rate_items_support
            rate_item_neibor.append(item_rate)
            for item in rate_items_support:
                rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[item]), 0)

        support_y_app = torch.FloatTensor(support_u_amazon_y[u_id])

        pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
        # pickle.dump(support_rate_app, open("{}/{}/support_x_rate_{}.pkl".format(output_dir, state, idx), "wb"))
        # pickle.dump(support_att_app, open("{}/{}/support_x_att_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))

        query_x_app = None

        user_idx = torch.Tensor([[u_id]]).long()
        for b_id in query_u_amazon[u_id]:
            tmp_x_converted = torch.cat((item_fea_homo[b_id], user_idx), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted

            att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[b_id]), 0)
            rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[b_id]), 0)
            att_items = att_item_neibors.get(b_id)
            item_n = [b_id] + att_items
            att_item_neibor.append(item_n)
            for item in att_items:
                att_item_tenor = torch.cat((att_item_tenor, item_fea_homo[item]), 0)

            rate_items_query = user_query_dic.get(b_id)
            item_rate = [b_id] + rate_items_query
            rate_item_neibor.append(item_rate)
            for item in rate_items_query:
                rate_item_tensor = torch.cat((rate_item_tensor, item_fea_homo[item]), 0)

        query_y_app = torch.FloatTensor(query_u_amazon_y[u_id])

        pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(rate_item_neibor, open("{}/{}/rate_neibor_item_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(att_item_neibor, open("{}/{}/att_neibor_item_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(rate_item_tensor, open("{}/{}/rate_item_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(att_item_tenor, open("{}/{}/att_item_{}.pkl".format(output_dir, state, idx), "wb"))
        pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))

    print(idx)
