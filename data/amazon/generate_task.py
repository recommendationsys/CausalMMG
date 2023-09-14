import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
from tqdm import tqdm
random.seed(13)
'''
石川给出的源码中只包含他划分的训练个四种场景下的测试数据也就是movielens文件夹下的所有json文件，并没有给出support和query的划分后的json文件
给到的json文件中是没有过滤掉不符合交互数在10-100之间的用户的，所以再通过json文件生成任务时，需要先将不符合标准的用户过滤掉
对于support和query的构建，以元训练为例，是通过在meta_training.json文件中先得到交互项目数在10-100之间的哪些用户，每一个用户作为一个任务，其中从这些
留下的用户中随机选择10个项目作为该用户的query集，其余的作为support集

'''
input_dir = 'original/'
output_dir = './'  # 当前目录
melu_output_dir = './'   # ../ 当前目录上一级目录
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "meta_training","user_and_item_cold_testing"]
# states = [ "meta_training","user_and_item_cold_testing"]





def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

for state in states:
    state_data = json.load(open(output_dir+state+'.json','r'), object_hook=jsonKeys2int)
    state_data_y = json.load(open(output_dir + state + '_y.json', 'r'), object_hook=jsonKeys2int)
    support_dic = {}
    support_y_dic = {}
    query_dic = {}
    query_y_dic = {}
    for key, key_y in zip(state_data.keys(), state_data_y.keys()):
        train_x = state_data[key]
        train_y = state_data_y[key]

        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(train_x)
        random.seed(randnum)
        random.shuffle(train_y)
        query_dic[key] = train_x[0:10]
        support_dic[key] = train_x[10:]
        query_y_dic[key] = train_y[0:10]
        support_y_dic[key] = train_y[10:]
    with open( state + '/support_u_amazon.json', 'w', encoding='utf-8') as fp:
        json.dump(support_dic, fp, ensure_ascii=False)

    with open( state + '/support_u_amazon_y.json', 'w', encoding='utf-8') as fp:
        json.dump(support_y_dic, fp, ensure_ascii=False)

    with open(state + '/query_u_amazon.json', 'w', encoding='utf-8') as fp:
        json.dump(query_dic, fp, ensure_ascii=False)

    with open(state + '/query_u_amazon_y.json', 'w', encoding='utf-8') as fp:
        json.dump(query_y_dic, fp, ensure_ascii=False)