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
from random import sample

random.seed(13)
#生成项目之间的联系
input_dir = 'original/'
output_dir = './'  # 当前目录
state = 'meta_training'

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)  ###注意看K,V 互换了
    return dict(re_d)

def union_dic(dica,dicb):
    dic = {}
    for key in dica:
        dic[key] = {}
        if dicb.get(key):
            for key1 in dica[key]:
                if dicb[key].get(key1):
                    dic[key][key1] = dica[key][key1] + dicb[key][key1]
                else:
                    dic[key][key1] = dica[key][key1]
        else:
            dic[key] = dica[key]
    for key in dicb:
        if dica.get(key):
            for key1 in dica[key]:
                if dica[key].get(key1):
                    pass
                else:
                    dic[key][key1] = dicb[key][key1]
        else:
            dic[key] = {}
            dic[key] = dicb[key]
    return dic

# 读取评分数据
support_u_books = json.load(open(output_dir + state + '/support_u_amazon.json', 'r'), object_hook=jsonKeys2int)
query_u_books = json.load(open(output_dir + state + '/query_u_amazon.json', 'r'), object_hook=jsonKeys2int)
support_u_books_y = json.load(open(output_dir + state + '/support_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)
query_u_books_y = json.load(open(output_dir + state + '/query_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)

u_id_list = support_u_books.keys()
train_u_movies = {}
for idx, u_id in tqdm(enumerate(u_id_list)):  # 迭代器
    train_u_movies[int(u_id)] = []
    # 自己加的
    # if u_id not in query_u_movies:
    #     query_u_movies[int(u_id)] = []
    # if u_id not in support_u_movies:

    ##train_u_movies 是字典  键：train_u_movies[int(u_id)]   值：support_u_movies[u_id] + query_u_movies[u_id] 支持和查询集中对应的电影
    train_u_movies[int(u_id)] += support_u_books[u_id] + query_u_books[u_id]
print("len(train_u_movies):", len(train_u_movies))


#得到项目：点击过它的用户
support_m_u = reverse_dict(support_u_books)

userRateDictSupportTrain = {}
for k, k_y in zip(support_u_books, support_u_books_y):
    userRateDictSupportTrain[k] = {}
    for i in range(len(support_u_books_y[k_y])):
        if support_u_books_y[k_y][i] in userRateDictSupportTrain[k]:
            userRateDictSupportTrain[k][support_u_books_y[k_y][i]].append(support_u_books[k][i])
        else:
            userRateDictSupportTrain[k][support_u_books_y[k_y][i]] = [support_u_books[k][i]]

# 保存的为support中每一个用户点击的项目的增强的项目
m_m_rate_support = {}
for u, rate_item in userRateDictSupportTrain.items():
    m_m_rate_support[u] = {}
    for rat, movie in rate_item.items():
        for m in movie:
            m_m_rate_support[u][m] = set([m])
            m_m_rate_support[u][m].update(userRateDictSupportTrain[u][rat])
            # 得到跟用户u点击过同一个电影的用户（support中的，不能使用query中的数据）
            user_m = support_m_u[m]
            for u_ in user_m:
                if rat in userRateDictSupportTrain[u_]:
                    # 将跟用户u点击过同一个电影的用户还点击过的其他的项目作为这个项目的邻居
                    m_m_rate_support[u][m].update(userRateDictSupportTrain[u_][rat])
                else:
                    continue
            # m_m_rate_support[u][m] = list(m_m_rate_support[u][m])
            all_rate_nei_support = list(m_m_rate_support[u][m])
            if len(all_rate_nei_support) < 15:
                m_m_rate_support[u][m] = all_rate_nei_support
            else:
                m_m_rate_support[u][m] = sample(all_rate_nei_support, 15)

# 处理query中的数据
userRateDictQueryTrain = {}
for k, k_y in zip(query_u_books, query_u_books_y):
    userRateDictQueryTrain[k] = {}
    for i in range(len(query_u_books_y[k_y])):
        if query_u_books_y[k_y][i] in userRateDictQueryTrain[k]:
            userRateDictQueryTrain[k][query_u_books_y[k_y][i]].append(query_u_books[k][i])
        else:
            userRateDictQueryTrain[k][query_u_books_y[k_y][i]] = [query_u_books[k][i]]

# 保存的为support中每一个用户点击的项目的增强的项目
m_m_rate_query = {}
for u, rate_item in userRateDictQueryTrain.items():
    m_m_rate_query[u] = {}
    for rat, movie in rate_item.items():
        for m in movie:
            # 得到跟用户u点击过同一个电影的用户（support中的，不能使用query中的数据）
            # 先将该用户在support中点击过的与该项目评分相同的项目加进去
            m_m_rate_query[u][m] = set([m])
            if rat in userRateDictSupportTrain[u]:
                m_m_rate_query[u][m].update(userRateDictSupportTrain[u][rat])
            if m in support_m_u:
                # 在元训练的过程中只有suppor中的用户可以被看得见
                for u_ in support_m_u[m]:
                    if rat in userRateDictSupportTrain[u_]:
                        # 将跟用户u点击过同一个电影的用户还点击过的其他的项目作为这个项目的邻居
                        m_m_rate_query[u][m].update(userRateDictSupportTrain[u_][rat])
                    else:
                        continue
            # m_m_rate_query[u][m] = list(m_m_rate_query[u][m])
            all_rate_nei = list(m_m_rate_query[u][m])
            if len(all_rate_nei) < 15:
                m_m_rate_query[u][m] = list(m_m_rate_query[u][m])
            else:
                m_m_rate_query[u][m] = sample(all_rate_nei, 15)


train_rate_dic = union_dic(userRateDictSupportTrain,userRateDictQueryTrain)
# 将数据保存至文件中，格式为字典格式
with open(output_dir + state + '/support_u_rate_amazon15.json', 'w', encoding='utf-8') as fp:
    json.dump(m_m_rate_support, fp, ensure_ascii=False)

with open(output_dir + state + '/query_u_rate_amazon15.json', 'w', encoding='utf-8') as fp:
    json.dump(m_m_rate_query, fp, ensure_ascii=False)



states = ["warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]

for state in states:
    print(state)
    # 读取评分数据
    support_u_books = json.load(open(output_dir + state + '/support_u_amazon.json', 'r'), object_hook=jsonKeys2int)
    query_u_books = json.load(open(output_dir + state + '/query_u_amazon.json', 'r'), object_hook=jsonKeys2int)
    support_u_books_y = json.load(open(output_dir + state + '/support_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)
    query_u_books_y = json.load(open(output_dir + state + '/query_u_amazon_y.json', 'r'), object_hook=jsonKeys2int)

    # train_u_movies中保存的为元训练的support和query中所对应的用户所点击过的所有电影
    cur_train_u_movies = train_u_movies.copy()

    if support_u_books.keys() == query_u_books.keys():
        u_id_list = support_u_books.keys()
    print(len(u_id_list))
    for idx, u_id in tqdm(enumerate(u_id_list)):
        if u_id not in cur_train_u_movies:
            cur_train_u_movies[u_id] = []
        cur_train_u_movies[u_id] += support_u_books[u_id]

    print(len(cur_train_u_movies), len(train_u_movies))

    #support_m_u = reverse_dict(support_u_movies)
    # 得到项目：点击过它的用户(包含训练集以及该测试场景下的support中的用户）
    cur_train_m_users = reverse_dict(cur_train_u_movies)

    userRateDictSupport = {}
    for k, k_y in zip(support_u_books, support_u_books_y):
        userRateDictSupport[k] = {}
        for i in range(len(support_u_books_y[k_y])):
            if support_u_books_y[k_y][i] in userRateDictSupport[k]:
                userRateDictSupport[k][support_u_books_y[k_y][i]].append(support_u_books[k][i])
            else:
                userRateDictSupport[k][support_u_books_y[k_y][i]] = [support_u_books[k][i]]

    # 保存了元训练与当前测试环境的support集中的所有用户与点击过的不同评分的项目
    cur_RateDict = union_dic(userRateDictSupport,train_rate_dic)

    # 保存的为support中每一个用户点击的项目的增强的项目
    m_m_rate_support = {}
    for u, rate_item in userRateDictSupport.items():
        m_m_rate_support[u] = {}
        for rat, movie in rate_item.items():
            for m in movie:
                m_m_rate_support[u][m] = set([m])
                m_m_rate_support[u][m].update(cur_RateDict[u][rat])
                # 得到跟用户u点击过同一个电影的用户（support中的，不能使用query中的数据）
                user_m = cur_train_m_users[m]
                for u_ in user_m:
                    if rat in cur_RateDict[u_]:
                        # 将跟用户u点击过同一个电影的用户还点击过的其他的项目作为这个项目的邻居
                        m_m_rate_support[u][m].update(cur_RateDict[u_][rat])
                    else:
                        continue
                all_rate_nei_support = list(m_m_rate_support[u][m])
                if len(all_rate_nei_support) < 15:
                    m_m_rate_support[u][m] = all_rate_nei_support
                else:
                    m_m_rate_support[u][m] = sample(all_rate_nei_support, 15)

    # 处理query中的数据
    userRateDictQuery = {}
    for k, k_y in zip(query_u_books, query_u_books_y):
        userRateDictQuery[k] = {}
        for i in range(len(query_u_books_y[k_y])):
            if query_u_books_y[k_y][i] in userRateDictQuery[k]:
                userRateDictQuery[k][query_u_books_y[k_y][i]].append(query_u_books[k][i])
            else:
                userRateDictQuery[k][query_u_books_y[k_y][i]] = [query_u_books[k][i]]

    # 保存的为support中每一个用户点击的项目的增强的项目
    m_m_rate_query = {}
    for u, rate_item in userRateDictQuery.items():
        m_m_rate_query[u] = {}
        for rat, movie in rate_item.items():
            for m in movie:
                # 得到跟用户u点击过同一个电影的用户（support中的，不能使用query中的数据）
                m_m_rate_query[u][m] = set([m])
                if rat in cur_RateDict[u]:
                    m_m_rate_query[u][m].update(cur_RateDict[u][rat])
                if m in cur_train_m_users:
                    #在元训练的过程中只有suppor中的用户可以被看得见
                    for u_ in cur_train_m_users[m]:
                        if rat in cur_RateDict[u_]:
                            # 将跟用户u点击过同一个电影的用户还点击过的其他的项目作为这个项目的邻居
                            if m in m_m_rate_query[u]:
                                m_m_rate_query[u][m].update(cur_RateDict[u_][rat])
                            else:
                                m_m_rate_query[u][m] = set(cur_RateDict[u_][rat])
                        else:
                            continue
                all_rate_nei = list(m_m_rate_query[u][m])
                if len(all_rate_nei) < 15:
                    m_m_rate_query[u][m] = list(m_m_rate_query[u][m])
                else:
                    m_m_rate_query[u][m] = sample(all_rate_nei, 15)

    # 将数据保存至文件中，格式为字典格式
    with open(output_dir + state + '/support_u_rate_amazon15.json', 'w', encoding='utf-8') as fp:
        json.dump(m_m_rate_support, fp, ensure_ascii=False)

    with open(output_dir + state + '/query_u_rate_amazon15.json', 'w', encoding='utf-8') as fp:
        json.dump(m_m_rate_query, fp, ensure_ascii=False)