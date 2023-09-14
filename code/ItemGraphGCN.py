import json
import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from GCN import GraphConvolution
import warnings
warnings.filterwarnings("ignore")

class LoaderData():
    def __init__(self, config, userid, item_ids,rate_item,neibor_item,item_emb, path="../data/movielens/meta_training", device='cpu'):
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.m_item = 0
        self.path = path
        self.device = device
        self.item_emb = item_emb

        Item, RateItem = [], []
        UniqueItems = set()
        all_second_items = set()
        self.dataSizeItem = 0
        self.dataSizeUser = 0

        for item in neibor_item:
            UniqueItems.update(item)
            item_list = [int(i) for i in item[1:]]
            Item.extend([item[0]] * len(item_list))
            all_second_items.update(item_list)
            RateItem.extend(item_list)
            self.dataSizeItem += len(item)

        self.UniqueItems = list(UniqueItems)
        att_item_array = np.array(rate_item)
        self.all_second_items = list(all_second_items)

        att_item_df = pd.DataFrame(att_item_array)
        att_item_df = att_item_df.drop_duplicates()
        att_item_df.set_index(att_item_df.iloc[:, 0], inplace=True)
        att_data = att_item_df.loc[self.UniqueItems, :]
        att_feature = att_data.values.tolist()

        self.item_embedding = self.item_emb.forward(torch.tensor(att_feature))

        self.m_item = len(UniqueItems)
        item_map_dic = dict(zip(self.UniqueItems, range(self.m_item)))
        self.item_map_dic = item_map_dic

        df_item = pd.DataFrame(Item, columns=['item'])
        df_item = df_item.assign(item_id=[item_map_dic[iid] for iid in df_item.item])
        df_rate_item = pd.DataFrame(RateItem, columns=['item'])
        df_rate_item = df_rate_item.assign(item_id=[item_map_dic[iid] for iid in df_rate_item.item])
        ItemMap = df_item['item_id'].values.tolist()
        RateItemMap = df_rate_item['item_id'].values.tolist()
        self.Item = np.array(ItemMap)
        self.RateItem = np.array(RateItemMap)
        self.ItemItemNet = csr_matrix((np.ones(len(self.Item)), (self.Item, self.RateItem)),
                                      shape=(self.m_item, self.m_item))
        self.items_D = np.array(self.ItemItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self.Graph = None
        self.allPosItem = self.getItemPosItems(list(range(self.m_item)))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = self.m_item// self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def getSparseGraph(self):
        if self.Graph is None:
            itemAdj = self.ItemItemNet.tolil()
            adj_mat = itemAdj
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.device)
                #print("don't split the matrix")
        return self.Graph

    def getItemPosItems(self, items):
        posItems = []
        for item in items:
            posItems.append(self.ItemItemNet[item].nonzero()[1])
        return posItems

    def jsonKeys2int(self,x):
        if isinstance(x, dict):
            return {int(k): v for k, v in x.items()}
        return x
    


class ItemGraphGCN(torch.nn.Module):
    def __init__(self,config):
        super(ItemGraphGCN, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.n_layers = self.config['layer']
        gcn_layer = {}
        for i in range(self.n_layers):
            gcn_layer[i] = GraphConvolution(config, config['item_embedding_dim'], config['item_embedding_dim'])
        self.gcn_layer = gcn_layer

    def one_user(self, path, userid, item_ids, rate_item, neibor_item, item_emb):
        self.userid = userid
        self.item_ids = item_ids
        self.dataset = LoaderData(self.config, userid, item_ids, rate_item, neibor_item, item_emb, path)
        self.__init_weight()

    def __init_weight(self):
        self.num_items = self.dataset.m_item
        self.item_map_dic = self.dataset.item_map_dic
        self.item_bedding = self.dataset.item_embedding
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        #print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def second_item_neibor(self):
        all_second_items = self.dataset.all_second_items
        return all_second_items

    def computer(self, way="training"):
        """
        propagate methods for lightGCN
        """
        node_embedding = self.item_bedding
        embs = [node_embedding]
        g_droped = self.Graph

        for i in range(self.n_layers):
            node_embedding = self.gcn_layer[i](node_embedding, g_droped)
            embs.append(node_embedding)
        embs = torch.stack(embs, dim=1)

        items = torch.mean(embs, dim=1)
        items_split = embs
        if way == "getItem":
            return items
        elif way == "getSplitItem":
            return items_split
        else:
            return items


    def item_idx_map(self,item):
        df_item = pd.DataFrame(item, columns=['item'])
        df_item = df_item.assign(item_id=[self.item_map_dic[iid] for iid in df_item.item])
        ItemMap = df_item['item_id'].values.tolist()

        return ItemMap
    def emb_item(self,item):
        item_map = self.item_idx_map(item)
        all_items = self.computer()
        return all_items[item_map]


    def emb_item_split(self,item):
        way = "getSplitItem"
        all_items = self.computer(way)
        item_map = self.item_idx_map(item)
        return all_items[item_map]
