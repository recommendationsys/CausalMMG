import numpy as np
import torch
from torch.nn import functional as F
from Evaluation import Evaluation
from MetaLearner_new import MetapathLearner, MetaLearner
from ItemGraphGCN import ItemGraphGCN


class CausalMMG(torch.nn.Module):
    def __init__(self, config):
        super(CausalMMG, self).__init__()
        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.layer = config['layer']

        if self.config['dataset'] == 'movielens':
            from EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML
            self.item_emb = ItemEmbeddingML(config)
            self.user_emb = UserEmbeddingML(config)
        elif self.config['dataset'] == 'yelp':
            from EmbeddingInitializer import UserEmbeddingYelp, ItemEmbeddingYelp
            self.item_emb = ItemEmbeddingYelp(config)
            self.user_emb = UserEmbeddingYelp(config)
        elif self.config['dataset'] == 'dbook':
            from EmbeddingInitializer import UserEmbeddingDB, ItemEmbeddingDB
            self.item_emb = ItemEmbeddingDB(config)
            self.user_emb = UserEmbeddingDB(config)
        elif self.config['dataset'] == 'amazon':
            from EmbeddingInitializer import Amazon_item, Amazon_user
            self.item_emb = Amazon_item(config)
            self.user_emb = Amazon_user(config)
        self.attribution_item_graph = ItemGraphGCN(config)
        self.rate_item_graph = ItemGraphGCN(config)
        self.vars = torch.nn.ParameterDict().to(self.device)
        self.vars_bn = torch.nn.ParameterList().to(self.device)

        w1 = torch.nn.Parameter(torch.ones([64, 64]))  # 64, 96
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(64))

        w2 = torch.nn.Parameter(torch.ones([64, 128]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(64))


        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)

        self.mp_lr = config['mp_lr']
        self.local_lr = config['local_lr']
        self.emb_dim = self.config['embedding_dim']

        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())

        self.transformer_liners = self.transform_mp2task()
        #
        self.meta_optimizer = torch.optim.RMSprop(self.parameters(), lr=config['lr'])

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        output_dim_of_mp = 64
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp,
                                                          np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def forward(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]

        for idx in range(1, self.config['local_update']):  # for the current task, locally update
            support_set_y_pred = self.meta_learner( support_item_emb, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)  # calculate loss on support set
            grad = torch.autograd.grad(loss, fast_weights.values(),
                                       create_graph=True)  # calculate gradients w.r.t. model parameters

            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights


    def get_id(self,support):
        item_ids = support[:, 0].tolist()
        user_idx = support[:, -1].tolist()[0]
        return user_idx,item_ids

    def update_user(self, support_set_x, support_set_y, query_set_x, query_set_y):

        support_layer_enhanced_user_emb_s, query_layer_enhanced_user_emb_s = [], []
        layer_task_fast_weights_s = {}
        layer_task_loss_s = {}

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()

        userId_s, itemIds_s = self.get_id(support_set_x)
        userId_q, itemIds_q = self.get_id(query_set_x)

        rate_item_graph = self.rate_item_graph
        rate_support_item_emb = rate_item_graph.emb_item(itemIds_s)
        rate_support_user_emb = rate_item_graph.emb_item_split(itemIds_s)  # 前26位 最初的项目特征嵌入
        rate_query_item_emb = rate_item_graph.emb_item(itemIds_q)
        rate_query_user_emb = rate_item_graph.emb_item_split(itemIds_q)

        att_item_graph = self.attribution_item_graph
        att_support_item_emb = att_item_graph.emb_item(itemIds_s)
        att_support_user_emb = att_item_graph.emb_item_split(itemIds_s)
        att_query_item_emb = att_item_graph.emb_item(itemIds_q)
        att_query_user_emb = att_item_graph.emb_item_split(itemIds_q)

        support_user_emb = self.item_map(rate_support_user_emb, att_support_user_emb)
        query_user_emb = self.item_map(rate_query_user_emb, att_query_user_emb)
        support_item_emb = self.item_map(rate_support_item_emb, att_support_item_emb)
        query_item_emb = self.item_map(rate_query_item_emb, att_query_item_emb)

        user_feature_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])


        for layer in range(self.layer + 1):
            support_layer_emb = support_user_emb[:, layer, :]
            query_layer_emb = query_user_emb[:, layer, :]
            support_layer_enhanced_user_emb = self.mp_learner(user_feature_emb, support_layer_emb)
            support_set_y_pred = self.meta_learner(support_item_emb, support_layer_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values(), create_graph=True)
            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

            for idx in range(1, self.config['layer_update']):
                support_layer_enhanced_user_emb = self.mp_learner(user_feature_emb, support_layer_emb,
                                                                  vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_item_emb, support_layer_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]

            support_layer_enhanced_user_emb = self.mp_learner(user_feature_emb, support_layer_emb,vars_dict=fast_weights)
            support_layer_enhanced_user_emb_s.append(support_layer_enhanced_user_emb)
            query_layer_enhanced_user_emb = self.mp_learner(user_feature_emb, query_layer_emb, vars_dict=fast_weights)
            query_layer_enhanced_user_emb_s.append(query_layer_enhanced_user_emb)

            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                f_fast_weights[w] = ml_initial_weights[w] * \
                                    torch.sigmoid(liner(support_layer_enhanced_user_emb.mean(0))). \
                                        view(ml_initial_weights[w].shape)
            # f_fast_weights = None
            # # the current mp ---> task update
            layer_task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                                   support_layer_enhanced_user_emb, vars_dict=f_fast_weights)
            layer_task_fast_weights_s[layer] = layer_task_fast_weights

            query_set_y_pred = self.meta_learner(query_item_emb, query_layer_enhanced_user_emb,vars_dict=layer_task_fast_weights)
            q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            layer_task_loss_s[layer] = q_loss.data  # movielens: 0.8126 dbook 0.6084

        mp_att = F.softmax(-torch.stack(list(layer_task_loss_s.values())), dim=0)  # movielens: 0.80781 lr0.001

        agg_task_fast_weights = self.aggregatorGcn(layer_task_fast_weights_s, mp_att)
        agg_mp_emb = torch.stack(query_layer_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        query_y_pred = self.meta_learner(query_item_emb, query_agg_enhanced_user_emb,
                                         vars_dict=agg_task_fast_weights)

        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5


    def aggregatorGcn(self, task_weights_s, att):
        for idx in range(self.layer + 1):
            if idx == 0:
                att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[idx].items()})
                continue
            tmp_att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[idx].items()})
            att_task_weights = dict(zip(att_task_weights.keys(),
                                        list(map(lambda x: x[0] + x[1],zip(att_task_weights.values(), tmp_att_task_weights.values())))))

        return att_task_weights

    def item_fusion(self,att_embedding,rate_embedding):
        x_att = torch.relu(F.linear(att_embedding, self.vars['ml_fc_w1'], self.vars['ml_fc_b1']))
        x_att_ = torch.tanh(F.linear(x_att, self.vars['ml_fc_w2'], self.vars['ml_fc_b2']))
        x_rate = torch.relu(F.linear(rate_embedding, self.vars['ml_fc_w1'], self.vars['ml_fc_b1']))
        x_rate_ = torch.tanh(F.linear(x_rate, self.vars['ml_fc_w2'], self.vars['ml_fc_b2']))

        att_att = torch.exp(x_att_)
        att_rate = torch.exp(x_rate_)
        att_att_ = att_att / (att_att + att_rate)
        att_rate_ = att_rate / (att_att + att_rate)
        all_emb = att_att_ * att_embedding + att_rate_ * rate_embedding
        return all_emb

    def item_map(self,att_embedding,rate_embedding):
        # x = torch.cat((att_embedding, rate_embedding), -1)
        # all_emb = F.linear(x, self.vars['ml_fc_w2'], self.vars['ml_fc_b2'])

        x_att = torch.sigmoid(F.linear(att_embedding, self.vars['ml_fc_w1'], self.vars['ml_fc_b1']))
        weight = torch.sigmoid(F.linear(torch.cat((rate_embedding,x_att),-1), self.vars['ml_fc_w2'], self.vars['ml_fc_b2']))
        all_emb = weight * x_att + (1-weight) * rate_embedding
        # x_rate = F.relu(F.linear(rate_embedding, self.vars['ml_fc_w1'], self.vars['ml_fc_b1']))
        # weight = torch.sigmoid(F.linear(torch.cat((att_embedding, x_rate), -1), self.vars['ml_fc_w2'], self.vars['ml_fc_b2']))
        # all_emb = weight * x_rate + (1 - weight) * att_embedding

        return all_emb


    def global_update(self, gcn_data_dir, state, support_xs, support_ys, query_xs, query_ys, att_item,rate_item,att_neibor_item,rate_neibor_item, device='cpu'):
        """
        """
        batch_sz = len(support_xs)
        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []
        for i in range(batch_sz):  # each task in a batch
            gcn_data_path = gcn_data_dir + '/' + state
            userId_s, itemIds_s = self.get_id(support_xs[i])
            userId_q, itemIds_q = self.get_id(query_xs[i])
            userId = userId_s
            itemIds = itemIds_s + itemIds_q

            self.attribution_item_graph.one_user(gcn_data_path, userId, itemIds, att_item[i], att_neibor_item[i],
                                                 self.item_emb)
            self.rate_item_graph.one_user(gcn_data_path, userId, itemIds, rate_item[i], rate_neibor_item[i],
                                          self.item_emb)
            _loss, _mae, _rmse, _ndcg_5 = self.update_user(support_xs[i].to(device), support_ys[i].to(device),
                                                         query_xs[i].to(device), query_ys[i].to(device))
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)

        loss = torch.stack(loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(self, gcn_data_dir, state, support_x, support_y, query_x, query_y, att_item,rate_item,att_neibor_item,rate_neibor_item, device='cpu'):
        """
        """
        gcn_data_path = gcn_data_dir + '/' + state
        userId_s, itemIds_s = self.get_id(support_x)
        userId_q, itemIds_q = self.get_id(query_x)
        userId = userId_s
        itemIds = itemIds_s + itemIds_q
        self.attribution_item_graph.one_user(gcn_data_path, userId, itemIds, att_item, att_neibor_item,
                                             self.item_emb)
        self.rate_item_graph.one_user(gcn_data_path, userId, itemIds, rate_item, rate_neibor_item, self.item_emb)

        _, mae, rmse, ndcg_5 = self.update_user(support_x.to(device), support_y.to(device),
                                              query_x.to(device), query_y.to(device))

        return mae, rmse, ndcg_5

