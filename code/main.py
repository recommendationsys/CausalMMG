import random
import time
import numpy as np
import torch
from CausalMMG import CausalMMG
from DataHelper import DataHelper
np.random.seed(13)
torch.manual_seed(13)



def training(model, model_save=True, model_file=None, device = 'cpu'):
    print('training model...')
    if config['use_cuda']:
        model.cuda()
    model.train()

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']
    state = 'meta_training'
    for _ in range(num_epoch):  # 20
        loss, mae, rmse = [], [], []
        ndcg_at_5 = []
        start = time.time()

        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size)  # ~80
        supp_xs_s, supp_ys_s, query_xs_s, query_ys_s,att_item_s, rate_item_s, att_neibor_item_s,rate_neibor_item_s = zip(*train_data)
        for i in range(num_batch):  # each batch contains some tasks (each task contains a support set and a query set)
            support_xs = list(supp_xs_s[batch_size * i:batch_size * (i + 1)])
            support_ys = list(supp_ys_s[batch_size * i:batch_size * (i + 1)])
            query_xs = list(query_xs_s[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i:batch_size * (i + 1)])
            att_item = list(att_item_s[batch_size * i:batch_size * (i + 1)])
            rate_item = list(rate_item_s[batch_size * i:batch_size * (i + 1)])
            att_neibor_item = list(att_neibor_item_s[batch_size * i:batch_size * (i + 1)])
            rate_neibor_item = list(rate_neibor_item_s[batch_size * i:batch_size * (i + 1)])
            _loss, _mae, _rmse, _ndcg_5 = model.global_update(gcn_data_dir, state, support_xs, support_ys,
                                                              query_xs, query_ys, att_item, rate_item, att_neibor_item,
                                                              rate_neibor_item, device)
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_5.append(_ndcg_5)

        print('epoch: {}, loss: {:.6f}, cost time: {:.1f}s, mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
              format(_, np.mean(loss), time.time() - start,
                     np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)))

        if _ % 10 == 0 and _ != 0:
            testing(model, device)
            model.train()

    if model_save:
        print('saving model...')
        torch.save(model.state_dict(), model_file+ '.pkl')


def testing(model, device = "cpu"):
    # testing
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()
    for state in ["user_and_item_cold_testing","item_cold_testing","user_cold_testing","warm_up"]:
        if state == 'meta_training':
            continue
        print(state + '...')
        evaluate(model, state, device)


def evaluate(model, state, device='cpu'):
    test_data = data_helper.load_data(data_set=data_set, state=state,
                                      load_from_file=True)
    supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, att_item_s, rate_item_s, att_neibor_item_s,rate_neibor_item_s = zip(*test_data)
    loss, mae, rmse = [], [], []
    ndcg_at_5 = []

    for i in range(len(test_data)):  # each task
        _mae, _rmse, _ndcg_5 = model.evaluation(gcn_data_dir, state, supp_xs_s[i], supp_ys_s[i],
                                                query_xs_s[i], query_ys_s[i], att_item_s[i], rate_item_s[i],
                                                att_neibor_item_s[i], rate_neibor_item_s[i], device)
        mae.append(_mae)
        rmse.append(_rmse)
        ndcg_at_5.append(_ndcg_5)
    print('mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
          format(np.mean(mae), np.mean(rmse),np.mean(ndcg_at_5)))



if __name__ == "__main__":
    # data_set = 'dbook'
    # data_set = 'movielens'
    # data_set = 'yelp'
    data_set = 'amazon'

    input_dir = '../data/'
    output_dir = '../data/'
    res_dir = '../res/'+data_set
    load_model = False

    if data_set == 'movielens':
        from Config import config_ml as config
    elif data_set == 'yelp':
        from Config import config_yelp as config
    elif data_set == 'dbook':
        from Config import config_db as config
    elif data_set == 'amazon':
        from Config import config_amazon as config
    cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(config)

    model_filename = "{}/causalMMG.pkl".format(res_dir)
    data_helper = DataHelper(input_dir, output_dir, config)

    gcn_data_dir = output_dir + data_set
    # training model.
    model_name = 'layer_update'

    cmmg = CausalMMG(config).cuda()

    print('--------------- {} ---------------'.format(model_name))

    if not load_model:
        # Load training dataset
        print('loading train data...')
        train_data = data_helper.load_data(data_set=data_set,state='meta_training',load_from_file=True)
        training(cmmg, model_save=True, model_file=model_filename,device = cuda_or_cpu)
    else:
        trained_state_dict = torch.load(model_filename)
        cmmg.load_state_dict(trained_state_dict)

    # testing
    testing(cmmg,device = cuda_or_cpu)
    print('--------------- {} ---------------'.format(model_name))

