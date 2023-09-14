import gc
import glob
import os
import pickle
from tqdm import tqdm



class DataHelper:
    def __init__(self, input_dir, output_dir, config):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config

    def load_data(self, data_set, state, load_from_file=True):
        data_dir = os.path.join(self.output_dir,data_set)
        supp_xs_s = []
        supp_ys_s = []
        query_xs_s = []
        query_ys_s = []
        att_item_s = []
        rate_item_s = []
        att_neibor_item_s = []
        rate_neibor_item_s = []


        training_set_size = int(len(glob.glob("{}/{}/*.pkl".format(data_dir,state))) / self.config['file_num'])  # support, query
        for idx in tqdm(range(training_set_size)):
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            att_item_s.append(pickle.load(open("{}/{}/att_item_{}.pkl".format(data_dir, state, idx), "rb")))
            rate_item_s.append(pickle.load(open("{}/{}/rate_item_{}.pkl".format(data_dir, state, idx), "rb")))
            att_neibor_item_s.append(
                pickle.load(open("{}/{}/att_neibor_item_{}.pkl".format(data_dir, state, idx), "rb")))
            rate_neibor_item_s.append(
                pickle.load(open("{}/{}/rate_neibor_item_{}.pkl".format(data_dir, state, idx), "rb")))

        print('#support set: {}, #query set: {}'.format(len(supp_xs_s), len(query_xs_s)))
        total_data = list(zip(supp_xs_s, supp_ys_s,
                              query_xs_s, query_ys_s, att_item_s, rate_item_s, att_neibor_item_s,
                              rate_neibor_item_s))  # all training tasks
        del (
        supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, att_item_s, rate_item_s, att_neibor_item_s, rate_neibor_item_s)
        gc.collect()
        return total_data





