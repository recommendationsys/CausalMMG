import torch
from torch.nn import functional as F


class MetaLearner(torch.nn.Module):
    def __init__(self,config):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = 64 + config['item_embedding_dim']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        # prediction parameters
        self.vars = torch.nn.ParameterDict().to(self.device)
        self.vars_bn = torch.nn.ParameterList().to(self.device)

        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim,self.fc1_in_dim]))  # 64, 96
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim,self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars['ml_fc_w3'] = w3
        self.vars['ml_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_emb, user_neigh_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_neigh_emb

        x = torch.cat((x_i, x_u), 1)  # ?, item_emb_dim+user_emb_dim+user_emb_dim
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1'])).to(self.device)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2'])).to(self.device)
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3']).to(self.device)
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class MetapathLearner(torch.nn.Module):
    def __init__(self,config):
        super(MetapathLearner, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        # meta-path parameters
        self.vars = torch.nn.ParameterDict().to(self.device)
        neigh_w = torch.nn.Parameter(torch.ones([64,config['item_embedding_dim']]))
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(64))

        neigh_w1 = torch.nn.Parameter(torch.ones([64, 96]))
        torch.nn.init.xavier_normal_(neigh_w1)
        self.vars['neigh_w1'] = neigh_w1
        self.vars['neigh_b1'] = torch.nn.Parameter(torch.zeros(64))

    def forward(self, user_emb, neighs_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b']).to(
            self.device)  # (#neighbors, item_emb_dim)
        output_emb = F.relu(torch.mean(agg_neighbor_emb, 0)).to(self.device)  # (#sample, user_emb_dim)
        x = torch.cat((user_emb[0], output_emb), -1)
        agg_user_emb = F.linear(x, vars_dict['neigh_w1'], vars_dict['neigh_b1']).repeat(neighs_emb.shape[0], 1).to(self.device)

        return agg_user_emb

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

