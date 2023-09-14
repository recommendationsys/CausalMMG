config_db = {
    'dataset': 'dbook',
    'use_cuda': True,
    'file_num': 8,  # each task contains 10 files
    # user
    'num_location': 453,
    'num_author' : 10544,

    # item
    'num_publisher': 1698,

    'item_fea_len': 3,
    'embedding_dim': 64,
    'user_embedding_dim': 64*1,  # 1 features
    'item_embedding_dim': 64*1,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 1e-4,
    'mp_lr': 1e-3,
    'local_lr': 1e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 120,
    'neigh_agg': 'mean',
    'mp_agg': 'mean',
    'layer' : 4,
}


config_ml = {
    'dataset': 'movielens',
    'use_cuda': True,
    'file_num': 8,  # each task contains 12 files for movielens

    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_fea_item': 2,
    'item_fea_len': 27,

    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 5,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*4,  # 4 features
    'item_embedding_dim': 32*2,  # 2 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 1e-4,
    'mp_lr': 1e-4,
    'local_lr': 1e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 100,
    'neigh_agg': 'mean',
    'mp_agg': 'mean',
    'layer' : 4,
}


config_yelp = {
    'dataset': 'yelp',
    'use_cuda': True,
    'file_num': 8,  # each task contains 12 files

    # item
    'num_stars': 9,
    'num_postalcode': 6133,
    'num_fea_item': 2,
    'item_fea_len': 6,

    # user
    'num_fans': 412,
    'num_avgrating': 359,
    'num_fea_user': 3,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*2,  # 1 features
    'item_embedding_dim': 32*2,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 1e-4,
    'mp_lr': 1e-4,
    'local_lr': 1e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 100,
    'neigh_agg': 'mean',
    'mp_agg': 'mean',
    'layer' : 4,
}

config_amazon = {
    'dataset': 'amazon',
    'use_cuda': True,
    'file_num': 8,  # each task contains 12 files

    # item
    'num_brand': 334,
    'num_category': 22,
    'item_fea_len': 3,

    # user
    'num_user': 6170,
    'num_fea_user': 1,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*1,  # 1 features
    'item_embedding_dim': 32*2,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'layer_update': 1,
    'local_update': 1,
    'lr': 1e-4,
    'mp_lr': 1e-4,
    'local_lr': 1e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 150,
    'neigh_agg': 'mean',
    'mp_agg': 'mean',
    'layer' : 3,
    'A_split': False,
    'A_n_fold': 200,

}

states = ["meta_training","warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]
