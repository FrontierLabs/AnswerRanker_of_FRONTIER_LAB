# -*- coding: utf-8 -*-

from config_local import base_model_folder

from context_lasagne import *
from experiment_base import ExpBase
from experiment_base_ubuntu import UbuntuDataLoader, Vocab


def get_model(context_num, vocab_size, max_len):
    vocab = Vocab()
    EMBEDDING_SIZE = 50
    vocab.load_vocab('/home/v-huixue/workspace/ubuntu/data/new_embed', EMBEDDING_SIZE)
    model = DefaultRelevanceModel(context_num + 1, 10, max_len, vocab_size, True, EMBEDDING_SIZE,
        base_model_folder + 'relevance.ubuntu.sm_gru.eye', reg_rate=1e-4,
        embedding_w=vocab.get_weight()[0],
        kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
        kwargs4predict={'sm_len': 100})
    model_log = 'context size %d, model:relevance.douban.sm_gru.eye, gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0], eye[]' % context_num
    # model = DefaultMultiTurnModel(context_num + 1, 10, max_len, vocab.size, True, 100,
    #     base_model_folder + 'relevance.douban.sm_gru.dense', reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
    #     kwargs4predict={'sm_len': 100, 'mlp_hidden': 50})
    # model = MultiEyeMultiTurnModel(context_num + 1, 10, max_len, vocab.size, True, 100,
    #     base_model_folder + 'context.douban.sm_gru.eye', reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2},
    #     kwargs4predict={'sm_len': 100})
    # model = MemoryNetworkMultiTrunModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + 'context.douban.sm_gru.mm', reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
    #     kwargs4predict={'sm_len': 100})
    # model_log = 'context size %d, model:context.douban.sm_gru.mm, gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0.2], mm[]' % context_num
    return model, model_log

if __name__ == '__main__':
    context_num = 1
    percent = 0.5
    data_loader = UbuntuDataLoader(context_num, batch_size=10, percent=percent, max_len_max=40)
    vocab_size = data_loader.vocab_size
    max_len = data_loader.max_len
    model, model_log = get_model(context_num, vocab_size, max_len)
    exp = ExpBase(model, model_log, data_loader, 10)
    # exp.train(epoch=10)
    # exp.test(epoch2model=9)
    exp.test_p_at_k(epoch2model=6, balance_test=False, k_list=[1, 2, 5])
