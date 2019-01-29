# -*- coding: utf-8 -*-
# nohup python experiment_douban.py > train.cnn.realmean_max.rnn.log 2>&1 &
# 38090 pid

import os

import lasagne

from config_local import base_model_folder

from context_lasagne import *
from experiment_base import ExpBase
from experiment_base_douban import DoubanDataLoader


def load_pre_trained_lstm():
    params = cPickle.load(open(base_model_folder + 'pre2.100.model'))
    emb_weights = params[0]
    lstm_weights = params[1:]
    return emb_weights, lstm_weights


def get_model(context_num, vocab_size, max_len):
    # model_file = 'relevance.douban.sm_gru.eye'
    # model = DefaultRelevanceModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + model_file, reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2},
    #     kwargs4predict={'sm_len': 100})
    # model_config_log = 'gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0], eye[]'
    #
    # emb_weights, lstm_weights = load_pre_trained_lstm()
    # model_file = 'context.douban.sm_prelstm.mlp'
    # model = DefaultMultiTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + model_file, reg_rate=1e-4,
    #     embedding_trainable=False, embedding_w=emb_weights,
    #     kwargs4sm={'name': 'lstm', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2, 'weights': lstm_weights},
    #     kwargs4predict={'sm_len': 100, 'mlp_hidden': 50})
    # model_config_log = 'lstm[pre2.100.model], mlp[h:50]'
    #
    # model_file = 'context.douban.gru.mlp.large_head2'
    # model = DefaultMultiTurnModel(context_num + 1, 10, max_len, vocab_size, True, 50,
    #     base_model_folder + model_file, reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 50, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
    #     kwargs4predict={'sm_len': 50, 'mlp_hidden': 50})
    # model_config_log = 'emb50, gru[pre2.50.model], mlp[h:50]'
    #
    # model_file = 'context.douban.sm_gru.eye'
    # model = MultiEyeMultiTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + 'context.douban.sm_gru.eye', reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2},
    #     kwargs4predict={'sm_len': 100})
    # model_config_log = 'gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0]'
    #
    # model = MemoryNetworkMultiTrunModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + 'context.douban.sm_gru.mm.x', reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
    #     kwargs4predict={'sm_len': 100})
    # model_log = 'context size %d, model:context.douban.sm_gru.mm, gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0.2], mm[]' % context_num
    #
    # model_file = 'context.douban.sm_rnn.rnn'
    # model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + model_file, reg_rate=1e-4,
    #     kwargs4sm={'name': 'rnn', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'eye': True},
    #     kwargs4predict={'sm_len': 100, 'rnn_num_unit': 100, 'dense_num_unit': 1, 'type': 'rnn', 'eye': True})
    # model_config_log = 'rnn[100, l2:1e-4, drop_sm:0.2, drop_embed:0], rnn[100, eye]'
    #
    # model_file = 'context.douban.sm_gru.rnn'
    # model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #     base_model_folder + model_file, reg_rate=1e-4,
    #     kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'eye': True},
    #     kwargs4predict={'sm_len': 100, 'rnn_num_unit': 100, 'dense_num_unit': 1, 'type': 'rnn', 'eye': True})
    # model_config_log = 'gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0], rnn[100, eye]'
    #
    model_file = 'context.douban.sm_gru.gru'
    model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
        base_model_folder + model_file, reg_rate=1e-4,
        kwargs4sm={'name': 'gru', 'n_hidden': 100, 'l2_reg': True, 'drop_sm': 0.2, 'drop_embedding': 0.2},
        kwargs4predict={'sm_len': 100, 'rnn_num_unit': 100, 'dense_num_unit': 1, 'type': 'gru'})
    model_config_log = 'gru[100, l2:1e-4, drop_sm:0.2, drop_embed:0], gru[100]'
    #
    # average_exc_pad
    # model_file = 'context.douban.sm_cnn.realmean2.rnn'
    # model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #                           base_model_folder + model_file, reg_rate=1e-4,
    #                           kwargs4sm={'name': 'cnn', 'mode': 'cnn_mc', 'conv_mode': 'realmean',
    #                                      'num_filters': 1, 'filters': [2, 3, 5],
    #                                      'drop_embedding': 0.2, 'drop_sm': 0.2},
    #                           kwargs4predict={'sm_len': 300, 'rnn_num_unit': 300, 'dense_num_unit': 1, 'type': 'rnn', 'eye': True})
    # model_config_log = 'cnn[mc, num_filters:1, filters:[2,3,5], drop_sm:0.2, drop_embed:0.2, realmean], rnn[300, eye]'
    #
    # model_file = 'context.douban.sm_cnn_attention_f2.masked.realmean.rnn.200w'
    # model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #                           base_model_folder + model_file, reg_rate=1e-4,
    #                           kwargs4sm={'name': 'cnn_attention', 'weight_reg': True, 'attention_method': 2, 'step2': False,
    #                               'conv_mode': 'realmean', 'filters': [2, 3, 5],
    #                               'drop_embed': 0.2, 'drop_sm': 0.2, 'drop_cnn': 0.2},
    #                           kwargs4predict={'sm_len': 300, 'rnn_num_unit': 300, 'dense_num_unit': 1, 'type': 'rnn'}, learning_rate=0.001)
    # model_config_log = 'cnn_attention[filters:[2,3,5], drop_embed:0.2, drop_sm:0.2, drop_cnn:0.2, realmean, l2reg:1e-4, type2], rnn[300, eye]'
    #
    # model_file = 'context.douban.sm_gru_attention.masked.rnn.200w'
    # model = RNNMutilTurnModel(context_num + 1, 10, max_len, vocab_size, True, 100,
    #                           base_model_folder + model_file, reg_rate=1e-4,
    #                           kwargs4sm={'name': 'gru_attention', 'weight_reg': True, 'attention_method': 1, 'step2': False,
    #                               'n_hidden': 100, 'l2_reg': True,
    #                               'drop_embed': 0.2, 'drop_sm': 0.2, 'drop_before_att': 0.2},
    #                           kwargs4predict={'sm_len': 100, 'rnn_num_unit': 100, 'dense_num_unit': 1, 'type': 'rnn'}, learning_rate=0.001)
    # model_config_log = 'gru_attention[drop_embed:0.2, drop_sm:0.2, drop_before_att:0.2, hidden:100, l2reg:1e-4, type1], rnn[100, eye]'
    #
    if not os.path.exists(base_model_folder + model_file):
        os.mkdir(base_model_folder + model_file)
    model_log = 'context size %d, model:%s, %s' % (
        context_num, model_file, model_config_log)
    return model, model_log


def attention_matrix(exp, epoch2model):
    Xs_test, X_masks_test, y_test = exp._load_test_data(None, None)
    exp._load_weights(epoch2model)
    # print exp.model.masked
    attention_probs = exp.model.CNN_attention_getatt_prob(
        Xs_test, X_masks_test)
    # print exp.model.attention_M.get_value()
    for prob in attention_probs:
        print prob


def get_rnn_weights(exp, epoch2model):
    exp._load_weights(epoch2model)
    params = lasagne.layers.get_all_params(model.l_sm, trainable=True)
    for param in params:
        print param.get_value().shape


def get_weights(exp, epoch2model):
    exp._load_weights(epoch2model)
    # rnn
    params = lasagne.layers.get_all_params(model.l_sm, trainable=True)
    fo = open('./weights/all.para', 'w')
    for i, param in enumerate(params):
        print param.get_value().shape
        if i != 0:
            np.savetxt('./weights/tmp.%d' % ((i - 1) % 3), param.get_value())
        if i != 0 and (i - 1) % 3 == 2:
            for j in xrange(3):
                with open('./weights/tmp.%d' % j, 'r') as fp:
                    lines_tmp = ''
                    for line in fp:
                        if j != 2:
                            fo.write(line)
                        else:
                            lines_tmp += line.strip() + ' '
                    if j == 2:
                        fo.write(lines_tmp[:-1] + '\n')
    # mlp
    params1 = model.l_dense_hidden.get_params(trainable=True)
    params2 = model.l_dense_out.get_params(trainable=True)
    w1, b1 = params1[0].get_value(), params1[1].get_value()
    w2, b2 = params2[0].get_value(), params2[1].get_value()
    print w1.shape
    print b1.shape
    for w, b in [(w1, b1), (w2, b2)]:
        for i in xrange(w.shape[0]):
            fo.write(' '.join([str(x) for x in w[i]]) + '\n')
        fo.write(' '.join([str(x) for x in b]) + '\n')
    fo.close()


if __name__ == '__main__':
    context_num = 3
    max_len_max = 50
    # vocab_path = None
    vocab_path = base_model_folder + 'douban.200w.vocab'
    # vocab_path = base_model_folder + 'douban.large.10mhead.vocab'
    # save_path = base_model_folder + 'douban.large.10mhead.vocab'
    data_loader = DoubanDataLoader(
        context_num, max_len_max=max_len_max, char=True, vocab_path=vocab_path, save_path=None)
    vocab_size = data_loader.vocab_size
    max_len = data_loader.max_len
    model, model_log = get_model(context_num, vocab_size, max_len)
    exp = ExpBase(model, model_log, data_loader, 8)
    # attention_matrix(exp, 6)
    exp.train(epoch=10, shuffle=True)
    # exp.continue_train(epoch=19, last_epoch=0, shuffle=True)
    # exp.test(epoch2model=9)
    # exp.test_p_at_k(epoch2model=7, balance_test=True, k_list=[1, 2])
    # get_rnn_weights(exp, 9)
    # exp.back_embedding(epoch2model=9, vocab=data_loader.line_obj.vocab, backfile="./embedding.epoch9.txt")
    # test_data = '/mnt/sdb/share/context_online/context.ranker.char.txt'
    # test_data = '/mnt/sdb/share/context_online/context.detection.2.editor_labeled.char.txt'
    # exp.predict(epoch2model=2, backfile=test_data.replace('char', 'score.large'), testname=test_data, name_non_path=False)
    # exp.test(epoch2model=2, testname=test_data, name_non_path=False)
    # exp.back_embedding(epoch2model=2, vocab=data_loader.line_obj.vocab, backfile="./weights/embedding.epoch2.txt")
    # get_weights(exp, 2)
