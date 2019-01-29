# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle

import lasagne
from utils.keras_generic_utils import Progbar

from abc import ABCMeta, abstractmethod

import heapq

from itertools import izip


# TODO: remove sm_len param in kwargs4sm, use l_sm.output_shape instead

class EmbeddingMaskLayer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, mask_input, embedding_size, **kwargs):
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(EmbeddingMaskLayer, self).__init__(incomings, **kwargs)
        self.embedding_size = embedding_size

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = inputs[1]
        mask = T.extra_ops.repeat(mask, self.embedding_size).reshape((mask.shape[0], mask.shape[1], self.embedding_size))
        return mask * input
# end EmbeddingMaskLayer


class NLPMeanPool2DLayer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, mask_input, max_len, kernel_size, **kwargs):
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(NLPMeanPool2DLayer, self).__init__(incomings, **kwargs)
        self.max_len = max_len
        self.kernel_size = kernel_size
        self.current_len = max_len - kernel_size + 1

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0] * self.current_len
        mask = inputs[1]
        reallen = 1 / (T.nnet.relu(mask.sum(axis=1) - self.kernel_size + 1) + 1)
        reallen4multi = T.extra_ops.repeat(reallen, input.shape[1] * input.shape[2] * input.shape[3]).reshape(input.shape)
        return input * reallen4multi
# end NLPMeanPool2DLayer


class ContextModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, conversation_len, batch_size, max_len, vocab_size, masked, embedding_size, model_folder,
                 embedding_w=lasagne.init.Normal(), embedding_trainable=True, learning_rate=0.0005, reg_rate=0.,
                 kwargs4sm={}, kwargs4predict={}):
        # save the params
        self.conversation_len = conversation_len
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.masked = masked
        self.learning_rate = learning_rate
        self.model_folder = model_folder
        # init matrixs
        self.conversations = []
        if self.masked:
            self.conversation_masks = []
        for i in xrange(conversation_len):
            self.conversations.append(T.imatrix('c%d' % i))
            if self.masked:
                self.conversation_masks.append(T.fmatrix('c_mask%d' % i))
        self.y = T.ivector('y')
        # theano shared
        self.givens = {}
        for i in xrange(conversation_len):
            self.givens['c%d' % i] = theano.shared(
                np.zeros((batch_size, max_len), dtype=np.int32))
            if self.masked:
                self.givens['c_mask%d' % i] = theano.shared(
                    np.zeros((batch_size, max_len), dtype=np.int32))
        self.givens['y'] = theano.shared(
            np.zeros((batch_size,), dtype=np.int32))
        # init embeddings
        self.l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len))
        if self.masked:
            self.l_mask = lasagne.layers.InputLayer(
                shape=(batch_size, max_len))
        self.l_emb = lasagne.layers.EmbeddingLayer(
            self.l_in, vocab_size, embedding_size, W=embedding_w)
        if not embedding_trainable:
            self.l_emb.params[self.l_emb.W].remove('trainable')
        # build sentece model and get context embeddings
        sm_params, sm_params_contain_emb, reg_sm = self.build_sentence_model(kwargs4sm)
        self.params = sm_params if sm_params_contain_emb else (sm_params + lasagne.layers.get_all_params(self.l_emb, trainable=True))
        # get outputs and define quantizations
        params, reg_pred = self.build_predict(kwargs4predict)
        self.params += params
        self.reg = reg_rate * (reg_sm + reg_pred)
        train_cost, train_acc, train_pred, train_probas = self.get_predict_and_loss(train=True)
        train_loss = train_cost + self.reg
        self.updates = lasagne.updates.adam(
            train_loss, self.params, learning_rate=learning_rate)
        test_cost, test_acc, test_pred, test_probas = self.get_predict_and_loss(train=False)
        # define functions
        inputs = []
        inputs += self.conversations
        if self.masked:
            inputs += self.conversation_masks
        self.train_func = theano.function(
            inputs + [self.y], [train_cost, train_acc], updates=self.updates)
        self.test_func = theano.function(
            inputs + [self.y], [test_cost, test_acc])
        self.predict_func = theano.function(inputs, [test_pred, test_probas])

    def build_context_embedding(self, train=True):
        e_conversations = []
        for i in xrange(self.conversation_len):
            e_conversations.append(self.get_sentence_embedding(i, train=train))
        return e_conversations

    def get_predict_and_loss(self, train=True):
        e_conversations = self.build_context_embedding(train=train)
        o = self.get_predict(e_conversations, train=train)
        probas = T.concatenate(
            [(1 - o).reshape((-1, 1)), o.reshape((-1, 1))], axis=1)
        pred = T.argmax(probas, axis=1)
        cost = T.nnet.binary_crossentropy(o, self.y).mean()
        acc = T.mean(T.eq(pred, self.y))
        return cost, acc, pred, probas

    @abstractmethod
    def build_sentence_model(self, kwargs4sm={}):
        '''
        Build the sentence model based on the embedding layer(self.l_emb)

        @Return: [params], [params contains embedding] and [regularization, default use 0]
        '''
        pass

    @abstractmethod
    def get_sentence_embedding(self, conversation_idx, train=True):
        '''
        Get the sentence embedding based on built model by build_sentence_model method

        @param conversation_idx: get output for `conversation_idx`th sentence
        @param train:            get output for train or test

        @Return: [output]
        '''
        pass

    @abstractmethod
    def build_predict(self, kwargs4predict={}):
        '''
        Build model to predict and the params for updates

        @param kwargs4predict:  param dict for build model

        @Return: [params] and [regularization, default use 0]
        '''
        pass

    @abstractmethod
    def get_predict(self, e_conversations, train=True):
        '''
        Get the predict result

        @param e_conversations: list of conversation's sentence embeddings
        @param train:            get output for train or test

        @Return: [the predict result]
        '''
        pass

    def _build_features(self, current_batch, Xs, y=None, indices=None, X_masks=[]):
        i = current_batch
        if indices is not None:
            indexs = indices[i * self.batch_size:(i + 1) * self.batch_size]
        else:
            indexs = range(i * self.batch_size, (i + 1) * self.batch_size)
        features = [Xs[j][indexs] for j in xrange(self.conversation_len)]
        if self.masked:
            features += [X_masks[j][indexs] for j in xrange(self.conversation_len)]
        if y is not None:
            features += [y[indexs]]
        return features

    def _train_epoch(self, Xs, y, X_masks=[], progbar=None, train=True, shuffle=True, log=False):
        log_values = []
        cost, acc = 0., 0.
        batch = Xs[0].shape[0] / self.batch_size
        if train and shuffle:
            indices = range(Xs[0].shape[0])
            indices = np.random.permutation(indices)
        else:
            indices = None
        for i in xrange(batch):
            features = self._build_features(
                i, Xs, y=y, indices=indices, X_masks=X_masks)
            func = self.train_func if train else self.test_func
            tcost, tacc = func(*features)
            cost += self.batch_size * tcost
            acc += self.batch_size * tacc
            if train or log:
                log_values = []
                log_values.append(('loss', tcost))
                log_values.append(('acc', tacc))
                if i != batch - 1:
                    progbar.update((i + 1) * self.batch_size, log_values)
        return cost / Xs[0].shape[0], acc / Xs[0].shape[0], log_values

    def train(self, num_epochs, Xs, y, Xs_val, y_val, Xs_test, y_test, X_masks=[], X_masks_val=[], X_masks_test=[], shuffle=True, last_epoch=-1):
        assert Xs[0].shape[0] % self.batch_size == 0
        start_epoch = last_epoch + 1
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print 'Epoch:', str(epoch + 1) + '/' + str(start_epoch + num_epochs)
            progbar = Progbar(target=Xs[0].shape[0], verbose=True)
            train_cost, train_acc, log_values = self._train_epoch(Xs, y, X_masks=X_masks, progbar=progbar, train=True, shuffle=shuffle)
            val_cost, val_acc, tmp = self._train_epoch(Xs_val, y_val, X_masks=X_masks_val, train=False)
            test_cost, test_acc, tmp = self._train_epoch(Xs_test, y_test, X_masks=X_masks_test, train=False)
            log_values.append(('loss', train_cost))
            log_values.append(('acc', train_acc))
            log_values.append(('val_loss', val_cost))
            log_values.append(('val_acc', val_acc))
            log_values.append(('test_loss', test_cost))
            log_values.append(('test_acc', test_acc))
            progbar.update(Xs[0].shape[0], log_values)
            # save model
            params_values = []
            for param in self.params:
                params_values.append(param.get_value())
            fo = open(self.model_folder + '/%d.model' % epoch, 'w')
            cPickle.dump(params_values, fo)
            fo.close()

    def test(self, Xs, y, X_masks=[]):
        progbar = Progbar(target=Xs[0].shape[0], verbose=True)
        cost, acc, log_values = self._train_epoch(Xs, y, X_masks=X_masks, train=False, progbar=progbar, log=True)
        log_values.append(('loss', cost))
        log_values.append(('acc', acc))
        progbar.update(Xs[0].shape[0], log_values)
        return cost, acc

    def predict(self, Xs, X_masks=[]):
        preds, probas = np.zeros((Xs[0].shape[0], ), dtype=np.float32), np.zeros((Xs[0].shape[0], 2), dtype=np.float32)
        batch = Xs[0].shape[0] / self.batch_size
        for i in xrange(batch):
            features = self._build_features(i, Xs, X_masks=X_masks)
            tpreds, tprobas = self.predict_func(*features)
            preds[i * self.batch_size:(i + 1) * self.batch_size] = tpreds
            probas[i * self.batch_size:(i + 1) * self.batch_size] = tprobas
        return preds, probas

    def p_at_k(self, probas, group_size, k, balance=True):
        acc, total = 0, 0
        for i in xrange(probas.shape[0] / group_size):
            pres = []
            heapq.heappush(pres, probas[i * group_size][0])
            for j in xrange(1, group_size):
                if (balance and j % 2 == 1) or (not balance):
                    heapq.heappush(pres, probas[i * group_size + j][0])
            topk = set([heapq.heappop(pres) for z in xrange(k)])
            if probas[i * group_size][0] in topk:
                acc += 1
            total += 1
        return float(acc) / total

    def p_1in2_at_1(self, probas, group_size, balance=True):
        acc, total = 0, 0
        for i in xrange(probas.shape[0] / group_size):
            pos = probas[i * group_size][0]
            for j in xrange(1, group_size):
                if (balance and j % 2 == 1) or (not balance):
                    neg = probas[i * group_size + j][0]
                    if neg > pos:
                        acc += 1
                    total += 1
        return float(acc) / total

    def precision_recall(self, Xs, y, X_masks=[], threshold=0.5):
        preds, probas = self.predict(Xs, X_masks=X_masks)
        right4zero, right4one = 0., 0.
        total4zero, total4one = 0, 0
        total2zero, total2one = 0, 0
        for proba, ty in izip(probas, y):
            pred = 1 if proba[1] > threshold else 0
            if ty == 0:
                total4zero += 1
                if ty == pred:
                    right4zero += 1
                    total2zero += 1
                else:
                    total2one += 1
            if ty == 1:
                total4one += 1
                if ty == pred:
                    right4one += 1
                    total2one += 1
                else:
                    total2zero += 1
        return probas, [(right4zero / total2zero, right4zero / total4zero), (right4one / total2one, right4one / total4one)]

    def load_weigths(self, epoch):
        fp = open(self.model_folder + '/%d.model' % epoch, 'r')
        params_values = cPickle.load(fp)
        for p, v in zip(self.params, params_values):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    def back_embedding_weights(self, vocab, backup_file):
        weights = self.l_emb.get_params()[0].get_value()
        print weights.shape
        with open(backup_file, 'w') as fo:
            fo.write(str(vocab.size) + ' ' + str(self.embedding_size) + '\n')
            for i in xrange(0, vocab.size):
                fo.write(vocab.index2word[i] + ' ' + ' '.join(map(str, weights[i])) + '\n')

    def _get_lasagne_sm_output(self, layer, conversation_idx, train):
        i = conversation_idx
        if self.masked:
            return lasagne.layers.get_output(layer, inputs={self.l_in: self.conversations[i], self.l_mask: self.conversation_masks[i]}, deterministic=(not train))
        else:
            return lasagne.layers.get_output(layer, inputs={self.l_in: self.conversations[i]}, deterministic=(not train))
# end ContextModel


class CommonSMContextModel(ContextModel):

    '''
    Abstract class for implement common used sentence model
    '''

    __metaclass__ = ABCMeta

    def build_sentence_model(self, kwargs4sm={}):
        '''
        Build the sentence model based on the embedding layer(self.l_emb)

        Param must contain in kwargs4sm:
        @param name: sentence model name

        Param can use in kwargs4sm:
        @param drop_embedding: dropout rate for embeddings
        @param drop_sm:        dropout rate for sentence model

        @Return: [params], [params contains embedding] and [regularization, default use 0]

        @Warning: the implements should use `l_emb_copy` to replace `l_emb` for drop_embedding
        '''
        assert 'name' in kwargs4sm, 'No sentence model name define by key `name` in kwargs4sm'
        self.sm_dict = {
            'rnn': (self.RNN, self.RNNBase_out),
            'gru': (self.GRU, self.RNNBase_out),
            'gru_attention': (self.GRU, self.Attention_out),
            'lstm': (self.LSTM, self.RNNBase_out),
            'cnn': (self.CNN, self.CNN_out),
            'cnn_attention': (self.CNN, self.Attention_out)
        }
        assert kwargs4sm[
            'name'] in self.sm_dict, 'No sentence model can support the name `%s`' % kwargs4sm['name']
        self.sm_name = kwargs4sm['name']
        self.l_emb_copy = self.l_emb
        if 'drop_embedding' in kwargs4sm:
            drop_rate = float(kwargs4sm['drop_embedding'])
            self.l_emb_copy = lasagne.layers.DropoutLayer(self.l_emb, p=drop_rate)
        params, params_contain_emb, reg = self.sm_dict[kwargs4sm['name'].lower()][0](kwargs4sm)
        return params, params_contain_emb, reg

    def get_sentence_embedding(self, conversation_idx, train=True):
        '''
        Get the sentence embedding based on built model by build_sentence_model method

        @param conversation_idx: get output for `conversation_idx`th sentence
        @param train:            get output for train or test

        @Return: [output]
        '''
        return self.sm_dict[self.sm_name][1](conversation_idx, train=train)

    def RNNBase(self, kwargs4sm, build_func):
        '''
        RNN Based model for sentence model (for RNN/GRU/LSTM)

        Params must contain in kwargs4sm:
        @param n_hidden: number of hidden node for the gate

        Params can use in kwargs4sm:
        @param grad_clip:       clip grad, default is 100
        @param gradient_steps:  max step can generate gradient, default is 20
        @param l2_reg:          use l2 regularization(should set the reg_rate param for the whole model)
        @param l1_reg:          use l1 regularization(should set the reg_rate param for the whole model)
        '''
        assert 'n_hidden' in kwargs4sm, 'must contain `n_hidden` value for GRU in kwargs4sm'
        if self.masked:
            l_mask = self.l_mask
        else:
            l_mask = None
        grad_clip = int(
            kwargs4sm['grad_clip']) if 'grad_clip' in kwargs4sm else 100
        gradient_steps = int(
            kwargs4sm['gradient_steps']) if 'gradient_steps' in kwargs4sm else 20
        # check attention
        drop_before_att = 0
        if 'attention_method' in kwargs4sm:
            self.attention_method = int(kwargs4sm['attention_method'])
            # current support from 1 to 2
            assert self.attention_method > 0 and self.attention_method <= 2
            assert 'pool_size' not in kwargs4sm
            assert 'sm_dense' not in kwargs4sm
            self.step2_attention = kwargs4sm['step2'] if 'step2' in kwargs4sm else False
            if 'drop_before_att' in kwargs4sm:
                drop_before_att = float(kwargs4sm['drop_before_att'])
        else:
            self.attention_method = -1
        l_recurrent = build_func(
            kwargs4sm, self.l_emb_copy, kwargs4sm[
                'n_hidden'], mask_input=l_mask, unroll_scan=False,
            only_return_final=(True if self.attention_method == -1 else False),
            grad_clipping=grad_clip, gradient_steps=gradient_steps)
        # drop out for all h before attention
        if drop_before_att != 0:
            l_recurrent4att = lasagne.layers.DropoutLayer(l_recurrent, p=drop_before_att)
        else:
            l_recurrent4att = l_recurrent
        print l_recurrent.output_shape
        if 'weights' in kwargs4sm:
            params = l_recurrent.get_params()
            for i, (param, value) in enumerate(izip(params, kwargs4sm['weights'])):
                param.set_value(value)
                if 'trainable' in l_recurrent.params[param]:
                    l_recurrent.params[param].remove('trainable')
        reg = 0
        if 'l2_reg' in kwargs4sm and kwargs4sm['l2_reg']:
            reg = lasagne.regularization.regularize_layer_params(l_recurrent, lasagne.regularization.l2)
        if 'l1_reg' in kwargs4sm and kwargs4sm['l1_reg']:
            reg = lasagne.regularization.regularize_layer_params(l_recurrent, lasagne.regularization.l1)
        if 'drop_sm' in kwargs4sm:
            self.drop_rate = float(kwargs4sm['drop_sm'])
            l_sm_drop = lasagne.layers.DropoutLayer(l_recurrent, p=self.drop_rate)
            params = lasagne.layers.get_all_params(l_sm_drop, trainable=True)
            self.l_sm = l_sm_drop
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        else:
            self.drop_rate = 0
            params = lasagne.layers.get_all_params(l_recurrent, trainable=True)
            self.l_sm = l_recurrent
        # build for attention
        if self.attention_method != -1:
            dict_idx2matrix = {}
            for istrain in [True, False]:
                idx2matrix = {}
                for conversation_idx in xrange(self.conversation_len - 2):
                    idx2matrix[conversation_idx] = [self._get_lasagne_sm_output(l_recurrent4att, conversation_idx, istrain)]
                dict_idx2matrix[istrain] = idx2matrix
            return self.Sub_element4attention(params, dict_idx2matrix,
                self.attention_method, self.step2_attention, self.embedding_size, kwargs4sm['n_hidden'],
                'weight_reg' in kwargs4sm, for_rnn=True)
        return params, True, reg

    def RNNBase_out(self, conversation_idx, train=True):
        return self._get_lasagne_sm_output(self.l_sm, conversation_idx, train)

    def RNN(self, kwargs4sm):
        return self.RNNBase(kwargs4sm, self.RNN_build_func)

    def RNN_build_func(self, kwargs4sm, l_emb, n_hidden, mask_input, only_return_final, unroll_scan, grad_clipping, gradient_steps):
        if 'eye' in kwargs4sm:
            assert self.embedding_size == n_hidden
            W_in_to_hid = np.eye(n_hidden).astype(theano.config.floatX)
            W_hid_to_hid = np.eye(n_hidden).astype(theano.config.floatX)
        else:
            W_in_to_hid = lasagne.init.Uniform()
            W_hid_to_hid = lasagne.init.Uniform()
        return lasagne.layers.RecurrentLayer(
            l_emb, n_hidden, mask_input=mask_input, only_return_final=only_return_final, unroll_scan=unroll_scan,
            grad_clipping=grad_clipping, gradient_steps=gradient_steps,
            W_in_to_hid=W_in_to_hid, W_hid_to_hid=W_hid_to_hid)

    def GRU(self, kwargs4sm):
        return self.RNNBase(kwargs4sm, self.GRU_build_func)

    def GRU_build_func(self, kwargs4sm, l_emb, n_hidden, mask_input, only_return_final, unroll_scan, grad_clipping, gradient_steps):
        return lasagne.layers.GRULayer(
            l_emb, n_hidden, mask_input=mask_input, only_return_final=only_return_final, unroll_scan=unroll_scan,
            grad_clipping=grad_clipping, gradient_steps=gradient_steps)

    def LSTM(self, kwargs4sm):
        return self.RNNBase(kwargs4sm, self.LSTM_build_func)

    def LSTM_build_func(self, kwargs4sm, l_emb, n_hidden, mask_input, only_return_final, unroll_scan, grad_clipping, gradient_steps):
        return lasagne.layers.LSTMLayer(
            l_emb, n_hidden, mask_input=mask_input, only_return_final=only_return_final, unroll_scan=unroll_scan,
            grad_clipping=grad_clipping, gradient_steps=gradient_steps)

    def CNN(self, kwargs4sm):
        assert 'filters' in kwargs4sm, 'must contain `filters` value for CNN in kwargs4sm'

        # CNN related config
        filters = kwargs4sm['filters']
        pool_size = kwargs4sm['pool_size'] if 'pool_size' in kwargs4sm else -1
        conv_mode = kwargs4sm['conv_mode'] if 'conv_mode' in kwargs4sm else 'max'
        sm_dense = kwargs4sm['sm_dense'] if 'sm_dense' in kwargs4sm else -1

        # Asserts and config for cnn and attention
        if 'attention_method' in kwargs4sm:
            self.attention_method = int(kwargs4sm['attention_method'])
            # current support from 1 to 2
            assert self.attention_method > 0 and self.attention_method <= 2
            assert 'pool_size' not in kwargs4sm
            assert 'sm_dense' not in kwargs4sm
            self.step2_attention = kwargs4sm['step2'] if 'step2' in kwargs4sm else False
            mode = 'cnn_mc'
            num_filters = 1
            if 'drop_cnn' in kwargs4sm:
                self.cnn_drop = float(kwargs4sm['drop_cnn'])
            else:
                self.cnn_drop = 0
        else:
            self.attention_method = -1
            assert 'mode' in kwargs4sm, 'must contain `mode` value for CNN in kwargs4sm'
            assert 'num_filters' in kwargs4sm, 'must contain `num_filters` value for CNN in kwargs4sm'
            mode = kwargs4sm['mode']
            num_filters = kwargs4sm['num_filters']

        l_cnn = []
        l_pool = []

        # embedding
        if self.masked:
            l_emb_masked = EmbeddingMaskLayer(self.l_emb_copy, self.l_mask, self.embedding_size)
            l_reshape = lasagne.layers.DimshuffleLayer(l_emb_masked, (0, 'x', 1, 2))
        else:
            l_reshape = lasagne.layers.DimshuffleLayer(self.l_emb_copy, (0, 'x', 1, 2))

        # cnn
        if mode == 'cnn_mc':
            for i, filter_size in enumerate(filters):
                # scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=2./3, scale_out=1.7159)
                # l_cnn_temp = lasagne.layers.Conv2DLayer(l_reshape, 1, (filter_size, 1), pad='valid', nonlinearity=scaled_tanh)
                l_cnn_temp = lasagne.layers.Conv2DLayer(l_reshape, num_filters, (filter_size, 1), pad='valid')
                if self.cnn_drop > 0:
                    l_cnn_drop = lasagne.layers.DropoutLayer(l_cnn_temp, p=self.cnn_drop)
                    l_cnn.append(l_cnn_drop)
                else:
                    l_cnn.append(l_cnn_temp)
                pool_size_tmp = pool_size if pool_size != -1 else (self.max_len - filter_size + 1)
                if conv_mode == 'realmean':
                    l_pool_temp = lasagne.layers.Pool2DLayer(l_cnn_temp, (pool_size_tmp, 1), mode='average_exc_pad')
                    l_pool_mean = NLPMeanPool2DLayer(l_pool_temp, self.l_mask, self.max_len, filter_size)
                    l_pool.append(l_pool_mean)
                else:
                    l_pool_temp = lasagne.layers.Pool2DLayer(l_cnn_temp, (pool_size_tmp, 1), mode=conv_mode)
                    l_pool.append(l_pool_temp)

        # concat and flatten cnn outputs
        l_concat = lasagne.layers.ConcatLayer(l_pool, axis=2)
        l_flatten = lasagne.layers.FlattenLayer(l_concat)
        if sm_dense != -1:
            l_flatten = lasagne.layers.DenseLayer(l_flatten, sm_dense, nonlinearity=lasagne.nonlinearities.sigmoid)
        if 'drop_sm' in kwargs4sm:
            self.drop_rate = float(kwargs4sm['drop_sm'])
            l_sm_drop = lasagne.layers.DropoutLayer(l_flatten, p=self.drop_rate)
            self.l_sm = l_sm_drop
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        else:
            self.drop_rate = 0
            self.l_sm = l_flatten

        # get params and check attention result or not
        params = lasagne.layers.get_all_params(self.l_sm, trainable=True)
        if self.attention_method > 0:
            dict_idx2matrix = {}
            for istrain in [True, False]:
                idx2matrix = {}
                for conversation_idx in xrange(self.conversation_len - 2):
                    idx2matrix[conversation_idx] = []
                    for j, cnn in enumerate(l_cnn):
                        cnn_o = self._get_lasagne_sm_output(cnn, conversation_idx, istrain)
                        cnn_o = cnn_o.reshape((cnn_o.shape[0], cnn_o.shape[2], cnn_o.shape[3]))
                        idx2matrix[conversation_idx].append(cnn_o)
                dict_idx2matrix[istrain] = idx2matrix
            return self.Sub_element4attention(params, dict_idx2matrix,
                self.attention_method, self.step2_attention, self.embedding_size, self.l_sm.output_shape[-1], 'weight_reg' in kwargs4sm)
        return params, True, 0

    def CNN_out(self, conversation_idx, train=True):
        return self._get_lasagne_sm_output(self.l_sm, conversation_idx, train)

    def Sub_element4attention(self, params, dict_idx2matrix, attention_method, step2_attention, size4beforeatt, size4behindatt, weight_reg, for_rnn=False):
        '''Build attention function and weights

        params:
            dict_idx2matrix : conversation idx to matrix for attention avg

        Example:
            dict_idx2matrix : {True(means train): {1: [list of cnn output for train], 2: [lstm of rnn hs for train]}, False : {...}}
        '''
        self.attention_method = attention_method
        self.step2_attention = step2_attention
        reg = 0
        rng = np.random.RandomState(1)
        if self.attention_method == 1:
            W_bound = 4 * np.sqrt(6. / (size4beforeatt + size4behindatt))
            M = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(size4beforeatt, size4behindatt)), dtype='float32')
            self.attention_M = theano.shared(M.astype(theano.config.floatX))
            params += [self.attention_M]

            if self.step2_attention:
                M4score = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                    size=(size4beforeatt + size4behindatt + 1, 1)), dtype='float32')
                self.attention_M4score = theano.shared(M4score.astype(theano.config.floatX))
                params += [self.attention_M4score]

            if weight_reg:
                reg += lasagne.regularization.l2(self.attention_M)
                if self.step2_attention:
                    reg += lasagne.regularization.l2(self.attention_M4score)
        elif self.attention_method == 2:
            # using by "Teaching Machines to Read and Comprehend"
            W_bound = 4 * np.sqrt(6. / (size4beforeatt + size4beforeatt / 2))
            M1 = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(size4beforeatt, size4beforeatt / 2)), dtype='float32')
            self.attention_M1 = theano.shared(M1.astype(theano.config.floatX))
            params += [self.attention_M1]

            W_bound = 4 * np.sqrt(6. / (size4behindatt + size4beforeatt / 2))
            M2 = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(size4behindatt, size4beforeatt / 2)), dtype='float32')
            self.attention_M2 = theano.shared(M2.astype(theano.config.floatX))
            params += [self.attention_M2]

            W_bound = 4 * np.sqrt(6. / (size4beforeatt + 1))
            M3 = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(size4beforeatt, 1)), dtype='float32')
            self.attention_M3 = theano.shared(M3.astype(theano.config.floatX))
            params += [self.attention_M3]

            if weight_reg:
                reg += lasagne.regularization.l2(self.attention_M1)
                reg += lasagne.regularization.l2(self.attention_M2)
                reg += lasagne.regularization.l2(self.attention_M3)

        self.attention_probs = []
        self.embeddings_train = self.Sub_element4attention_out_pre(dict_idx2matrix, train=True, for_rnn=for_rnn)
        self.embeddings_test = self.Sub_element4attention_out_pre(dict_idx2matrix, train=False, for_rnn=for_rnn)
        inputs = []
        inputs += self.conversations
        if self.masked:
            inputs += self.conversation_masks
        self.get_attention_func = theano.function(inputs, self.attention_probs, on_unused_input='ignore')
        return params, True, reg

    def Sub_element4attention_out_pre(self, dict_idx2matrix, train=True, for_rnn=False):
        if for_rnn:
            question_embed = self._get_lasagne_sm_output(self.l_sm, self.conversation_len - 2, train).dimshuffle(1, 0, 2)[-1]
        else:
            question_embed = self._get_lasagne_sm_output(self.l_sm, self.conversation_len - 2, train)
        question_reshape = question_embed.dimshuffle(0, 1, 'x')
        embeddings = []
        idx2matrix = dict_idx2matrix[train]
        for i in xrange(self.conversation_len):
            if i == self.conversation_len - 2:
                embeddings.append(question_embed)
            elif i == self.conversation_len - 1:
                tmp_embed = self._get_lasagne_sm_output(self.l_sm, i, train)
                if for_rnn:
                    tmp_embed = tmp_embed.dimshuffle(1, 0, 2)[-1]
                embeddings.append(tmp_embed)
            else:
                attention_prob = []
                outs = []
                for matrix in idx2matrix[i]:
                    o, emb = self.Sub_element4attention_func(matrix, question_embed, question_reshape, train)
                    attention_prob.append(o)
                    outs.append(emb)
                if len(idx2matrix[i]) > 1:
                    e_concat = T.concatenate(outs, axis=1)
                else:
                    e_concat = outs[0]
                embeddings.append(e_concat)
                if not train:
                    self.attention_probs += attention_prob
        return embeddings

    def Sub_element4attention_func(self, matrix, question_embed, question_reshape, train, mask_softmax=True):
        if self.attention_method == 1:
            dp = T.batched_dot(T.dot(matrix, self.attention_M), question_reshape)
            if mask_softmax:
                masked_cnn_o = 1 - T.eq(dp, T.zeros_like(dp)).flatten(ndim=2)
            if self.step2_attention:
                dp = T.nnet.sigmoid(dp)
                q_repeated = question_embed.repeat(matrix.shape[1], axis=0).reshape(
                    (matrix.shape[0], matrix.shape[1], question_embed.shape[1]))
                combined_feature = T.concatenate([dp, matrix, q_repeated], axis=2)
                dp = T.dot(combined_feature, self.attention_M4score)
            dp = T.tanh(dp).flatten(ndim=2)
        elif self.attention_method == 2:
            matrix_sum = matrix.sum(axis=2)
            masked_cnn_o = 1 - T.eq(matrix_sum, T.zeros_like(matrix_sum)).flatten(ndim=2)
            format_1 = T.tanh(T.dot(matrix, self.attention_M1))
            format_2 = T.tanh(T.dot(question_embed.repeat(matrix.shape[1], axis=0).reshape(
                (matrix.shape[0], matrix.shape[1], question_embed.shape[1])), self.attention_M2))
            combined_feature = T.concatenate([format_1, format_2], axis=2)
            dp = T.dot(combined_feature, self.attention_M3).flatten(ndim=2)
        else:
            assert False, 'not support attention method now'
        if mask_softmax:
            masked_o = masked_cnn_o * T.exp(dp - dp.max(axis=1, keepdims=True))
            o = T.switch(masked_cnn_o, masked_o / masked_o.sum(axis=1, keepdims=True), masked_o)
        else:
            o = T.nnet.softmax(dp)
        o = o.reshape((o.shape[0], 1, o.shape[1]))
        emb = T.batched_dot(o, matrix)
        emb = emb.flatten(ndim=2)
        if self.drop_rate != 0 and train:
            return o, emb / (1 - self.drop_rate) * self._srng.binomial(emb.shape, p=1 - self.drop_rate, dtype=theano.config.floatX)
        return o, emb

    def Attention_out(self, conversation_idx, train=True):
        if train:
            return self.embeddings_train[conversation_idx]
        else:
            return self.embeddings_test[conversation_idx]

    def Attention_getatt_prob(self, Xs, X_masks=[]):
        attention_probs = []
        batch = Xs[0].shape[0] / self.batch_size
        for i in xrange(batch):
            features = self._build_features(
                i, Xs, y=None, indices=None, X_masks=X_masks)
            attention_prob = tuple(self.get_attention_func(*features))
            attention_probs.append(attention_prob)
        return attention_probs
# end CommonSMContextModel


class DefaultRelevanceModel(CommonSMContextModel):

    def build_predict(self, kwargs4predict={}):
        assert 'sm_len' in kwargs4predict, 'default predict should define `sm_len` in kwargs4predict'
        self.M = theano.shared(np.eye(int(kwargs4predict['sm_len'])).astype(theano.config.floatX), borrow=True)
        params = [self.M]
        return params, 0

    def get_predict(self, e_conversations, train=True):
        assert len(e_conversations) == 2, 'relevance model just handle the context with only one sentence'
        dp = T.batched_dot(e_conversations[0], T.dot(e_conversations[1], self.M.T))
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0 - 1e-7)
        return o
# end DefaultRelevanceModel


class DefaultMultiTurnModel(CommonSMContextModel):

    def build_predict(self, kwargs4predict={}):
        assert 'sm_len' in kwargs4predict, 'default predict should define `sm_len` in kwargs4predict'
        mlp_hidden_size = kwargs4predict['mlp_hidden'] if 'mlp_hidden' in kwargs4predict else kwargs4predict['sm_len']
        self.l_dense_out_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.conversation_len * kwargs4predict['sm_len']))
        self.l_dense_hidden = lasagne.layers.DenseLayer(self.l_dense_out_in, mlp_hidden_size, nonlinearity=lasagne.nonlinearities.sigmoid)
        self.l_dense_out = lasagne.layers.DenseLayer(self.l_dense_hidden, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
        params = lasagne.layers.get_all_params(self.l_dense_out, trainable=True)
        return params, 0

    def get_predict(self, e_conversations, train=True):
        e_concat = T.concatenate(e_conversations, axis=1)
        o = lasagne.layers.get_output(self.l_dense_out, inputs={self.l_dense_out_in: e_concat}, deterministic=(not train)).flatten()
        o = T.clip(o, 1e-7, 1.0 - 1e-7)
        return o
# end DefaultRelevanceModel


class MultiEyeMultiTurnModel(CommonSMContextModel):

    def build_predict(self, kwargs4predict={}):
        assert 'sm_len' in kwargs4predict, 'default predict should define `sm_len` in kwargs4predict'
        self.M = theano.shared(np.eye(int(kwargs4predict['sm_len'])).astype(theano.config.floatX), borrow=True)
        self.l_dense_out_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.conversation_len - 1))
        self.l_dense_out = lasagne.layers.DenseLayer(self.l_dense_out_in, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
        params = [self.M] + lasagne.layers.get_all_params(self.l_dense_out, trainable=True)
        return params, 0

    def get_predict(self, e_conversations, train=True):
        sims = []
        for i in xrange(self.conversation_len - 1):
            dp = T.batched_dot(e_conversations[i], T.dot(e_conversations[-1], self.M.T))
            o = T.nnet.sigmoid(dp)
            sims.append(o.dimshuffle(0, 'x'))
        e_concat = T.concatenate(sims, axis=1)
        o = lasagne.layers.get_output(self.l_dense_out, inputs={self.l_dense_out_in: e_concat}, deterministic=(not train)).flatten()
        o = T.clip(o, 1e-7, 1.0 - 1e-7)
        return o
# end MultiEyeMultiTurnModel


class MemoryNetworkMultiTrunModel(CommonSMContextModel):

    def build_predict(self, kwargs4predict={}):
        assert 'sm_len' in kwargs4predict, 'default predict should define `sm_len` in kwargs4predict'
        self.M = theano.shared(np.eye(int(kwargs4predict['sm_len'])).astype(theano.config.floatX), borrow=True)
        self.M2 = theano.shared(np.eye(int(kwargs4predict['sm_len'])).astype(theano.config.floatX), borrow=True)
        self.sentence_len = int(kwargs4predict['sm_len'])
        params = [self.M, self.M2]
        reg = 0
        if 'l2_reg' in kwargs4predict and kwargs4predict['l2_reg']:
            reg = lasagne.regularization.l2(self.M) + lasagne.regularization.l2(self.M2)
        return params, reg

    def get_predict(self, e_conversations, train=True):
        sims = []
        for i in xrange(self.conversation_len - 1):
            dp = T.batched_dot(e_conversations[i], T.dot(e_conversations[-1], self.M.T))
            o = T.nnet.sigmoid(dp)
            sims.append(o.dimshuffle(0, 'x'))
        e_sims = T.concatenate(sims, axis=1).dimshuffle(0, 'x', 1)
        e_concate = T.concatenate(e_conversations[:-1], axis=1).reshape((self.batch_size, self.conversation_len - 1, self.sentence_len))
        e_context = T.batched_dot(e_sims, e_concate).reshape((self.batch_size, self.sentence_len))
        o = T.batched_dot(e_context, T.dot(e_conversations[-1], self.M2.T))
        o = T.clip(o, 1e-7, 1.0 - 1e-7)
        return o
# end MemoryNetworkMultiTrunModel


class RNNMutilTurnModel(CommonSMContextModel):

    def build_predict(self, kwargs4predict={}):
        assert 'sm_len' in kwargs4predict, 'default predict should define `sm_len` in kwargs4predict'

        self.sm_len = kwargs4predict['sm_len']
        rnn_num_unit = kwargs4predict['rnn_num_unit']
        dense_num_unit = kwargs4predict['dense_num_unit']
        reg = 0

        self.l_rnn_pred_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.conversation_len, self.sm_len))
        if 'type' not in kwargs4predict or kwargs4predict['type'] == 'lstm':
            l_rnn = lasagne.layers.LSTMLayer(self.l_rnn_pred_in, rnn_num_unit, only_return_final=True)
        elif kwargs4predict['type'] == 'gru':
            l_rnn = lasagne.layers.GRULayer(self.l_rnn_pred_in, rnn_num_unit, only_return_final=True)
        elif kwargs4predict['type'] == 'rnn':
            if 'eye' in kwargs4predict:
                assert self.sm_len == rnn_num_unit
                W_in_to_hid = np.eye(self.sm_len).astype(theano.config.floatX)
                W_hid_to_hid = np.eye(self.sm_len).astype(theano.config.floatX)
            else:
                W_in_to_hid = lasagne.init.Uniform()
                W_hid_to_hid = lasagne.init.Uniform()
            l_rnn = lasagne.layers.RecurrentLayer(self.l_rnn_pred_in, rnn_num_unit, only_return_final=True,
                W_in_to_hid=W_in_to_hid, W_hid_to_hid=W_hid_to_hid)
        else:
            raise 'not support RNN type', kwargs4predict['type']
        self.l_dense_out = lasagne.layers.DenseLayer(l_rnn, dense_num_unit, nonlinearity=lasagne.nonlinearities.sigmoid)

        if 'l2_reg' in kwargs4predict and kwargs4predict['l2_reg']:
            reg = lasagne.regularization.regularize_layer_params(self.l_lstm, lasagne.regularization.l2)
        if 'l1_reg' in kwargs4predict and kwargs4predict['l1_reg']:
            reg = lasagne.regularization.regularize_layer_params(self.l_lstm, lasagne.regularization.l1)

        params = lasagne.layers.get_all_params(self.l_dense_out, trainable=True)
        return params, reg

    def get_predict(self, e_conversations, train=True):
        e_concat = T.concatenate(e_conversations, axis=1)
        e_reshape = T.reshape(e_concat, (self.batch_size, self.conversation_len, self.sm_len))
        o = lasagne.layers.get_output(self.l_dense_out, inputs={self.l_rnn_pred_in: e_reshape}).flatten()
        o = T.clip(o, 1e-7, 1.0 - 1e-7)
        return o
# end RNNMutilTurnModel
