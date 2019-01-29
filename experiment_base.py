# -*- coding: utf-8 -*-

# TODO:
# comments

class ExpBase(object):

    def __init__(self, model, model_log, data_loader, group_size):
        self.model = model
        self.model_log = model_log
        self.data_loader = data_loader
        self.group_size = group_size

    def train(self, epoch, shuffle=True):
        print 'Loading training data'
        Xs, X_masks, y = self.data_loader.load_train()
        print 'Loading validation data'
        Xs_val, X_masks_val, y_val = self.data_loader.load_valid()
        print 'Loading testing data'
        Xs_test, X_masks_test, y_test = self.data_loader.load_test()
        print 'Start train:'
        print self.model_log
        self.model.train(epoch, Xs, y, Xs_val, y_val, Xs_test, y_test, X_masks=X_masks, X_masks_val=X_masks_val, X_masks_test=X_masks_test, shuffle=shuffle)

    def continue_train(self, epoch, last_epoch, shuffle=True):
        print 'Loading training data'
        Xs, X_masks, y = self.data_loader.load_train()
        print 'Loading validation data'
        Xs_val, X_masks_val, y_val = self.data_loader.load_valid()
        print 'Loading testing data'
        Xs_test, X_masks_test, y_test = self.data_loader.load_test()
        self._load_weights(last_epoch)
        print 'Start train:'
        print self.model_log
        self.model.train(epoch, Xs, y, Xs_val, y_val, Xs_test, y_test, X_masks=X_masks, X_masks_val=X_masks_val, X_masks_test=X_masks_test, shuffle=shuffle, last_epoch=last_epoch)

    def _load_test_data(self, testname, name_non_path):
        print 'Loading testing data'
        Xs_test, X_masks_test, y_test = self.data_loader.load_test(testname=testname, name_non_path=name_non_path)
        return Xs_test, X_masks_test, y_test

    def _load_weights(self, epoch2model):
        print 'Load weights from model of epoch', epoch2model
        self.model.load_weigths(epoch2model)

    def test(self, epoch2model, testname=None, name_non_path=True):
        Xs_test, X_masks_test, y_test = self._load_test_data(testname, name_non_path)
        self._load_weights(epoch2model)
        print 'Start test'
        cost, acc = self.model.test(Xs_test, y_test, X_masks=X_masks_test)
        print cost
        print acc

    def test_p_at_k(self, epoch2model, balance_test, testname=None, name_non_path=True, k_list=[1, 2]):
        Xs_test, X_masks_test, y_test = self._load_test_data(testname, name_non_path)
        self._load_weights(epoch2model)
        print 'Start predict'
        preds, probas = self.model.predict(Xs_test, X_masks=X_masks_test)
        print 'Start evaluate'
        for k in k_list:
            print 'P@%d: %f' % (k, self.model.p_at_k(probas, self.group_size, k, balance=balance_test))
        print '1in2P@1: %f' % (self.model.p_1in2_at_1(probas, self.group_size, balance=balance_test))

    def test_pr(self, epoch2model, testname=None, name_non_path=True, threshold=0.5):
        Xs_test, X_masks_test, y_test = self._load_test_data(testname, name_non_path)
        self._load_weights(epoch2model)
        print 'Start predict with threshold', threshold
        probas, res = self.model.precision_recall(Xs_test, y_test, X_masks=X_masks_test, threshold=threshold)
        print res

    def predict(self, epoch2model, backfile, testname=None, name_non_path=True):
        Xs_test, X_masks_test, y_test = self._load_test_data(testname, name_non_path)
        self._load_weights(epoch2model)
        print 'Start predict'
        preds, probas = self.model.predict(Xs_test, X_masks=X_masks_test)
        print 'Start back up'
        with open(backfile, 'w') as fo:
            for i in xrange(probas.shape[0]):
                fo.write(str(probas[i][1]) + '\n')

    def back_embedding(self, epoch2model, vocab, backfile):
        self._load_weights(epoch2model)
        self.model.back_embedding_weights(vocab, backfile)
# end ExpBase


class DataLoaderBase(object):

    def __init__(self, line_obj, base_train_file, support_head=False, name_list=['train', 'dev', 'test'], heads=[-1, -1, -1]):
        self.line_obj = line_obj
        self.support_head = support_head
        self.base_train_file = base_train_file
        assert len(name_list) == 3 and 'train' == name_list[0]
        assert len(heads) == 3
        self.train_f = base_train_file
        self.valid_f = base_train_file.replace(name_list[0], name_list[1])
        self.test_f = base_train_file.replace(name_list[0], name_list[2])
        self.train_head = heads[0]
        self.valid_head = heads[1]
        self.test_head = heads[2]

    def _load(self, fname, head=-1):
        assert head == -1 or self.support_head
        self.line_obj.source = fname
        if head != -1:
            self.line_obj.head = head
        return self.line_obj.load()

    def load_train(self):
        return self._load(self.train_f, self.train_head)

    def load_valid(self):
        return self._load(self.valid_f, self.valid_head)

    def load_test(self, testname=None, name_non_path=True):
        fname = self.test_f
        head = self.test_head
        if testname is not None:
            head = -1
            if name_non_path:
                fname = self.base_train_file.replace('train', testname)
            else:
                fname = testname
        return self._load(fname, head)
# end DataLoaderBase
