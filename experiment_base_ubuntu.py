# -*- coding: utf-8 -*-

import numpy as np

from utils.data_loader_utils import pad_into_matrix_test

from config_local import machine

from experiment_base import DataLoaderBase


if machine == 'GPU':
    data_folder = '/mnt/sdb/xuehui/workspace/ubuntu_original/data/train.change.data'
    balance_data_folder = '/mnt/sdb/xuehui/workspace/ubuntu/data/train.triple.data'
elif machine == 'shareGPU':
    data_folder = '/home/bowen/data/context_ubuntu/train.change.data'
    balance_data_folder = '/home/bowen/data/context_ubuntu/train.triple.data'
elif machine == 'vm302':
    data_folder = '/home/bowenwu/dev/data/ubuntu/train.change.data'
    balance_data_folder = data_folder.replace('change', 'triple')
elif machine == 'other':
    data_folder = None
    balance_data_folder = None
else:
    raise Exception('not support for machine ' + machine + ' now')


class Vocab:
    def load_vocab(self, embedding_file_path, dimension):
        self.weight = []
        print embedding_file_path
        file = open(embedding_file_path, 'r')
        for i, line in enumerate(file):
            temp_list = line.strip().split()
            # index = temp_list[0]
            value = temp_list[1:]
            list = [float(x) for x in value]
            self.weight.append(np.asarray(list, dtype="float32"))
        self.weight = np.array(self.weight, dtype='float32')

        file.close()
        return

    def get_weight(self):
        return [self.weight]
# end Vocab


class UbuntuLines(object):

    def __init__(self, source, context_num, max_len=100, head=None):
        self.max_len = max_len
        self.context_num = context_num
        self.source = source
        self.head = head

    def format_x(self, lines):
        sentence_lines = [[] for i in xrange(self.context_num + 1)]
        for line in lines:
            real_line = line.split(' ', 2)[2]
            sentences = real_line.split(' 77475')[-self.context_num - 2:-1]
            emptys = self.context_num + 1 - len(sentences)
            for i in xrange(self.context_num + 1):
                if i < emptys:
                    sentence_lines[i].append([0])
                else:
                    sentence_lines[i].append(
                        map(int, sentences[i - emptys].strip().split(' ')))
        batches = []
        for lines in sentence_lines:
            batches.append(np.array(pad_into_matrix_test(
                lines, None, maxlen=self.max_len), dtype=np.int32))
        return batches

    def format(self, lines, ys):
        assert len(lines) == len(ys)
        batches = self.format_x(lines)
        batches_mask = []
        for batch in batches:
            batches_mask.append(np.array(batch != 0, dtype=np.float32))
        batch_Y = np.array(map(int, ys), dtype=np.int32)
        return batches, batches_mask, batch_Y

    def load(self):
        need_close = False
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            iters = self.source
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            iters = open(self.source, 'r')
            need_close = True

        lines, ys = [], []
        for i, line in enumerate(iters):
            y, line = line.strip().split(' ', 1)
            ys.append(y)
            lines.append(line)
            if self.head is not None and i + 1 >= self.head:
                break
        if need_close:
            iters.close()
        return self.format(lines, ys)
# end UbuntuLines


class UbuntuDataLoader(DataLoaderBase):

    def __init__(self, context_num, percent, batch_size, balance=False, max_len_max=40):
        self.vocab_size = 769155
        self.max_len = max_len_max
        train_f = balance_data_folder if balance else data_folder
        train_len, dev_len, test_len = 9192244, 480490, 476690
        if balance:
            train_len = 13788366
        l = UbuntuLines(train_f, max_len=self.max_len, context_num=context_num)
        super(UbuntuDataLoader, self).__init__(l, train_f, support_head=True, name_list=['train', 'valid', 'test'],
                                               heads=[int(train_len * percent) / batch_size * batch_size,
                                                      int(dev_len * percent) /
                                                      batch_size * batch_size,
                                                      int(test_len * percent) / batch_size * batch_size])
# end UbuntuDataLoader
