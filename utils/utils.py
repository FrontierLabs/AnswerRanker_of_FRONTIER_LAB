# -*- coding: utf-8 -*-

from data_loader_utils import Vocab, pad_into_matrix_test

import numpy as np

"""
Load all text data into memory, mostly for validation set and test set
"""


class Lines(object):

    """
    Simple format:
        one features = one line;
        words already preprocessed and separated by whitespace;
        y in the fist column split by \t
    """

    def __init__(self, source, vocab, ytype='bool', contain_y=True, max_len=100):
        """
        'source' can be either a string or a file object.
        'vocab' Vocab object
        'ytype' in bool/int/float

        Example::
            sentences = Lines('myfile.txt', vocab)
        """
        self.source = source
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.contain_y = contain_y
        self.max_len = max_len
        assert ytype in ['bool', 'int', 'float']
        if ytype == 'bool' or ytype == 'int':
            self.y_format = lambda y: int(y)
            self.y_nptype = 'int32' if ytype == 'int' else 'bool'
        elif ytype == 'float':
            self.y_format = lambda y: float(y)
            self.y_nptype = 'float32'

    def format(self, lines, ys):
        assert len(lines) == len(ys)
        batch_X = np.array(pad_into_matrix_test(
            map(self.vocab, lines), self.vocab_size, maxlen=self.max_len), dtype=np.int32)
        batch_Y = np.array(map(self.y_format, ys), dtype=self.y_nptype)
        return batch_X, batch_Y

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
        for line in iters:
            if self.contain_y:
                try:
                    y, l = line.strip().split('\t', 1)
                except:
                    print l
                y, line = line.strip().split('\t', 1)
                ys.append(y)
            lines.append(line)
        if need_close:
            iters.close()
        return self.format(lines, ys)
# end Lines


class TestLines(Lines):

    def format(self, lines, ys):
        batch_X = np.array(pad_into_matrix_test(
            map(self.vocab, lines), self.vocab_size, maxlen=self.max_len), dtype=np.int32)
        return batch_X
# end TestLines


def build_vocab(base_path, max_len_max=100):
    vocab = Vocab()
    max_len = 0
    with open(base_path + 'train.txt', 'r') as fp:
        for line in fp:
            words = line.strip().split('\t')[1].split()
            vocab.add_words(words)
            max_len = max(max_len, len(words))
    max_len = min(max_len_max, max_len)
    return vocab, max_len
