# -*- coding:utf-8 -*-

import numpy as np
import keras_sequence


class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]

    def __init__(self, index2word=None):
        self.word2index = {}
        self.index2word = []
        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0

        if index2word is not None:
            self.add_words(index2word)

    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    def __call__(self, line):
        """
        Convert from numerical representation to words and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
        else:
            line = line.split(" ")
        indices = [0] * len(line)
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
        return indices

    @property
    def size(self):
        return len(self.index2word)

    def __len__(self):
        return len(self.index2word)

    def save_vocab(self, vocab_file):
        fvocab = open(vocab_file, 'w')
        for windx, word in enumerate(self.index2word):
            fvocab.write('%d\t%s\n' % (windx, word))
            fvocab.flush()
        fvocab.close()

    def load_vocab(self, vocab_file):
        self.word2index = {}
        self.index2word = []
        self.unknown = 0

        fin = open(vocab_file)
        for line in fin:
            items = line.strip().split('\t')
            self.word2index[items[1]] = int(items[0])
        fin.close()
        word_num = len(self.word2index)
        self.index2word = [''] * word_num
        for word, idx in self.word2index.items():
            self.index2word[idx] = word


def pad_into_matrix_pre_text(rows, vocab_size):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    # height = len(rows)
    np.random.shuffle(rows)
    X = []
    Y = np.zeros((len(rows), vocab_size), dtype=np.bool)
    for i, row in enumerate(rows):
        X.append(row[:-1])
        Y[i, row[-1]] = 1
    X = keras_sequence.pad_sequences(X, maxlen=width)
    return X, Y


def pad_into_matrix(rows, vocab_size, maxlen=100):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    # height = len(rows)
    np.random.shuffle(rows)
    X = []
    Y = np.zeros((len(rows), vocab_size), dtype=np.bool)
    for i, row in enumerate(rows):
        X.append(row[:-1])
        Y[i, row[-1]] = 1
    X = keras_sequence.pad_sequences(X, maxlen=maxlen)
    return X, Y


def pad_into_matrix_test(rows, vocab_size, maxlen=100):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    X = []
    for i, row in enumerate(rows):
        if len(row) > maxlen:
            row = row[:maxlen]
        X.append(row)
    X = keras_sequence.pad_sequences(X, maxlen=maxlen)
    return X


def pad_into_matrix_test_line(rows, vocab_size, maxlen=100):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    X = []
    for i, row in enumerate(rows):
        if len(row) > width:
            row = row[:width]
        X.append(row)
    X = keras_sequence.pad_sequences(X, maxlen=width)
    return X


def word2utf8(word):
    word = word.encode('utf-8')
    if word.isdigit() or word.isalpha():
        return None
    return word


def text2word(srcfile, outfile):
    #
    fin = open(srcfile)
    fout = open(outfile, 'w')
    for line in fin:
        try:
            words = unicode(line.strip(), encoding='utf-8')
            fout.write("%s ã€‚\n" % (' '.join(map(word2utf8, words))))
        except:
            pass
    fin.close()
    fout.close()

"""
For heavy task, the training dataset is very big. So loading all to memory challenges the device pro.

"""


class LinesSlice(object):

    """
    Simple format:
        one features = one line;
        words already preprocessed and separated by whitespace;
        y in the fist column split by \t
    """

    def __init__(self, source, vocab, line_count=100, ytype='bool', contain_y=True):
        """
        'source' can be either a string or a file object.
        'line_count' return lines each time
        'vocab' Vocab object
        'ytype' in bool/int/float

        Example::
            sentences = LinesSlice('myfile.txt', vocab)
        """
        self.source = source
        self.line_count = line_count
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.contain_y = contain_y
        assert ytype in ['bool', 'int', 'float']
        if ytype == 'bool' or ytype == 'int':
            self.y_format = lambda y: int(y)
            self.y_nptype = 'int32' if ytype == 'int' else 'bool'
        elif ytype == 'float':
            self.y_format = lambda y: float(y)
            self.y_nptype = 'float32'
        self.split = lambda lines: map(float, lines.split())

    def _yield_batch(self, lines, ys):
        assert len(lines) == len(ys)
        batch_X = np.array(map(self.split, lines), dtype='float32')
        batch_Y = np.array(map(self.y_format, ys), dtype=self.y_nptype)
        return batch_X, batch_Y

    def __iter__(self):
        """Iterate through the lines in the source."""
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

        num_flag = 0
        lines, ys = [], []
        for line in iters:
            num_flag += 1
            if num_flag == self.line_count:
                yield self._yield_batch(lines, ys)
                num_flag = 0
                lines = []
                ys = []
            if self.contain_y:
                y, line = line.strip().split('\t', 1)
                ys.append(y)
            lines.append(line)
        yield self._yield_batch(lines, ys)
        if need_close:
            iters.close()
# end LinesSlice


class LinesSliceTest(LinesSlice):

    def __init__(self, source, vocab, line_count=100, ytype='bool'):
        super(LinesSliceTest, self).__init__(
            source, vocab, line_count=line_count, ytype=ytype, contain_y=False)

    def _yield_batch(self, lines, ys):
        batch_X = np.array(map(float, lines), dtype='float32')
        return batch_X
# end LinesSliceTest


class LinesSlice4Sentence(LinesSlice):

    def format(self, lines):
        return map(self.vocab, lines)

    def _yield_batch(self, lines, ys):
        assert len(lines) == len(ys)
        batch_X = pad_into_matrix_test(self.format(lines), self.vocab_size)
        batch_Y = np.array(map(self.y_format, ys), dtype=self.y_nptype)
        return batch_X, batch_Y
# end LinesSlice4Sentence


class LinesSliceTest4Sentence(LinesSlice4Sentence):

    def __init__(self, source, vocab, line_count=100, ytype='bool'):
        super(LinesSliceTest4Sentence, self).__init__(
            source, vocab, line_count=line_count, ytype=ytype, contain_y=False)

    def _yield_batch(self, lines, ys):
        batch_X = pad_into_matrix_test(self.format(lines), self.vocab_size)
        return batch_X
# end LinesSliceTest4Sentence


class LinesSlice4LM(LinesSlice4Sentence):

    def __init__(self, source, vocab, line_count=100, ytype='bool'):
        super(LinesSlice4LM, self).__init__(
            source, vocab, line_count=line_count, ytype=ytype, contain_y=False)

    def _yield_batch(self, lines, ys):
        batch_X, batch_Y = pad_into_matrix(self.format(lines), self.vocab_size)
        return batch_X, batch_Y
# end LinesSlice4LM


class LinesSliceTest4LM(LinesSlice4LM):

    def _yield_batch(self, lines, ys):
        batch_X = pad_into_matrix_test(self.format(lines), self.vocab_size)
        return batch_X
# end LinesSliceTest4LM


def sliceIter(slices):
    for batch_X, batch_Y in slices:
        yield batch_X, batch_Y


def sliceIterTest(slices):
    for batch_X in slices:
        yield batch_X
