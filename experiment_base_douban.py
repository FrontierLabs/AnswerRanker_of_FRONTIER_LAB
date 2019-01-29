# -*- coding: utf-8 -*-


import numpy as np

from utils.data_loader_utils import Vocab, pad_into_matrix_test

from config_local import machine

from experiment_base import DataLoaderBase

from utils.utils import Lines


if machine == 'GPU':
    # data_folder = '/mnt/sdb/share/context.large/10mhead/train.txt'
    data_folder = '/mnt/sdb/share/context/train.small.txt'
elif machine == 'shareGPU':
    data_folder = '/home/bowen/data/context_douban/train.small.txt'
elif machine == 'vm302':
    data_folder = '/home/bowenwu/dev/data/context/train.small.txt'
elif machine == 'other':
    data_folder = '/home/bowen/dev/data/context/train.small.txt'
else:
    raise Exception('not support for machine ' + machine + ' now')


def is_chinese(uchar):
    try:
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False
    except:
        return False


def is_eng(uchar):
    if uchar > u'a' and uchar < u'z':
        return True
    if uchar > u'A' and uchar < u'Z':
        return True
    return False


def split_sentence2char(sentence, tostr=False):
    res = []
    for w in sentence.strip().decode('utf-8'):
        if w.strip() != '' and (is_chinese(w) or is_eng(w)):
            res += [w.encode('utf-8')]
    return res if (not tostr) else ' '.join(res)


def build_vocab(filename, max_len_max=100, char=True):
    vocab = Vocab()
    max_len = 0
    with open(filename, 'r') as fp:
        for line in fp:
            for sentence in line.strip().split('\t')[1:]:
                if char:
                    words = split_sentence2char(sentence)
                else:
                    words = sentence.split()
                vocab.add_words(words)
                max_len = max(max_len, len(words))
    max_len = min(max_len_max, max_len)
    return vocab, max_len


class ContextLines(Lines):

    def __init__(self, source, vocab, context_num, ytype='int32', contain_y=True, max_len=100, char=True):
        super(ContextLines, self).__init__(
            source, vocab, ytype=ytype, contain_y=contain_y, max_len=max_len)
        self.char = char
        self.context_num = context_num

    def format_x(self, lines):
        sentence_lines = [[] for i in xrange(self.context_num + 1)]
        for line in lines:
            sentences = line.split('\t')[-self.context_num - 1:]
            format_func = split_sentence2char if self.char else (
                lambda x: x.split())
            for i, sentence in enumerate(sentences):
                sentence_lines[i].append(format_func(sentence))
        batches = []
        for lines in sentence_lines:
            batches.append(np.array(pad_into_matrix_test(
                map(self.vocab, lines), self.vocab_size, maxlen=self.max_len), dtype=np.int32))
        return batches

    def format(self, lines, ys):
        assert len(lines) == len(ys)
        batches = self.format_x(lines)
        batches_mask = []
        for batch in batches:
            batches_mask.append(np.array(batch != 0, dtype=np.float32))
        batch_Y = np.array(map(self.y_format, ys), dtype=self.y_nptype)
        return batches, batches_mask, batch_Y
# end ContextLines


class DoubanDataLoader(DataLoaderBase):

    def __init__(self, context_num, max_len_max=100, char=True, vocab_path=None, save_path=None):
        print 'Building vocab'
        if vocab_path is None:
            vocab, max_len = build_vocab(
                data_folder, max_len_max=max_len_max, char=char)
            if save_path is not None:
                print 'saving vocab to', save_path
                vocab.save_vocab(save_path)
        else:
            vocab = Vocab()
            vocab.load_vocab(vocab_path)
            max_len = max_len_max
        print '\t vocab size %d, max len %d' % (vocab.size, max_len)
        self.vocab_size = vocab.size
        self.max_len = max_len
        l = ContextLines(
            data_folder, vocab, context_num, max_len=max_len, ytype='int')
        super(DoubanDataLoader, self).__init__(l, data_folder, support_head=False, name_list=[
            'train', 'dev', 'test'], heads=[-1, -1, -1])
# end DoubanDataLoader
