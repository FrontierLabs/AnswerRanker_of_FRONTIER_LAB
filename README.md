# AnswerRanker

A Architecture for design answer ranker by seperate the sentence modeling and sequence modeling.

Contained sentence model include:  
Simple RNN  
GRU  
LSTM  
GRU with attention  
CNN  
CNN with attention  

Contained sequence model include:  
Relevance(two sentence) - S1<sup>T</sup> ⋅ M ⋅ S2  
Multi Relevance         - MLP ahead Relevance between answer and each context
MLP beyond concanted  
One layer Memory Network with context as Memory  
Simple RNN  
GRU  
LSTM


## Architecture

```
|── utils  
|  |── data_loader_utils.py       # vocab and padding function.
|  |── keras_generic_utils.py     # copy the generic_utils from keras/utils.
|  |── keras_sequence.py          # copy the sequence.py from keras/utils.
|  |── utils.py                   # based class for generate batched data as model input from formated text files.
├── config_local.py               # config for model folder.
├── context_lasagne.py            # models.
├── experiment_base.py            # base class for experiment include train/continue_train/test/test_p@k/test_pr/predict/backup embedding
├── experiment_base_douban.py     # data loader for douban corpus
├── experiment_base_ubuntu.py 	  # data loader for Ubuntu corpus
├── experiment_douban.py          # example to experiment for douban corpus.
├── experiment_ubuntu.py          # example to experiment for Ubuntu corpus.
```

## How to run

Run experiment_xxx.py directly.

## Data format

A sample contained in one line, for each line, the format is:  
label[0/1] \t sentence1 \t sentence2 \t ...

## Dependency
- [NumPy](http://www.numpy.org) : normal computing package
- [Theano](http://deeplearning.net/software/theano/) : Based graph computing package
- [Lasagne](http://lasagne.readthedocs.io/en/latest/) : Based DL package

## Reference
- [Ranking Responses Oriented to Conversational Relevance in Chat-bots](https://www.aclweb.org/anthology/C/C16/C16-1063.pdf)
- [End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf)
- [The Ubuntu Dialogue Corpus - A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://www.sigdial.org/workshops/conference16/proceedings/pdf/SIGDIAL40.pdf)
