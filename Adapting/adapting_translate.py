import numpy
import os
import theano
import cPickle as pkl
import ctypes
import re
import argparse

from post_nmt import build_pmodel, post_train


# tian jia xiangsidu daima
# zhi geng xin decoder duan canshu
# geng xing guo de ju zi jin xin dao cun
# geng ju xiang si du que ding yon na zu can shu


def main(adapting_params):
    ret_params = build_pmodel(saveto=adapting_params['model'][0],
                              dim_word=adapting_params['dim_word'][0],
                              n_words=adapting_params['n-words'][0],
                              n_words_src=adapting_params['n-words'][0],
                              dim=adapting_params['dim'][0],
                              decay_c=adapting_params['decay-c'][0],
                              clip_c=adapting_params['clip-c'][0],
                              lrate=adapting_params['learning-rate'][0],
                              optimizer=adapting_params['optimizer'][0],
                              patience=10,
                              maxlen=50,
                              dictionaries=adapting_params['dictionaries'],
                              use_dropout=adapting_params['use-dropout'][0],
                              max_epochs=1)

    post_train(adapting_params, ret_params)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('adapting_src', type=str)
    parser.add_argument('adapting_trg', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    adapting_params = {'model': [args.model],
                       'dim_word': [512],
                       'dim': [1024],
                       'n-words': [30000],
                       'optimizer': ['adam'],
                       'decay-c': [0.],
                       'clip-c': [1.],
                       'use-dropout': [False],
                       'learning-rate': [0.00001],
                       'dictionaries': [args.dictionary, args.dictionary_target],
                       'beam_size': [5],
                       'adapting_datasets': [args.adapting_src, args.adapting_trg],
                       'test_sets':[args.test],
                       'save_path': [args.saveto],
                       }
    main(adapting_params)
