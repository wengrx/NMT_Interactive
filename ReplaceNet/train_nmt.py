import argparse
import numpy
import os

from nmt import train


def main(job_id, params):

    # print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][1],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=200,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=10000,
                     sampleFreq=100,
                     datasets=[params['train'][0],
                               params['train'][1]],
                     valid_datasets=[params['dev'][0],
                                     params['dev'][1]],
                     dictionaries=[params['dictionaries'][0],
                                   params['dictionaries'][1]],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     valid_save=params['model'][0] + '.valid',
                     pretrain=params['model'][1])
    return validerr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_train', type=str)
    parser.add_argument('trg_train', type=str)
    parser.add_argument('dictionary_src', type=str)
    parser.add_argument('dictionary_trg', type=str)
    parser.add_argument('src_dev', type=str)
    parser.add_argument('trg_dev', type=str)
    parser.add_argument('pretrain', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    params = {'train': [args.src_train, args.trg_train],
              'dev': [args.src_dev, args.trg_dev],
              'dictionaries': [args.dictionary_src, args.dictionary_trg],
              'model': [args.saveto, args.pretrain],
              'dim_word': [512],
              'dim': [1024],
              'n-words': [30000, 30000],
              'optimizer': ['adadelta'],
              'decay-c': [0.],
              'clip-c': [1.],
              'use-dropout': [False],
              'learning-rate': [0.0001],
              'reload': [True],
              }

    main(0, params)
