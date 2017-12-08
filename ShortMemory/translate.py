'''
Translates a source file using a translation model.
'''
import argparse
import theano
import numpy
import os
import cPickle as pkl
import re
import ctypes

theano.config.floatX = 'float32'
from nmt import (build_sampler, gen_sample, load_params,
                         init_params, init_tparams)


def translate_model(tp, tparams, options):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    srcindex = word_index(tp['src_path'][0], tp['dictionaries'][0], options)
    hisindex = word_index(tp['his_path'][0], tp['dictionaries'][2], options)

    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate():
        trans_seqs = []
        # f_att = open(target_file + '.att', 'w')
        for i, seq in enumerate(srcindex):
            if i == 0:
                continue
            print 'the number of sequence: %s' % i

            # modify
            sample, score = gen_sample(tparams, f_init, f_next,
                                       numpy.array(seq).reshape([len(seq), 1]),numpy.array(hisindex[i-1]).reshape([len(hisindex[i-1]), 1]),
                                       options, trng=trng, k=tp['beam_size'][0], maxlen=200,
                                       stochastic=False, argmax=False)

            if sample == [[]]:
                sample = [1, 1, 1, 1, 1, 0]
                trans_seqs.append(' '.join(index_word(sample, tp['dictionaries'][3])))
            else:
                score = numpy.array(score)
                lengths = numpy.array([len(s) for s in sample])
                score = score / lengths
                sidx = numpy.argmin(score)
                sample_new = sample[sidx]
                if sample[sidx] == [0]:
                    sample_new = [1, 1, 1, 1, 1, 0]
                # else:
                #     save_att(numpy.array(attention[sidx]), seq, sample_new, f_att, word_idict, word_idict_trg)
                print ' '.join(index_word(sample_new, tp['dictionaries'][3]))
                trans_seqs.append(' '.join(index_word(sample_new, tp['dictionaries'][3])))
        with open(tp['save_path'][0] + '.lctok', 'w') as fw:
            fw.writelines('\n'.join(trans_seqs))
            # f_att.close()

    _translate()
    return


def word_index(filepaths, word_dict, options, chr_level=False):
    w2i = []
    with open(filepaths, 'r') as f:
        for idx, line in enumerate(f):
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
            x += [0]
            w2i.append(x)
    return w2i


def save_att(alpha, src, tgt, fatt, id2word_src=None, id2word_trg=None):
    src = [id2word_src[w] for w in src]
    tgt = [id2word_trg[w] for w in tgt]
    line = ['0', '|||'] \
           + tgt + ['|||', '0', '|||'] + src \
           + ['|||', '%d %d\n' % (len(src), len(tgt))]
    fatt.write(' '.join(line))
    numpy.savetxt(fatt, alpha, fmt='%s ' * (alpha.size / len(alpha)))
    fatt.write('\n')


def index_word(caps, word_idict_trg):
    capsw = []
    seqlen = len(caps)
    if caps[-1] == 0:
        seqlen -= 1
    for w in xrange(seqlen):
        capsw.append(word_idict_trg[caps[w]])
    return capsw


def main(tp):
    # load source dictionary and invert
    print 'load data......',
    with open('%s.pkl' % tp['model'][0], 'rb') as f:
        options = pkl.load(f)
    params = load_params(tp['model'][0])
    tparams = init_tparams(params)
    print 'done'

    with open(tp['dict_path'][0], 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(tp['dict_path'][1], 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    tp['dictionaries'] = [word_dict, word_idict, word_dict_trg, word_idict_trg]

    print 'start translate'
    translate_model(tp, tparams, options)
    print 'done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('history', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    tp = {'model': [args.model],
          'beam_size': [5],
          'normalize': [True],
          'dict_path': [args.dictionary, args.dictionary_target],
          'src_path': [args.source],
          'his_path': [args.history],
          'save_path': [args.saveto]}

    print tp

    main(tp)
