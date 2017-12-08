
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
    refindex = word_index(tp['trg_path'][0]+'/ref0',tp['dictionaries'][2], options)

    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate():
        trans_seqs = []
        # f_att = open(target_file + '.att', 'w')
        for i, seq in enumerate(srcindex):
            print 'the number of sequence: %s' % i

            # modify
            sample, score, _ = gen_sample(tparams, f_init, f_next,
                                       numpy.array(seq).reshape([len(seq), 1]),
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
        with open(tp['save_path'][0] + '_0.lctok', 'w') as fw:
            fw.writelines('\n'.join(trans_seqs))
            # f_att.close()


    def preword(transeqs, reference):

        change_seq = [[-1]]*len(reference)
        change_index = [-1]*len(reference)

        for i, refer in enumerate(reference):
            for j, word in enumerate(refer):
                if j >= len(transeqs[i]):
                    change_seq[i] = refer[j:j+3]
                    change_index[i] = j
                    break
                else:
                    if transeqs[i][j] != word:
                        change_seq[i] = refer[j:j + 1]
                        change_index[i] = j
                        break
        # if -1 not in tpi:
        #     id = numpy.argmax(numpy.array(tpi))
        #     index.append(tpi[id])
        #     chword.append(tp[id][tpi[id]])
        # else:
        #     index.append(-1)
        #     chword.append(-1)
        return change_seq, change_index

    def l2r_translate(times):
        # import ipdb

        pre_result = word_index(tp['save_path'][0]+'_'+str(times-1)+'.lctok', tp['dictionaries'][2], options)
        for i, line in enumerate(pre_result):
            pre_result[i]=line[:-1]

        change_word, change_index = preword(pre_result, refindex)

        # ipdb.set_trace()

        transdoc = []
        correct_sequence = 0
        for i in change_index:
            if i == -1:
                correct_sequence +=1
        if correct_sequence == len(change_index):
            return 
        for i, seq in enumerate(pre_result):

            # if i < 240:
            #     continue

            if change_index[i] == -1:
                print 'this sequence is true'
                seq = index_word(seq, tp['dictionaries'][3])
                transdoc.append(' '.join(seq))
            else:
                print 'the times : %s  number of sequence: %s' % (times, i)
                pre_seq = seq[:change_index[i]] + change_word[i]

                # seq[change_index[i]:change_index[i]+len(change_word[i])] = change_word[i]

                # sample_new = seq
                # print '----------------------------------------------'
                # print ' '.join(one4index2en(seq,word_idict_trg))
                # print ' '.join(one4index2en(seqtar, word_idict_trg))
                # print '----------------------------------------------


                sample, score, _ = gen_sample(tparams, f_init, f_next,
                                              numpy.array(srcindex[i]).reshape([len(srcindex[i]), 1]),
                                              options, trng=trng, k=tp['beam_size'][0], maxlen=200,
                                              stochastic=False, argmax=False, seqchange=pre_seq, word_idict_trg=None)
                if sample == [[]]:
                    sample = [1, 1, 1, 1, 1, 0]
                    transdoc.append(' '.join(index_word(sample, tp['dictionaries'][3])))
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
                    transdoc.append(' '.join(index_word(sample_new, tp['dictionaries'][3])))
                # transdoc.append(opti_seq(sample, score, word_idict_trg))
                # print sample
                # xxx = opti_bleu(sample, my_reference[i], word_idict_trg)
                # xxx.reverse()
                # transdoc.append(xxx)
                # print transdoc[i]

        with open(tp['save_path'][0] + '_' + str(times) + '.lctok', 'w') as fw:
            fw.writelines('\n'.join(transdoc))

    if not os.path.exists(tp['save_path'][0]+'_0.lctok'):
        _translate()

    for times in range(tp['max_times'][0]):
        if os.path.exists(tp['save_path'][0] + '_'+str(times+1)+'.lctok'):
            continue
        else:
            l2r_translate(times+1)

    return


def word_index(filepaths, word_dict, options):
    w2i = []
    with open(filepaths, 'r') as f:
        for idx, line in enumerate(f):
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
    for index in caps:
        if index == 0:
            break
        capsw.append(word_idict_trg[index])
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
    parser.add_argument('reference',type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    tp = {'model': [args.model],
          'beam_size': [5],
          'normalize': [True],
          'dict_path': [args.dictionary, args.dictionary_target],
          'src_path': [args.source],
          'trg_path': [args.reference],
          'save_path': [args.saveto],
          'max_times': [3]}

    print tp

    main(tp)


