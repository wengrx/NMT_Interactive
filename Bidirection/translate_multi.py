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
from math import fabs

theano.config.floatX = 'float32'

from nmt import (build_sampler, gen_sample, load_params, init_tparams)

# from bleu_a import (find_bleu_score)
from bleu import (find_bleu_score_one)


def translate_model(translate_params, dictionaries, tparams, options, tparams_r, options_r):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    print 'bulid_sampler'
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)
    f_init_r, f_next_r = build_sampler(tparams_r, options_r, trng, use_noise)
    print 'done'

    srcinfo = src2index(translate_params['test_datasets'][0], dictionaries[0])

    # search all mistakes in a translated result
    def changewords_a(transeqs, reference_index):

        seq_change = []

        for i, trans_seq in enumerate(transeqs):
            scha = []
            for j, refer_seq in enumerate(reference_index[i]):
                flag = [-1] * len(trans_seq)  # lenth == translate sequence
                change = [-1] * len(refer_seq)  # lenth == reference sequence
                for k, refer_word in enumerate(refer_seq):
                    if refer_word in trans_seq and refer_word != 1:  #
                        index = []
                        for ii, trans_word in enumerate(trans_seq):
                            if trans_word == refer_word:
                                index.append(ii)
                        for _, iii in enumerate(index):
                            if flag[iii] == -1:
                                change[k] = iii
                                flag[iii] = 1
                                break
                scha.append(change)
            seq_change.append(scha)

        change_words = []
        change_indexs = []
        change_dir = []

        for i, seqone_c in enumerate(seq_change):

            chwords = []
            chindexs = []
            chdirs = []
            for k, seq in enumerate(seqone_c):
                chword = []
                chindex = []
                chdir = []
                # the first word is true or not
                if seq[0] != 0 and reference_index[i][k][0] != 1:
                    chword.append(reference_index[i][k][0])
                    chindex.append(0)
                    chdir.append('l')

                for j in xrange(len(seq) - 1):
                    if seq[j] != -1 and (seq[j + 1] != seq[j] + 1) and reference_index[i][k][j + 1] != 1:
                        chword.append(reference_index[i][k][j + 1])
                        chindex.append(seq[j] + 1)
                        chdir.append('l')

                    if seq[j + 1] != -1 and (seq[j + 1] != seq[j] + 1) and reference_index[i][k][j] != 1:
                        chword.append(reference_index[i][k][j])
                        chindex.append(seq[j + 1] - 1)
                        chdir.append('r')

                if seq[-1] != (len(reference_index[i][k])) and reference_index[i][k][-1] != 1:
                    chword.append(reference_index[i][k][-1])
                    chindex.append(len(transeqs[i]) - 1)
                    chdir.append('r')

                for h, ww in enumerate(transeqs[i]):
                    if ww == reference_index[i][k][-1]:
                        chword.append(0)
                        chindex.append(h + 1)
                        chdir.append('l')

                chwords.append(chword)
                chindexs.append(chindex)
                chdirs.append(chdir)

            change_words.append(chwords)
            change_indexs.append(chindexs)
            change_dir.append(chdirs)

        return change_words, change_indexs, change_dir

    # left to right interactive translate
    def l2r_interactive(times):

        reference_index = ref2index(translate_params['test_datasets'][1], dictionaries[2])
        reference_word = get_reference(translate_params['test_datasets'][1])

        transeqs = []
        with open(translate_params['trgpath'][0] + '_' + str(times) + '.lctok', 'r') as f:
            for line in f:
                sequence = line.strip().split()
                transeqs.append(sequence)
        transeqs = seqs4en2index(transeqs, dictionaries[2])

        pre_words = []
        with open(translate_params['trgpath'][0] + '_' + str(times) + '.chooseword', 'r') as f_word:
            for line in f_word:
                ss = line.strip().split()
                pre_words.append(ss)

        for i in xrange(len(pre_words)):
            for j in xrange(len(pre_words[i])):
                pre_words[i][j] = int(pre_words[i][j])

        choose_ref = []
        with open(translate_params['trgpath'][0] + '_' + str(times) + '.refnum', 'r') as f_ref:
            for line in f_ref:
                choose_ref.append(line.strip())

        for i in xrange(len(choose_ref)):
            choose_ref[i] = int(choose_ref[i])

        exist_seq = 0
        if os.path.exists(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok'):
            with open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok', 'r') as f:
                for _ in f:
                    exist_seq += 1

        # search all mistake: words, index, left or right
        chwords, indexs, dir = changewords_a(transeqs, reference_index)

        chwords_or = []
        indexs_or = []
        dir_or = []

        for i, num in enumerate(choose_ref):
            if num == -1:
                chwords_or.append([-1])
                indexs_or.append([-1])
                dir_or.append([-1])
            else:
                chwords_or.append(chwords[i][num])
                indexs_or.append(indexs[i][num])
                dir_or.append(dir[i][num])

        # for i in xrange(len(chwords_or)):
        #     if len(chwords_or[i]) != len(indexs_or[i]):
        #         print 'word lenth is not equal with index lenth'
        #         print i

        # todo
        for i, seq in enumerate(transeqs):

            if i < exist_seq:
                continue

            origin_bleu = only_bleu(seq, reference_word[i], dictionaries[3])

            f_result = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok', 'a')
            f_wordindex = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.chooseword', 'a')
            f_refnum = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.refnum', 'a')

            if chwords_or[i] == [-1] or (origin_bleu == 1.0) or choose_ref[i] == -1:
                print 'this sequence is true'
                print >> f_result, ' '.join(one4index2en(seq, dictionaries[3]))
                print >> f_wordindex, -1
                print >> f_refnum, -1
            else:

                print 'the number of sequence: %s' % i
                print origin_bleu
                print ' '.join(one4index2en(seq, dictionaries[3]))

                hyp_seq = []
                hyp_loc = []

                for j, word in enumerate(chwords_or[i]):

                    bigger_index = []
                    smaller_index = []

                    for pre_index in pre_words[i]:
                        if pre_index > indexs_or[i][j]:
                            bigger_index.append(pre_index)
                        else:
                            smaller_index.append(pre_index)
                    print bigger_index
                    print smaller_index

                    if dir_or[i][j] == 'l' or dir_or[i][j] == 'r':

                        pre_seq = seq[:indexs_or[i][j]] + [word]
                        if indexs_or[i][j] < 0:
                            pre_seq = [word]
                        pre_lenth = len(pre_seq)

                        candidate = pre_seq

                        if bigger_index != [] and bigger_index[-1] + 1 < len(seq):

                            sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                                                                numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                                                                options, trng=trng, k=translate_params['beam_size'][0],
                                                                maxlen=200,
                                                                stochastic=False, argmax=False, preseq=pre_seq)
                            _, sidx = opti_seq(sample, score, dictionaries[3])

                            # print len(sample[sidx][:-1])
                            # print pre_lenth
                            # print indexs_or[i][j]
                            # print bigger_index[-1]

                            compare_part = seq[pre_lenth:bigger_index[-1] + 1]

                            # print ' '.join(one4index2en(sample[sidx], dictionaries[3]))
                            # print ' '.join(one4index2en(compare_part, dictionaries[3]))

                            temp_lenth = len(sample[sidx]) - pre_lenth - len(compare_part)
                            temp_lenth_r = pre_lenth - len(compare_part) - 1

                            temp_seq = [sample[sidx][:pre_lenth] + compare_part]
                            temp_num = [0]
                            if temp_lenth > 0:
                                if temp_lenth > 3:
                                    temp_lenth = 3
                                for ii in xrange(temp_lenth):
                                    pre_sen = sample[sidx][pre_lenth + ii:pre_lenth + ii + len(compare_part)]
                                    temp_seq.append(pre_sen)
                                    temp_num.append(ii)

                            if temp_lenth_r >0:
                                if temp_lenth_r >3:
                                    temp_lenth_r = 3
                                for ii in xrange(temp_lenth_r):
                                    pre_sen = sample[sidx][pre_lenth - ii:pre_lenth - ii + len(compare_part)]
                                    temp_seq.append(pre_sen)
                                    temp_num.append(-ii)

                            for pp, sen in enumerate(temp_seq):
                                if sen == [] or sen ==[0]:
                                    temp_seq[pp] = [1,1,1,1,1,1]
                            print temp_seq

                            _, number = opti_bleu(temp_seq, [one4index2en(compare_part, dictionaries[3])],
                                                  dictionaries[3])

                            number = temp_num[number]
                            if number >= 0:
                                candidate = sample[sidx][:pre_lenth + number] + compare_part
                            else:
                                candidate = sample[sidx][:pre_lenth]+compare_part[-number:]

                            print ' '.join(one4index2en(candidate, dictionaries[3]))

                        sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                                                            numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                                                            options, trng=trng, k=translate_params['beam_size'][0],
                                                            maxlen=200,
                                                            stochastic=False, argmax=False, preseq=candidate)

                        _, sidx = opti_seq(sample, score, dictionaries[3])

                        print ' '.join(one4index2en(sample[sidx], dictionaries[3]))

                        print '----the left part is finished-----'

                        # hyp_seq.append(sample[sidx])
                        # hyp_loc.append([pre_lenth - 1, len(candidate) - 1])

                        lenth_left = len(sample[sidx][:-1])

                        prob_r = prob[sidx][:pre_lenth - 1]
                        prob_r.reverse()
                        pre_seq_r = sample[sidx][pre_lenth - 1:-1]
                        pre_seq_r.reverse()
                        candidate_r = pre_seq_r

                        pre_lenth_r = len(pre_seq_r)

                        seq_r = []
                        for word_index in seq:
                            seq_r.append(word_index)
                        seq_r.reverse()

                        smaller_index_r = []
                        for w, num in enumerate(smaller_index):
                            smaller_index_r.append(lenth_left - smaller_index[
                                w] - 1)

                        if smaller_index_r != [] and smaller_index_r[-1] + 1 < len(seq) and smaller_index_r[-1] > pre_lenth_r:

                            sample_r, score_r, _, prob_r = gen_sample(tparams_r, f_init_r, f_next_r,
                                                                      numpy.array(srcinfo[i]).reshape(
                                                                          [len(srcinfo[i]), 1]),
                                                                      options_r, trng=trng,
                                                                      k=translate_params['beam_size'][0],
                                                                      maxlen=200,
                                                                      stochastic=False, argmax=False, preseq=pre_seq_r,
                                                                      pre_prob=prob_r)
                            #
                            _, sidx_r = opti_seq(sample_r, score_r, dictionaries[3])
                            #
                            #     # prob = prob[sidx][pre_lenth:]
                            #
                            print lenth_left
                            print smaller_index[-1]
                            print smaller_index_r[-1]
                            print pre_lenth_r
                            compare_part = seq_r[pre_lenth_r:smaller_index_r[-1] + 1]
                            #
                            print ' '.join(one4index2en(sample_r[sidx], dictionaries[3]))

                            print ' '.join(one4index2en(compare_part, dictionaries[3]))
                            #
                            # compare_lenth = bigger_index[-1] - pre_lenth + 1
                            #
                            temp_lenth = len(sample_r[sidx]) - pre_lenth_r - len(compare_part)
                            #
                            #     print temp_lenth
                            #     print '----compare------'
                            #     # candidate = None
                            if temp_lenth > 0:
                                temp_seq = []
                                for ii in xrange(temp_lenth):
                                    pre_sen = sample_r[sidx][pre_lenth_r + ii:pre_lenth_r + ii + len(compare_part)]
                                    #             print one4en2index(pre_sen, dictionaries[3])
                                    temp_seq.append(pre_sen)
                                _, number = opti_bleu(temp_seq, [one4index2en(compare_part, dictionaries[3])],
                                                      dictionaries[3])
                                candidate_r = sample_r[sidx][:pre_lenth_r + number] + compare_part
                            else:
                                candidate_r = sample_r[sidx][:pre_lenth_r] + compare_part

                            print ' '.join(one4index2en(candidate_r, dictionaries[3]))
                            prob_r = prob[sidx][len(candidate_r) - 1:]



                        print '---------hahahha--------------'

                        sample, score, _, prob = gen_sample(tparams_r, f_init_r, f_next_r,
                                                            numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                                                            options_r, trng=trng, k=translate_params['beam_size'][0],
                                                            maxlen=200,
                                                            stochastic=False, argmax=False, preseq=candidate_r,
                                                            pre_prob=prob_r)
                        _, sidx = opti_seq(sample, score, dictionaries[3])
                        print sample
                        sample_r = sample[sidx][:-1]
                        sample_r.reverse()
                        print ' '.join(one4index2en(sample_r, dictionaries[3]))
                        hyp_seq.append(sample_r)
                        hyp_loc.append([len(sample_r) - len(candidate_r) + 1,
                                        pre_lenth - len(sample_r) + pre_lenth_r + len(candidate) - 1])

                        # if dir_or[i][j] == 'r':
                        #     hyp_seq.append(seq)
                        #     hyp_loc.append(-1)



                        # there also have any problems
                        #     pre_seq = seq[:indexs_or[i][j]] + [word]
                        #     pre_lenth = len(pre_seq)
                        #     l_sample = pre_seq
                        #
                        #     for h, preword in enumerate(bigger_word):
                        #         sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                        #                                             numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                        #                                             options, trng=trng, k=translate_params['beam_size'][0],
                        #                                             maxlen=200,
                        #                                             stochastic=False, argmax=False, preseq=pre_seq)
                        #         _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #         prob = prob[sidx][pre_lenth:]
                        #
                        #         prob_suf = [pp[preword] * (bigger_index[h] - pre_lenth + 1) / ((
                        #                                                                            (bigger_index[
                        #                                                                                 h] - pre_lenth + 1) + fabs(
                        #                                                                                kk + pre_lenth -
                        #                                                                                bigger_index[
                        #                                                                                    h])) * 1.0)
                        #                     for kk, pp in enumerate(prob)]
                        #
                        #         useloc = prob_suf.index(max(prob_suf))
                        #
                        #         pre_seq = sample[sidx][:pre_lenth + useloc] + [preword]
                        #         pre_lenth = len(pre_seq)
                        #
                        #         minus = pre_lenth - 1 - bigger_index[h]
                        #         for w, num in enumerate(bigger_index[h:]):
                        #             bigger_index[w] = num + minus
                        #         l_sample = pre_seq
                        #
                        #     print 'first part left'
                        #
                        #     sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                        #                                         numpy.array(srcinfo[i]).reshape(
                        #                                             [len(srcinfo[i]), 1]),
                        #                                         options, trng=trng,
                        #                                         k=translate_params['beam_size'][0], maxlen=200,
                        #                                         stochastic=False, argmax=False, preseq=l_sample)
                        #
                        #     _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #     prob_result = prob[sidx][:pre_lenth - 2]
                        #
                        #     if indexs_or[i][j] == 0:
                        #         hyp_seq.append(sample[sidx])
                        #         hyp_loc.append(smaller_index + [pre_lenth - 1] + bigger_index)
                        #         continue
                        #
                        #     prob_result.reverse()
                        #
                        #     left_result = sample[sidx]
                        #     del left_result[-1]
                        #     left_result.reverse()
                        #
                        #     # reverse the index for the smalle group
                        #     for iii, num in enumerate(smaller_index):
                        #         smaller_index[iii] = len(left_result) - smaller_index[
                        #             iii] - 1
                        #
                        #     pre_seq_r = left_result[:len(left_result) - pre_lenth + 2]  # the pre-sequence for reverse
                        #     r_sample = pre_seq_r
                        #
                        #     pre_lenth_r = len(pre_seq_r)
                        #
                        #     for h, preword in enumerate(smaller_word):
                        #         sample, score, _, prob = gen_sample(tparams_r, f_init_r, f_next_r,
                        #                                             numpy.array(srcinfo[i]).reshape(
                        #                                                 [len(srcinfo[i]), 1]),
                        #                                             options_r, trng=trng,
                        #                                             k=translate_params['beam_size'][0], maxlen=200,
                        #                                             stochastic=False, argmax=False,
                        #                                             preseq=pre_seq_r, pre_prob=prob_result)
                        #
                        #         _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #         prob_result = prob[sidx][pre_lenth_r:]
                        #
                        #         prob_suf = [pp[preword] * ((smaller_index[h] - pre_lenth_r + 1) / (
                        #             (smaller_index[h] - pre_lenth_r + 1) + fabs(kk + pre_lenth_r) - smaller_index[
                        #                 h]) * 1.0)
                        #                     for kk, pp in enumerate(prob_result)]
                        #
                        #         useloc = prob_suf.index(max(prob_suf))
                        #         prob_result = prob_result[useloc:]
                        #         pre_seq_r = sample[sidx][:pre_lenth_r + useloc] + [preword]
                        #         pre_lenth_r = len(pre_seq_r)
                        #
                        #         minus = pre_lenth_r - 1 - smaller_index[h]
                        #         for c, num in enumerate(smaller_index[h:]):
                        #             smaller_index[c] = num + minus
                        #
                        #         r_sample = pre_seq_r
                        #
                        #     sample, score, _, _ = gen_sample(tparams_r, f_init_r, f_next_r,
                        #                                      numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                        #                                      options_r, trng=trng, k=translate_params['beam_size'][0],
                        #                                      maxlen=200,
                        #                                      stochastic=False, argmax=False, preseq=r_sample,
                        #                                      pre_prob=prob_result)
                        #
                        #     _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #     sample_result_r = sample[sidx]
                        #     del sample_result_r[-1]
                        #
                        #     seq_lenth_r = len(sample_result_r)
                        #     sample_result_r.reverse()
                        #
                        #     minuses = seq_lenth_r - pre_lenth_r - pre_lenth
                        #     for iiii in xrange(len(bigger_index)):
                        #         bigger_index[iiii] = bigger_index[iiii] + minuses
                        #
                        #     for ij, num in enumerate(smaller_index):
                        #         smaller_index[ij] = seq_lenth_r - smaller_index[ij] - 1
                        #     hyp_seq.append(sample_result_r)
                        #
                        #     hyp_loc.append(smaller_index + [seq_lenth_r - pre_lenth_r + 1] + bigger_index)
                        #
                        #     print 'second part left'
                        #
                        #
                        #
                        #
                        #
                        # elif dir_or[i][j] == 'r':
                        #
                        #     hyp_seq.append(seq)
                        #     hyp_loc.append(0)

                        # pre_seq = seq[:indexs_or[i][j]] + [word]
                        # pre_lenth = len(pre_seq)
                        # l_sample = pre_seq
                        #
                        # for h, preword in enumerate(bigger_word):
                        #     sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                        #                                         numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                        #                                         options, trng=trng, k=translate_params['beam_size'][0],
                        #                                         maxlen=200,
                        #                                         stochastic=False, argmax=False, preseq=pre_seq)
                        #
                        #     _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #     prob = prob[sidx][pre_lenth:]
                        #
                        #     # for k, p in enumerate(prob):
                        #     #     print k
                        #     #     print len(p)
                        #
                        #     prob_suf = [
                        #         pp[preword] * ((bigger_index[h] - pre_lenth + 1) / (kk + 1) * 1.0) \
                        #         for kk, pp in enumerate(prob)]
                        #     useloc = prob_suf.index(max(prob_suf))
                        #
                        #     pre_seq = sample[sidx][:pre_lenth + useloc] + [preword]
                        #     pre_lenth = len(pre_seq)
                        #
                        #     minus = pre_lenth - 1 - bigger_index[h]
                        #     for iii, num in enumerate(bigger_index[h:]):
                        #         bigger_index[iii] = num + minus
                        #     l_sample = pre_seq
                        #
                        # print 'frist part right'
                        #
                        # sample, score, _, prob = gen_sample(tparams, f_init, f_next,
                        #                                     numpy.array(srcinfo[i]).reshape(
                        #                                         [len(srcinfo[i]), 1]),
                        #                                     options, trng=trng,
                        #                                     k=translate_params['beam_size'][0], maxlen=200,
                        #                                     stochastic=False, argmax=False, preseq=l_sample)
                        #
                        # _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        # prob_result = prob[sidx][:pre_lenth - 2]
                        #
                        # if indexs[i][j] == 0:
                        #     hyp_seq.append(sample[sidx])
                        #     hyp_loc.append(smaller_index + [pre_lenth - 1] + bigger_index)
                        #     continue
                        #
                        # prob_result.reverse()
                        #
                        # left_result = sample[sidx]
                        # del left_result[-1]
                        # left_result.reverse()
                        #
                        # # reverse the index for the smalle group
                        # for iiii, num in enumerate(smaller_index):
                        #     smaller_index[iiii] = len(left_result) - smaller_index[
                        #         iiii] - 1
                        #
                        # pre_seq_r = left_result[:len(left_result) - pre_lenth + 2]  # the pre-sequence for reverse
                        # r_sample = pre_seq_r
                        #
                        # pre_lenth_r = len(pre_seq_r)
                        #
                        # for h, preword in enumerate(smaller_word):
                        #     sample, score, _, prob = gen_sample(tparams_r, f_init_r, f_next_r,
                        #                                         numpy.array(srcinfo[i]).reshape(
                        #                                             [len(srcinfo[i]), 1]),
                        #                                         options_r, trng=trng,
                        #                                         k=translate_params['beam_size'][0], maxlen=200,
                        #                                         stochastic=False, argmax=False,
                        #                                         preseq=pre_seq_r, pre_prob=prob_result)
                        #
                        #     _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #     prob_suf = [pp[preword] * ((smaller_index[h] - pre_lenth_r + 1) / (kk + 1) * 1.0) \
                        #                 for kk, pp in enumerate(prob[sidx][pre_lenth_r:])]
                        #
                        #     useloc = prob_suf.index(max(prob_suf))
                        #     prob_result = prob_result[useloc:]
                        #     pre_seq_r = sample[sidx][:pre_lenth_r + useloc] + [preword]
                        #     pre_lenth_r = len(pre_seq_r)
                        #
                        #     minus = pre_lenth_r - 1 - smaller_index[h]
                        #     for ij, num in enumerate(smaller_index[h:]):
                        #         smaller_index[ij] = num + minus
                        #
                        #     r_sample = pre_seq_r
                        #
                        # sample, score, _, _ = gen_sample(tparams_r, f_init_r, f_next_r,
                        #                                  numpy.array(srcinfo[i]).reshape([len(srcinfo[i]), 1]),
                        #                                  options_r, trng=trng, k=translate_params['beam_size'][0],
                        #                                  maxlen=200,
                        #                                  stochastic=False, argmax=False, preseq=r_sample,
                        #                                  pre_prob=prob_result)
                        #
                        # _, sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        # sample_result_r = sample[sidx]
                        # del sample_result_r[-1]
                        #
                        # seq_lenth_r = len(sample_result_r)
                        # sample_result_r.reverse()
                        #
                        # minuses = seq_lenth_r - pre_lenth_r - pre_lenth
                        # for ijj in xrange(len(bigger_index)):
                        #     bigger_index[ijj] = bigger_index[ijj] + minuses
                        #
                        # for ijjj, num in enumerate(smaller_index):
                        #     smaller_index[ijjj] = seq_lenth_r - smaller_index[ijjj] - 1
                        # hyp_seq.append(sample_result_r)
                        #
                        # hyp_loc.append(smaller_index + [seq_lenth_r - pre_lenth_r + 1] + bigger_index)
                        #
                        #
                        # print 'second part right'
                        #
                        #     left_bleu = only_bleu(sample_w, my_reference[i], word_idict_trg)
                        #
                        #     temp_result.append(sample_w)
                        #     temp_score.append(left_bleu)
                        #
                        #     temp_score = numpy.array(temp_score)
                        #     id = numpy.argmax(temp_score)
                        #
                        #     hypseq_allref.append(temp_result[id])

                for hhh in hyp_seq:
                    if hhh == []:
                        del hyp_seq[hyp_seq.index(hhh)]
                best_sequece, id_s = opti_bleu(hyp_seq, reference_word[i], dictionaries[3])

                print >> f_result, best_sequece
                f_result.close()
                print >> f_wordindex, hyp_loc[id_s]
                f_wordindex.close()
                print >> f_refnum, choose_ref[i]
                f_refnum.close()
                # print >> f_refnum, hyp_refnum[id_s]
                # f_refnum.close()

    for times in xrange(translate_params['maxround'][0]):
        l2r_interactive(times + 1)
    return


def src2index(filepaths, word_dict):
    srcw2i = []
    with open(filepaths, 'r') as f:
        for idx, line in enumerate(f):
            words = line.strip().split()
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < 30000 else 1, x)
            x += [0]
            srcw2i.append(x)
    return srcw2i


def get_reference(filepaths):
    reference = dict()
    files = []
    for filename in os.listdir(filepaths):
        if re.search(r'^ref', filename) != None:
            files.append(filepaths + '/' + filename)
    files = sorted(files)
    for file in files:
        with open(file, 'r') as f:
            for idx, line in enumerate(f):
                if idx not in reference:
                    reference[idx] = []
                words = line.split()
                reference[idx].append(words)
    return reference


def ref2index(filepaths, word_dict_trg):
    reference = dict()
    files = []
    for filename in os.listdir(filepaths):
        if re.search(r'^ref', filename) != None:
            files.append(filepaths + '/' + filename)
    files = sorted(files)
    for file in files:
        with open(file, 'r') as f:
            for idx, line in enumerate(f):
                if idx not in reference:
                    reference[idx] = []
                words = line.decode('utf-8').split()
                word_dict_trg.iterkeys()
                x = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                x = map(lambda ii: ii if ii < 30000 else 1, x)
                # x += [0]
                reference[idx].append(x)
    return reference


def opti_bleu(sample, my_reference, word_idict_trg):
    bleu = []
    for i in xrange(len(sample)):
        samle_w = one4index2en(sample[i], word_idict_trg)
        bleu.append(find_bleu_score_one(samle_w, my_reference))
    score = numpy.array(bleu)
    sidx = numpy.argmax(score)

    seq = ' '.join(one4index2en(sample[sidx], word_idict_trg))
    print '------------------------------------------------'
    print score[sidx]
    print seq
    print '------------------------------------------------'
    return seq, sidx


def only_bleu(sequence, my_reference, word_idict_trg):
    samle_w = one4index2en(sequence, word_idict_trg)
    bleu = find_bleu_score_one(samle_w, my_reference)

    return bleu


def opti_seq(sample, score, word_idict_trg):
    if sample == [[]]:
        sample = [1, 1, 1, 1, 1, 1, 0]
        return ' '.join(one4index2en(sample, word_idict_trg)), 10000
    else:
        score = numpy.array(score)
        lengths = numpy.array([len(s) for s in sample])
        score = score / lengths
        sidx = numpy.argmin(score)
        return ' '.join(one4index2en(sample[sidx], word_idict_trg)), sidx


def seqs4en2index(caps, word_dict_trg):
    capsw = []
    for w in caps:
        x = []
        for ww in w:
            x.append(word_dict_trg[ww])
        capsw.append(x)
    return capsw


def one4index2en(caps, word_idict_trg):
    capsw = []
    if caps[-1] == 0:
        for w in xrange(len(caps) - 1):
            capsw.append(word_idict_trg[caps[w]])
    else:
        for w in xrange(len(caps)):
            capsw.append(word_idict_trg[caps[w]])
    return capsw


def one4en2index(caps, word_dict_trg):
    capsw = []
    for w in xrange(len(caps)):
        capsw.append(word_dict_trg[caps[w]])
    return capsw


def main(translate_params):
    print 'load model.....',
    with open('%s.pkl' % translate_params['model'][0], 'rb') as f:
        options = pkl.load(f)
    params = load_params(translate_params['model'][0])
    tparams = init_tparams(params)

    with open('%s.pkl' % translate_params['model'][1], 'rb') as f:
        options_r = pkl.load(f)
    iparams = load_params(translate_params['model'][1])
    tparams_r = init_tparams(iparams)
    print 'done'

    # load source dictionary and invert
    with open(translate_params['dictionaries'][0], 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(translate_params['dictionaries'][1], 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    dictionaries = [word_dict, word_idict, word_dict_trg, word_idict_trg]

    translate_model(translate_params, dictionaries, tparams, options, tparams_r, options_r)
    print 'done'


if __name__ == "__main__":
    prepath = '/home/wengrx/dl4mt_interactive_revise1_oneref'

    translate_params = {'model': [prepath + '/param/model_hal.npz',
                                  prepath + '/param/model_hal_r.npz'],
                        'beam_size': [5],
                        'dictionaries': [prepath + '/param/cn.tok.pkl',
                                         prepath + '/param/en.tok.pkl'],
                        'test_datasets': [prepath + '/data/source/MT03.cn.dev',
                                          prepath + '/data/reference/MT03'],
                        'trgpath': [prepath + '/data/translate/transmt03'],
                        'maxround': [20]}

    main(translate_params)
