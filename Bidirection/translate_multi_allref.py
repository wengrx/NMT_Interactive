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

from nmt import (build_sampler, gen_sample_force, load_params, init_tparams)
from bleu import (find_bleu_score_one)


def translate_model(translate_params, dictionaries, tparams, options):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    print 'bulid_sampler\n'
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)
    f_init_r, f_next_r = build_sampler(tparams, options, trng, use_noise, prefix='decoder_r2l')
    print 'done'

    src_index = src2index(translate_params['test_datasets'][0], dictionaries[0])

    # n-times
    for times in xrange(translate_params['maxround'][0]):
        _interactive(tparams, options, trng, dictionaries, src_index, translate_params, f_init, f_init_r, f_next,
                     f_next_r, times + 1)
    return


def _interactive(tparams, options, trng, dictionaries, srcinfo, translate_params, f_init, f_init_r, f_next, f_next_r,
                 times):
    # get mistake and compute bleu
    reference_index = ref2index(translate_params['test_datasets'][1], dictionaries[2])
    reference_word = get_reference(translate_params['test_datasets'][1])

    pre_seqs = []
    with open(translate_params['trgpath'][0] + '_' + str(times) + '.lctok', 'r') as f:
        for line in f:
            sequence = line.strip().split()
            pre_seqs.append(sequence)
        pre_seqs = seqs4en2index(pre_seqs, dictionaries[2])

    # the location of history revision
    pre_loc = []
    with open(translate_params['trgpath'][0] + '_' + str(times) + '.picklocation', 'r') as f_word:
        for line in f_word:
            ss = line.strip().split()
            pre_loc.append(ss)
        for i in xrange(len(pre_loc)):
            for j in xrange(len(pre_loc[i])):
                pre_loc[i][j] = int(pre_loc[i][j])

    # the word index of history revision
    pre_words = []
    with open(translate_params['trgpath'][0] + '_' + str(times) + '.reviseword', 'r') as f_word:
        for line in f_word:
            ss = line.strip().split()
            pre_words.append(ss)
        for i in xrange(len(pre_words)):
            for j in xrange(len(pre_words[i])):
                pre_words[i][j] = int(pre_words[i][j])

    exist_seq = 0
    if os.path.exists(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok'):
        with open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok', 'r') as f:
            for _ in f:
                exist_seq += 1

    # search all mistake: words, index, left or right
    print 'search all mistaks....',
    chwords, indexs, _ = search_all_mistakes(pre_seqs, reference_index)
    print 'done'

    for i, seq in enumerate(pre_seqs):
        if i < exist_seq:
            continue
        # the BLEU of original sequence
        origin_bleu = only_bleu(seq, reference_word[i], dictionaries[3])
        print 'Times is: %s' % (times + 1)
        print 'No. %s' % i
        print 'Origin Sequence:', ' '.join(one4index2en(seq, dictionaries[3]))
        print 'BLEU:', origin_bleu

        f_result = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.lctok', 'a')
        f_wordindex = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.reviseword', 'a')
        f_location = open(translate_params['trgpath'][0] + '_' + str(times + 1) + '.picklocation', 'a')

        # need not revise
        if [] in chwords[i] or (origin_bleu == 1.0) or -100 in seq:
            print 'This sequence is true'
            print >> f_result, ' '.join(one4index2en(seq, dictionaries[3]))
            print >> f_wordindex, -100
            print >> f_location, -100
        else:
            hyp_seq = [seq]
            hyp_loc = [pre_loc[i]]
            hyp_word = [pre_words[i]]

            for q, chwords_or in enumerate(chwords[i]):
                for j, word in enumerate(chwords_or):
                    location = []
                    if indexs[i][q][j] == -10:
                        continue

                    location.append(indexs[i][q][j])

                    hyp_word.append(word)

                    bigger_index = []
                    smaller_index = []

                    bigger_word = []
                    smaller_word = []

                    # the pre-revise's location
                    # the pre_revise can be changed when last pre-revise think this revise is wrong
                    for index_, index in enumerate(pre_loc[i]):
                        if index > indexs[i][q][j]:
                            bigger_index.append(index)
                            bigger_word.append(pre_words[i][index_])
                        elif index < indexs[i][q][j]:
                            smaller_index.append(index)
                            smaller_word.append(pre_words[i][index_])

                    # prefix if the revised word in the front of the first word
                    if indexs[i][q][j] < 0:
                        pre_seq = [word]
                    else:
                        pre_seq = seq[:indexs[i][q][j]] + [word]

                    pre_length = len(pre_seq)

                    # if bigger is none, this means pre_revise can't happened L2R grid search (jiu shi zhe yang)
                    if bigger_index is not []:
                        sample, score, prob, location = iteration_decoding(tparams, options, translate_params, dictionaries,trng,
                                                                           srcinfo[i], bigger_index, bigger_word,
                                                                           pre_length,
                                                                           pre_seq, f_init, f_next, location)
                        # for hisindex, hisword in zip(bigger_index,bigger_word):
                        #     if hisindex == pre_length:
                        #         pre_seq.append(hisword)
                        #         pre_length +=1
                        #         location.append(hisindex)
                        #     else:
                        #         sample, score, probs = gen_sample_force(tparams, f_init, f_next,
                        #                                                numpy.array(srcinfo[i]).reshape(
                        #                                                    [len(srcinfo[i]), 1]),
                        #                                                options, trng=trng,
                        #                                                k=translate_params['beam_size'][0],
                        #                                                maxlen=200,
                        #                                                stochastic=False, argmax=False,
                        #                                                preseq=pre_seq)
                        #
                        #         sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #         pre_seq = grid_search(sample[sidx],probs[sidx],hisindex,hisword)
                        #         pre_length = len(pre_seq)
                        #         location.append(pre_length-1)
                        #
                        # sample, score, prob = gen_sample_force(tparams, f_init, f_next,
                        #                                        numpy.array(srcinfo[i]).reshape(
                        #                                            [len(srcinfo[i]), 1]),
                        #                                        options, trng=trng,
                        #                                        k=translate_params['beam_size'][0],
                        #                                        maxlen=200,
                        #                                        stochastic=False, argmax=False, preseq=pre_seq)

                    else:
                        sample, score, prob = gen_sample_force(tparams, f_init, f_next,
                                                               numpy.array(srcinfo[i]).reshape(
                                                                   [len(srcinfo[i]), 1]),
                                                               options, trng=trng,
                                                               k=translate_params['beam_size'][0],
                                                               maxlen=200,
                                                               stochastic=False, argmax=False, preseq=pre_seq)

                    sidx = opti_seq(sample, score, dictionaries[3])


                    if indexs[i][q][j] <= 0:
                        hyp_seq.append(sample[sidx])
                        hyp_loc.append(location)
                        continue

                    # suf_probs = prob_result[::-1]
                    suf_probs = prob[sidx][:pre_length - 1][::-1]
                    pre_seq_r = sample[sidx][pre_length - 1: -1][::-1]
                    pre_length_r = len(pre_seq_r)
                    seq_length = len(sample[sidx][:-1])

                    # R2L part
                    location_pre = []
                    if smaller_index is not []:
                        # reverse
                        length_left = len(sample[sidx][:-1])
                        smaller_index_r = [length_left - num - 1 for num in smaller_index]
                        smaller_word_r = smaller_word[::-1]
                        sample, score, prob, location_pre = iteration_decoding(tparams, options, translate_params, dictionaries,trng,
                                                                           srcinfo[i], smaller_index_r, smaller_word_r,
                                                                           pre_length_r,
                                                                           pre_seq_r, f_init_r, f_next_r, location_pre)
                        # for hisindex, hisword in zip(smaller_index_r, smaller_word_r):
                        #     if hisindex == pre_length_r:
                        #         location_pre.append(hisindex)
                        #         pre_seq_r.append(hisword)
                        #         pre_length_r += 1
                        #
                        #     else:
                        #         sample, score, probs = gen_sample_force(tparams, f_init_r, f_next_r,
                        #                                                 numpy.array(srcinfo[i]).reshape(
                        #                                                     [len(srcinfo[i]), 1]),
                        #                                                 options, trng=trng,
                        #                                                 k=translate_params['beam_size'][0],
                        #                                                 maxlen=200,
                        #                                                 stochastic=False, argmax=False,
                        #                                                 preseq=pre_seq_r)
                        #
                        #         sidx = opti_seq(sample, score, dictionaries[3])
                        #
                        #         pre_seq_r = grid_search(sample[sidx], probs[sidx], hisindex, hisword)
                        #         pre_length_r = len(pre_seq_r)
                        #         location_pre.append(pre_length_r - 1)
                        #
                        # sample, score, prob = gen_sample_force(tparams, f_init_r, f_next_r,
                        #                                        numpy.array(srcinfo[i]).reshape(
                        #                                            [len(srcinfo[i]), 1]),
                        #                                        options, trng=trng,
                        #                                        k=translate_params['beam_size'][0],
                        #                                        maxlen=200,
                        #                                        stochastic=False, argmax=False, preseq=pre_seq_r)

                    else:
                        sample, score, prob = gen_sample_force(tparams, f_init_r, f_next_r,
                                                               numpy.array(srcinfo[i]).reshape(
                                                                   [len(srcinfo[i]), 1]),
                                                               options, trng=trng,
                                                               k=translate_params['beam_size'][0],
                                                               maxlen=200,
                                                               stochastic=False, argmax=False, preseq=pre_seq_r,
                                                               suf_probs=suf_probs)

                    sidx = opti_seq(sample, score, dictionaries[3])

                    sample_result_r = sample[sidx][:-1][::-1]

                    seq_length_r = len(sample_result_r)

                    location_pre = [seq_length_r - lo - 1 for lo in location_pre]
                    location = [seq_length_r - seq_length + loca for loca in location]

                    hyp_seq.append(sample_result_r)
                    hyp_loc.append(location_pre + location)

            for h, hhh in enumerate(hyp_seq):
                if hhh is []:
                    hyp_seq[h] = seq
                    hyp_loc[h] = [-100]

            # print len(hyp_seq)
            # print len(hyp_loc)
            # print len(hyp_word)

            best_sequece, id_s = opti_bleu(hyp_seq, reference_word[i], dictionaries[3])

            print 'Best Sequence:', best_sequece
            print 'BLEU:', only_bleu(hyp_seq[id_s], reference_word[i], dictionaries[3])
            print 'Revise Location', hyp_loc[id_s]
            # print 'Revise Word', one4index2en(hyp_word[id_s],dictionaries[3])

            print >> f_result, best_sequece
            f_result.close()
            # print hyp_loc
            loc = [str(p) for p in hyp_loc[id_s]]
            # pre_index[i].append(loc[id_s])
            print >> f_location, ' '.join(loc)
            f_location.close()

            pre_words[i].append(hyp_word[id_s])
            revise_words = [str(i) for i in pre_words[i]]
            print >> f_wordindex, ' '.join(revise_words)
            f_wordindex.close()


def iteration_decoding(tparams, options, translate_params, dictionaries, trng, src_seq, index, word, pre_length, pre_seq, f_init,
                       f_next, location):
    for hisindex, hisword in zip(index, word):
        if hisindex == pre_length:
            pre_seq.append(hisword)
            pre_length += 1
            location.append(hisindex)
        else:
            sample, score, probs = gen_sample_force(tparams, f_init, f_next,
                                                    numpy.array(src_seq).reshape(
                                                        [len(src_seq), 1]),
                                                    options, trng=trng,
                                                    k=translate_params['beam_size'][0],
                                                    maxlen=200,
                                                    stochastic=False, argmax=False,
                                                    preseq=pre_seq)

            sidx = opti_seq(sample, score, dictionaries[3])

            pre_seq = grid_search(sample[sidx], probs[sidx], hisindex, hisword)
            pre_length = len(pre_seq)
            location.append(pre_length - 1)

    sample, score, prob = gen_sample_force(tparams, f_init, f_next,
                                           numpy.array(src_seq).reshape(
                                               [len(src_seq), 1]),
                                           options, trng=trng,
                                           k=translate_params['beam_size'][0],
                                           maxlen=200,
                                           stochastic=False, argmax=False, preseq=pre_seq)
    return sample, score, prob, location


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


# search the position
def grid_search(sample, probs, index, word):
    if index >= len(sample) - 1:
        # print 'origin index :', index
        # print 'new index :', index
        # print 'hypothesis probability :', 1
        return sample[:-1] + [word]
    else:
        hyploc = index - 1
        hyppro = probs[index - 1][word]
        for i in range(index, index + 2):
            if hyppro < probs[i][word]:
                hyploc = i
                hyppro = probs[i][word]

        # print 'origin index :', index
        # print 'new index :', hyploc
        # print 'hypothesis probability :', hyppro
        return sample[:hyploc] + [word]


# search all mistakes in a translated result
def search_all_mistakes(transeqs, reference_index):
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
                # the the range of index is [-1, len(seq)]

                if seq[j] != -1 and (seq[j + 1] != seq[j] + 1) and reference_index[i][k][j + 1] != 1:
                    chword.append(reference_index[i][k][j + 1])
                    chindex.append(seq[j] + 1)
                    chdir.append('l')

                if seq[j + 1] != -1 and (seq[j + 1] != seq[j] + 1) and reference_index[i][k][j] != 1:
                    chword.append(reference_index[i][k][j])
                    chindex.append(seq[j + 1] - 1)
                    chdir.append('r')

            if seq[-1] != (len(reference_index[i][k]) - 1) and reference_index[i][k][-1] != 1:
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

    for h, cws in enumerate(change_words):
        for i in xrange(3):
            for j in xrange(len(cws[i])):
                for k in xrange(3 - i):
                    for w in xrange(len(cws[k + 1 + i])):
                        if cws[i][j] == cws[k + i + 1][w] and change_indexs[h][i][j] == change_indexs[h][k + i + 1][w]:
                            change_indexs[h][k + i + 1][w] = -10
    count = 0
    count_wipe = 0
    for h, index in enumerate(change_indexs):
        for i, iindex in enumerate(index):
            for j, iiindex in enumerate(iindex):
                count += 1
                if iiindex == -10:
                    count_wipe += 1
    print count
    print count_wipe
    # print change_words[0]
    # print change_indexs[0]

    return change_words, change_indexs, change_dir


# word
def get_reference(filepaths):
    reference = dict()
    files = []
    for filename in os.listdir(filepaths):
        if re.search(r'^ref', filename) is not None:
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


# index
def ref2index(filepaths, word_dict_trg):
    reference = dict()
    files = []
    for filename in os.listdir(filepaths):
        if re.search(r'^ref', filename) is not None:
            files.append(filepaths + '/' + filename)
    files = sorted(files)
    for file_ in files:
        with open(file_, 'r') as f:
            for idx, line in enumerate(f):
                if idx not in reference:
                    reference[idx] = []
                words = line.decode('utf-8').split()
                word_dict_trg.iterkeys()
                x = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                x = map(lambda ii: ii if ii < 30000 else 1, x)
                reference[idx].append(x)
    return reference


# sequence is indexs
# my_reference is words
def opti_bleu(sample, my_reference, word_idict_trg):
    bleu = []
    for i in xrange(len(sample)):
        samle_w = one4index2en(sample[i], word_idict_trg)
        bleu.append(find_bleu_score_one(samle_w, my_reference))
    score = numpy.array(bleu)
    sidx = numpy.argmax(score)

    seq = ' '.join(one4index2en(sample[sidx], word_idict_trg))
    return seq, sidx


# sequence is indexs
# my_reference is words
def only_bleu(sequence, my_reference, word_idict_trg):
    samle_w = one4index2en(sequence, word_idict_trg)
    bleu = find_bleu_score_one(samle_w, my_reference)
    return bleu


def opti_seq(sample, score, word_idict_trg):
    if sample == [[]]:
        return 10000
    else:
        score = numpy.array(score)
        lengths = numpy.array([len(s) for s in sample])
        score = score / lengths
        sidx = numpy.argmin(score)
        return sidx


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

    translate_model(translate_params, dictionaries, tparams, options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('reference', type=str)
    parser.add_argument('saveto', type=str)
    args = parser.parse_args()

    translate_params = {'model': [args.model],
                        'beam_size': [5],
                        'dictionaries': [args.dictionary, args.dictionary_target],
                        'test_datasets': [args.source, args.reference],
                        'trgpath': [args.saveto],
                        'maxround': [2]}

    print translate_params

    main(translate_params)
