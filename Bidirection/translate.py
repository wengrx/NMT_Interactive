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

from nmt import (build_sampler, gen_sample_force, load_params, init_tparams)

from bleu_a import (find_bleu_score)
from bleu import (find_bleu_score_one)


def translate_model(translate_params, dictionaries, tparams, options):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))
    flag = 1

    if os.path.exists(translate_params['trgpath'][0] + '_0.lctok'):
        flag = 0

    print 'bulid_sampler.....\n',
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)
    f_init_r, f_next_r = build_sampler(tparams, options, trng, use_noise, prefix='decoder_r2l')
    print 'done'

    print 'source data processing ....',
    src_sequences = src2index(translate_params['test_datasets'][0], dictionaries[0])
    print 'done'

    while True:
        if flag == 1:
            _translate(tparams, options, translate_params, dictionaries, trng, f_init, f_next, src_sequences)
            _interactive(tparams, options, translate_params, dictionaries, trng, f_init, f_next, f_init_r, f_next_r,
                         src_sequences)
            break
        else:
            _interactive(tparams, options, translate_params, dictionaries, trng, f_init, f_next, f_init_r, f_next_r,
                         src_sequences)
            break
    return


def search_all_mistakes(his_seqs, reference_index):
    seq_change = []

    for i, his_seq in enumerate(his_seqs):
        scha = []
        for j, refer_seq in enumerate(reference_index[i]):
            flag = [-1] * len(his_seq)  # lenth == translate sequence
            change = [-1] * len(refer_seq)  # lenth == reference sequence
            for k, refer_word in enumerate(refer_seq):
                if refer_word in his_seq and refer_word != 1:
                    index = []
                    for ii, trans_word in enumerate(his_seq):
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
                chindex.append(len(his_seqs[i]) - 1)
                chdir.append('r')

            for h, ww in enumerate(his_seqs[i]):
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
                        if cws[i][j] == cws[k + i + 1][w] and change_indexs[h][i][j] == change_indexs[h][k + i + 1][w] and change_dir[h][i][j] == change_dir[h][k + i + 1][w]:
                            change_indexs[h][k + i + 1][w] = -1
    count = 0
    count_wipe = 0
    for h, index in enumerate(change_indexs):
        for i, iindex in enumerate(index):
            for j, iiindex in enumerate(iindex):
                count += 1
                if iiindex == -1:
                    count_wipe += 1
    print count
    print count_wipe

    return change_words, change_indexs, change_dir


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


def _translate(tparams, options, translate_params, dictionaries, trng, f_init, f_next, src_sequences):
    transseqs = []
    for i, seq in enumerate(src_sequences):
        print 'No. %s' % i
        sample, score, _ = gen_sample_force(tparams, f_init, f_next,
                                            numpy.array(seq).reshape([len(seq), 1]),
                                            options, trng=trng, k=translate_params['beam_size'][0], maxlen=200,
                                            stochastic=False, argmax=False)
        sample_result, _ = opti_seq(sample, score, dictionaries[3])
        transseqs.append(sample_result)
    with open(translate_params['trgpath'][0] + '_0.lctok', 'w') as fw:
        fw.writelines('\n'.join(transseqs))


def _interactive(tparams, options, translate_params, dictionaries, trng, f_init, f_next, f_init_r, f_next_r,
                 src_sequences):
    reference_index = ref2index(translate_params['test_datasets'][1], dictionaries[2])
    reference_word = get_reference(translate_params['test_datasets'][1])

    transeqs = []
    with open(translate_params['trgpath'][0] + '_0.lctok', 'r') as f:
        for line in f:
            sequence = line.split()
            transeqs.append(sequence)
    transeqs = seqs4en2index(transeqs, dictionaries[2])

    exist_seq = 0
    if os.path.exists(translate_params['trgpath'][0] + '_1.lctok'):
        with open(translate_params['trgpath'][0] + '_1.lctok', 'r') as f:
            for _ in f:
                exist_seq += 1

    # search all mistake: words, index, left or right
    print 'search all mistaks....',
    chwords, indexs, _ = search_all_mistakes(transeqs, reference_index)
    print 'done'

    for i, seq in enumerate(transeqs):
        if i < exist_seq:
            continue

        print 'No. %s' % i
        origin_bleu = compute_bleu(seq, reference_word[i], dictionaries[3])
        print 'Original Sequence:', ' '.join(one4index2en(seq, word_idict_trg=dictionaries[3]))
        print 'BLEU:', origin_bleu

        f_result = open(translate_params['trgpath'][0] + '_1.lctok', 'a')
        f_reviseword = open(translate_params['trgpath'][0] + '_1.reviseword', 'a')
        f_refnum = open(translate_params['trgpath'][0] + '_1.refnum', 'a')
        f_length = open(translate_params['trgpath'][0] + '_1.length', 'a')
        f_picklocation = open(translate_params['trgpath'][0] + '_1.picklocation', 'a')

        if [] in chwords[i] or (origin_bleu == 1.0):
            print 'This sequence is true'
            print >> f_result, ' '.join(one4index2en(seq, dictionaries[3]))
            print >> f_reviseword, -100
            print >> f_refnum, -100
            print >> f_picklocation, -100
        else:

            hyp_seq = []
            hyp_word = []
            hyp_loc = []
            hyp_refnum = []

            for h in xrange(len(chwords[i])):
                for j, word in enumerate(chwords[i][h]):
                    if indexs[i][h][j] == -1:
                        continue

                    hyp_word.append(word)
                    hyp_refnum.append(h)

                    pre_seq = seq[:indexs[i][h][j]] + [word]
                    pre_length = len(pre_seq)

                    sample, score, prob = gen_sample_force(tparams, f_init, f_next,
                                                           numpy.array(src_sequences[i]).reshape(
                                                               [len(src_sequences[i]), 1]),
                                                           options, trng=trng,
                                                           k=translate_params['beam_size'][0], maxlen=200,
                                                           stochastic=False, argmax=False, preseq=pre_seq)
                    _, sidx = opti_seq(sample, score, dictionaries[3])

                    prob_result = prob[sidx][:pre_length - 1]

                    # if the location of the revised word is zero
                    if indexs[i][h][j] == 0:
                        hyp_seq.append(sample[sidx])
                        hyp_loc.append(pre_length - 1)
                        continue

                    # the inputs of R2L decoder
                    suf_probs = prob_result[::-1]
                    pre_seq_r = sample[sidx][pre_length - 1: -1][::-1]

                    pre_length_r = len(pre_seq_r)

                    sample, score, _ = gen_sample_force(tparams, f_init_r, f_next_r,
                                                        numpy.array(src_sequences[i]).reshape(
                                                            [len(src_sequences[i]), 1]),
                                                        options, trng=trng, k=translate_params['beam_size'][0],
                                                        maxlen=200,
                                                        stochastic=False, argmax=False, preseq=pre_seq_r,
                                                        sufprob=suf_probs)

                    _, sidx = opti_seq(sample, score, dictionaries[3])

                    sample_result_r = sample[sidx][:-1][::-1]

                    seq_length = len(sample_result_r)

                    hyp_seq.append(sample_result_r)
                    hyp_loc.append(seq_length - pre_length_r)

            for iii, hhh in enumerate(hyp_seq):
                if hhh is []:
                    hyp_seq[iii] = seq
                    hyp_refnum[iii] = -100
                    hyp_loc[iii] = -100
                    hyp_word[iii] = -100

            best_sequece, id_s = opti_bleu(hyp_seq, reference_word[i], dictionaries[3])

            print 'Best Sequence:', best_sequece
            print 'BLEU:', compute_bleu(hyp_seq[id_s], reference_word[i], dictionaries[3])
            print >> f_result, best_sequece
            f_result.close()
            print >> f_picklocation, hyp_loc[id_s]
            f_picklocation.close()
            print >> f_refnum, hyp_refnum[id_s]
            f_refnum.close()
            print >> f_length, len(hyp_seq[id_s])
            f_length.close()
            print >> f_reviseword, hyp_word[id_s]
            f_reviseword.close()


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


def ref2index(filepaths, word_dict_trg):
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
                words = line.decode('utf-8').split()
                word_dict_trg.iterkeys()
                x = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                x = map(lambda ii: ii if ii < 30000 else 1, x)
                reference[idx].append(x)
    return reference


def opti_bleu(sample, reference, word_idict_trg):
    bleu = []
    for i in xrange(len(sample)):
        samle_w = one4index2en(sample[i], word_idict_trg)
        bleu.append(find_bleu_score_one(samle_w, reference))
    score = numpy.array(bleu)
    sidx = numpy.argmax(score)
    seq = ' '.join(one4index2en(sample[sidx], word_idict_trg))

    return seq, sidx


def compute_bleu(sequence, reference, word_idict_trg):
    samle_w = one4index2en(sequence, word_idict_trg)
    bleu = find_bleu_score_one(samle_w, reference)
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
                        'trgpath': [args.saveto]}

    print translate_params

    main(translate_params)
