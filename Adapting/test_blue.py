import re
import os
from bleu_oneseq import (find_bleu_score_one)

f_inter = open('/home/wengrx/dl4mt_interactive_adapting/data/hyps/transmt03_1.lctok')
seq_inter = []
while True:
    seq = f_inter.readline()
    if seq == "":
        break
    seq_inter.append(seq.split())

f_adapt = open('/home/wengrx/dl4mt_interactive_adapting/data/translate/transmt03_adapt.lctok')
seq_adapt = []
while True:
    seq = f_adapt.readline()
    if seq == "":
        break
    seq_adapt.append(seq.split())

f_origin = open('/home/wengrx/dl4mt_interactive_adapting/data/translate/transmt03_0.lctok')
seq_origin = []
while True:
    seq = f_origin.readline()
    if seq == "":
        break
    seq_origin.append(seq.split())


def get_reference(filepaths):
    reference = dict()
    files = []
    for filename in os.listdir(filepaths):
        if re.search(r'.lctok$', filename) != None:
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

my_reference = get_reference('/home/wengrx/dl4mt_interactive_adapting/data/reference/MT03')


f = open('result_origin_adapt','w')
for i in xrange(len(seq_adapt)):
    print ' '.join(seq_origin[i])
    print ' '.join(seq_adapt[i])
    print ' '.join(seq_inter[i])
    print find_bleu_score_one(seq_origin[i], my_reference[i])
    print find_bleu_score_one(seq_adapt[i], my_reference[i])
    print find_bleu_score_one(seq_inter[i], my_reference[i])
    f.writelines(' '.join(seq_origin[i])+'\n')
    f.writelines(' '.join(seq_adapt[i]) + '\n')
    f.writelines(' '.join(seq_inter[i]) + '\n')
    f.writelines(str(find_bleu_score_one(seq_origin[i], my_reference[i])) + '\n')
    f.writelines(str(find_bleu_score_one(seq_adapt[i], my_reference[i])) + '\n')
    f.writelines(str(find_bleu_score_one(seq_inter[i], my_reference[i])) + '\n')
    f.writelines('-----------------------------------------------\n')
