import sys

def reverse(input_file):
    r_sequence=[]
    with open(input_file,'r') as f:
        for line in f:
            r_sequence.append(' '.join(line.strip().split()[::-1])+'\n')

    with open(input_file,'w') as f:
        f.writelines(r_sequence)

    print 'done'


if __name__=='__main__':
    input_file=sys.argv[1]
    reverse(input_file)