from kaldi_io import read_vec_flt, write_vec_flt, open_or_fd, write_mat
import sys
import numpy as np
from collections import defaultdict

# first read the dev/test set
with open(sys.argv[3], 'r') as f:
    content = f.readlines()
content = [x.strip() for x in content]

dev_test_spk = defaultdict(list)
for uttid in content:
    spk = uttid.split('_')[0]
    dev_test_spk[spk].append(uttid)

# read utterance embeddings
with open(sys.argv[1], 'r') as f:
    content = f.readlines()
content = [x.strip() for x in content]

# speaker to utterances mapping
spk2mat = defaultdict(list)
for line in content:
    (key,rxfile) = line.split()
    spk = key.split('-')[0]
    if spk in dev_test_spk.keys():
        uttid = key.split('-')[1] + '_' + key.split('-')[2]
        if uttid not in dev_test_spk[spk]:
            continue
    spk2mat[spk].append(read_vec_flt(rxfile))

#for i in spk2mat.keys():
#    if i in dev_test_spk.keys():
#        print(len(spk2mat[i]))

# create speaker embeddings
out_file = sys.argv[2]
ark_scp_output = 'ark:| copy-vector ark:- ark,scp:' + out_file + '.ark,' + out_file + '.scp'
with open_or_fd(ark_scp_output, 'wb') as f:
    for spk,mat in spk2mat.items():
        spk_emb = np.mean(mat, axis=0).reshape(-1,) # get speaker embedding (vector)
        #print(spk_emb.shape)
        #print(spk)
        #print(spk_emb.shape)
        write_vec_flt(f, spk_emb, key=spk)
