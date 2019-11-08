from kaldi_io import read_vec_flt, write_vec_flt, open_or_fd, write_mat
import sys
import numpy as np
from collections import defaultdict

dev_test_spk = ['p311', 'p226', 'p303', 'p234', 'p302', 'p237', 'p294', 'p225']

with open(sys.argv[1], 'r') as f:
    content = f.readlines()
content = [x.strip() for x in content]

spk2mat = defaultdict(list)
for line in content:
    (key,rxfile) = line.split()
    spk = key.split('-')[0]
    if spk in dev_test_spk:
        seg = int(key.split('-')[2])
        if seg < 25:
            continue
    spk2mat[spk].append(read_vec_flt(rxfile))

out_file = sys.argv[2]
ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_file + '.ark,' + out_file + '.scp'
with open_or_fd(ark_scp_output, 'wb') as f:
    for spk,mat in spk2mat.items():
        spk_emb = np.mean(mat, axis=0).reshape(-1, 1)
        #print(spk)
        #print(spk_emb.shape)
        write_mat(f, spk_emb, key=spk)
