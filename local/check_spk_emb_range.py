import sys
import numpy as np
from collections import defaultdict
import kaldi_io

for key, mat in kaldi_io.read_vec_flt_scp(sys.argv[1]):
    mean = np.mean(mat, axis=0)
    std  = np.std(mat, axis=0)
    print('key %s has mean %f and std %f' % (key, mean, std))
