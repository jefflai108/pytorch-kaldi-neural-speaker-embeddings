import numpy as np
from torch.utils.data import Dataset
import kaldi_io

# Author: Nanxin Chen, Cheng-I Lai

class SequenceDataset(Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, utt2spkid_file, min_length):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles, self.labels, self.utt2spkid = [], [], {}
        
        # balanced training 
        id_count = {}
        for line in open(utt2spkid_file):
            utt, label = line.rstrip().split()
            self.utt2spkid[utt] = int(label)
            if not int(label) in id_count:
                id_count[int(label)] = 0
            id_count[int(label)] += 1
        max_id_count = int((max(id_count.values())+1)/2)
        
        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            label = self.utt2spkid[utt]
            repetition = max(1, max_id_count // id_count[label])
            self.rxfiles.extend([rxfile] * repetition)
            self.labels.extend([label] * repetition)
        
        self.rxfiles = np.array(self.rxfiles)
        self.labels  = np.array(self.labels, dtype=np.int)
        self.seq_len = min_length
        print("Totally "+str(len(self.rxfiles))+" samples with at most "+
            str(max_id_count)+" samples for one class")
    
    def __len__(self):
        """Return number of samples 
        """
        return len(self.labels)

    def update(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop 
        once per training iteration. 
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        rxfile  = self.rxfiles[index]
        full_mat = kaldi_io.read_mat(rxfile)
        assert len(full_mat) >= self.seq_len
        pin = np.random.randint(0, len(full_mat) - self.seq_len + 1)
        chunk_mat = full_mat[pin:pin+self.seq_len, :]
        y = np.array(self.labels[index])
        
        return chunk_mat, y
