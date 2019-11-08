import time
import sys
import numpy as np
from struct import unpack
import argparse
from model import NeuralSpeakerModel
import torch
from kaldi_io import read_mat_scp
torch.backends.cudnn.benchmark = True

def printf(format, *args):
    sys.stdout.write(format % args)

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--spk_num', type=int, help='number of speakers')
parser.add_argument('--model', type=str, required=True, help='feature extractor model type')
parser.add_argument('--input-dim', type=int, required=True, help='input feature dimension')
parser.add_argument('--D', type=int, required=True, help='LDE dictionary components')
parser.add_argument('--hidden-dim', type=int, required=True, help='speaker embedding dimension')
parser.add_argument('--pooling', type=str, required=True, help='mean or mean+std')
parser.add_argument('--network-type', type=str, required=True, help='lde or att')
parser.add_argument('--distance-type', type=str, required=True, help='sqr or norm')
parser.add_argument('--asoftmax', required=True, help='True or False')
parser.add_argument('--m', type=int, help='m for A-softmax')
parser.add_argument('--model-path', help='trained model (.h5)')
parser.add_argument('--decode-scp', help='decode.scp')
parser.add_argument('--out-path', help='output file path')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model=NeuralSpeakerModel(model=args.model, input_dim=args.input_dim, output_dim=args.spk_num, D=args.D, hidden_dim=args.hidden_dim, \
    pooling=args.pooling, network_type=args.network_type, distance_type=args.distance_type, asoftmax=args.asoftmax, m=args.m).to(device)

checkpoint = torch.load(args.model_path, lambda a,b:a)
model.load_state_dict(checkpoint['state_dict'])

def SequenceGenerator(file_name, out_file):
    model.eval()
    f = open(out_file, 'w')
    with torch.no_grad():
        for lab, x in read_mat_scp(file_name):
            y=[x]
            y_pred=model.predict(torch.from_numpy(np.array(y,dtype=np.float32)).to(device)).cpu().data.numpy().flatten()
            f.write(lab+' [ '+' '.join(map(str, y_pred.tolist()))+' ]\n')
    f.close()

SequenceGenerator(args.decode_scp, args.out_path)
