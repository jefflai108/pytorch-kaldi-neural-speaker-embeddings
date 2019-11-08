import sys
import os
import shutil
import torch
from model import NeuralSpeakerModel, AngleLoss
import numpy as np
import argparse
import torch.optim as optim
import torch.nn.functional as F
from datasets import SequenceDataset

# Author: Nanxin Chen

class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train', type=str, help='training scp')
    parser.add_argument('--cv', type=str, help='cv scp')
    parser.add_argument('--utt2spkid', type=str, help='utt2spkid')
    parser.add_argument('--spk_num', type=int, help='number of speakers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-n_warmup_steps', type=int, default=8000)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, required=True, help='feature extractor model type')
    parser.add_argument('--input-dim', type=int, required=True, help='input feature dimension')
    parser.add_argument('--D', type=int, required=True, help='LDE dictionary components')
    parser.add_argument('--hidden-dim', type=int, required=True, help='speaker embedding dimension')
    parser.add_argument('--pooling', type=str, required=True, help='mean or mean+std')
    parser.add_argument('--network-type', type=str, required=True, help='lde or att')
    parser.add_argument('--distance-type', type=str, required=True, help='sqr or norm')
    parser.add_argument('--asoftmax', required=True, help='True or False')
    parser.add_argument('--m', type=int, help='m for A-softmax')
    parser.add_argument('--min-chunk-size', type=int, required=True, help='minimum feature map length')
    parser.add_argument('--max-chunk-size', type=int, required=True, help='maximum feature map length')
    parser.add_argument('--log-dir', type=str, required=True, help='logging directory')
    parser.add_argument('--pretrain-model-pth', type=str)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use cuda is %s' % use_cuda)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    tmp = torch.Tensor([2]).to(device)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train = SequenceDataset(scp_file=args.train, utt2spkid_file=args.utt2spkid, min_length=args.max_chunk_size)
    val   = SequenceDataset(scp_file=args.cv, utt2spkid_file=args.utt2spkid, min_length=args.max_chunk_size)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model=NeuralSpeakerModel(model=args.model, input_dim=args.input_dim, output_dim=args.spk_num, D=args.D, hidden_dim=args.hidden_dim, \
            pooling=args.pooling, network_type=args.network_type, distance_type=args.distance_type, asoftmax=args.asoftmax, m=args.m).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('===> Model total parameter: {}'.format(model_params))

    optimizer = ScheduledOptim( # Transformer optimizer
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    start_epoch = 1
    best = 0
    best_epoch = -1

    if args.pretrain_model_pth is not None:
        if os.path.isfile(args.pretrain_model_pth):
            print('loading pre-trained model from %s' % args.pretrain_model_pth)
            model_dict = model.state_dict()
            checkpoint = torch.load(args.pretrain_model_pth, map_location=lambda storage, loc: storage) # load for cpu
            start_epoch = checkpoint['epoch']
            best_epoch = start_epoch
            best = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("===> no checkpoint found at '{}'".format(args.pretrain_model_pth))
            exit()

    if args.asoftmax == 'True': # angular-softmax
        print('training with Angular Softmax')
        criterion = AngleLoss()
    else:
        print('training with Softmax')
        criterion = torch.nn.NLLLoss()

    # ------------------
    # main training loop
    # ------------------

    for epoch in range(start_epoch, start_epoch+args.epochs):

        print('Epoch %d' % epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, non_blocking=True).view((-1,))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), lr, loss.item()))
            train.update(np.random.randint(args.min_chunk_size, args.max_chunk_size+1)) # 3-8s chunk
            del data, target, output, loss

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device, non_blocking=True).view((-1,))
                output = model(data)
                test_loss += criterion(output, target).item() # sum up batch loss
                if args.asoftmax == 'True': # angular-softmax
                    output = output[0] # 0=cos_theta 1=phi_theta
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        if 100. * correct / len(val_loader.dataset) > best:
            best = 100. * correct / len(val_loader.dataset)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': best,
                'optimizer' : optimizer.state_dict(),
            }, args.log_dir + str(epoch) + "_" + str(int(100. * correct / len(val_loader.dataset))) + ".h5")
            print("===> save to checkpoint at {}\n".format(args.log_dir + 'model_best.pth.tar'))
            shutil.copyfile(args.log_dir + str(epoch) + "_" + str(int(100. * correct / len(val_loader.dataset))) +
                    ".h5", args.log_dir + 'model_best.pth.tar')
            best_epoch = epoch
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch
