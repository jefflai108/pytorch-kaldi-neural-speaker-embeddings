import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from scipy import linalg as la
import torch.nn.functional as F
from densenet import densenet62

# Author: Nanxin Chen, Cheng-I Lai

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ThinResNet(nn.Module):
    """ResNet with smaller channel dimensions
    """
    def __init__(self, block, layers):
        self.inplanes = 8
        super(ThinResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 3))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 3))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape) # 128, 1, 800, 30
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape) # 128, 128, 100, 4

        x = self.avgpool(x)
        #print(x.shape) # 128, 128, 100, 1
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)
        #print(x.shape) # 128, 100, 128

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def thin_resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ThinResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class LDE(nn.Module):
    def __init__(self, D, input_dim, with_bias=False, distance_type='norm', network_type='att', pooling='mean'):
        """LDE layer
        """
        super(LDE, self).__init__()
        self.dic = nn.Parameter(torch.randn(D, input_dim)) # input_dim by D (dictionary components)
        nn.init.uniform_(self.dic.data, -1, 1)
        self.wei = nn.Parameter(torch.ones(D)) # non-negative assigning weight in Eq(4) in LDE paper
        if with_bias: # Eq(4) in LDE paper
            self.bias = nn.Parameter(torch.zeros(D))
        else:
            self.bias = 0
        assert distance_type == 'norm' or distance_type == 'sqr'
        if distance_type == 'norm':
            self.dis = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dis = lambda x: torch.sum(x**2, dim=-1)
        assert network_type == 'att' or network_type == 'lde'
        if network_type == 'att':
            self.norm = lambda x: F.softmax(-self.dis(x) * self.wei + self.bias, dim = -2)
        else:
            self.norm = lambda x: F.softmax(-self.dis(x) * (self.wei ** 2) + self.bias, dim = -1)
        assert pooling == 'mean' or pooling == 'mean+std'
        self.pool = pooling

    def forward(self, x):
        #print(x.size()) # (B, T, F)
        #print(self.dic.size()) # (D, F)
        r = x.view(x.size(0), x.size(1), 1, x.size(2)) - self.dic # residaul vector
        #print(r.size()) # (B, T, D, F)
        w = self.norm(r).view(r.size(0), r.size(1), r.size(2), 1) # numerator without r in Eq(5) in LDE paper
        #print(self.norm(r).size()) # (B, T, D)
        #print(w.size()) # (B, T, D, 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-9) #batch_size, timesteps, component # denominator of Eq(5) in LDE paper
        if self.pool == 'mean':
            x = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
        else:
            x1 = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
            x2 = torch.sqrt(torch.sum(w * r ** 2, dim=1)+1e-8) # std vector
            x = torch.cat([x1, x2], dim=-1)
        return x.view(x.size(0), -1)


class NeuralSpeakerModel(nn.Module):
    """Neural Speaker Model 
    @model: resnet model
    @input_dim: feature dim
    @output_dim: number of speakers
    @D: LDE dictionary components
    @hidden_dim: speaker embedding dim
    @distance_tpye: 1) norm (Frobenius Norm) or 2) sqr (square norm) --> distance metric in Eq(4) in LDE paper, for calculating the weight over the residual vectors
    @network_type: 1) att (multi-head attention, or attention over T) or 2) lde (LDE, or attention over dictionary components).
    @pooling: aggregation step over the residual vectors 1) mean only or 2) mean and std
    @m: m for A-Softmax
    Note: use the pairing ('norm', 'att') and ('sqr', 'lde')
    """
    def __init__(self, model, input_dim, output_dim, D, hidden_dim=128, distance_type='norm', network_type='att', pooling='mean', asoftmax=False, m=2):
        super(NeuralSpeakerModel, self).__init__()
        if model == 'resnet34':
            self.res = resnet34()
            _feature_dim = 128
        elif model == 'thin-resnet34':
            self.res = thin_resnet34()
            _feature_dim = 64
        elif model == 'se-resnet34':
            self.res = se_resnet34()
            _feature_dim = 128
        elif model == 'densenet62':
            self.res = densenet62()
            _feature_dim = 128
        else:
            raise NotImplementedError

        self.pool = LDE(D, _feature_dim, distance_type=distance_type, network_type=network_type, pooling=pooling, with_bias=False)
        if pooling=='mean':
            self.fc11  = nn.Linear(_feature_dim*D, hidden_dim)
        if pooling=='mean+std':
            self.fc11  = nn.Linear(_feature_dim*2*D, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.asoftmax = asoftmax
        self.m = m
        self.mlambda = [
                lambda x: x**0,
                lambda x: x**1,
                lambda x: 2*x**2-1,
                lambda x: 4*x**3-3*x,
                lambda x: 8*x**4-8*x**2+1,
                lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, x):
        x = self.res(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.fc11(x)
        #print(x.shape)
        x = self.bn1(x)
        if self.asoftmax == 'True':
            # source: https://github.com/clcarwin/sphereface_pytorch
            # AngleLinear class
            w = torch.transpose(self.fc2.weight, 0, 1) # size=(F,Classnum) F=in_features Classnum=out_features
            ww = w.renorm(2,1,1e-5).mul(1e5)
            xlen = x.pow(2).sum(1).pow(0.5) # size=B
            wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
            cos_theta = x.mm(ww) # size=(B,Classnum)

            cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
            cos_theta = cos_theta.clamp(-1,1)
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = torch.cuda.FloatTensor(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
            cos_theta = cos_theta * xlen.view(-1,1)
            phi_theta = phi_theta * xlen.view(-1,1)
            #print(cos_theta.shape, phi_theta.shape)
            return (cos_theta, phi_theta)
    
        else:
            x = F.relu(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=-1)

    def predict(self, x):
        x = self.res(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0]
        x = self.fc11(x)
        return x


class AngleLoss(nn.Module):
    # source: https://github.com/clcarwin/sphereface_pytorch
    # AngleLoss class
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().detach()
        #index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

