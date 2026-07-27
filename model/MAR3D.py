import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
from functools import partial
from torch.nn import MultiheadAttention
class Attention(nn.Module):
    def __init__(self, dim, dim2, plane, heads, dim_head, dropout = 0.):
        super().__init__()
        self.pool_size1 = 8
        self.pool_size2 = 529
        self.dim_head = dim_head
        self.heads = heads
        self.inner_dim = dim_head *  heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.pool_size2)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.plane = plane
        self.pool = nn.MaxPool2d((plane//self.pool_size1, dim//self.pool_size2), return_indices = True)
        self.unpool = nn.MaxUnpool2d((plane//self.pool_size1, dim//self.pool_size2))
        #self.conv1 = nn.Conv3d(plane, plane//4, kernel_size=(self.ks+1,self.ks+1,self.ks+1), stride=(self.ks,self.ks,self.ks))
        #self.unconv = nn.ConvTranspose3d(plane//4, plane, kernel_size=(self.ks+1,self.ks+1,self.ks+1), stride=(self.ks,self.ks,self.ks))
        #self.conv1 = nn.Conv2d(4, 4, kernel_size=(plane//self.pool_size1+1, dim//self.pool_size2+1), stride=(plane//self.pool_size1,dim//self.pool_size2), padding=1)
        #self.unconv = nn.ConvTranspose2d(4, 4, kernel_size=(plane//mathself.pool_size1+1, dim//self.pool_size2+1), stride=(plane//self.pool_size1,dim//self.pool_size2), padding=1)
        self.to_qkv = nn.Linear(self.pool_size2, self.pool_size2 * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(self.pool_size2, self.pool_size2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, length, weight, height = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.view(b,c,-1)
        x = self.norm(x)
        x, indices  = self.pool(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q1 = q.squeeze(dim=0).cpu()
        k1 = k.squeeze(dim=0).cpu()
        v1 = v.squeeze(dim=0).cpu()
        multi_attn = MultiheadAttention(self.dim_head, self.heads)
        out_f, weight_f = multi_attn(q1,k1,v1)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.norm2(out)
        out = self.unpool(out, indices)
        out = out.view(b, c, length, weight, height)
        return out
        
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, imgsize, imgsize2, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.attention1 = Attention(imgsize, imgsize2, planes * 4, heads = 23, dim_head = 23, dropout = 0.1)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        #out += residual
        residual = residual + self.attention1(residual)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, imgsize, imgsize2, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.attention1 = Attention(imgsize, imgsize2, planes * 4, heads = 23, dim_head = 23, dropout = 0.1)
        #self.attention2 = Attention(imgsize, heads = 12, dim_head = 64, dropout = 0.1)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
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
        else:
            residual = x
        residual = residual + self.attention1(residual)
        out += residual
        out = self.relu(out)
        return out


class MAR3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda=False):
        plt.figure(figsize=(12,5))
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(MAR3D, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.imgsize1 = 46*55*46
        self.subimgsize1 = 46
        self.imgsize2 = 23*28*23
        self.subimgsize2 = 23
        self.layer1 = self._make_layer(block, 64, self.imgsize1, self.subimgsize1, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, self.imgsize2, self.subimgsize2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, self.imgsize2, self.subimgsize2, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, self.imgsize2, self.subimgsize2, layers[3], shortcut_type, stride=1, dilation=4)
        #self.layer5 = self._make_layer(
        #    block, 1024, 1, shortcut_type, stride=2)
        self.pooling_layer = nn.AvgPool3d(kernel_size=(23, 28, 23), stride=2, padding=0)
        #self.attention = Attention(2016, heads = 32, dim_head = 64, dropout = 0.1)
        if block == Bottleneck:
            self.output_layer = nn.Linear(2048,1)
        else:
            self.output_layer = nn.Linear(512,1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, imgsize, imgsize2, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, imgsize, imgsize2, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, imgsize, imgsize2, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling_layer(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x


def MAR3D_10(**kwargs):
    model = MAR3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def MAR3D_18(**kwargs):
    model = MAR3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def MAR3D_34(**kwargs):
    model = MAR3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def MAR3D_50(**kwargs):
    model = MAR3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def MAR3D_101(**kwargs):
    model = MAR3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model