import os

import torch
import torch.nn as nn
from torch.nn import Upsample


class Residual(nn.Module):
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        # 卷积模块
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs//2,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs//2,3,1,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs,1)
        )
        # 跳层
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs

    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class Lin(nn.Module):
    def __init__(self,numIn,numout):
        super(Lin, self).__init__()
        self.conv = nn.Conv2d(numIn, numout, 1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class HourGlass(nn.Module):
    def __init__(self,n=4, f=256):

        super(HourGlass,self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        # 上分支
        setattr(self,'res'+str(n)+'_1',Residual(f,f))
        # 下分支
        setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
        setattr(self,'res'+str(n)+'_2',Residual(f,f))
        if n > 1:
            self._init_layers(n-1,f)
        else:
            self.res_center = Residual(f,f)
        setattr(self,'res'+str(n)+'_3',Residual(f,f))
        # setattr(self,'SUSN'+str(n),UpsamplingNearest2d(scale_factor=2))
        setattr(self,'SUSN'+str(n),Upsample(scale_factor=2))

    def _forward(self,x,n,f):
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'SUSN'+str(n)).forward(low3)

        return up1+up2

    def forward(self,x):
        return self._forward(x,self._n,self._f)


class StackedHourGlass(nn.Module):
    def __init__(self, nFeats=256, nStack=8, nJoints=18):

        super(StackedHourGlass,self).__init__()
        self._nFeats = nFeats
        self._nStack = nStack
        self._nJoints = nJoints
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.res1 = Residual(64,128)
        self.pool1 = nn.MaxPool2d(2,2)
        self.res2 = Residual(128,128)
        self.res3 = Residual(128,self._nFeats)
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self._nStack):
            setattr(self,'hg'+str(i), HourGlass(4, self._nFeats))
            setattr(self,'hg'+str(i)+'_res1',Residual(self._nFeats,self._nFeats))
            setattr(self,'hg'+str(i)+'_lin1',Lin(self._nFeats,self._nFeats))
            setattr(self, 'hg' + str(i) + '_conv1', nn.Conv2d(self._nFeats, self._nFeats, 1))
            if i < self._nStack - 1:
                setattr(self, 'hg' + str(i) + '_conv_pred', nn.Conv2d(self._nFeats, self._nJoints, 1))
                setattr(self,'hg'+str(i)+'_conv2',nn.Conv2d(self._nJoints,self._nFeats,1))

    def forward(self,x):
        # 初始图像处理
        x = self.relu1(self.bn1(self.conv1(x))) #(n,64,128,128)
        x = self.res1(x)                        #(n,128,128,128)
        x = self.pool1(x)                       #(n,128,64,64)
        x = self.res2(x)                        #(n,128,64,64)
        x = self.res3(x)                        #(n,256,64,64)

        out = []
        inter = x

        for i in range(self._nStack):
            hg = eval('self.hg'+str(i))(inter)
            # Residual layers at output resolution
            ll = hg
            ll = eval('self.hg'+str(i)+'_res1')(ll)
            # Linear layer to produce first set of predictions
            ll = eval('self.hg'+str(i)+'_lin1')(ll)

            # Add predictions back
            if i < self._nStack - 1:
                # Predicted heatmaps
                tmpOut = eval('self.hg' + str(i) + '_conv_pred')(ll)
                out.append(tmpOut)
                # Add predictions back
                ll_ = eval('self.hg'+str(i)+'_conv1')(ll)
                tmpOut_ = eval('self.hg'+str(i)+'_conv2')(tmpOut)
                inter = inter + ll_ + tmpOut_
            else:
                ll_ = eval('self.hg' + str(i) + '_conv1')(ll)
                inter = inter + ll_
        return out, inter

    def init_weights(self, pretrained=None):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if pretrained is not None:
            if os.path.isfile(pretrained):
                pretrained_state_dict = torch.load(pretrained)
                print('=> loading pretrained model {}'.format(pretrained))

                self.load_state_dict(pretrained_state_dict, strict=False)
            elif pretrained:
                print('=> please download pre-trained models first!')
                raise ValueError('{} is not exist!'.format(pretrained))


def Get_Hourglass(d_model, Num_landmarks, Num_block):
    model = StackedHourGlass(nFeats=d_model, nStack=Num_block, nJoints=Num_landmarks)

    model.init_weights(None)

    return model

if __name__ == '__main__':
    model = Get_Hourglass(256, 98, 2)
    input_tensor = torch.rand((2, 3, 256, 256))
    out_tensor = model(input_tensor)
    print(out_tensor[1].size())