# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:38:33 2020

@author: Administrator
"""

# import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

#定义一个3*3的卷积模板，步长为1，并且使用大小为1的zeropadding
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#定义基础模块BasicBlock
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
def conv3x3_(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#定义基础模块BasicBlock
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class BasicBlock_p(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_p, self).__init__()
        self.conv1 = conv3x3_(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_(planes, planes)
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
#不做修改的层不能乱取名字，否则预训练的权重参数无法传入
class ResidualBlock(nn.Module):
     def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
     def forward(self, x):
         y=self.bn1(x)
         y =self.relu(y)
         y = self.conv1(y)
         y =self.bn1(y)
         y = self.conv2(y)
         y+=  x
         y = self.relu(y)
         return y
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)
class CNN(nn.Module):

    def __init__(self, block, block_p,layers, num_classes=1000):
        self.inplanes = 64
        super(CNN, self).__init__()
        self.conv1_a= nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_a = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_a = self._make_layer(block, 64, layers[0])
        self.layer2_a = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_a = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_a = self._make_layer(block, 512, layers[3], stride=2)
        
        self.inplanes = 64
        self.conv1_p = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1_p = nn.BatchNorm2d(64)
        self.relu_p = nn.ReLU(inplace=True)
        self.maxpool_p = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_p = self._make_layer(block_p, 64, layers[0])
        self.layer2_p = self._make_layer(block_p, 128, layers[1], stride=2)
        self.layer3_p = self._make_layer(block_p, 256, layers[2], stride=2)
        self.layer4_p = self._make_layer(block_p, 512, layers[3], stride=2)
        
        # self.conv2vd= nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        # self.conv2vd2= nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=2)
        self.ca = ChannelAttention(self.inplanes*2)
        self.sa = SpatialAttention()
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.bn1_f = nn.BatchNorm2d(1024)
        # self.res = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=3, stride=1,
        #              padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1024, 1024, kernel_size=3, stride=1,
        #              padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
           
        #     )
        
        # self.incep1 = InceptionA(in_channels=1024)
        # self.rblock1 = ResidualBlock(channels=1024)
       # 去掉原来的fc层，新增一个fclass层
        # self.conf=nn.Conv2d( 512,1024, kernel_size=1, stride=1, bias=False)
        # self.confc =nn.Conv2d( 1024,512, kernel_size=1, stride=1, bias=False)
       
        self.classifier = nn.Sequential(
        #     # nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(),
            nn.Linear(1024, 512),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(552, 276),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)
        # self.fc_a = nn.Linear(1024, 276)
        self.fc_f = nn.Linear(512, 276)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
	#这一步用以设置前向传播的顺序，可以自行调整，前提是合理
    def forward(self, x_a,x_p):
        x_a = self.conv1_a(x_a)
        x_a = self.bn1_a(x_a)
        x_a = self.relu(x_a)
        x_a = self.maxpool(x_a)
        x_a = self.layer1_a(x_a)
        x_a = self.layer2_a(x_a)
        x_a = self.layer3_a(x_a)
        x_a = self.layer4_a(x_a)
        
        x_p= self.conv1_p(x_p)
        x_p= self.bn1_p(x_p)
        x_p= self.relu_p(x_p)
        x_p= self.maxpool_p(x_p)        
        x_p= self.layer1_p(x_p)
        x_p= self.layer2_p(x_p)
        x_p= self.layer3_p(x_p)
        x_p= self.layer4_p(x_p)
        # x=torch.add(x_a,x_p)/2
        x_f= torch.cat([x_a,x_p],dim=1)
        x_f = self.bn1_f(x_f)
        x_f = self.ca(x_f) *  x_f
        x_f = self.sa(x_f) * x_f
        
        # x_f = self.rblock1(x_f)
        # x_a = self.conv2vd(x_a)
        x_a = self.avgpool(x_a)
        x_f= self.avgpool(x_f)
        # x_f=self.confc(x_f)
        #x_f = F.dropout(x_f, training=self.training)
        
        # x_a=0.75*x_a+0.25*x_f
       
        x_f = x_f.view(x_f.size(0), -1)
        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.fc(x_a)
        x_f =  self.classifier(x_f) 
        x_f = self.fc_f(x_f)
        # x=torch.cat([x_a,x_f],dim=1)
        
        return x_a,x_f
'''
  #%%

new_file = "new_44.pth"
model_ft = CNN(BasicBlock, BasicBlock_p, [2, 2, 2, 2])
# model_ft.load_state_dict(torch.load(new_file))

pretrained_dict=torch.load(new_file)
model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model_ft.load_state_dict(model_dict)
# print(resnet34)

print(model_ft)
input_tensor = torch.zeros(1, 3, 100, 100)
input_tensor2 = torch.zeros(1, 3, 100, 100)
model_ft.eval()
out,out2 = model_ft(input_tensor,input_tensor2)
print("out:", out.shape, out[0, 0:10])
print("out2:", out2.shape, out2[0, 0:10])

   #%%
 
def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string
 
 
def modify_model(pretrained_file, model):
    
   
   
   
    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    state_dict = modify_state_dict(pretrained_dict, model_dict)
    model.load_state_dict(state_dict)
    return model
 
 
def modify_state_dict(pretrained_dict, model_dict):
       
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            o=str(k).split('.')[0]
            new_string= o +"_a"
            kk1 = string_rename(k, new_string, start=0, end=len(o))
            new_stringp= o +"_p"
            kk2 = string_rename(k, new_stringp, start=0, end=len(o))
            # print("rename layer modules:{}-->{}".format(k, kk1))
            # print("rename layer modules:{}-->{}".format(k, kk2))
            state_dict[kk1] = v
            state_dict[kk2] = v
    return state_dict
if __name__ == "__main__":
    input_tensor = torch.zeros(1, 3, 100, 100)
    print('input_tensor:', input_tensor.shape)
    pretrained_file = "resnet18-5c106cde.pth"
    # model = models.resnet18()
    # model.load_state_dict(torch.load(pretrained_file))
    # model.eval()
    # out = model(input_tensor)
    # print("out:", out.shape, out[0, 0:10])
    #
    # model1 = resnet18()
    # model1 = transfer_model(pretrained_file, model1)
    # out1 = model1(input_tensor)
    # print("out1:", out1.shape, out1[0, 0:10])
    #
    new_file = "new_44.pth"
    model = CNN(BasicBlock, [2, 2, 2, 2])
    new_model = modify_model(pretrained_file, model)
    torch.save(new_model.state_dict(), new_file)  
    #%%

new_file = "new_44.pth"  
model_ft = CNN(BasicBlock,  [2, 2, 2, 2])
model_ft.load_state_dict(torch.load(new_file))   
input_tensor = torch.zeros(16, 3, 224, 224)
input_tensor2 = torch.zeros(16, 3, 224, 224)
model_ft.eval()
out = model_ft(input_tensor,input_tensor2)
print("out:", out.shape, out[0, 0:10])
    #%%

# new_file  = "resnet18-5c106cde.pth"
new_file = "new_44.pth"
model_ft = CNN(BasicBlock,  [2, 2, 2, 2])
# model_ft.load_state_dict(torch.load(new_file))

pretrained_dict=torch.load(new_file)
model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model_ft.load_state_dict(model_dict)
# print(resnet34)

print(model_ft)
input_tensor = torch.zeros(1, 3, 100, 100)
input_tensor2 = torch.zeros(1, 3, 100, 100)
model_ft.eval()
out = model_ft(input_tensor,input_tensor2)
print("out:", out.shape, out[0, 0:10])
 
#%%

import torch
from torchvision import models
import torch.nn as nn
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("-----device:{}".format(device))
    print("-----Pytorch version:{}".format(torch.__version__))
 
    input_tensor = torch.zeros(1, 3, 100, 100)
    print('input_tensor:', input_tensor.shape)
    pretrained_file = "resnet34-333f7ec4.pth"
    model = models.resnet34()
    model.load_state_dict(torch.load(pretrained_file))
    model.eval()
    out = model(input_tensor)
    print("out:", out.shape, out[0, 0:10])'''