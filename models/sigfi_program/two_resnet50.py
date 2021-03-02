# import torchvision.models as models
import torch
import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
#Bottleneck是一个class 里面定义了使用1*1的卷积核进行降维跟升维的一个残差块，可以在github resnet pytorch上查看
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
#不做修改的层不能乱取名字，否则预训练的权重参数无法传入
class CNN50(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(CNN50, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # 新增一个反卷积层
        # self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0,
                                                 # groups=1, bias=False, dilation=1)
        # 新增一个最大池化层
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 去掉原来的fc层，新增一个fclass层
        
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
        self.layer1_p = self._make_layer(block, 64, layers[0])
        self.layer2_p = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_p = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_p = self._make_layer(block, 512, layers[3], stride=2)
        
        self.ca = ChannelAttention(self.inplanes*2)
        self.sa = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.bn1_f = nn.BatchNorm2d(4096)
        self.classifier = nn.Sequential(
        #     # nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(),
              nn.Linear(4096, 2048),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(552, 276),
              nn.ReLU(inplace=True))
        self.fc = nn.Linear(2048, num_classes)
        self.fc_f = nn.Linear(2048, 150)
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
       
        
        x_a = self.avgpool(x_a)
        x_f= self.avgpool(x_f)
        # x_f = F.dropout(x_f, training=self.training)
        x_f = x_f.view(x_f.size(0), -1)
        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.fc(x_a)
        x_f =  self.classifier(x_f) 
        x_f = self.fc_f(x_f)
        # x=torch.cat([x_a,x_f],dim=1)
        
        return x_a,x_f
'''    
#%%

new_file = "two_50.pth"
model_ft = CNN50(Bottleneck, [3, 4, 6, 3])
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
 
 
# def modify_model(pretrained_file, model):
def modify_model(model):   
    # pretrained_dict = torch.load(pretrained_file)
    resnet50 = models.resnet50(pretrained=True)
    pretrained_dict = resnet50.state_dict()
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
            print("rename layer modules:{}-->{}".format(k, kk1))
            print("rename layer modules:{}-->{}".format(k, kk2))
            state_dict[kk1] = v
            state_dict[kk2] = v
    return state_dict
if __name__ == "__main__":
    input_tensor = torch.zeros(1, 3, 100, 100)
    print('input_tensor:', input_tensor.shape)
    # pretrained_file = "resnet18-5c106cde.pth"
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
    new_file = "two_50.pth"
    model = CNN50(Bottleneck, [3, 4, 6, 3])
    # new_model = modify_model(pretrained_file, model)
    new_model = modify_model(model)
    torch.save(new_model.state_dict(), new_file) 


#%%
pretrained_file = "two_50.pth"
pretrained_dict = torch.load(pretrained_file)
for k, v in pretrained_dict.items():
    if(k=='layer3_a.1.bn1.weight'):
        print(v)
#%%
resnet50 = models.resnet50(pretrained=True)
resnet50_dict = resnet50.state_dict()
for j, i in resnet50_dict.items():
    if(j=='layer3.1.bn1.weight'):
        print(i.data)
        #%%
new_file  = "two_50.pth"
model_ft = CNN50(Bottleneck, [3, 4, 6, 3])
pretrained_dict=torch.load(new_file)
model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model_ft.load_state_dict(model_dict)
model_dict = model_ft.state_dict()
o=[]
for k, v in model_dict.items() :
    if(k=='layer3_a.1.bn1.weight'):
        o.append(v)
    if(k=='layer3_p.1.bn1.weight'):
        o.append(v)  
print(o[0]==o[1])
#%%    
# 加载model
resnet50 = models.resnet50(pretrained=True)
#3 4 6 3 分别表示layer1 2 3 4 中Bottleneck模块的数量。res101则为3 4 23 3 
cnn = CNN50(Bottleneck, [3, 4, 6, 3])
# 读取参数
pretrained_dict = resnet50.state_dict()
model_dict = cnn.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
cnn.load_state_dict(model_dict)
# print(resnet50)
print(cnn)'''