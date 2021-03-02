import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import collections

__all__ = ['TwoStream_ResNet', 'twostream_resnet18', 'twostream_resnet34', 'twostream_resnet50', \
            'twostream_resnet50_aux', 'twostream_resnet101', 'twostream_resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

class TwoStream_ResNet(nn.Module):

    def __init__(self, block, layers, rgb_channel=3, flow_channel=20, num_classes=1000):
        self.inplanes = 64
        super(TwoStream_ResNet, self).__init__()
        # for rgb
        self.conv1_a = nn.Conv2d(rgb_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_a = nn.BatchNorm2d(64)
        self.relu_a = nn.ReLU(inplace=True)
        self.maxpool_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_a = self._make_layer(block, 64, layers[0])
        self.layer2_a = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_a = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_a = self._make_layer(block, 512, layers[3], stride=2)

        self.ca_a = ChannelAttention(self.inplanes)
        self.sa_a = SpatialAttention()
        # for optical flow x, y 
        self.inplanes = 64
        self.conv1_b = nn.Conv2d(flow_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_b = nn.BatchNorm2d(64)
        self.relu_b = nn.ReLU(inplace=True)
        self.maxpool_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_b = self._make_layer(block, 64, layers[0])
        self.layer2_b = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_b = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_b = self._make_layer(block, 512, layers[3], stride=2)

        self.ca_b = ChannelAttention(self.inplanes)
        self.sa_b = SpatialAttention()
        
        self.bn_f1 = nn.BatchNorm2d(512*2 * block.expansion)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc_aux = nn.Linear(512 * block.expansion, 101)
        self.dp = nn.Dropout(p=0.8)
        self.fc_f = nn.Linear(512*2 * block.expansion, num_classes)
        self.fc_a = nn.Linear(512* block.expansion, num_classes)

        # self.bn_final = nn.BatchNorm1d(512*2 * block.expansion)
        # self.fc2 = nn.Linear(512*2 * block.expansion, 100)
        # self.fc_final = nn.Linear(100, num_classes)

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

    def forward(self, x_a, x_b):
        ## for stream a, rgb, channel 3
        x_a = self.conv1_a(x_a)
        x_a = self.bn1_a(x_a)
        x_a = self.relu_a(x_a)
        x_a = self.maxpool_a(x_a)

        x_a = self.layer1_a(x_a)
        x_a = self.layer2_a(x_a)
        x_a = self.layer3_a(x_a)
        x_a = self.layer4_a(x_a)

        # x_a = self.ca_a(x_a) *  x_a  # attention, not good
        # x_a = self.sa_a(x_a) * x_a   # attention

        ## for stream b, optical flow, channel 20, 10x, 10y
        x_b = self.conv1_b(x_b)
        x_b = self.bn1_b(x_b)
        x_b = self.relu_b(x_b)
        x_b = self.maxpool_b(x_b)

        x_b = self.layer1_b(x_b)
        x_b = self.layer2_b(x_b)
        x_b = self.layer3_b(x_b)
        x_b = self.layer4_b(x_b)
        
        # x_b = self.ca_b(x_b) *  x_b  # attention
        # x_b = self.sa_b(x_b) * x_b   # attention

        ## fusion, concate a, b stream features
        x_f= torch.cat([x_a, x_b],dim=1)
        x_f = self.bn_f1(x_f)

        x_f = self.avgpool(x_f)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = self.dp(x_f)
        x_f = F.softmax(self.fc_f(x_f),dim =1)
        
        x_a = self.avgpool(x_a)
        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.dp(x_a)
        x_a = F.softmax(self.fc_a(x_a),dim =1)
        # print(x_a)

        # x_f = self.bn_final(x_f)
        # x_f = self.fc2(x_f)
        # x_f = F.relu(x_f)
        # x_f = self.fc_final(x_f)

        return x_a, x_f

def gen_two_stream_pretrained(old_params, flow_channel, segments=1):
    print('gen_two_stream_pretrained...segments ', segments)
    new_params_rgb = collections.OrderedDict()
    new_params_flow = collections.OrderedDict()
    
    # for rgb stream
    layer_count = 0
    allKeyList = old_params.keys()
    allKeyList = list(allKeyList)
    for layer_key in allKeyList[:-2]:
        lay= layer_key.split('.', 1)
        new_layer_name = '_a.'.join(lay)
        if layer_count == 0 and segments > 1:
            new_params_rgb[new_layer_name] = old_params[layer_key].repeat(1, segments, 1, 1)
        else:
            new_params_rgb[new_layer_name] = old_params[layer_key]
        layer_count += 1
            
    # for flow stream
    layer_count = 0        
    for layer_key in allKeyList[:-2]:
        lay= layer_key.split('.', 1)
        new_layer_name = '_b.'.join(lay)       
        if layer_count == 0:
            # print('old_params shape', old_params[layer_key].shape)
            rgb_weight = old_params[layer_key]
            # print(type(rgb_weight))
            rgb_weight_mean = torch.mean(rgb_weight, dim=1)
            # TODO: ugly fix here, why torch.mean() turn tensor to Variable
            # print(type(rgb_weight_mean))
            flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, flow_channel, 1, 1)
            new_params_flow[new_layer_name] = flow_weight
            layer_count += 1
            # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
        else:
            new_params_flow[new_layer_name] = old_params[layer_key]
            layer_count += 1
            # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
    
    new_params_rgb.update(new_params_flow)
    return new_params_rgb        
        
def set_freeze_layers(model):
    freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
    for name, param in model.named_parameters():
        param.requires_grad = True 
        for ll in freeze_layers:
            if ll in name.split('.')[0]:
                param.requires_grad = False
                break    

def twostream_resnet18(pretrained=False, rgb_channel=3, flow_channel=20, segments=1, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStream_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet18']))
    return model


def twostream_resnet34(pretrained=False, rgb_channel=3, flow_channel=20, segments=1, train_all=True, **kwargs):
    model = TwoStream_ResNet(BasicBlock, [3, 4, 6, 3], rgb_channel, flow_channel, **kwargs)
    if pretrained:
       
        pretrained_dict = load_state_dict_from_url(model_urls['resnet34'])
        # print("model_zoo pretrained_dict's state_dict:", len(pretrained_dict))
#         print(type(pretrained_dict))
#         print('------'*20)
#         print("pretrained_dict's state_dict:")
#         for param_tensor in pretrained_dict:
#             print(param_tensor, "\t", pretrained_dict[param_tensor].size())
            
        model_dict = model.state_dict()

        new_pretrained_dict = gen_two_stream_pretrained(pretrained_dict, flow_channel, segments)
        # print('------'*20)
        # print("gen_two_stream_pretrained pretrained_dict's state_dict:", len(new_pretrained_dict))
#         for param_tensor in new_pretrained_dict:
#             print(param_tensor, "\t", new_pretrained_dict[param_tensor].size())

        
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # print('------'*20)
        # print("filter out unnecessary keys from new_pretrained_dict:", len(new_pretrained_dict))
#         for i, param_tensor in enumerate(new_pretrained_dict):
#             print(i, param_tensor, "\t", new_pretrained_dict[param_tensor].size())
            
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # print('------'*20)
        # print("update model_dict's state_dict:", len(model_dict))
#         for i, param_tensor in enumerate(model_dict):
#             print(i, param_tensor, "\t", model_dict[param_tensor].size())

        # 3. load the new state dict
#         print(model)
        model.load_state_dict(model_dict)

        if not train_all: 
            set_freeze_layers(model)           

        
    return model


def twostream_resnet50(pretrained=False, rgb_channel=3, flow_channel=20, segments=1, train_all=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStream_ResNet(Bottleneck, [3, 4, 6, 3], rgb_channel, flow_channel, **kwargs)
    if pretrained:
       
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])
        # print("model_zoo pretrained_dict's state_dict:", len(pretrained_dict))
#         print(type(pretrained_dict))
#         print('------'*20)
#         print("pretrained_dict's state_dict:")
#         for param_tensor in pretrained_dict:
#             print(param_tensor, "\t", pretrained_dict[param_tensor].size())
            
        model_dict = model.state_dict()

        new_pretrained_dict = gen_two_stream_pretrained(pretrained_dict, flow_channel, segments)
        # print('------'*20)
        # print("gen_two_stream_pretrained pretrained_dict's state_dict:", len(new_pretrained_dict))
#         for param_tensor in new_pretrained_dict:
#             print(param_tensor, "\t", new_pretrained_dict[param_tensor].size())

        
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # print('------'*20)
        # print("filter out unnecessary keys from new_pretrained_dict:", len(new_pretrained_dict))
#         for i, param_tensor in enumerate(new_pretrained_dict):
#             print(i, param_tensor, "\t", new_pretrained_dict[param_tensor].size())
            
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # print('------'*20)
        # print("update model_dict's state_dict:", len(model_dict))
#         for i, param_tensor in enumerate(model_dict):
#             print(i, param_tensor, "\t", model_dict[param_tensor].size())

        # 3. load the new state dict
#         print(model)
        model.load_state_dict(model_dict)

        if not train_all:
            set_freeze_layers(model)  
        
    return model

def twostream_resnet50_aux(pretrained=False, rgb_channel=3, flow_channel=20, segments=1, train_all=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStream_ResNet(Bottleneck, [3, 4, 6, 3], rgb_channel, flow_channel, **kwargs)
    if pretrained:
        # model.load_state_dict(load_state_dict_from_url(model_urls['resnet50']))
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])

        model_dict = model.state_dict()
        fc_origin_weight = pretrained_dict["fc.weight"].data.numpy()
        fc_origin_bias = pretrained_dict["fc.bias"].data.numpy()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # print(model_dict)
        fc_new_weight = model_dict["fc_aux.weight"].numpy() 
        fc_new_bias = model_dict["fc_aux.bias"].numpy() 

        fc_new_weight[:1000, :] = fc_origin_weight
        fc_new_bias[:1000] = fc_origin_bias

        model_dict["fc_aux.weight"] = torch.from_numpy(fc_new_weight)
        model_dict["fc_aux.bias"] = torch.from_numpy(fc_new_bias)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def twostream_resnet101(pretrained=False, rgb_channel=3, flow_channel=20, train_all=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStream_ResNet(Bottleneck, [3, 4, 23, 3], rgb_channel, flow_channel, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet101']))
    return model


def twostream_resnet152(pretrained=False, rgb_channel=3, flow_channel=20, train_all=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStream_ResNet(Bottleneck, [3, 8, 36, 3], rgb_channel, flow_channel, **kwargs)
    if pretrained:
        # model.load_state_dict(load_state_dict_from_url(model_urls['resnet152']))
        pretrained_dict = load_state_dict_from_url(model_urls['resnet152'])
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model