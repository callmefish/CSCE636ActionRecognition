import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import collections
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'VGG', 'vgg16', 'vgg16_bn']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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


class ResNet(nn.Module):

    def __init__(self, block, layers, nb_classes=2, channel=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        self.sig = nn.Sigmoid()
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

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        #out = self.sig(out)
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.8),
        )
        # Change the dropout value to 0.9 and 0.8 for flow model
        self.fc_action = nn.Linear(4096, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_make_layers(cfg, channel=20, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(channel, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            channel = v
    return nn.Sequential(*layers)


def vgg_change_key_names(old_params, channel):
    new_params = collections.OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count < len(allKeyList) - 2:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1,channel,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())
        else:
            continue

    return new_params


def vgg16(pretrained=False, channel=20, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(vgg_make_layers(cfg['D'], channel=channel), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        new_pretrained_dict = vgg_change_key_names(pretrained_dict, channel)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def vgg16_bn(pretrained=False, channel=20, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
       No pretrained model available.
    """
    model = VGG(vgg_make_layers(cfg['D'], channel=channel, batch_norm=True), **kwargs)
    return model
    
    
def resnet18(pretrained=False, channel= 20, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes=2, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet18'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=False, channel= 20, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes=2, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet34'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, channel= 20, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes=2, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet50'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, channel= 20, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3],nb_classes=2, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet101'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)

    return model


def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict

#Test network
if __name__ == '__main__':
    model = vgg16(pretrained=True, channel=20)
    print(model)
     
