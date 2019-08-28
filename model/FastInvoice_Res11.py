import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _xe_loss(pred, gt):
    assert pred.size(1)==2,print('pred channel should be 2')
    pred = F.log_softmax(pred, 1)
    pred = pred.permute([1, 0, 2, 3]).contiguous().view([2, -1])
    gt = gt.view(-1)
    loss = -(gt * pred[1] + (1 - gt) * pred[0])

    return loss.mean()

class Xe_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fun = _xe_loss

    def forward(self, out, target):
        return self.loss_fun(out,target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, cfg=None,input_inplane=3, activate_func='relu',first_stride=1):
        assert first_stride in [1,2], print('first stride error, should be one or two')
        if activate_func == 'relu':
            self.activate_func = F.relu

        self.inplanes = 64
        super().__init__()

        if cfg == None:
            cfg = [[64] * layers[0], [128] * layers[1], [256] * layers[2], [512] * layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(input_inplane, 64, kernel_size=5, stride=first_stride, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 0
        self.layer1 = self._make_layer(block, 64, layers[0], cfg[:layers[0]])
        count += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[count:count + layers[1]], stride=2)
        count += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[count:count + layers[2]], stride=2)
        count += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[count:count + layers[3]], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)    # /2
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)  # /2
        x3 = self.layer3(x2)  # /2
        x4 = self.layer4(x3)  # /2
        return (x1,x2,x3,x4,)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2, input_3, input_4):
        x = torch.cat((input_1, input_2, input_3, input_4), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class Decoder(nn.Module):

    def __init__(self, channels):
        super().__init__()
        x1_channel, x2_channel, x3_channel, x4_channel = channels
        self.conv1 = nn.Conv2d(x1_channel, 32*4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(x2_channel, 32*16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(x3_channel, 32*64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(x4_channel, 32*256, 3, stride=1, padding=1)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        self.dp1 = nn.Dropout(0.3,)
        self.dp3 = nn.Dropout(0.3,)
        self.dp4 = nn.Dropout(0.3,)

        self.arm1 = AttentionRefinementModule(64, 64)
        self.arm2 = AttentionRefinementModule(128, 128)
        self.arm3 = AttentionRefinementModule(256,256)
        self.arm4 = AttentionRefinementModule(512, 512)

        self.fusenet = FeatureFusionModule(128,128)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(128, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.conv_ck = nn.Sequential(
            nn.Conv2d(128, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))

        self.dist_conv = nn.Sequential(
            nn.Conv2d(128, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
        )


    def forward(self, xs):
        x1,x2,x3,x4 = xs

        x1 = self.dp1(x1)
        x3 = self.dp3(x3)
        x4 = self.dp4(x4)

        t1 = self.prelu1(self.conv1(self.arm1(x1)))
        t2 = self.prelu2(self.conv2(self.arm2(x2)))
        t3 = self.prelu3(self.conv3(self.arm3(x3)))
        t4 = self.prelu4(self.conv4(self.arm4(x4)))

        # s1 = t1
        s1 = F.pixel_shuffle(t1, upscale_factor=2)
        s2 = F.pixel_shuffle(t2, upscale_factor=4)
        s3 = F.pixel_shuffle(t3, upscale_factor=8)
        s4 = F.pixel_shuffle(t4, upscale_factor=16)

        fusion = self.fusenet(s1,s2,s3,s4)

        cls = self.conv_cls(fusion)
        cks = self.conv_ck(fusion)
        dist = self.dist_conv(fusion)

        output = {'cls':cls,'cks':cks,'dist':dist}
        return output

class Net(nn.Module):
    def __init__(self,cfg=None):
        super().__init__()
        self.backbone = ResNet(BasicBlock,[3,2,4,2],cfg,first_stride=2)
        self.decoder = Decoder([64,128,256,512])

        self.xe_loss = Xe_loss()

    def _compute_logits(self,x):
        xs = self.backbone(x)
        output = self.decoder(xs)
        return output

    def forward(self, x):

        output = self._compute_logits(x)

        salient_map = output['cls']
        ck_map = output['cks']
        dist = output['dist']

        predict_ck_map = F.softmax(ck_map, dim=1)
        predict_obj_map = F.softmax(salient_map, dim=1)
        predict_dist_map = dist

        return {'obj':predict_obj_map,'ct':predict_ck_map,'dist':predict_dist_map}

    def compute(self, x,labels):
        output = self._compute_logits(x)

        salient_map = output['cls']
        ct = output['cks']
        dist = output['dist']

        ck_loss = self.xe_loss(ct,labels['label_ct'])
        obj_loss = F.cross_entropy(salient_map, labels['label_obj'])

        dist_loss = F.smooth_l1_loss(dist,labels['label_dist'],reduction='none')
        dist_mask = (labels['label_ct']!=0).unsqueeze(1).float()

        dist_loss = (dist_loss * dist_mask).sum()/dist_mask.sum()

        loss = obj_loss + ck_loss+ dist_loss

        predict_obj_map = F.softmax(salient_map, dim=1)
        predict_ck_map = F.softmax(ct, dim=1)
        dist = dist.view([dist.size(0),4,2,dist.size(2),dist.size(3)])

        return {'loss':loss, 'obj':predict_obj_map, 'ct':predict_ck_map, 'dist':dist}