import os
import numpy as np
from importlib import import_module
import torch
import torch.nn as nn
import cv2
import math

def restore(net,model_name,on_gpu=True):

    path = os.path.join('pretrained_model',model_name)

    if not on_gpu: pth = torch.load(path, map_location='cpu')
    else: pth = torch.load(path)

    net.load_state_dict(pth['model_state_dict'],strict=False)
    print('load model: '+path)

def get_model(
          model_name=None,
          pretrained_model=None,
          on_gpu=True,
          ):

    # assert os.path.exists(os.path.join('model',model_name)), print(model_name+' not exist')
    Model = import_module('model.'+model_name, model_name)
    net = Model.Net()
    if on_gpu is False: net.cpu()
    if pretrained_model:
        restore(net,pretrained_model,on_gpu=on_gpu)
    return net


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk_channel(scores, K=64):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def center_points(hmap,mask,thred=0.4):

    heat = _nms(hmap.unsqueeze(0).unsqueeze(0), kernel=3)
    # print(heat.size(),mask.size())
    masked_heat = heat * (mask!=0).float()
    scores, inds, ys, xs = _topk_channel(masked_heat, K=70)

    st = 0
    for s in scores[0, 0]:
        if s >= thred:
            st += 1
        else:
            break

    ys = ys[0, 0, :st]
    xs = xs[0, 0, :st]

    dots = np.stack([xs, ys], -1).astype(np.int)

    return dots

def infer(net,image,on_gpu=True):
    if on_gpu:
        net.cuda()
        image = image.cuda()
    else:
        net.cpu()

    net.eval()
    with torch.no_grad():
        predict = net(image)
    return predict

def apply(net,image,resize_to=(512,512),on_gpu=False):

    assert isinstance(image,np.ndarray) and len(image.shape) == 3, \
        print('input image has type '+str(type(image)))
    # image = Image.open(image_name).convert('RGB')
    image = cv2.resize(image, resize_to)
    image = np.array(image, dtype=np.float32) / 255
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    res = infer(net,image,on_gpu=on_gpu)
    return res