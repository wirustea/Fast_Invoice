from utils.net import *
import cv2
import numpy as np
import torch
import os
from PIL import Image
import json
import time

### 
mapping = {
    '税额-内容': 1,
 '购买方-开户行及账号-标题': 2,
 '数量-内容': 3,
 '税率': 4,
 '密码区-内容': 5,
 '序号-右侧': 6,
 '货物或应税劳务服务名称': 7,
 '购买方-地址电话-内容': 8,
 '备注': 9,
 '销售方': 10,
 '销售方-开户行及账号-标题': 11,
 '购买方-名称-标题': 12,
 '复核-标题': 13,
 '浙江增值税专用发票': 14,
 '序号': 15,
 '合计-税额-内容': 16,
 '收款人-内容': 17,
 '税率-内容': 18,
 '开票日期-内容': 19,
 '价税合计-大写-标题': 20,
 '规格型号': 21,
 '金额-内容': 22,
 '购买方': 23,
 '价税合计-小写-标题': 24,
 '收款人-标题': 25,
 '价税合计-小写-内容': 26,
 '购买方-地址电话-标题': 27,
 '税额': 28,
 '销售方-名称-标题': 29,
 '价税合计-大写-内容': 30,
 '购买方-纳税人识别号-标题': 31,
 '销售方-纳税人识别号-内容': 32,
 '规格型号-内容': 33,
 '密码区-标题': 34,
 '购买方-纳税人识别号-内容': 35,
 '销售方-纳税人识别号-标题': 36,
 '销售方-名称-内容': 37,
 '开票人-标题': 38,
 '单价-内容': 39,
 '开票人-内容': 40,
 '销售方-开户行及账号-内容': 41,
 '税总函': 42,
 '备注-内容': 43,
 '单位-内容': 44,
 '合计': 45,
 '单价': 46,
 '编号': 47,
 '购买方-名称-内容': 48,
 '开票日期-标题': 49,
 '第三联': 50,
 '数量': 51,
 '发票联': 52,
 '合计-金额-内容': 53,
 '复核-内容': 54,
 '购买方-开户行及账号-内容': 55,
 '销售方-地址电话-标题': 56,
 '销售方-地址电话-内容': 57,
 '校验码': 58,
 '单位': 59,
 '编号-右侧': 60,
 '销售方-章-标题': 61,
 '货物或应税劳务服务名称-内容': 62,
 '金额': 63}
random_color = \
    [[ 52,  71, 253],
    [197, 252,  73],
    [ 97, 138, 221],
    [118, 126,  75],
    [219, 247,  45],
    [122, 133,  85],
    [145, 223, 232],
    [129, 173, 246],
    [171, 179, 238],
    [127, 142, 198],
    [169, 148, 207],
    [132,  61, 176],
    [246, 121, 152],
    [ 91, 155, 212],
    [114, 236,  95],
    [ 72, 127, 192],
    [ 92,  68, 229],
    [128, 106, 102],
    [ 41, 143,  69],
    [174, 193, 145],
    [207, 211, 143],
    [ 42, 196, 109],
    [173, 197, 140],
    [169,  86,  96],
    [ 72,  59, 142],
    [100, 191, 251],
    [173, 159, 234],
    [131, 127, 221],
    [ 86, 185, 150],
    [217, 225, 188],
    [167, 171, 237],
    [126, 241, 187],
    [243, 214,  34],
    [129, 196, 186],
    [ 53,  30, 179],
    [189, 105, 193],
    [174, 179,  77],
    [239,  59,  49],
    [ 78, 237, 218],
    [ 41, 164,  38],
    [191, 177,  79],
    [ 52, 210, 225],
    [215, 210,  48],
    [237,  33,  72],
    [172,  33, 199],
    [163, 157,  46],
    [107, 125,  34],
    [221,  72, 122],
    [203, 171 , 61],
    [102, 123, 217],
    [151, 204, 161],
    [ 71, 209, 134],
    [232,  32,  85],
    [254,  92, 128],
    [ 78, 208, 226],
    [ 53, 155, 109],
    [203, 239,  27],
    [148, 231, 101],
    [235, 151,  60],
    [229, 168, 172],
    [ 35,  44,  61],
    [248, 183, 159],
    [184, 214, 229],
    [121,  59,  45],
    [104, 192, 137],
    [186, 174,  65],
    [174, 191,  15],
    [ 79,  51,  58],
]
mapping = dict(zip(mapping.values(),mapping.keys()))
###

class Detection():
    def __init__(self,model_name:str,pretrained_model:str,on_gpu=False,center_points_thred=0.3,_nms_kernel=3,echo=False,K=110):
        self.net = get_model(model_name,pretrained_model,on_gpu)
        self.on_gpu = on_gpu
        self.center_points_thred= center_points_thred
        self._nms_kernel=_nms_kernel
        self.mapping = mapping
        self.echo = echo
        self.K = K

    def _detect(self,image, draw_rects_on_image=False):
        t0 = time.time()
        output = apply(self.net,image ,on_gpu=self.on_gpu)
        t1 = time.time()
        cls_map, ck_map, dist_map = output['obj'][0], output['ct'][0][1], output['dist'][0]
        dots = center_points(ck_map * (cls_map.argmax(0) != 0).float(), cls_map.argmax(0),thred=self.center_points_thred,topK=self.K,kernel_size=self._nms_kernel)
        
        if self.echo:
            print('%d items found'%(len(dots),))
        # distance to four corners

        bias = dist_map.cpu().detach().numpy()
        rects = [np.ceil(bias[:, i, j].reshape(4, 2) + [j, i]).astype('int') for j, i in dots]

        factor = np.array([image.shape[1]/256,image.shape[0]/256])

        if draw_rects_on_image:
            original_image = image.copy()

        t2 = time.time()

        results = {}
        for rect in rects:
            h0, w0 = rect.min(0)
            h1, w1 = rect.max(0)

            h1 = min(h1, 255)
            w1 = min(w1, 255)
            h0 = max(h0, 0)
            w0 = max(w0, 0)

            canvas = np.zeros([w1 - w0, h1 - h0])
            cv2.fillPoly(canvas, [rect - [h0, w0]], 1)
            cls = ((cls_map[:, w0:w1, h0:h1] * torch.from_numpy(canvas).to(cls_map.device).float()).mean((1, 2))[1:].argmax()).detach().numpy() + 1

            rect = cv2.minAreaRect(np.ceil(rect* factor).astype(int))
            rect = np.int0(cv2.boxPoints(rect))
            
            if not self.mapping[cls]:
                results[self.mapping[cls]] = []
            results[self.mapping[cls]] = rect.tolist()

            if draw_rects_on_image:
                cv2.polylines(original_image, [np.int0(rect)], 1, random_color[cls], int(factor.min()))

        if self.echo:
            print('Time: %.3fs (detection:%.3fs; post-processing:%.3fs)'%(t2-t0,t1-t0,t2-t1))

        if draw_rects_on_image:
            return original_image,results
        else:
            return results

    def detect(self,input:str,visualize:bool):
        if not os.path.exists('result'):
            os.mkdir('result')

        ext = os.path.splitext(input)[-1]
        if len(ext)==0:
            names = os.listdir(input)
            dd = {}
            for name in names:
                if name.split('.')[-1] in ['png','jpg','jpeg','PNG','JPG','JPEG']:
                    path = os.path.join(input,name)
                    image = np.array(Image.open(path).convert('RGB'))
                    if visualize:
                        to_show,bbox = self._detect(image,draw_rects_on_image=True)
                        Image.fromarray(to_show).save(os.path.join('result',name))
                    else:
                        bbox = self._detect(image)
                    dd[name] = bbox
            with open('result/bbox.json','w') as f:
                json.dump(dd,f)

        elif ext in ['.avi','.mp4','.mpeg','.mov']:
            cap = cv2.VideoCapture(input)
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)
            out = cv2.VideoWriter('result/output.mp4', \
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    frame = cv2.resize(frame, size)
                    to_show,bbox = self._detect(frame,draw_rects_on_image=True)
                    out.write(to_show)
            cap.release()
            out.release()

        elif ext in ['.png','.jpg','.jpeg','.PNG','.JPG','.JPEG']:
            name = os.path.join('result',input.split('/')[-1])
            image = np.array(Image.open(input).convert('RGB'))
            if visualize:
                to_show,bbox = self._detect(image,draw_rects_on_image=True)
                Image.fromarray(to_show).save(name)
            else:
                bbox = self._detect(image)

            with open('result/bbox.json','w') as f:
                json.dump({name: bbox},f)

        else: raise Exception
        


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Fast Invoice')

    parser.add_argument('--path', type=str)
    parser.add_argument('--use_gpu', action='store_true',)
    parser.add_argument('--model_name', type=str, default='FastInvoice_Res11')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--visualize',action='store_true',)
    parser.add_argument('--center_points_thred',type=float,default=0.25)
    parser.add_argument('--echo',action='store_true',)
    parser.add_argument('--K',type=int,default=110)
    
    args = parser.parse_args()

    detection = Detection(args.model_name, args.pretrained_model, on_gpu=args.use_gpu,center_points_thred=args.center_points_thred,_nms_kernel=3,echo=args.echo,K=args.K)
    detection.detect(args.path,visualize=args.visualize)