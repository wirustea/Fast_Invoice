# Fast-Invoice
A fast&simple model for multi-scenario multi-class invoices detection (only localization)

## Introduction
This model is designed for information localization on **invoice-like images** which have dense, very long, blurred and overlapped text. There were many excellent models for scene text detection (), but they are not very suitable for this high accuracy required and classification task. So we designed our model for various kinds of invoices.
Our model is based on semantic segmentation and center points prediction. It is composed of an Encoder and a Decoder. Encoder is for feature extraction, while Decoder is for pixel classification, center points prediction, and distance estimation. For most data, our model could precisely find center points. So non-maxima suppression for bounding box can be removed. We have provide pretrained models for added-value tax and taxi invoice. Lite models will be released soon.

<!-- ![demo](https://github.com/wirustea/Fast_Invoice/blob/master/demo1.jpg) -->

<!-- <img src="https://github.com/wirustea/Fast_Invoice/blob/master/demo1.jpg" width = "200" height = "200"> -->

<figure class="half">
    <img src="https://github.com/wirustea/Fast_Invoice/blob/master/demo1.jpg" width = "360" height = "221">
</figure>

<!-- <figure class="half">
    <img src="https://github.com/wirustea/Fast_Invoice/blob/master/demo1.jpg" width = "200" height = "200">
    <img src="https://github.com/wirustea/Fast_Invoice/blob/master/demo2.jpg" width = "200" height = "200">
</figure> -->

## Quick start
### Install
- Install PyTorch>=0.4.1 following the [official instructions](https://pytorch.org/)

````bash
git clone https://github.com/wirustea/Fast_Invoice
pip install -r requirements.txt
````

## Pretrained models
on added-value tax invoice dataset

| model |num classes |#Params | GFLOPs | Multi-scale | mIoU_for_Seg | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |:--: |
| FastInvoice_Res11 | 64 | - | - | YES | - | [BaiDuYun (key:ey4g)](https://pan.baidu.com/s/1UKKf_N_uj8suse3lm2L8Mg) |
| Lite-FastInvoice_Res11 | 64 | - | - | YES | - | - |

on multi-invoice(added-value-tax and taxi) dataset

| model |num classes |#Params | GFLOPs | Multi-scale | mIoU_for_Seg | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |:--: |
| FastInvoice_Res18 | 64 | - | - | YES | - | - |

## Test
first download pretrained models, and move to folder **PROJECT_ROOT/pretrained_model**.
````bash
python test.py --path IMAGE_PATH/VIDEO_PATH/FOLDER_PATH --model_name MODEL_NAME --pretrained_model PTH_NAME
````
1. if MODEL_NAME is not given, it will use **FastInvoice_Res11** as default.
2. if gpu is avilable, add option **--use_gpu**
3. visulalize bounding boxes on input, add option **--visualize**
4. you can find bounding boxes and visualized version for every image in folder **PROJECT_ROOT/result**

if you just want to call detection function in projects
````python
from test import Detection
detection = Detection(model_name:str, pretrained_model:str, on_gpu=False)
test.detection.detect(input:numpy.ndarray, visualize:bool)
````

## Train
### Data preparation
Your directory tree and label file(json) should be look like this:
````bash
$PROJECT_ROOT/dataset
├── 512_train
│   ├── IMAGE_NAME_1
│   ├── IMAGE_NAME_2
│   ├── ...
│   ├── label.json 
├── 512_test
│   ├── IMAGE_NAME_1
│   ├── IMAGE_NAME_2
│   ├── ...
│   ├── label.json 
├── mapping.json

label.json
{
    'IMAGE_NAME_1':{
        'TAG_ONG':[
            [[12,45],[68,90],[12,90],[25,68]],
            [[12,45],[68,90],[12,90],[25,68]]
        ],
        'TAG_TWO':[
            [[12,45],[68,90],[12,90],[25,68]],
            [[12,45],[68,90],[12,90],[25,68]]
        ]
    },
}

mapping.json
{
    'TAG_ONG':1, # background is 0
    'TAG_two':2,
    ...
    'TAG_N':N
}
````
### labeling tools
we provide a tool for invoice data labeling .