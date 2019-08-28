# Fast-Invoice
A fast model for multi-scenario multi-class invoices detection (only localization)

## Introduction
This model 

## Quick start
### Install
1. Install PyTorch>=0.4.1 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/wirustea/Fast_Invoice
3. nstall dependencies: pip install -r requirements.txt

## Pretrained models
on added-value tax invoice dataset

| model |num classes |#Params | GFLOPs | Multi-scale | mIoU_for_Seg | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |:--: |
| FastInvoice_Res11 | 64 | - | - | YES | - | - |
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
````bash
from test import Detection
detection = Detection(model_name:str, pretrained_model:str, on_gpu=False)
test.detection.detect(input:numpy.ndarray, visualize:bool)
````

## Train
### Data preparation
Your directory tree should be look like this:
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