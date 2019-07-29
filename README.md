## Domain adaptive Object detection for autonomous navigation 

This repository provides base support for performing object detection. Features for domain adaptation are in experimental phase.
## Prerequisites
- Pytorch >=1.1
- torchvision ==0.3

## Datasets
This work provides support for these datasets related to object detection:
- Cityscapes 
- India Driving Dataset
- Berkeley Deep drive

## Documentation

### Datalists
We use datalists. Datalists are lists which contains path to images and labels. This is because some of the images don't have proper labels. Datalists ensure that the lists only contain structured usable data (dataloader would work seamlessly). Data cleaning happens in the process.
Use these scripts to generate your own datalists. 

You need to specify a proper path in the following scripts and then it would run seamlessly.

For BDD100K, use :
$ python3 get_datalists_bdd100k.py

For IDD, use:
$ python3 get_datalists_idd_non_hq.py   # For non HQ image set
$ python3 get_datalists_idd_hq.py       # For HQ image set

For cityscapes
$ python3 get_datalists_cityscapes.py

### Datasets
It assumes that datalists have been created. This step ensures that you won't get bad samples while dataloader iterates. Create a dir named `data` and put all datasets inside it.
This library uses a common API (similar to torchvision). 
All datasets class expect the same inputs:
Input:
    idd_image_path_list
    idd_anno_path_list
    get_transform: A transformation function.
Output:
    A dict containing boxes, labels, image_id, area, iscrowd

- IDD

`dset = IDD(idd_image_path_list,idd_anno_path_list,transforms=None)`

- BDD100K 

`dset = BDD(bdd_img_path_list,train_anno_json_path,transforms=None)`

BDD100k doesn't provide individual ground truths. A single JSON file is provided. So creating dataset takes a little longer than usual for parsing JSON.

- Cityscapes

`dset = Cityscapes(image_path_list,target_path_list, split='train',transforms=None)`

This was tested for Citypersons (GTs for person class). You can extract GTs from segmentation as well, but user would have to manage datalists.

### Transforms
- `get_transforms(bool:train)` 

converts images into tensors and applies Random Horizontal flipping on input data.

### Model
Any detection model can be used (Yolo,FasterRCNN,SSD). Currently we provide support from torchvision.
```
from train_baseline import get_model
model = get_model(len(classes))    # Returns a Faster RCNN with Resnet 50 as backbone pretrained on COCO.
```

### Training
Support for baseline has been added. Domain adaptive features will be added later.
Users need to specify the path in the script (in user defined settings section) and dataset 

`
$ python train_baseline.py
`

### Evaluation
It performs evaluation in COCO style. Users need to specify saved model path on which evaluation is supposed to occur. CocoAPI needs to be compiled.
`
$ cd cocoapi/PythonAPI
$ python setup.py build_ext install
`

Now evaluation can be performed.

`
$ python3 evaluation_baseline.py
`

### Visualization

Refer to `inference.ipynb` for plotting images with model's predictions.

### Example

![img](assets/eval_baseline_idd.PNG)