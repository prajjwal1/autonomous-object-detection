## Object detection for autonomous navigation 
This repository provides core support for performing object detection on navigation datasets. Support for 3D object detection and domain adaptation are in experimental phase and will be added later. This project provides support for training, evaluation, inference, visualization.

### This repo also contains the code for:
- [On Generalizing Detection Models for Unconstrained Environments (ICCV W 2019)](https://arxiv.org/abs/1909.13080) in `exp`

#### NEW: Pretrained models are now available

## Prerequisites
- Pytorch >= 1.1
- torchvision >= 0.3
- tensorboardX (optional, required for visualizing)

## Datasets
This work provides support for the following datasets (related to object detection for autonomous navigation):
- [India Driving Dataset](https://idd.insaan.iiit.ac.in/)
- [Berkeley Deep drive](https://bdd-data.berkeley.edu/)
- [Cityscapes](https://www.cityscapes-dataset.com/) 

Directory structure :
```
+-- data
|   +-- bdd100k
|   +-- IDD_Detection
|   +-- cityscapes
+-- autonmous-object-detection
.......
```
### Getting started
1. Download the required dataset
2. Setup dataset paths in `cfg.py`
3. Create datalists
4. Start training and evaluating

## Documentation

### Setting up Config
By default, all paths and hyperparameters are loaded from `cfg.py`. Users are required to specify paths of dataset and hyperparameters once.
This can also be overriden by user 

### Datalists
We use something called datalists. Datalists are lists which contains path to images and labels. This is because some of the images don't have proper labels. Datalists ensure that the lists only contain structured usable data (dataloader would work seamlessly). Data cleaning happens in the process.

You need to specify a proper path and `ds` variable in the `cfg.py` to specify the dataset you want to use.
```
python3 get_datalists.py
```

### Datasets
It assumes that datalists have been created. This step ensures that you won't get bad samples while dataloader iterates. Create a dir named `data` and put all datasets inside it.
This library uses a common API (similar to torchvision). 
All datasets class expect the same inputs:
```
Input:
    idd_image_path_list
    idd_anno_path_list
    get_transform: A transformation function.
```
```
Output:
    A dict containing boxes, labels, image_id, area, iscrowd inside a torch.tensor.
```
- IDD

```
dset = IDD(idd_image_path_list,idd_anno_path_list,transforms=None)
```

- BDD100K 

```
dset = BDD(bdd_img_path_list,train_anno_json_path,transforms=None)
```

BDD100k doesn't provide individual ground truths. A single JSON file is provided. So creating dataset takes a little longer than usual for parsing JSON.

- Cityscapes

```
dset = Cityscapes(image_path_list,target_path_list, split='train',transforms=None)
```

This was tested for Citypersons (GTs for person class). You can extract GTs from segmentation as well, but user would have to manage datalists.

### Transforms
- ```get_transforms(bool:train)```

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

```
$ python train_baseline.py
```

### Evaluation
Evaluation in performed in COCO format. Users need to specify saved `model_name` in `cfg.py`on which evaluation is supposed to occur.

CocoAPI needs to be compiled. first download it from [here](https://github.com/cocodataset/cocoapi)
```
$ cd cocoapi/PythonAPI
$ python setup.py build_ext install
```

Now evaluation can be performed.

```
$ python3 evaluation_baseline.py
```

## Pretrained models
Pretrained Models for IDD and BDD100k are available [here](https://drive.google.com/open?id=1EGMce4aHlo7QpvMsxXgato87gQo8aYrk). For BDD100k, you can straightaway use the model. This model was used to perform incremental learning as mentioned in the paper on IDD. As a result, the base network (model for BDD100k) was reused with new task specific layers to train on IDD. 

## Incremental learning support
Please refer to `exp` directory, jupyter notebooks are self explanatory. Here are the results from the paper.

| S and T                      | Epoch               | Active Components (with LR)                            | LR Range            | map (%) at specified epochs                          |
|------------------------------|---------------------|--------------------------------------------------------|---------------------|------------------------------------------------------|
| <br>BDD -> IDD<br>IDD -> BDD | <br>5<br>Eval       | +ROI Head(1e-3)                                        | <br>1e-3, 6e-3<br>- | <br>24.3<br>45.7                                     |
| BDD -> IDD<br>IDD -> BDD     | <br>5,9<br>Eval     | +RPN (1e-4)<br>+ROI head (1e-3)                        | <br>1e-4, 6e-4<br>- | <br>24.7, 24.9<br><br>45.3, 45.0<br>                 |
| BDD -> IDD<br>IDD -> BDD     | <br>1,5,6,7<br>Eval | <br>+RPN (1e-4)+ROI head (1e-3)                        | <br>1e-4, 6e-3<br>- | <br>24.3, 24.9, 24.9, 25.0<br>45.7, 44.8, 44.7, 44.7 |
| BDD -> IDD<br>IDD -> BDD     | <br>1,5,10<br>Eval  | <br>+ROI head (1e-3)<br><br>+RPN (4e-4) +FPN(2e-4)<br> | <br>1e-4, 6e-3<br>- | <br>24.9, 25.4, 25.9<br><br>45.2, 43.9, 43.3<br>     |

### Inference

Refer to `inference.ipynb` for plotting images with model's predictions.

### Visualization

By default, tensorboard will start logging `loss` and `learning_rate` in `engine.py`. You can start by using
```
$ tensorboard /path/ --port=8888
```

### Example

![img](assets/eval_baseline_idd.png)

### Contribtuion

Feel free to send PRs related to any bugs, support for more datasets etc. 
