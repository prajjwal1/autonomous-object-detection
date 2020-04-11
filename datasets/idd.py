import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms

import torch
import transforms as T
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch import FloatTensor, Tensor
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.Normalize(mean=(0.3520, 0.3520, 0.3520),std=(0.2930, 0.2930, 0.2930)))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class IDD(torch.utils.data.Dataset):
    def __init__(self, list_img_path, list_anno_path, transforms=None):
        super(IDD, self).__init__()
        self.img = list_img_path
        self.anno = list_anno_path
        self.transforms = transforms
        self.classes = {
            "person": 0,
            "rider": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
            "motorcycle": 5,
            "bicycle": 6,
            "autorickshaw": 7,
            "animal": 8,
            "traffic light": 9,
            "traffic sign": 10,
            "vehicle fallback": 11,
            "caravan": 12,
            "trailer": 13,
            "train": 14,
        }

    def __len__(self):
        return len(self.img)

    def get_height_and_width(self, idx):
        img_path = os.path.join(img_path, self.img[idx])
        img = Image.open(img_path).convert("RGB")
        dim_tensor = torchvision.transforms.ToTensor()(img).shape
        height, width = dim_tensor[1], dim_tensor[2]
        return height, width

    def get_label_bboxes(self, xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(self.classes[object_present])
            bboxes.append((xmin, ymin, xmax, ymax))
        return Tensor(objects), Tensor(bboxes)

    def __getitem__(self, idx):
        img_path = self.img[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.get_label_bboxes(self.anno[idx])[0]
        bboxes = self.get_label_bboxes(self.anno[idx])[1]
        labels = labels.type(torch.int64)
        img_id = Tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        iscrowd = torch.zeros(len(bboxes,), dtype=torch.int64)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
