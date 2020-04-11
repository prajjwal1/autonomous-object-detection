import json
import os
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch
import transforms as T
import utils
from torch import Tensor, nn
from torch.utils.data import Dataset


def get_ground_truths(train_img_path_list, anno_data):

    bboxes, total_bboxes = [], []
    labels, total_labels = [], []
    classes = {
        "bus": 0,
        "traffic light": 1,
        "traffic sign": 2,
        "person": 3,
        "bike": 4,
        "truck": 5,
        "motor": 6,
        "car": 7,
        "train": 8,
        "rider": 9,
        "drivable area": 10,
        "lane": 11,
    }

    for i in tqdm(range(len(train_img_path_list))):
        for j in range(len(anno_data[i]["labels"])):
            if "box2d" in anno_data[i]["labels"][j]:
                xmin = anno_data[i]["labels"][j]["box2d"]["x1"]
                ymin = anno_data[i]["labels"][j]["box2d"]["y1"]
                xmax = anno_data[i]["labels"][j]["box2d"]["x2"]
                ymax = anno_data[i]["labels"][j]["box2d"]["y2"]
                bbox = [xmin, ymin, xmax, ymax]
                category = anno_data[i]["labels"][j]["category"]
                cls = classes[category]

                bboxes.append(bbox)
                labels.append(cls)

        total_bboxes.append(Tensor(bboxes))
        total_labels.append(Tensor(labels))
        bboxes = []
        labels = []

    return total_bboxes, total_labels


def _load_json(path_list_idx):
    with open(path_list_idx, "r") as file:
        data = json.load(file)
    return data


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class BDD(torch.utils.data.Dataset):
    def __init__(
        self, img_path, anno_json_path, transforms=None
    ):  # total_bboxes_list,total_labels_list,transforms=None):
        super(BDD, self).__init__()
        self.img_path = img_path
        self.anno_data = _load_json(anno_json_path)
        self.total_bboxes_list, self.total_labels_list = get_ground_truths(
            self.img_path, self.anno_data
        )
        self.transforms = transforms
        self.classes = {
            "bus": 0,
            "traffic light": 1,
            "traffic sign": 2,
            "person": 3,
            "bike": 4,
            "truck": 5,
            "motor": 6,
            "car": 7,
            "train": 8,
            "rider": 9,
            "drivable area": 10,
            "lane": 11,
        }

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.total_labels_list[idx]
        bboxes = self.total_bboxes_list[idx]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        img_id = Tensor([idx])
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
