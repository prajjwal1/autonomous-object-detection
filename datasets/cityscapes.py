import json
import os
import pickle

import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms

import torch
import transforms as T
import utils
from torch import FloatTensor, Tensor
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.dataloader import default_collate
from transforms import *


class Cityscapes(torch.utils.data.Dataset):
    def __init__(
        self, image_path_list, target_path_list, split="train", transforms=None
    ):
        super(Cityscapes, self).__init__()
        self.images = image_path_list
        self.targets = target_path_list
        self.transforms = transforms
        self.classes = {
            "pedestrian": 0,
            "rider": 1,
            "person group": 2,
            "person (other)": 3,
            "sitting person": 4,
            "ignore": 5,
        }

    def get_label_bboxes(self, label):
        """
        Bounding boxes are in the form [x0,y0.w,h]
        """
        bboxes = []
        labels = []
        for data in label["objects"]:
            x0 = data["bbox"][0]
            y0 = data["bbox"][1]
            x1 = x0 + data["bbox"][2]
            y1 = y0 + data["bbox"][3]
            bbox_list = [x0, y0, x1, y1]
            labels.append(self.classes[data["label"]])
            bboxes.append(bbox_list)
        return Tensor(bboxes), Tensor(labels)

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path_list_idx):
        with open(path_list_idx, "r") as file:
            data = json.load(file)
        return data

    def __getitem__(self, idx):

        image = Image.open(self.images[idx]).convert("RGB")

        data = self._load_json(self.targets[idx])

        labels = self.get_label_bboxes(data)[1]
        bboxes = self.get_label_bboxes(data)[0]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros(len(bboxes,), dtype=torch.int64)

        img_id = Tensor([idx])
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
