import json
import math
import os
import pickle
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch
import transforms as T
import utils
from engine import *
from torch import FloatTensor, Tensor, nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)

COLOR = "yellow"
matplotlib.rcParams["text.color"] = COLOR
