import pickle,shutil
import torch,os
from pathlib import Path
import xml.etree.ElementTree as ET
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils,math
import transforms as T
from os import path
from tqdm import tqdm
from torch import Tensor
################################            User defined settings      ############################
path = '/home/jupyter/autonue/data/IDD_Detection'
################################################################################################
root_anno_path = os.path.join(path,'Annotations')
root_img_path = os.path.join(path,'JPEGImages')
orientations = ['frontFar','frontNear','rearNear','sideLeft','sideRight']
## Images
##Here we'll first get the paths of images and annotations in list

frontFar_img_dirs = os.listdir(os.path.join(root_img_path,orientations[0]))
frontNear_img_dirs = os.listdir(os.path.join(root_img_path,orientations[1]))
rearNear_img_dirs = os.listdir(os.path.join(root_img_path,orientations[2]))
sideLeft_img_dirs = os.listdir(os.path.join(root_img_path,orientations[3]))
sideRight_img_dirs = os.listdir(os.path.join(root_img_path,orientations[4]))

frontFar_anno_dirs = os.listdir(os.path.join(root_anno_path,orientations[0]))
frontNear_anno_dirs = os.listdir(os.path.join(root_anno_path,orientations[1]))
rearNear_anno_dirs = os.listdir(os.path.join(root_anno_path,orientations[2]))
sideLeft_anno_dirs = os.listdir(os.path.join(root_anno_path,orientations[3]))
sideRight_anno_dirs = os.listdir(os.path.join(root_anno_path,orientations[4]))

#Sanity check
print("FrontFar_imgs_dirs", len(frontFar_img_dirs))
print("frontFar_anno_dirs", len(frontFar_anno_dirs))
print("frontNear_imgs_dirs", len(frontNear_img_dirs))
print("frontNear_anno_dirs", len(frontNear_anno_dirs))
print("rearNear_imgs_dirs", len(rearNear_img_dirs))
print("rearNear_anno_dirs", len(rearNear_anno_dirs))
print("sideLeft_img_dirs",len(sideLeft_img_dirs))
print("sideLeft_anno_dirs",len(sideLeft_anno_dirs))
print("sideRight_img_dirs",len(sideRight_img_dirs))
print("sideRight_anno_dirs",len(sideRight_anno_dirs))

## As evident, the number of images and annotations directories aren't same. So we'll take the intersection of image and directory path
frontFar_img_dirs = list(set(frontFar_img_dirs)&set(frontFar_anno_dirs))
frontFar_anno_dirs = list(set(frontFar_img_dirs)&set(frontFar_anno_dirs))
frontNear_img_dirs = list(set(frontNear_img_dirs)&set(frontNear_anno_dirs))
frontNear_anno_dirs = list(set(frontNear_img_dirs)&set(frontNear_anno_dirs))
rearNear_img_dirs = list(set(rearNear_img_dirs)&set(rearNear_anno_dirs))
rearNear_anno_dirs = list(set(rearNear_img_dirs)&set(rearNear_anno_dirs))
sideLeft_img_dirs = list(set(sideLeft_img_dirs)&set(sideLeft_anno_dirs))
sideLeft_anno_dirs = list(set(sideLeft_img_dirs)&set(sideLeft_anno_dirs))
sideRight_img_dirs = list(set(sideRight_img_dirs)&set(sideRight_anno_dirs))
sideRight_anno_dirs = list(set(sideRight_img_dirs)&set(sideRight_anno_dirs))

# We now ensure that the number of directories of images and annotations are same
assert len(frontFar_img_dirs)==len(frontFar_anno_dirs)
assert len(frontNear_img_dirs),len(frontNear_anno_dirs)
assert len(rearNear_img_dirs),len(rearNear_anno_dirs)
assert len(sideLeft_img_dirs),len(sideLeft_anno_dirs)
assert len(sideRight_img_dirs),len(sideRight_anno_dirs)

frontFar_img_path = []
frontNear_img_path = []
rearNear_img_path = []
sideLeft_img_path = []
sideRight_img_path = []

for i,x in enumerate(frontFar_img_dirs):
    path = os.path.join(root_img_path,orientations[0],frontFar_img_dirs[i])
    for i in os.listdir(path):
        frontFar_img_path.append(os.path.join(path,i))

for i,x in enumerate(frontNear_img_dirs):
    path = os.path.join(root_img_path,orientations[1],frontNear_img_dirs[i])
    for i in os.listdir(path):
        frontNear_img_path.append(os.path.join(path,i))
        
for i,x in enumerate(rearNear_img_dirs):
    path = os.path.join(root_img_path,orientations[2],rearNear_img_dirs[i])
    for i in os.listdir(path):
        rearNear_img_path.append(os.path.join(path,i))
        
for i,x in enumerate(sideLeft_img_dirs):
    path = os.path.join(root_img_path,orientations[3],sideLeft_img_dirs[i])
    for i in os.listdir(path):
        sideLeft_img_path.append(os.path.join(path,i))
        
for i,x in enumerate(sideRight_img_dirs):
    path = os.path.join(root_img_path,orientations[4],sideRight_img_dirs[i])
    for i in os.listdir(path):
        sideRight_img_path.append(os.path.join(path,i))
        
frontFar_img_path = sorted(frontFar_img_path)
frontNear_img_path = sorted(frontNear_img_path)
rearNear_img_path = sorted(rearNear_img_path)
sideLeft_img_path = sorted(sideLeft_img_path)
sideRight_img_path = sorted(sideRight_img_path)

## Processing Annotations 
frontFar_anno_path = []
frontNear_anno_path = []
rearNear_anno_path = []
sideLeft_anno_path = []
sideRight_anno_path = []

for i,x in enumerate(frontFar_anno_dirs):
    path = os.path.join(root_anno_path,orientations[0],frontFar_anno_dirs[i])
    for i in os.listdir(path):
        frontFar_anno_path.append(os.path.join(path,i))

for i,x in enumerate(frontNear_anno_dirs):
    path = os.path.join(root_anno_path,orientations[1],frontNear_anno_dirs[i])
    for i in os.listdir(path):
        frontNear_anno_path.append(os.path.join(path,i))
        
for i,x in enumerate(rearNear_anno_dirs):
    path = os.path.join(root_anno_path,orientations[2],rearNear_anno_dirs[i])
    for i in os.listdir(path):
        rearNear_anno_path.append(os.path.join(path,i))
        
for i,x in enumerate(sideLeft_anno_dirs):
    path = os.path.join(root_anno_path,orientations[3],sideLeft_anno_dirs[i])
    for i in os.listdir(path):
        sideLeft_anno_path.append(os.path.join(path,i))
        
for i,x in enumerate(sideRight_anno_dirs):
    path = os.path.join(root_anno_path,orientations[4],sideRight_anno_dirs[i])
    for i in os.listdir(path):
        sideRight_anno_path.append(os.path.join(path,i))
#######
frontFar_anno_path = sorted(frontFar_anno_path)
frontNear_anno_path = sorted(frontNear_anno_path)
rearNear_anno_path = sorted(rearNear_anno_path)
sideLeft_anno_path = sorted(sideLeft_anno_path)
sideRight_anno_path = sorted(sideRight_anno_path)

len(frontFar_img_path),len(frontNear_img_path) ,len(rearNear_img_path),len(sideLeft_img_path) ,len(sideRight_img_path) 
len(frontFar_anno_path),len(frontNear_anno_path) ,len(rearNear_anno_path),len(sideLeft_anno_path) ,len(sideRight_anno_path) 

classes = {'person':0,'rider':1,'car':2,'truck':3,
         'bus':4,'motorcycle':5,'bicycle':6,'autorickshaw':7,'animal':8,'traffic light':9,
          'traffic sign':10,'vehicle fallback':11,'caravan':12,'trailer':13,'train':14}

def get_obj_bboxes(xml_obj):
    xml_obj = ET.parse(xml_obj)
    objects,bboxes = [],[]
    
    for node in xml_obj.getroot().iter('object'):
        object_present = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)
        objects.append(object_present)
        bboxes.append((xmin,ymin,xmax,ymax))
    return objects,bboxes

def get_label_bboxes(xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects,bboxes = [],[]
    
        for node in xml_obj.getroot().iter('object'):
            object_present = node.find('name').text
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)
            objects.append(classes[object_present])
            bboxes.append((xmin,ymin,xmax,ymax))
        return Tensor(objects),Tensor(bboxes)
    
print("Testing")
## Visualizing
obj_anno_0 = get_obj_bboxes(frontFar_anno_path[100])
obj_anno_0[0],obj_anno_0[1]
print("######################")

## Removing empty stuff
cnt=0
for i,a in tqdm(enumerate(frontFar_anno_path)):
    obj_anno_0 = get_obj_bboxes(frontFar_anno_path[i])
    if not obj_anno_0[0]:
        frontFar_anno_path.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        frontFar_img_path.remove(a)
        cnt+=1
        #print("Problematic", a)
        
for i,a in tqdm(enumerate(frontNear_anno_path)):
    obj_anno_0 = get_obj_bboxes(frontNear_anno_path[i])
    if not obj_anno_0[0]:
        frontNear_anno_path.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        frontNear_img_path.remove(a)
        cnt+=1
        #print("Problematic", a)
        
for i,a in tqdm(enumerate(rearNear_anno_path)):
    obj_anno_0 = get_obj_bboxes(rearNear_anno_path[i])
    if not obj_anno_0[0]:
        rearNear_anno_path.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        rearNear_img_path.remove(a)
        cnt+=1
        #print("Problematic", a)
        
for i,a in tqdm(enumerate(sideLeft_anno_path)):
    obj_anno_0 = get_obj_bboxes(sideLeft_anno_path[i])
    if not obj_anno_0[0]:
        sideLeft_anno_path.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        sideLeft_img_path.remove(a)
        cnt+=1
        #print("Problematic", a)
        
for i,a in tqdm(enumerate(sideRight_anno_path)):
    obj_anno_0 = get_obj_bboxes(sideRight_anno_path[i])
    if not obj_anno_0[0]:
        sideRight_anno_path.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        sideRight_img_path.remove(a)
        cnt+=1
        #print("Problematic", a)
print("Images without annotations ", cnt)
print("Doing sanity check again")
## Matching Images and Annotations
print(frontFar_img_path[100]) 
print(frontFar_anno_path[100])

labels = {'person':0,'rider':1,'car':2,'truck':3,
         'bus':4,'motorcycle':5,'bicycle':6,'autorickshaw':7,'animal':8,'traffic light':9,
          'traffic sign':10,'vehicle fallback':11,'caravan':12,'trailer':13,'train':14}

print("Annotation check")
obj_anno_0 = get_label_bboxes(frontFar_anno_path[128])
obj_anno_0[0],obj_anno_0[1]


print("Creating Dataset path list")

idd_images_path_list = frontFar_img_path+frontNear_img_path+rearNear_img_path+sideLeft_img_path+sideRight_img_path
idd_anno_path_list = frontFar_anno_path+frontNear_anno_path+rearNear_anno_path+sideLeft_anno_path+sideRight_anno_path

with open("datalists/idd_images_path_list.txt", "wb") as fp:
        pickle.dump(idd_images_path_list, fp)

with open("datalists/idd_anno_path_list.txt", "wb") as fp:
    pickle.dump(idd_anno_path_list, fp)


print("Successfully completed")
