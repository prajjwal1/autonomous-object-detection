from imports import *
from datasets.idd import *
from tqdm import tqdm
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
###########################        User defined settings ############################
path = '/ml/temp/autonue/data/IDD_Detection/'
######################################################################################
root_anno_path = os.path.join(path,'Annotations','highquality_16k')
root_img_path = os.path.join(path,'JPEGImages','highquality_16k') 

img_id = os.listdir(root_img_path)
anno_id = os.listdir(root_anno_path)

img_idxs = [value for value in img_id if value in anno_id] 
anno_idxs = [value for value in anno_id if value in img_idxs] 

img_paths = []
for i in range(len(img_idxs)):
    img_paths.append(os.path.join(root_img_path,img_idxs[i]))
assert len(img_paths)==len(img_idxs)
total_img_paths = []
for i in tqdm(range(len(img_paths))):
    img_names = os.listdir(img_paths[i])
    for j in range(len(img_names)):
        img_name = os.path.join(img_paths[i],img_names[j])
        total_img_paths.append(img_name)
        
anno_paths = []
for i in range(len(anno_idxs)):
    anno_paths.append(os.path.join(root_anno_path,anno_idxs[i]))
assert len(anno_paths)==len(anno_idxs)
total_anno_paths = []
for i in tqdm(range(len(anno_paths))):
    anno_names = os.listdir(anno_paths[i])
    for j in range(len(anno_names)):
        anno_name = os.path.join(anno_paths[i],anno_names[j])
        #print(img_name)
        total_anno_paths.append(anno_name)

total_img_paths,total_anno_paths = sorted(total_img_paths),sorted(total_anno_paths)
len(total_img_paths),len(total_anno_paths)

###############################################################
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
        objects.append(labels[object_present])
        bboxes.append((xmin,ymin,xmax,ymax))
    return Tensor(objects),Tensor(bboxes)

##############################################################

print("######### Check ############")
print(total_img_paths[100],total_anno_paths[100])

print("Problematic images Found, fixing them")
cnt=0
for i,a in tqdm(enumerate(total_anno_paths)):
    obj_anno_0 = get_obj_bboxes(total_anno_paths[i])
    if not obj_anno_0[0]:
        total_anno_paths.remove(a)
        a = a.replace('Annotations','JPEGImages')
        a = a.replace('xml','jpg')
        total_img_paths.remove(a)
        #print("Problematic", a)
        cnt+=1
        
print('Number of problematic images: '+str(cnt))

#total_img_paths = total_img_paths[:10000]
#total_anno_paths = total_anno_paths[:10000]
print(total_img_paths[2000],total_anno_paths[2000])

assert len(total_anno_paths)==len(total_img_paths)

with open("datalists/idd_hq_images_path_list.txt", "wb") as fp:
    pickle.dump(total_img_paths, fp)

with open("datalists/idd_hq_anno_path_list.txt", "wb") as fp:
    pickle.dump(total_anno_paths, fp)
   
print("Saved successfully", "datalists/idd_hq_images_path_list.txt")