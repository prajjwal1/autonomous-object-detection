from datasets.cityscapes import *
from tqdm import tqdm
########################      User Defined Settings ############################
path = '/ml/temp/autonue/data/cityscapes'
split='train'
##################################################################################
root = path
images_dir = os.path.join(root, 'images', split)
targets_dir = os.path.join(root, 'bboxes', split)
images_val_dir = os.path.join(root,'images','val')
targets_val_dir = os.path.join(root,'bboxes','val')

images,targets = [],[]
val_images,val_targets = [],[]

print("Images Directory", images_dir)
print("Targets Directory", targets_dir)
print("Validation Images Directory", images_val_dir)
print("Validation Targets Directory", targets_val_dir)


if split not in ['train', 'test', 'val']:
    raise ValueError('Invalid split for mode "fine"! Please use split="train", split="test"'
                             ' or split="val"')
            
if not os.path.isdir(images_dir) or not os.path.isdir(targets_dir):
    raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

#####################  For Training Set ###################################
for city in os.listdir(images_dir):
    img_dir = os.path.join(images_dir, city)
    target_dir = os.path.join(targets_dir, city)
            
    for file_name in os.listdir(img_dir):
        #target_types = []
        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],'gtBboxCityPersons.json')
        targets.append(os.path.join(target_dir, target_name))

        images.append(os.path.join(img_dir, file_name))
        #targets.append(target_types)
        
###################### For Validation Set ##########################

for city in os.listdir(images_val_dir):
    img_val_dir = os.path.join(images_val_dir, city)
    target_val_dir = os.path.join(targets_val_dir, city)
            
    for file_name in os.listdir(img_val_dir):
        #target_types = []
        target_val_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],'gtBboxCityPersons.json')
        val_targets.append(os.path.join(target_val_dir, target_val_name))

        val_images.append(os.path.join(img_val_dir, file_name))
#######################################################################

print("Length of images and targets", len(images),len(targets))
print("Lenght of Validation images and targets",len(val_images),len(val_targets))

images,targets=sorted(images),sorted(targets)
val_images, val_targets = sorted(val_images), sorted(val_targets)

cityscapes_classes = {'pedestrian':0,'rider':1,'person group':2,'person (other)':3,'sitting person':4,'ignore':5}

def _load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def get_label_bboxes(label):
    bboxes = []
    labels = []
    for data in label['objects']:
        bboxes.append(data['bbox'])
        labels.append(cityscapes_classes[data['label']])
    return bboxes,labels

##################################### Fixing annotations with empty labels ########################3
empty_target_paths = []

for i in tqdm(range(2975)):
    data = _load_json(targets[i])
    obj,bbox_coords = get_label_bboxes(data)[0],get_label_bboxes(data)[1]
    if len(bbox_coords)==0:       #Check if the list is empty
        fname = targets[i]
        empty_target_paths.append(fname)
        
print("Length of Empty targets: ", len(empty_target_paths))

img_files_to_remove = []

for i in range(len(empty_target_paths)):
    fname = empty_target_paths[i]
    fname = fname.replace('json','png')
    fname = fname.replace('gtBboxCityPersons','leftImg8bit')
    fname = fname.replace('bboxes','images')
    img_files_to_remove.append(fname)

print("Image files to remove",len(img_files_to_remove))
print(empty_target_paths[0])
print(img_files_to_remove[0])

for i in range(len(empty_target_paths)):
    target_fname = empty_target_paths[i]
    image_fname = img_files_to_remove[i]
    if target_fname in targets:
        targets.remove(target_fname)
    if image_fname in images:
        images.remove(image_fname)
#################################### Validation Set : Fixing annotations ################################
val_target_files_to_remove = []

for i in tqdm(range(500)):
    data = _load_json(val_targets[i])
    obj,bbox_coords = get_label_bboxes(data)[0],get_label_bboxes(data)[1]
    if len(bbox_coords)==0:       #Check if the list is empty
        fname = val_targets[i]
        val_target_files_to_remove.append(fname)
        
print("Length of Empty targets: ", len(val_target_files_to_remove))

val_img_files_to_remove = []

for i in range(len(val_target_files_to_remove)):
    fname = val_target_files_to_remove[i]
    fname = fname.replace('json','png')
    fname = fname.replace('gtBboxCityPersons','leftImg8bit')
    fname = fname.replace('bboxes','images')
    #fname = fname.replace('train','val')
    val_img_files_to_remove.append(fname)

print("Image files to remove",len(val_img_files_to_remove))
print(val_target_files_to_remove[0])
print(val_img_files_to_remove[0], val_images[0])

for i in range(len(val_img_files_to_remove)):
    target_fname = val_target_files_to_remove[i]
    image_fname = val_img_files_to_remove[i]
    
    if image_fname in val_images:
        val_images.remove(image_fname)
        
    if target_fname in val_targets:
        val_targets.remove(target_fname)
        
###############################################################################################################

print("Updated Length",len(images),len(targets))
#assert len(val_images)==len(val_targets)==500
print("Length of Validation set", len(val_images))

with open("datalists/cityscapes_images_path.txt", "wb") as fp:
    pickle.dump(images, fp)
    
with open("datalists/cityscapes_targets_path.txt", "wb") as fp:
    pickle.dump(targets, fp)
    
with open("datalists/cityscapes_val_images_path.txt", "wb") as fp:
    pickle.dump(val_images, fp)
    
with open("datalists/cityscapes_val_targets_path.txt", "wb") as fp:
    pickle.dump(val_targets, fp)
################################################################################################
print("Done")

