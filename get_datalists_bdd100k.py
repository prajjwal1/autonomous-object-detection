from imports import *

########### User defined setting ##########################
path = '/home/jupyter/autonue/data/bdd100k'
###########################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

root_img_path = os.path.join(path,'images','100k')
root_anno_path = os.path.join(path,'labels')

train_img_path = root_img_path+'/train/'
val_img_path = root_img_path+'/val/'

train_anno_json = root_anno_path+'/bdd100k_labels_images_train.json'
val_anno_json = root_anno_path+'/bdd100k_labels_images_val.json'


def _load_json(path_list_idx):
    with open(path_list_idx, 'r') as file:
        data = json.load(file)
    return data

train_anno_data = _load_json(train_anno_json)

img_datalist = []
for i in range(len(train_anno_data)):
    img_path = train_img_path+train_anno_data[i]['name']
    img_datalist.append(img_path)

val_anno_data = _load_json(val_anno_json)
    
val_datalist = []

for i in range(len(val_anno_data)):
    img_path = val_img_path + val_anno_data[i]['name']
    val_datalist.append(img_path)
    
try:
    os.mkdir('datalists')
except:
    pass

with open("datalists/bdd100k_train_images_path.txt", "wb") as fp:
    pickle.dump(img_datalist, fp)
    
with open("datalists/bdd100k_val_images_path.txt", "wb") as fp:
    pickle.dump(val_datalist, fp)
    
print("Done")
    
    
