from imports import *

########### User defined setting ##########################
path = '/ml/temp/autonue/data/bdd100k/'
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
    

with open("datalists/bdd100k_train_images_path.txt", "wb") as fp:
    pickle.dump(img_datalist, fp)
    
with open("datalists/bdd100k_val_images_path.txt", "wb") as fp:
    pickle.dump(val_datalist, fp)
    
print("Done")
    
    
    
# path = '/ml/temp/autonue/data/bdd100k/'

# root_img_path = os.path.join(path,'images','100k')
# root_anno_path = os.path.join(path,'labels')

# train_img_path = root_img_path+'/train'
# val_img_path = root_img_path+'/val'
# test_img_path = root_img_path+'/test'

# train_anno_json = root_anno_path+'/bdd100k_labels_images_train.json'
# val_anno_json = root_anno_path+'/bdd100k_labels_images_val.json'

# train_img_ids = os.listdir(train_img_path)
# val_img_ids = os.listdir(val_img_path)

# img_paths = []
# for i in range(len(train_img_ids)):
#     img_paths.append(os.path.join(train_img_path,train_img_ids[i]))
    
# def _load_json(path_list_idx):
#     with open(path_list_idx, 'r') as file:
#         data = json.load(file)
#     return data

# def get_img_id_list_from_json(train_json_file,val_json_file):
#     """
#     Takes in JSON file and returns Image IDs from (train and val) respectively
#     """
#     train_anno_data = _load_json(train_json_file)
#     val_anno_data = _load_json(val_json_file)
    
#     train_anno_img_id_list = []
#     val_anno_img_id_list = []
    
#     for i in tqdm(range(len(train_anno_data))):
#         fname = train_anno_data[i]['name']
#         train_anno_img_id_list.append(fname)
        
#     for i in tqdm(range(len(val_anno_data))):
#         fname = val_anno_data[i]['name']
#         val_anno_img_id_list.append(fname)
        
#     return train_anno_img_id_list,val_anno_img_id_list

# train_anno_img_id_list,val_anno_img_id_list = get_img_id_list_from_json(train_anno_json,val_anno_json)

# def intersection(lst1, lst2): 
#     lst3 = [value for value in lst1 if value in lst2] 
#     return lst3 

# train_img_ids = intersection(train_img_ids,train_anno_img_id_list)

# print("Length of Training Image IDs" , len(train_img_ids))

# train_img_ids,train_anno_img_id_list = sorted(train_img_ids),sorted(train_anno_img_id_list)

# for i in tqdm(range(len(val_anno_img_id_list))):
#     if val_img_ids[i] not in val_anno_img_id_list:
#         val_train_img_ids.remove(val_train_img_ids[i])
# assert len(val_img_ids)==len(val_anno_img_id_list)

# val_img_ids,val_anno_img_id_list = sorted(val_img_ids),sorted(val_anno_img_id_list)

# train_img_paths = []
# for i in range(len(train_img_ids)):
#     train_img_paths.append(os.path.join(train_img_path,train_img_ids[i]))
    
# assert len(train_img_paths)==len(train_anno_img_id_list)

# val_img_paths = []
# for i in range(len(val_img_ids)):
#     val_img_paths.append(os.path.join(val_img_path,val_img_ids[i]))
    
# assert len(val_img_paths)==len(val_anno_img_id_list)