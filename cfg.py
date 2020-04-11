##########  User specific settings ##########################
idd_path = "/home/jupyter/autonue/data/IDD_Detection/"
bdd_path = "/home/jupyter/autonue/data/bdd100k"
cityscapes_path = "/ml/temp/autonue/data/cityscapes"
cityscapes_split = "train"

idx = 1
batch_size = 8

num_epochs = 25
lr = 0.001
ckpt = False
idd_hq = False
model_name = "bdd100k_24.pth"
##############################################################

dset_list = ["bdd100k", "idd_non_hq", "idd_hq", "Cityscapes"]
ds = dset_list[idx]
