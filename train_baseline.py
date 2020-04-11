import pickle

from cfg import *
from datasets.bdd import *
from datasets.idd import *
from imports import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if ds == "bdd100k":
    root_img_path = os.path.join(bdd_path, "bdd100k", "images", "100k")
    root_anno_path = os.path.join(bdd_path, "bdd100k", "labels")

    train_img_path = root_img_path + "/train/"
    val_img_path = root_img_path + "/val/"

    train_anno_json_path = root_anno_path + "/bdd100k_labels_images_train.json"
    val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

    print("Loading files")

    with open("datalists/bdd100k_train_images_path.txt", "rb") as fp:
        train_img_path_list = pickle.load(fp)
    with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
        val_img_path_list = pickle.load(fp)

    dataset_train = dset = BDD(
        train_img_path_list, train_anno_json_path, get_transform(train=True)
    )
    dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

if ds in ["idd_non_hq", "idd_hq"]:

    with open("datalists/idd_images_path_list.txt", "rb") as fp:
        non_hq_img_paths = pickle.load(fp)
    with open("datalists/idd_anno_path_list.txt", "rb") as fp:
        non_hq_anno_paths = pickle.load(fp)

    if idd_hq == True:
        images = non_hq_img_paths + hq_img_paths
        annos = non_hq_anno_paths + hq_anno_paths
    else:
        images = non_hq_img_paths
        annos = non_hq_anno_paths
    dataset_train = IDD(images, annos, get_transform(train=True))
    dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

print("Loading done")


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )  # replace the pre-trained head with a new one
    return model.cuda()


print("Model initialization")
model = get_model(len(dataset_train.classes))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3)

try:
    os.mkdir("saved_models/")
except:
    pass


if ckpt:
    checkpoint = torch.load("saved_models/sideRight.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# epoch = checkpoint['epoch']


print("Training started")


for epoch in tqdm(range(num_epochs)):
    train_one_epoch(model, optimizer, dl, device, epoch, print_freq=200)
    lr_scheduler.step()

    if epoch == 5 or epoch == 10 or epoch == 15 or epoch == 20 or epoch == 24:
        save_name = "saved_models/bdd100k_" + str(epoch) + ".pth"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),},
            save_name,
        )
        print("Saved model", save_name)
