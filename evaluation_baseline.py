import pickle
import time

from cfg import *
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from datasets.bdd import *
from datasets.idd import *
from imports import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading files")

if ds in ["idd_non_hq", "idd_hq"]:
    print("Evaluation on India Driving dataset")
    with open("datalists/idd_images_path_list.txt", "rb") as fp:
        idd_image_path_list = pickle.load(fp)
    with open("datalists/idd_anno_path_list.txt", "rb") as fp:
        idd_anno_path_list = pickle.load(fp)

    val_img_paths = []
    with open(idd_path + "val.txt") as f:
        val_img_paths = f.readlines()
    for i in range(len(val_img_paths)):
        val_img_paths[i] = val_img_paths[i].strip("\n")
        val_img_paths[i] = val_img_paths[i] + ".jpg"
        val_img_paths[i] = os.path.join(idd_path + "JPEGImages", val_img_paths[i])

    val_anno_paths = []
    for i in range(len(val_img_paths)):
        val_anno_paths.append(val_img_paths[i].replace("JPEGImages", "Annotations"))
        val_anno_paths[i] = val_anno_paths[i].replace(".jpg", ".xml")

    val_img_paths, val_anno_paths = sorted(val_img_paths), sorted(val_anno_paths)

    assert len(val_img_paths) == len(val_anno_paths)
    val_img_paths = val_img_paths[:10]
    val_anno_paths = val_anno_paths[:10]

    val_dataset = IDD(val_img_paths, val_anno_paths, None)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

if ds == "bdd100k":
    print("Evaluation on Berkeley Deep Drive")
    root_img_path = os.path.join(bdd_path, "images", "100k")
    root_anno_path = os.path.join(bdd_path, "labels")

    val_img_path = root_img_path + "/val/"
    val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

    with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
        bdd_img_path_list = pickle.load(fp)
    # bdd_img_path_list = bdd_img_path_list[:10]
    val_dataset = BDD(bdd_img_path_list, val_anno_json_path)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

if ds == "Cityscapes":
    with open("datalists/cityscapes_val_images_path.txt", "rb") as fp:
        images = pickle.load(fp)
    with open("datalists/cityscapes_val_targets_path.txt", "rb") as fp:
        targets = pickle.load(fp)

    val_dataset = Cityscapes(images, targets)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

###################################################################################################3


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )  # replace the pre-trained head with a new one
    return model.cuda()


model = get_model(12)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

checkpoint = torch.load("saved_models/" + model_name)
model.load_state_dict(checkpoint["model"])
print("Model Loaded successfully")

print("##### Dataloader is ready #######")


print("Getting coco api from dataset")
coco = get_coco_api_from_dataset(val_dl.dataset)
print("Done")

print("Evaluation in progress")
evaluate(model, val_dl, device=device)
