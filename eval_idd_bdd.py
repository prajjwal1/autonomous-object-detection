# Adapted from torchvision, changes made to support evaluation on idd and bdd100k

import pickle
import time

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from datasets.bdd import *
from datasets.idd import *
from imports import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###########################    User Defined settings ########################
ds = "BDD"
bdd_path = "/home/jupyter/autonue/data/bdd100k/"
batch_size = 8
model_name = "bdd100k_24.pth"
idd_path = "/home/jupyter/autonue/data/IDD_Detection/"
# name = 'do_ft_trained_bdd_eval_idd_ready.pth'
use_checkpoint = False
################################     Dataset and Dataloader Management       ##########################################

print("Loading files")

if ds == "IDD":
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
    #     val_img_paths = val_img_paths[:10]
    #     val_anno_paths = val_anno_paths[:10]

    val_dataset = IDD_Test(val_img_paths, val_anno_paths)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

if ds == "BDD":
    print("Evaluation on Berkeley Deep Drive")
    root_img_path = os.path.join(bdd_path, "images", "100k")
    root_anno_path = os.path.join(bdd_path, "labels")

    val_img_path = root_img_path + "/val/"
    val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

    with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
        bdd_img_path_list = pickle.load(fp)

    val_dataset = BDD(bdd_img_path_list, val_anno_json_path)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

###################################################################################################3


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )  # replace the pre-trained head with a new one
    return model.cuda()


ckpt = torch.load("saved_models/ulm_det_ft0.pth")
model = get_model(15)
model.load_state_dict(ckpt["model"])

model_bdd = get_model(12)
ckpt2 = torch.load("saved_models/bdd100k_24.pth")
model_bdd.load_state_dict(ckpt2["model"])

model.roi_heads = model_bdd.roi_heads
model.roi_heads.load_state_dict(model_bdd.roi_heads.state_dict())

model.cuda()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

if use_checkpoint:
    checkpoint = torch.load("saved_models/" + model_name)
    model.load_state_dict(checkpoint["model"])
    print("Model Loaded successfully")


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


print("##### Dataloader is ready #######")
iou_types = _get_iou_types(model)

print("Getting coco api from dataset")
coco = get_coco_api_from_dataset(val_dl.dataset)
print("Done")


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.cuda()
    # coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # print(image)
        # image = torchvision.transforms.ToTensor()(image[0])  # Returns a scaler tuple
        # print(image.shape)                                # dim of image 1080x1920

        image = torchvision.transforms.ToTensor()(image[0]).to(device)
        # image = img.to(device) for img in image
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()

        outputs = model([image])

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


print("Evaluation in progress")
evaluate(model, val_dl, device=device)
