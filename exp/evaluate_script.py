from collections import OrderedDict

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cfg import *
from datasets.bdd import *
from datasets.idd import *
from imports import *

batch_size = 16


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cpu()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    ).cpu()  # replace the pre-trained head with a new one
    return model.cpu()


with open("datalists/idd_val_images_path_list.txt", "rb") as fp:
    val_img_paths = pickle.load(fp)

with open("datalists/idd_val_anno_path_list.txt", "rb") as fp:
    val_anno_paths = pickle.load(fp)
# val_img_paths = val_img_paths[:10]
# val_anno_paths = val_anno_paths[:10]
val_dataset_idd = IDD(val_img_paths, val_anno_paths)
val_dl_idd = torch.utils.data.DataLoader(
    val_dataset_idd,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

root_img_path = os.path.join(bdd_path, "images", "100k")
root_anno_path = os.path.join(bdd_path, "labels")

val_img_path = root_img_path + "/val/"
val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
    bdd_img_path_list = pickle.load(fp)
# bdd_img_path_list = bdd_img_path_list[:10]
val_dataset_bdd = BDD(bdd_img_path_list, val_anno_json_path)
val_dl_bdd = torch.utils.data.DataLoader(
    val_dataset_bdd,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn,
    pin_memory=True,
)

coco_idd = get_coco_api_from_dataset(val_dl_idd.dataset)
coco_bdd = get_coco_api_from_dataset(val_dl_bdd.dataset)


@torch.no_grad()
def evaluate_(model, coco_dset, data_loader, device):
    iou_types = ["bbox"]
    coco = coco_dset
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.to(device)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    to_tensor = torchvision.transforms.ToTensor()
    for image, targets in metric_logger.log_every(data_loader, 100, header):

        image = list(to_tensor(img).to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(image)

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


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


device = torch.device("cuda")

trained_models = [
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_0.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_1.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_2.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_2.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_3.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_4.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_5.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_6.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_7.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_8.pth',
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_9.pth',
    "task_2_1/s_bdd_t_idd_task_new_2_1_epoch_10.pth",
    #                  'task_2_1/s_bdd_t_idd_task_new_2_1_epoch_11.pth'
]

for idx in tqdm(range(0, len(trained_models))):
    model = get_model(15)
    ckpt = torch.load("saved_models/" + trained_models[idx])
    model.load_state_dict(ckpt["model"])

    model.to(device)

    print("##########  Evaluation of IDD  ", "###   IDX  ", trained_models[idx])

    evaluate_(model, coco_idd, val_dl_idd, device=torch.device("cuda"))

    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 12)

    model_bdd = get_model(12)
    checkpoint = torch.load("saved_models/" + "bdd100k_24.pth")
    model_bdd.load_state_dict(checkpoint["model"])

    model.roi_heads.load_state_dict(model_bdd.roi_heads.state_dict())

    model.cuda()

    for n, p in model.named_parameters():
        p.requires_grad = False  # Number of params in RPN = 593935

    for n, p in model.rpn.named_parameters():
        p.requires_grad = True

    for n, p in model.roi_heads.named_parameters():
        p.requires_grad = True  # Number of params in RPN = 593935

    print("##########  Evaluation of BDD  ", "###   IDX  ", trained_models[idx])
    evaluate_(model, coco_bdd, val_dl_bdd, device=torch.device("cuda"))

    del model, model_bdd
