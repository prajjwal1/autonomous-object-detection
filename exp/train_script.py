from collections import OrderedDict

from cfg import *
from datasets.bdd import *
from datasets.idd import *
from imports import *

batch_size = 16

with open("datalists/idd_images_path_list.txt", "rb") as fp:
    non_hq_img_paths = pickle.load(fp)
with open("datalists/idd_anno_path_list.txt", "rb") as fp:
    non_hq_anno_paths = pickle.load(fp)

with open("datalists/idd_hq_images_path_list.txt", "rb") as fp:
    hq_img_paths = pickle.load(fp)
with open("datalists/idd_hq_anno_path_list.txt", "rb") as fp:
    hq_anno_paths = pickle.load(fp)

trgt_images = non_hq_img_paths  # hq_img_paths
trgt_annos = non_hq_anno_paths  # hq_anno_paths + hq_anno_paths
trgt_dataset = IDD(trgt_images, trgt_annos, get_transform(train=True))
trgt_dl = torch.utils.data.DataLoader(
    trgt_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cpu()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    ).cpu()  # replace the pre-trained head with a new one
    return model.cpu()


model = get_model(15)
ckpt = torch.load("saved_models/task_2_1/s_bdd_t_idd_task_new_2_1_epoch_2.pth")
model.load_state_dict(ckpt["model"])

for n, p in model.backbone.body.named_parameters():
    p.requires_grad = False  # Number of params in RPN = 593935

for n, p in model.rpn.named_parameters():
    p.requires_grad = True

for n, p in model.backbone.fpn.named_parameters():
    p.requires_grad = True

for n, p in model.roi_heads.named_parameters():
    p.requires_grad = True  # Number of params in RPN = 593935

device = torch.device("cuda")
model.to(device)

optimizer = torch.optim.SGD(
    [
        {"params": model.backbone.body.parameters(), "lr": 1e-5},
        {"params": model.backbone.fpn.parameters(), "lr": 2e-4},
        {"params": model.rpn.parameters(), "lr": 4e-4},
        {"params": model.roi_heads.parameters(), "lr": 1e-3},
    ]
)

lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=6e-3)

for epoch in tqdm(range(3, 16)):
    train_one_epoch(model, optimizer, trgt_dl, device, epoch, print_freq=50)

    lr_scheduler.step()

    save_name = (
        "saved_models/task_2_1/s_bdd_t_idd_task_new_2_1_epoch_" + str(epoch) + ".pth"
    )
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),}, save_name
    )
    print("Saved model", save_name)
