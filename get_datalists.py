from cfg import *
from imports import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if ds == "bdd100k":
    print("Creating datalist for Berkeley Deep Drive")
    root_img_path = os.path.join(bdd_path, "images", "100k")
    root_anno_path = os.path.join(bdd_path, "labels")

    train_img_path = root_img_path + "/train/"
    val_img_path = root_img_path + "/val/"

    train_anno_json = root_anno_path + "/bdd100k_labels_images_train.json"
    val_anno_json = root_anno_path + "/bdd100k_labels_images_val.json"

    def _load_json(path_list_idx):
        with open(path_list_idx, "r") as file:
            data = json.load(file)
        return data

    train_anno_data = _load_json(train_anno_json)

    img_datalist = []
    for i in tqdm(range(len(train_anno_data))):
        img_path = train_img_path + train_anno_data[i]["name"]
        img_datalist.append(img_path)

    val_anno_data = _load_json(val_anno_json)

    val_datalist = []

    for i in range(len(val_anno_data)):
        img_path = val_img_path + val_anno_data[i]["name"]
        val_datalist.append(img_path)

    try:
        os.mkdir("datalists")
    except:
        pass

    with open("datalists/bdd100k_train_images_path.txt", "wb") as fp:
        pickle.dump(img_datalist, fp)

    with open("datalists/bdd100k_val_images_path.txt", "wb") as fp:
        pickle.dump(val_datalist, fp)

    print("Done")

if ds == "idd_non_hq":
    print("Creating datalist for India Driving Dataset (non HQ)")
    ######################################################################################
    root_anno_path = os.path.join(idd_path, "Annotations", "highquality_16k")
    root_img_path = os.path.join(idd_path, "JPEGImages", "highquality_16k")

    img_id = os.listdir(root_img_path)
    anno_id = os.listdir(root_anno_path)

    img_idxs = [value for value in img_id if value in anno_id]
    anno_idxs = [value for value in anno_id if value in img_idxs]

    img_paths = []
    for i in range(len(img_idxs)):
        img_paths.append(os.path.join(root_img_path, img_idxs[i]))
    assert len(img_paths) == len(img_idxs)
    total_img_paths = []
    for i in tqdm(range(len(img_paths))):
        img_names = os.listdir(img_paths[i])
        for j in range(len(img_names)):
            img_name = os.path.join(img_paths[i], img_names[j])
            total_img_paths.append(img_name)

    anno_paths = []
    for i in range(len(anno_idxs)):
        anno_paths.append(os.path.join(root_anno_path, anno_idxs[i]))
    assert len(anno_paths) == len(anno_idxs)
    total_anno_paths = []
    for i in tqdm(range(len(anno_paths))):
        anno_names = os.listdir(anno_paths[i])
        for j in range(len(anno_names)):
            anno_name = os.path.join(anno_paths[i], anno_names[j])
            # print(img_name)
            total_anno_paths.append(anno_name)

    total_img_paths, total_anno_paths = (
        sorted(total_img_paths),
        sorted(total_anno_paths),
    )
    len(total_img_paths), len(total_anno_paths)

    ###############################################################
    def get_obj_bboxes(xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(object_present)
            bboxes.append((xmin, ymin, xmax, ymax))
        return objects, bboxes

    def get_label_bboxes(xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(labels[object_present])
            bboxes.append((xmin, ymin, xmax, ymax))
        return Tensor(objects), Tensor(bboxes)

    ##############################################################

    print("######### Checking ############")
    print(total_img_paths[100], total_anno_paths[100])

    print("Images without annotations found, fixing them")
    cnt = 0
    for i, a in tqdm(enumerate(total_anno_paths)):
        obj_anno_0 = get_obj_bboxes(total_anno_paths[i])
        if not obj_anno_0[0]:
            total_anno_paths.remove(a)
            a = a.replace("Annotations", "JPEGImages")
            a = a.replace("xml", "jpg")
            total_img_paths.remove(a)
            # print("Problematic", a)
            cnt += 1

    print("Total number of images without annotations: " + str(cnt))

    # total_img_paths = total_img_paths[:10000]
    # total_anno_paths = total_anno_paths[:10000]
    print(total_img_paths[2000], total_anno_paths[2000])

    assert len(total_anno_paths) == len(total_img_paths)

    with open("datalists/idd_hq_images_path_list.txt", "wb") as fp:
        pickle.dump(total_img_paths, fp)

    with open("datalists/idd_hq_anno_path_list.txt", "wb") as fp:
        pickle.dump(total_anno_paths, fp)

    print("Saved successfully", "datalists/idd_hq_images_path_list.txt")

if ds == "idd_hq":
    print("Creating datalist for India Driving Dataset (HQ)")
    root_anno_path = os.path.join(idd_path, "Annotations", "highquality_16k")
    root_img_path = os.path.join(idd_path, "JPEGImages", "highquality_16k")

    img_id = os.listdir(root_img_path)
    anno_id = os.listdir(root_anno_path)

    img_idxs = [value for value in img_id if value in anno_id]
    anno_idxs = [value for value in anno_id if value in img_idxs]

    img_paths = []
    for i in range(len(img_idxs)):
        img_paths.append(os.path.join(root_img_path, img_idxs[i]))
    assert len(img_paths) == len(img_idxs)
    total_img_paths = []
    for i in tqdm(range(len(img_paths))):
        img_names = os.listdir(img_paths[i])
        for j in range(len(img_names)):
            img_name = os.path.join(img_paths[i], img_names[j])
            total_img_paths.append(img_name)

    anno_paths = []
    for i in range(len(anno_idxs)):
        anno_paths.append(os.path.join(root_anno_path, anno_idxs[i]))
    assert len(anno_paths) == len(anno_idxs)
    total_anno_paths = []
    for i in tqdm(range(len(anno_paths))):
        anno_names = os.listdir(anno_paths[i])
        for j in range(len(anno_names)):
            anno_name = os.path.join(anno_paths[i], anno_names[j])
            # print(img_name)
            total_anno_paths.append(anno_name)

    total_img_paths, total_anno_paths = (
        sorted(total_img_paths),
        sorted(total_anno_paths),
    )
    len(total_img_paths), len(total_anno_paths)

    ###############################################################
    def get_obj_bboxes(xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(object_present)
            bboxes.append((xmin, ymin, xmax, ymax))
        return objects, bboxes

    def get_label_bboxes(xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(labels[object_present])
            bboxes.append((xmin, ymin, xmax, ymax))
        return Tensor(objects), Tensor(bboxes)

    ##############################################################

    print("######### Checking ############")
    print(total_img_paths[100], total_anno_paths[100])

    print("images without annotations found, fixing them")
    cnt = 0
    for i, a in tqdm(enumerate(total_anno_paths)):
        obj_anno_0 = get_obj_bboxes(total_anno_paths[i])
        if not obj_anno_0[0]:
            total_anno_paths.remove(a)
            a = a.replace("Annotations", "JPEGImages")
            a = a.replace("xml", "jpg")
            total_img_paths.remove(a)
            # print("Problematic", a)
            cnt += 1

    print("Total number of images without annotations: " + str(cnt))

    # total_img_paths = total_img_paths[:10000]
    # total_anno_paths = total_anno_paths[:10000]
    print(total_img_paths[2000], total_anno_paths[2000])

    assert len(total_anno_paths) == len(total_img_paths)

    with open("datalists/idd_hq_images_path_list.txt", "wb") as fp:
        pickle.dump(total_img_paths, fp)

    with open("datalists/idd_hq_anno_path_list.txt", "wb") as fp:
        pickle.dump(total_anno_paths, fp)

    print("Saved successfully", "datalists/idd_hq_images_path_list.txt")

if ds == "Cityscapes":
    root = cityscapes_path
    images_dir = os.path.join(root, "images", cityscapes_split)
    targets_dir = os.path.join(root, "bboxes", cityscapes_split)
    images_val_dir = os.path.join(root, "images", "val")
    targets_val_dir = os.path.join(root, "bboxes", "val")

    images, targets = [], []
    val_images, val_targets = [], []

    print("Images Directory", images_dir)
    print("Targets Directory", targets_dir)
    print("Validation Images Directory", images_val_dir)
    print("Validation Targets Directory", targets_val_dir)

    if split not in ["train", "test", "val"]:
        raise ValueError(
            'Invalid split for mode "fine"! Please use split="train", split="test"'
            ' or split="val"'
        )

    if not os.path.isdir(images_dir) or not os.path.isdir(targets_dir):
        raise RuntimeError(
            "Dataset not found or incomplete. Please make sure all required folders for the"
            ' specified "split" and "mode" are inside the "root" directory'
        )

    #####################  For Training Set ###################################
    for city in os.listdir(images_dir):
        img_dir = os.path.join(images_dir, city)
        target_dir = os.path.join(targets_dir, city)

        for file_name in os.listdir(img_dir):
            # target_types = []
            target_name = "{}_{}".format(
                file_name.split("_leftImg8bit")[0], "gtBboxCityPersons.json"
            )
            targets.append(os.path.join(target_dir, target_name))

            images.append(os.path.join(img_dir, file_name))
            # targets.append(target_types)

    ###################### For Validation Set ##########################

    for city in os.listdir(images_val_dir):
        img_val_dir = os.path.join(images_val_dir, city)
        target_val_dir = os.path.join(targets_val_dir, city)

        for file_name in os.listdir(img_val_dir):
            # target_types = []
            target_val_name = "{}_{}".format(
                file_name.split("_leftImg8bit")[0], "gtBboxCityPersons.json"
            )
            val_targets.append(os.path.join(target_val_dir, target_val_name))

            val_images.append(os.path.join(img_val_dir, file_name))
    #######################################################################

    print("Length of images and targets", len(images), len(targets))
    print("Lenght of Validation images and targets", len(val_images), len(val_targets))

    images, targets = sorted(images), sorted(targets)
    val_images, val_targets = sorted(val_images), sorted(val_targets)

    cityscapes_classes = {
        "pedestrian": 0,
        "rider": 1,
        "person group": 2,
        "person (other)": 3,
        "sitting person": 4,
        "ignore": 5,
    }

    def _load_json(path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def get_label_bboxes(label):
        bboxes = []
        labels = []
        for data in label["objects"]:
            bboxes.append(data["bbox"])
            labels.append(cityscapes_classes[data["label"]])
        return bboxes, labels

    ##################################### Fixing annotations with empty labels ########################3
    empty_target_paths = []

    for i in tqdm(range(2975)):
        data = _load_json(targets[i])
        obj, bbox_coords = get_label_bboxes(data)[0], get_label_bboxes(data)[1]
        if len(bbox_coords) == 0:  # Check if the list is empty
            fname = targets[i]
            empty_target_paths.append(fname)

    print("Length of Empty targets: ", len(empty_target_paths))

    img_files_to_remove = []

    for i in range(len(empty_target_paths)):
        fname = empty_target_paths[i]
        fname = fname.replace("json", "png")
        fname = fname.replace("gtBboxCityPersons", "leftImg8bit")
        fname = fname.replace("bboxes", "images")
        img_files_to_remove.append(fname)

    print("Image files to remove", len(img_files_to_remove))
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
        obj, bbox_coords = get_label_bboxes(data)[0], get_label_bboxes(data)[1]
        if len(bbox_coords) == 0:  # Check if the list is empty
            fname = val_targets[i]
            val_target_files_to_remove.append(fname)

    print("Length of Empty targets: ", len(val_target_files_to_remove))

    val_img_files_to_remove = []

    for i in range(len(val_target_files_to_remove)):
        fname = val_target_files_to_remove[i]
        fname = fname.replace("json", "png")
        fname = fname.replace("gtBboxCityPersons", "leftImg8bit")
        fname = fname.replace("bboxes", "images")
        # fname = fname.replace('train','val')
        val_img_files_to_remove.append(fname)

    print("Image files to remove", len(val_img_files_to_remove))
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

    print("Updated Length", len(images), len(targets))
    # assert len(val_images)==len(val_targets)==500
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
