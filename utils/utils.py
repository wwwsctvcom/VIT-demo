import torch
import math
import numpy as np
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from functools import partial

CLASSES_PATH = str(Path.cwd() / "data/voc_classes.txt")


def get_cur_time():
    return datetime.now().strftime("%Y_%m_%d")

def get_cur_time_sec():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_classes(classes_path: str):
    with open(Path(classes_path), encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    id2label = {index: class_name for index, class_name in enumerate(class_names)}
    label2id = {class_name: index for index, class_name in enumerate(class_names)}
    return class_names, id2label, label2id


def get_bbox_and_class(xml: str = None) -> list:
    """
    get bbox and class for per bbox
    :param xml:
    :return:
    """
    classes, id2label, label2id = get_classes(CLASSES_PATH)

    bbox_and_cls_all = []
    if xml is None or Path(xml).is_dir():
        raise ValueError("path is None or path is not a dir")
    with open(Path(xml), encoding="utf-8") as _xml_read:
        tree = ET.parse(_xml_read)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = 0
            if obj.find("difficult") != -1:
                difficult = obj.find('difficult').text
            cls = obj.find("name").text
            if cls not in classes or int(difficult) == 1:
                continue
            # get cls and bbox
            cls_id = classes.index(cls)
            xml_bbox = obj.find('bndbox')
            bbox_and_cls = [int(float(xml_bbox.find('xmin').text)),
                            int(float(xml_bbox.find('ymin').text)),
                            int(float(xml_bbox.find('xmax').text)),
                            int(float(xml_bbox.find('ymax').text)),
                            cls_id]
            bbox_and_cls_all.append(bbox_and_cls)

    return bbox_and_cls_all


def load_voc_dataset(voc: str = None, data_type: str = None):
    """
    please input path like: ./data/VOCdevkit/VOC2007
    :param data_type: train, test, val
    :param voc:
    :return:
    """
    train_ds, test_ds, val_ds = [], [], []

    if voc is None or not Path(voc).is_dir():
        raise ValueError("path is None or path is not a dir")

    if data_type == "train":
        with open(Path(voc) / "ImageSets" / "Main" / "train.txt", mode="r", encoding="utf-8") as _train_read:
            for train_index in _train_read.readlines():
                train_index = train_index.strip()
                image_path = str(Path.cwd() / Path(voc) / "JPEGImages" / (train_index + ".jpg"))
                xml_path = str(Path.cwd() / Path(voc) / "Annotations" / (train_index + ".xml"))
                train_ds.append([image_path] + get_bbox_and_class(xml_path))
        return train_ds

    if data_type == "test":
        with open(Path(voc) / "ImageSets" / "Main" / "test.txt", mode="r", encoding="utf-8") as _test_read:
            for test_index in _test_read:
                test_index = test_index.strip()
                image_path = str(Path.cwd() / Path(voc) / "JPEGImages" / (test_index + ".jpg"))
                xml_path = str(Path.cwd() / Path(voc) / "Annotations" / (test_index + ".xml"))
                test_ds.append([image_path] + get_bbox_and_class(xml_path))
        return test_ds

    if data_type == "val":
        with open(Path(voc) / "ImageSets" / "Main" / "val.txt", mode="r", encoding="utf-8") as _val_read:
            for val_index in _val_read:
                val_index = val_index.strip()
                image_path = str(Path.cwd() / Path(voc) / "JPEGImages" / (val_index + ".jpg"))
                xml_path = str(Path.cwd() / Path(voc) / "Annotations" / (val_index + ".xml"))
                val_ds.append([image_path] + get_bbox_and_class(xml_path))
        return val_ds
    return []

def load_cat_dog_dataset(cat_dog_path: str = None, data_type: str = None) -> list:
    """
    cat: 0, dog: 1
    :param cat_dog_path:
    :param data_type: "train" or "test"
    :return: 
    """
    data_ds = []
    file_type = ['.jpg', '.png', '.jpeg']

    # cat
    path_cat = Path(cat_dog_path) / data_type / "cat"
    for image_name in path_cat.iterdir():
        if image_name.suffix not in file_type:
            continue
        image_path = image_name.absolute()
        data_ds.append([image_path, 0])

    # dog
    path_dog = Path(cat_dog_path) / data_type / "dog"
    for image_name in path_dog.iterdir():
        if image_name.suffix not in file_type:
            continue
        image_path = image_name.absolute()
        data_ds.append([image_path, 1])
    return data_ds

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def collate_fn_voc(batch):
    images = []
    labels = []
    for image, label in batch:
        # image [1, 3, w, h]
        images.append(image)
        labels.append(label)
    images = torch.cat(images, 0).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor).long()
    return images, labels

def collate_fn_cat_dog(batch):
    images = []
    labels = []
    for image, label in batch:
        # image [1, 3, w, h]
        images.append(image)
        labels.append(label)
    images = torch.cat(images, 0).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor).long()
    return images, labels

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# if __name__ == "__main__":
#     res = Path.cwd() / (get_cur_time() + f"_epoch_{1}_loss_{2}")
#     print(res)
