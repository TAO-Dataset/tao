from detectron2.data.datasets import register_coco_instances


def register_datasets():
    register_coco_instances(
        "lvis_v0.5_coco_2017_train", {},
        "data/detectron_datasets/lvis-coco/lvis-0.5_coco2017_train.json",
        "data/detectron_datasets/lvis-coco/train2017")
