# Training your own detectors

To train your own detectors, follow the steps below:

1. Download the LVIS v0.5 annotations and (LVIS v0.5 + COCO) training
   annotations from
   [here](https://drive.google.com/file/d/1rPSSIVSer7pweyJS-uqAfIF59uZVJ0Nx/view),
   and extract them to `./data/detectron_datasets/lvis-coco`.

1. Setup [detectron2](https://github.com/facebookresearch/detectron2).

1. Download the COCO `train2017` and `val2017` datasets, and link them to:

    ```
    ./data/detectron_datasets/lvis-coco/train2017
    ./data/detectron_datasets/lvis-coco/val2017
    ```

1. Use the provided `./scripts/detectors/detectron2_train_net.py` script to
   train your detector.

   ```
   python scripts/detectors/detectron2_train_net.py \
       --num-gpus 8 \
       --config-file ./data/configs/LVIS-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml \
       DATASETS.TRAIN "('lvis_v0.5_coco_2017_train', )" \
       OUTPUT_DIR /path/to/output-dir
   ```

   This script was tested with detectron2 commit id
   fd87af71eebc660dde2f50e4693869bb04f66015.

