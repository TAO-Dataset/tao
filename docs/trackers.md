# Running trackers on TAO

## SORT

Here, we will reproduce a simpler variant of the SORT result presented in TAO.
Specifically, we will reproduce the following row from Table 13 in our
supplementary material.

| NMS Thresh | Det / image | Det score | `max_age` | `min_hits` | `min_iou` | Track mAP |
| ---------- | ----------- | --------- | --------- | ---------- | --------- | --------- |
| 0.5 | 300 | 0.0005 | 100 | 1 | 0.1 | 11.3 |

### Run detectors

1. Download and decompress the detection model and config from [here](https://drive.google.com/file/d/13BdXSQDqK0t-LrF2CrwJtT9lFc48u83H/view?usp=sharing) or [here](https://cdn3.vision.in.tum.de/~tao/baselines/detector-r101-fpn-1x-lvis-coco.zip) to
   `$DETECTRON_MODEL`.

    If you would like to re-train the detector, please see [this doc](./detector_train.md).

1. Setup and install
   [detectron2](https://github.com/facebookresearch/detectron2)
1. Run the detector on TAO:

    ```
    python scripts/detectors/detectron2_infer.py \
        --gpus 0 1 2 3 \
        --root $TAO_ROOT/train \
        --output /path/to/detectron2/output/train \
        --config $DETECTRON_MODEL/config.yaml \
        --opts MODEL.WEIGHTS $DETECTRON_MODEL/model_final.pth
    ```

    On a machine with 4 2080TIs, the above took about 8 hours to run on the
    train set.

### Run [SORT](https://github.com/abewley/sort)

```
python scripts/trackers/sort/track.py \
    --detections-dir /path/to/detectron2/output/train \
    --annotations $TAO_ROOT/annotations/train.json \
    --output-dir /path/to/sort/output/train \
    --workers 8
```

On our machine, the above took about 11 hours to run on the train set.

### Evaluate

```
python scripts/evaluation/evaluate.py \
    $TAO_ROOT/annotations/train.json \
    /path/to/sort/output/train/results.json
```

This should report an AP of 11.3.

## Single-object trackers

Here we show how to run single-object trackers from the excellent PySOT tracking
repository.

### Setup

1. Download and setup the PySOT repository. This code was tested with PySOT at
    commit
    [052b96](https://github.com/STVIR/pysot/tree/052b9678a7ed336752f74dc6af31cc00eb004551).
    Please follow instructions from the PySOT repository for installation.
2. Ensure `pysot` to your `PYTHONPATH`. You can check that the following import
   works:

    ```bash
    python -c 'from pysot.core.config import cfg'
    ```

### Download model

Download configs and models from the PySOT [model
zoo](https://github.com/STVIR/pysot/blob/052b9678a7ed336752f74dc6af31cc00eb004551/MODEL_ZOO.md).

### Run tracker

```
python scripts/trackers/single_obj/pysot_trackers.py \
        --annotations ${TAO_ROOT}/annotations/train.json \
        --frames-dir ${TAO_ROOT}/train/ \
        --output-dir /path/to/output \
        --config-file /path/to/pysot/repo/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
        --model-path /path/to/pysot/model/siamrpn_r50_l234_dwxcorr_model.pth \
        --gpus 0 1 2 3 \
        --tasks-per-gpu 2
```

### Run tracker with "biggest" init strategy

To run a single object tracker using the "biggest" init, as in Table 5 of [our
paper](https://arxiv.org/pdf/2005.10356.pdf), you can add the `--init biggest`
flag, as shown below:

```
python scripts/trackers/single_obj/pysot_trackers.py \
        --annotations ${TAO_ROOT}/annotations/train.json \
        --frames-dir ${TAO_ROOT}/train/ \
        --output-dir /path/to/output \
        --config-file /path/to/pysot/repo/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
        --model-path /path/to/pysot/model/siamrpn_r50_l234_dwxcorr_model.pth \
        --init biggest \
        --gpus 0 1 2 3 \
        --tasks-per-gpu 2
```

