# Running trackers on TAO

## SORT

Here, we will reproduce a simpler variant of the SORT result presented in TAO.
Specifically, we will reproduce the following row from Table 13 in our
supplementary material.

| NMS Thresh | Det / image | Det score | `max_age` | `min_hits` | `min_iou` | Track mAP |
| ---------- | ----------- | --------- | --------- | ---------- | --------- | --------- |
| 0.5 | 300 | 0.0005 | 100 | 1 | 0.1 | 11.3 |

### Run detectors

1. Download and decompress the detection model and config from [https://drive.google.com/file/d/13BdXSQDqK0t-LrF2CrwJtT9lFc48u83H/view?usp=sharing] to
   `$DETECTRON_MODEL`.
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
