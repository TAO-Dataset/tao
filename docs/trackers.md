# Running trackers on TAO

## Multi-Object Trackers

### Run detectors

1. Download and decompress the detection model and config from [here] to
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
