# Evaluating Trackers

## Results format

The TAO toolkit expects results in the same format as COCO, but with additional
`track_id` and `video_id` fields. Specifically, `results.json` should have the
following format:

```
[{
    "image_id" : int,
    "category_id" : int,
    "bbox" : [x,y,width,height],
    "score" : float,
    "track_id": int,
    "video_id": int
}]
```


## Evaluation (toolkit)

The TAO toolkit provides code for evaluating tracker results.

```python
import logging
from tao.toolkit.tao import TaoEval

# TAO uses logging to print results. Make sure logging is set to show INFO
# messages, or you won't see any evaluation results.
logging.setLevel(logging.INFO)
tao_eval = TaoEval('/path/to/annotations.json', '/path/to/results.json')
tao_eval.run()
tao_eval.print_results()
```

## Evaluation (command-line)

TAO also comes with a higher-level `evaluate.py` script which incorporates
various additional features for evaluation.

In all the examples below, let -
- `$ANNOTATIONS` be the `/path/to/annotations.json`
- `$RESULTS` be the `/path/to/results.json`
- `$OUTPUT_DIR` be the `/path/to/output/logdir`.

We demonstrate some features below; for more, take a look at the config
description in [`./tao/utils/evaluation.py`](/tao/utils/evaluation.py).

- Simple evaluation, with logging to an output directory

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
    ```

- <details><summary>Classification oracle</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates ORACLE.TYPE class
    ```
    </p>

- <details><summary>Track oracle (for linking detections)</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates ORACLE.TYPE track
    ```
    </p>

- <details><summary>Evaluate MOTA</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates MOTA.ENABLED True
    ```
    </p>

- <details><summary>Evaluate at (3D) IoU threshold of 0.9</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates EVAL_IOUS "[0.9]"
    ```
    </p>

- <details><summary>Evaluate at multiple (3D) IoU thresholds</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates \
            EVAL_IOUS "[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]"
    ```
    </p>

- <details><summary>Category agnostic evaluation</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates CATEGORY_AGNOSTIC True
    ```
    </p>

- <details><summary>Report evaluation by source dataset</summary><p>

    ```bash
    python scripts/evaluate.py \
        $ANNOTATIONS $RESULTS --output-dir $OUTPUT_DIR \
        --config-updates EVAL_BY_DATASET True
    ```
    </p>

## Evaluation (challenge server)

For local evaluation, evaluate with steps above on the released validation
set. When submitting test set results to the
[challenge server](https://motchallenge.net/login/), follow same format for
json files as mentioned above.

The server requires you to submit train, validation and test set results.
We request you to submit these three json files for facilitating progress in
the tracking community. However, if absolutely necessary, submit empty json
files for train and validation. Create a .zip archive that deflates into the
following files

```bash
./TAO_test.json
./TAO_train.json
./TAO_val.json
```
