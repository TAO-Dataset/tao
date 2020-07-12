# Frequently asked questions

1. Why does the training set only contain 216 LVIS categories?

    TAO contains a total of 488 LVIS categories. However, not all categories
    are present in the train, val, and test sets. Instead, we encourage researchers to
    train detectors on the LVIS v0.5 dataset, which contains a superset of
    the 488 categories, and trackers on existing single-object tracking datasets.
    TAO is primarily a benchmark dataset, but we provide a small set of training videos
    for tuning trackers.

1. Why do the LVIS v1 dataset categories not match with the TAO categories?

    Tao was constructed to be aligned with the LVIS v0.5 dataset. The LVIS v1 update
    changes the category names and ids in the LVIS dataset. We are looking into updating
    TAO to use the LVIS v1 categories. For now, you may either train on the LVIS v0.5
    dataset, or construct your own mapping from LVIS v1 categories to TAO categories
    using the 'synset' field.

1. Is there any restriction on which data I can train on?

    The only restriction is that you may not train on videos in the TAO test set.
    You can see a list of videos in the TAO test set from the test set json file
    shared with the annotations. In particular, a number of LaSOT training videos
    are in the TAO test set, and must not be used for training.

    Apart from this, there are currently no restrictions on training datasets.

1. Are only LVIS categories evaluated in TAO?

    Currently (as of July 2020), we are focusing on the LVIS categories within TAO.
    The ECCV challenge will only evaluate on these categories. We intend to formalize
    a protocol for evaluation on the non-LVIS categories later this year.

1. Is there a single-object tracking track in the ECCV '20 challenge?

    Currently, there is no single-object / user-initialized tracking track in
    the challenge. We are looking into ways to host a challenge for user-initialized
    tracking on held out data (e.g., by asking researchers to submit code which we run
    locally on the held out test set). If you have any suggestions or
    feedback, please contact us!