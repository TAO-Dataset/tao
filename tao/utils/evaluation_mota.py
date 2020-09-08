import csv
import logging
import math
from collections import defaultdict

import motmetrics as mm
from tqdm import tqdm

MOTA_COUNT_FIELDS = {
    'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost',
    'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
    'num_transfer', 'num_ascend', 'num_migrate'
}
MOTA_PERCENT_FIELDS = {
    'idf1', 'idp', 'idr', 'recall', 'precision', 'mota', 'motp'
}


def merge_values(mota_key, mota_values):
    avg = sum(mota_values) / max(len(mota_values), 1e-9)
    if mota_key in MOTA_PERCENT_FIELDS:
        return (avg, -1)
    elif mota_key in MOTA_COUNT_FIELDS:
        return (avg, sum(mota_values))
    else:
        raise ValueError("Unknown key: {key}")


def merged_mota(mota_metrics):
    output = {}
    for key, values in mota_metrics.items():
        values = [x for x in values if not math.isnan(x)]
        if key in MOTA_PERCENT_FIELDS:
            output[key] = sum(values) / max(len(values), 1e-9)
        elif key in MOTA_COUNT_FIELDS:
            output[f'{key}'] = sum(values)
        else:
            raise ValueError(f"Unknown key: {key}")
    return output


def summarize_mota(group_videos, group_accumulators):
    summaries = {}
    for group, videos in tqdm(group_videos.items(), desc='Summarizing'):
        if not videos:
            continue
        metrics_host = mm.metrics.create()
        # MOTA code makes a bunch of unnecessary logs; disable them for now.
        old_level = logging.root.level
        logging.root.setLevel(logging.WARN)
        summaries[group] = metrics_host.compute_many(
            group_accumulators[group],
            metrics=mm.metrics.motchallenge_metrics,
            names=videos,
            generate_overall=True)
        logging.root.setLevel(old_level)
    return summaries


def evaluate_mota(tao_eval, cfg, logger=logging.root):
    track_threshold = cfg.MOTA.TRACK_THRESHOLD
    tao = tao_eval.tao_gt
    results = tao_eval.tao_dt

    seen_categories = {x['category_id'] for x in tao.anns.values()}
    if not cfg.CATEGORIES:
        categories = [
            x['id'] for x in tao.cats.values() if x['id'] in seen_categories
        ]
    else:
        categories = [
            x['id'] for x in tao.cats.values() if x['synset'] in cfg.CATEGORIES
        ]
    # Map category to list of accumulators
    mota_accumulators = defaultdict(list)
    video_ids = sorted(tao.vids.keys())
    valid_videos = defaultdict(list)
    for vid_id in tqdm(video_ids):
        video = tao.vids[vid_id]
        for category in categories:
            acc = mm.MOTAccumulator(auto_id=True)
            has_groundtruth = False
            has_predictions = False
            for image in tao.vid_img_map[video['id']]:
                groundtruth = [
                    x for x in tao.img_ann_map[image['id']]
                    if x['category_id'] == category
                ]
                predictions = [
                    x for x in results.img_ann_map[image['id']]
                    if x['category_id'] == category
                    and float(x['score']) > track_threshold
                ]
                if not groundtruth and not predictions:
                    continue
                if groundtruth:
                    has_groundtruth = True
                if predictions:
                    has_predictions = True
                # IoU is 1 - IoU here. MOT threshold here is IoU 0.5
                distances = mm.distances.iou_matrix(
                    [x['bbox'] for x in groundtruth],
                    [x['bbox'] for x in predictions],
                    max_iou=0.5)
                acc.update([x['track_id'] for x in groundtruth],
                           [x['track_id'] for x in predictions], distances)
            if not has_groundtruth:
                # MOTA is not defined for sequences without a groundtruth.
                if not cfg.MOTA.INCLUDE_NEGATIVE_VIDEOS:
                    continue
                elif not (has_predictions
                          and category in video['neg_category_ids']):
                    continue
            if category in video['not_exhaustive_category_ids']:
                # Remove false positives.
                inds = [
                    i for i, event in enumerate(acc._events)
                    if event[0] != 'FP'
                ]
                acc._indices = [acc._indices[i] for i in inds]
                acc._events = [acc._events[i] for i in inds]
                acc.cached_events_df = (
                    mm.MOTAccumulator.new_event_dataframe_with_data(
                        acc._indices, acc._events))
            valid_videos[category].append(video['name'])
            mota_accumulators[category].append(acc)

    summaries = []
    raw_summaries = {}
    headers = None
    category_summaries = summarize_mota(valid_videos, mota_accumulators)
    for category, summary in category_summaries.items():
        if headers is None:
            headers = summary.columns.values.tolist()
        summaries.append([tao.cats[category]['synset']] +
                         summary.loc['OVERALL'].values.tolist())
        raw_summaries[category] = summary

    merged = merged_mota({
        key: [x[i+1] for x in summaries]
        for i, key in enumerate(headers)
    })

    videos_by_dataset = defaultdict(list)
    for video in tao.vids.values():
        videos_by_dataset[video['metadata']['dataset']].append(video)

    if cfg.MOTA.EVAL_BY_DATASET:
        dataset_overall = {}
        for dataset, videos in tqdm(videos_by_dataset.items(),
                                    desc='Summarizing by dataset'):
            video_names = {v['name'] for v in videos}
            dataset_videos = defaultdict(list)
            dataset_accums = defaultdict(list)
            for c in valid_videos:
                for v, accum in zip(valid_videos[c], mota_accumulators[c]):
                    if v in video_names:
                        dataset_videos[c].append(v)
                        dataset_accums[c].append(accum)
            dataset_summaries = summarize_mota(dataset_videos, dataset_accums)
            mota_metrics_raw = {
                key: x.loc['OVERALL'].values[i+1]
                for x in dataset_summaries.values()
                for i, key in enumerate(headers)
            }
            dataset_overall[dataset] = merged_mota(mota_metrics_raw)
    else:
        dataset_overall = {}

    raw_summaries = {
        tao.cats[c]['synset']: v
        for c, v in raw_summaries.items()
    }
    metrics_headers = []
    for x in headers:
        metrics_headers.append(x)
    return {
        'summary_headers': headers,
        'summaries': summaries,
        'mota_headers': metrics_headers,
        'overall': merged,
        'track_threshold': track_threshold,
        'raw_summaries': summaries,
        'dataset_overall': dataset_overall
    }


def log_mota(eval_info, logger=logging.root, output_dir=None, log_path=None):
    track_threshold = eval_info['mota_eval']['track_threshold']
    headers = eval_info['mota_eval']['summary_headers']
    mota_headers = eval_info['mota_eval']['mota_headers']
    summaries = eval_info['mota_eval']['summaries']
    dataset_overall = eval_info['mota_eval']['dataset_overall']
    # Overall metrics
    overall = eval_info['mota_eval']['overall']

    if output_dir:
        category_headers = ['category'] + mota_headers
        with open(output_dir / 'summary.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=category_headers, restval=-1)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(dict(zip(category_headers, summary)))
            overall_row = {'category': 'OVERALL'}
            overall_row.update(overall)
            writer.writerow(overall_row)

        if dataset_overall:
            dataset_headers = ['dataset'] + mota_headers
            with open(output_dir / 'dataset_summaries.csv', 'w') as f:
                writer = csv.DictWriter(f,
                                        fieldnames=dataset_headers,
                                        restval=-1)
                writer.writeheader()
                for dataset, overall in dataset_overall.items():
                    overall_row = {'dataset': dataset}
                    overall_row.update(overall)
                    writer.writerow(overall_row)

    logger.info('Overall MOTA: %s', overall['mota'])
    first_keys = ['mota', 'idf1']
    ordered_keys = first_keys + [
        x for x in mota_headers[1:] if x not in first_keys
    ]
    log_keys = ['threshold'] + ordered_keys
    str_values = [str(track_threshold)]
    for k in ordered_keys:
        v = overall[k]
        if k in MOTA_PERCENT_FIELDS:
            v_str = f'{100*v:.2f}'
        elif k in MOTA_COUNT_FIELDS:
            v_str = str(int(v))
        str_values.append(v_str)
    if output_dir:
        log_keys += ['path']
        str_values += [log_path if log_path else output_dir]
    logger.info('Copy paste:\n%s\n%s', ','.join(log_keys),
                ','.join(str_values))
