import copy
import csv
import itertools
import json
import logging
import yaml
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from yacs.config import CfgNode as CN

from tao.toolkit.tao import Tao, TaoEval, TaoResults
from tao.toolkit.tao.eval import bb_intersect_union, compute_track_box_iou
from tao.utils import misc
from tao.utils.yacs_util import cfg_to_flat_dict

# Configuration
_C = CN()
# Categories to evaluate on.
_C.CATEGORIES = []
# YAML containing mapping from category to supercategory. If specified,
# evaluate by supercategory in addition to overall.
_C.SUPERCATEGORY_MAP = ''
# Predictions format. One of 'json', 'mat_dir', 'pkl_dir', 'pickle_dir'
_C.PREDICTIONS_FORMAT = 'json'
# One of '3d_iou', 'avg_iou', 'imagenetvid'
_C.IOU_3D_TYPE = '3d_iou'
# If True, additionally report evaluation by source dataset. Off by default as
# it slows down evaluation.
_C.EVAL_BY_DATASET = False

_C.ORACLE = CN()
# One of 'none', 'random', 'class', 'track', 'track_class'
_C.ORACLE.TYPE = 'none'
# If oracle type is class, whether to remove false positives
_C.ORACLE.REMOVE_FPS = False
_C.ORACLE.IOU_THRESH = 0.5

# Options for evaluating single object trackers
_C.SINGLE_OBJECT = CN()
_C.SINGLE_OBJECT.ENABLED = False
# One of:
# 'unmodified': Leave initial frame score as is in the evaluation.
#     By default, this will be infinity in our implementations.
# 'average': Set initial frame score to be the average of all annotations
#     with score above THRESHOLD. If all annotations are below THRESHOLD,
#     set the value to INIT_SCORE_CONSTANT.
# 'constant': Set initial frame score to be some constant.
# NOTE: Annotations json must contain a field "_single_object_init" for each
# field, indicating whether the annotation was used for init.
_C.SINGLE_OBJECT.INIT_SCORE_TYPE = 'constant'
_C.SINGLE_OBJECT.INIT_SCORE_CONSTANT = 1

_C.MOTA = CN()
_C.MOTA.ENABLED = False
_C.MOTA.TRACK_THRESHOLD = -1.0
# Whether to include negative videos for categories in MOTA. This should be
# true; at some point, this was False by default.
_C.MOTA.INCLUDE_NEGATIVE_VIDEOS = True
# If True, additionally report evaluation by source dataset. Off by default as
# it slows down evaluation.
_C.MOTA.EVAL_BY_DATASET = False

# Track scores are assigned by taking the average of the highest k% of
# annotation scores for each track. By default, we take the average of the
# scores of *all* annotations in a track.
_C.TRACK_SCORE_TOP_PERC = 100
_C.EVAL_IOUS = [0.5]
_C.CATEGORY_AGNOSTIC = False
_C.AREA_RNG = ['all', 'small', 'medium', 'large']
# Split up each detection into its own track. Useful for diagnostics.
_C.SPLIT_TRACKS = False
# If a track has multiple categories, split it into multiple tracks. This is
# useful for SORT runs, where we accidentally assigned a unique track per
# (class, video) pairs, but multiple classes could have a track with the same
# id.
_C.SPLIT_CLASS_TRACKS = False
# Per frame score threshold.
_C.THRESHOLD = -1.0


def verify_config_or_error(cfg):
    assert all(x in {'all', 'small', 'medium', 'large'} for x in cfg.AREA_RNG)
    assert cfg.ORACLE.TYPE in {
        'none', 'random', 'class', 'track', 'track_class'
    }
    assert cfg.PREDICTIONS_FORMAT in {
        'json', 'mat_dir', 'pickle_dir', 'pkl_dir'
    }
    assert cfg.IOU_3D_TYPE in {'3d_iou', 'avg_iou', 'imagenetvid'}
    assert cfg.SINGLE_OBJECT.INIT_SCORE_TYPE in ('unmodified', 'average',
                                                 'constant')


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def fill_video_ids_inplace(anns, tao):
    missing_video_id = [x for x in anns if 'video_id' not in x]
    if missing_video_id:
        image_id_to_video_id = {
            x['id']: x['video_id'] for x in tao.dataset['images']
        }
        for x in missing_video_id:
            x['video_id'] = image_id_to_video_id[x['image_id']]


def set_track_scores_inplace(results, top_k_percent):
    track_anns = defaultdict(list)
    for ann in results:
        track_anns[ann['track_id']].append(ann)
    for track_id, anns in track_anns.items():
        all_scores = np.array([x['score'] for x in anns])
        if top_k_percent == 100:
            score = np.mean(all_scores)
        else:
            k = int(round(top_k_percent / 100 * len(all_scores)))
            k = min(max(k, 1), len(all_scores))
            # Get top k elements
            score = np.mean(all_scores[np.argpartition(all_scores, -k)[-k:]])
        for ann in anns:
            ann['score'] = score


def make_track_ids_unique(result_anns):
    track_id_videos = {}
    track_ids_to_update = set()
    max_track_id = 0
    for ann in result_anns:
        t = ann['track_id']
        if t not in track_id_videos:
            track_id_videos[t] = ann['video_id']

        if ann['video_id'] != track_id_videos[t]:
            # Track id is assigned to multiple videos
            track_ids_to_update.add(t)
        max_track_id = max(max_track_id, t)

    if track_ids_to_update:
        next_id = itertools.count(max_track_id + 1)
        new_track_ids = defaultdict(lambda: next(next_id))
        for ann in result_anns:
            t = ann['track_id']
            v = ann['video_id']
            if t in track_ids_to_update:
                ann['track_id'] = new_track_ids[t, v]
    return len(track_ids_to_update)


def apply_oracle(tao,
                 results,
                 oracle_cfg,
                 category_agnostic,
                 logger=logging.root):
    if oracle_cfg.TYPE in ('track_class', 'track'):
        logger.info(
            'Average track length before oracle track assignment: %s',
            np.mean([
                len(track_ann) for track_ann in results.track_ann_map.values()
            ]))
        oracle_class = oracle_cfg.TYPE == 'track_class'
        # Pairs that can feasibly matched together.
        gt_det_pairs = []
        for img_id in tqdm(tao.imgs, desc='Loading annotations'):
            groundtruth = tao.load_anns(tao.get_ann_ids(img_ids=[img_id]))
            detections = results.load_anns(
                results.get_ann_ids(img_ids=[img_id]))
            if oracle_class or category_agnostic:
                gt_det_pairs.append((groundtruth, detections))
            else:
                # Match per category if we are not using a class oracle
                by_cat = defaultdict(lambda: ([], []))
                for ann in groundtruth:
                    by_cat[ann['category_id']][0].append(ann)
                for ann in detections:
                    by_cat[ann['category_id']][1].append(ann)
                gt_det_pairs.extend(by_cat.values())

        matched_detections = []
        for groundtruth, detections in tqdm(gt_det_pairs):
            if len(detections) == 0 or len(groundtruth) == 0:
                continue
            ious = np.zeros([len(detections), len(groundtruth)])
            for i, j in np.ndindex(ious.shape):
                intersect, union = bb_intersect_union(
                    [float(x) for x in detections[i]['bbox']],
                    [float(x) for x in groundtruth[j]['bbox']])
                ious[i, j] = intersect / max(union, 1e-9)
            dt_matches, gt_matches = linear_sum_assignment(-ious)
            for dt_index, gt_index in zip(dt_matches, gt_matches):
                # Directly modified annotation in results object
                if ious[dt_index, gt_index] < oracle_cfg.IOU_THRESH:
                    continue
                dt = detections[dt_index].copy()
                gt = groundtruth[gt_index]
                dt['track_id'] = gt['track_id']
                if oracle_class:
                    dt['category_id'] = gt['category_id']
                    if oracle_cfg.REMOVE_FPS:
                        dt['score'] = 1.0
                elif category_agnostic:
                    dt['category_id'] = 1
                else:
                    assert dt['category_id'] == gt['category_id']
                matched_detections.append(dt)
        results = TaoResults(tao, matched_detections)
        logger.info(
            'Average track length after oracle track assignment: %s',
            np.mean([
                len(track_ann) for track_ann in results.track_ann_map.values()
            ]))
        logger.info(
            'Number of tracks of length > 1: %s',
            len([x for x in results.track_ann_map.values() if len(x) > 1]))
        logger.info(
            'Average person length after: %s',
            np.mean([
                len(track_ann)
                for track, track_ann in results.track_ann_map.items()
                if results.tracks[track]['category_id'] == 805
            ]))
        return results
    elif oracle_cfg.TYPE == 'class':
        updated_detections = []
        for vid_id in tao.vids:
            gt_tracks = tao.group_ann_tracks(
                tao.load_anns(tao.get_ann_ids(vid_ids=[vid_id])))
            dt_tracks = results.group_ann_tracks(
                results.load_anns(results.get_ann_ids(vid_ids=[vid_id])))
            gt = [{g['image_id']: g['bbox']
                   for g in gt_track['annotations']} for gt_track in gt_tracks]
            dt = [{d['image_id']: d['bbox']
                   for d in dt_track['annotations']} for dt_track in dt_tracks]
            ious = np.zeros([len(dt), len(gt)])
            for i, j in np.ndindex(ious.shape):
                ious[i, j] = compute_track_box_iou(dt[i], gt[j])
            dt_matches, gt_matches = linear_sum_assignment(-ious)

            update_category = {
                dt_index: gt_tracks[gt_index]['category_id']
                for dt_index, gt_index in zip(dt_matches, gt_matches)
                if ious[dt_index, gt_index] > oracle_cfg.IOU_THRESH
            }

            for dt_index in range(len(dt_tracks)):
                track_anns = results.track_ann_map[dt_tracks[dt_index]['id']]
                if dt_index in update_category:
                    for ann in track_anns:
                        ann['category_id'] = update_category[dt_index]
                        if oracle_cfg.REMOVE_FPS:
                            ann['score'] = 1.0
                        updated_detections.append(ann)
                elif not oracle_cfg.REMOVE_FPS:
                    updated_detections.extend(track_anns)
        return TaoResults(tao, updated_detections)


def update_init_scores_inplace(results, single_obj_cfg):
    """Update scores for annotations used for init."""
    init_type = single_obj_cfg.INIT_SCORE_TYPE
    if init_type == 'unmodified':
        pass
    else:
        assert init_type in ('average', 'constant')
        track_ids = set()
        init_frames = defaultdict(list)
        noninit_scores = defaultdict(list)
        for ann in results:
            t = ann['track_id']
            track_ids.add(t)
            if ann['_single_object_init']:
                init_frames[t].append(ann)
            else:
                noninit_scores[t].append(ann['score'])
        missing_inits = set(init_frames.keys()) - track_ids
        if missing_inits:
            raise ValueError(
                f'Could not find any init frames for tracks: {missing_inits}')
        for track, inits in init_frames.items():
            if init_type == 'average' and len(noninit_scores[track]) > 0:
                score = np.mean(noninit_scores[track])
            else:
                score = single_obj_cfg.INIT_SCORE_CONSTANT
            for init in inits:
                init['_single_object_original_score'] = init['score']
                init['score'] = score


def evaluate(annotations, predictions, cfg, logger=logging.root):
    """
    Args:
        annotations (str, Path, or dict)
        predictions (str, Path or dict)
        cfg (ConfigNode)
    """
    logger.info(f'Evaluating predictions at path: {predictions}')
    logger.info(f'Using annotations at path: {annotations}')
    verify_config_or_error(cfg)
    if cfg.SUPERCATEGORY_MAP:
        assert not cfg.CATEGORY_AGNOSTIC, (
            '--category-agnostic is not valid if --supercategory-map is '
            'specified.')
        assert not cfg.CATEGORIES, (
            '--categories cannot be specified if --supercategory-map is '
            'specified.')

    if isinstance(annotations, dict):
        tao = annotations
    else:
        with open(annotations, 'r') as f:
            tao = json.load(f)

    # name_to_id = {x['name']: x['id'] for x in tao['categories']}
    merge_categories = Tao._construct_merge_map(tao)
    assert merge_categories

    for ann in tao['annotations'] + tao['tracks']:
        ann['category_id'] = merge_categories.get(ann['category_id'],
                                                  ann['category_id'])
    tao = Tao(tao)
    if cfg.PREDICTIONS_FORMAT == 'json':
        if isinstance(predictions, dict):
            results = predictions
        else:
            with open(predictions, 'r') as f:
                results = json.load(f)
        for x in results:
            x['score'] = float(x['score'])
        if cfg.THRESHOLD >= 0:
            results = [
                x for x in results if x['score'] >= cfg.THRESHOLD
            ]
    elif cfg.PREDICTIONS_FORMAT in ('mat_dir', 'pickle_dir', 'pkl_dir'):
        detection_format = cfg.PREDICTIONS_FORMAT.split('_')[0]
        results = misc.load_detection_dir_as_results(
            predictions,
            tao.dataset,
            score_threshold=cfg.THRESHOLD,
            detections_format=detection_format,
            show_progress=True)

    invalid_images = {
        x['image_id']
        for x in results if x['image_id'] not in tao.imgs
    }
    if invalid_images:
        logger.warning(f'Found invalid image ids: {invalid_images}')
        results = [x for x in results if x['image_id'] not in invalid_images]

    if cfg.CATEGORY_AGNOSTIC:
        for x in results:
            x['category_id'] = 1

    if cfg.SPLIT_CLASS_TRACKS:
        track_id_gen = itertools.count(1)
        unique_track_ids = defaultdict(lambda: next(track_id_gen))
        for x in results:
            x['track_id'] = unique_track_ids[(x['track_id'], x['category_id'])]

    if cfg.SPLIT_TRACKS:
        last_track_id = itertools.count(
            max([x['track_id'] for x in tao.anns.values()]) + 1)
        for x in results:
            x['track_id'] = next(last_track_id)

    for x in results:
        x['category_id'] = merge_categories.get(x['category_id'],
                                                x['category_id'])

    fill_video_ids_inplace(results, tao)

    if cfg.SINGLE_OBJECT.ENABLED:
        update_init_scores_inplace(results, cfg.SINGLE_OBJECT)

    num_updated_tracks = make_track_ids_unique(results)
    if num_updated_tracks:
        logger.info(
            f'Updating {num_updated_tracks} track ids to make them unique.')
    set_track_scores_inplace(results, cfg.TRACK_SCORE_TOP_PERC)

    results = TaoResults(tao, results)
    if cfg.ORACLE.TYPE != 'none':
        results = apply_oracle(tao,
                               results,
                               cfg.ORACLE,
                               cfg.CATEGORY_AGNOSTIC,
                               logger=logger)
    tao_eval = TaoEval(tao, results, iou_3d_type=cfg.IOU_3D_TYPE)
    if cfg.CATEGORY_AGNOSTIC:
        tao_eval.params.use_cats = 0
    if cfg.CATEGORIES:
        if cfg.CATEGORY_AGNOSTIC:
            raise ValueError(
                '--categories and --category-agnostic are mutually exclusive')
        cat_synset_to_id = {x['synset']: x['id'] for x in tao.cats.values()}
        cat_ids = []
        for x in cfg.CATEGORIES:
            if x not in cat_synset_to_id:
                raise ValueError(
                    f'Could not find category synset {x} (specified from '
                    f'--categories)')
            cat_ids.append(cat_synset_to_id[x])
        tao_eval.params.cat_ids = cat_ids

    tao_eval.params.area_rng = [
        x
        for x, l in zip(tao_eval.params.area_rng, tao_eval.params.area_rng_lbl)
        if l in cfg.AREA_RNG
    ]
    tao_eval.params.area_rng_lbl = cfg.AREA_RNG
    tao_eval.params.iou_thrs = cfg.EVAL_IOUS
    tao_eval.run()

    eval_info = {'tao_eval': tao_eval}
    if cfg.MOTA.ENABLED:
        from .evaluation_mota import evaluate_mota
        mota_info = evaluate_mota(tao_eval, cfg, logger)
        eval_info['mota_eval'] = mota_info
    return eval_info


def log_eval(eval_info,
             cfg,
             logger=logging.root,
             output_dir=None,
             log_path=None):
    tao_eval = eval_info['tao_eval']
    tao = tao_eval.tao_gt
    tao_eval.print_results()
    if log_path is None:
        log_path = output_dir if output_dir is not None else ''

    eval_keys = [
        ('AP', 'AP'),
        (('AP', 'time', 'short', 300), 'AP-short'),
        (('AP', 'time', 'medium', 300), 'AP-med'),
        (('AP', 'time', 'long', 300), 'AP-long'),
        # (('AP', 'area', 'small', 300), 'AP-small-area'),
        # (('AP', 'area', 'medium', 300), 'AP-med-area'),
        # (('AP', 'area', 'large', 300), 'AP-large-area'),
        ('AR@300', 'AR'),
        (('AR', 'time', 'short', 300), 'AR-short'),
        (('AR', 'time', 'medium', 300), 'AR-med'),
        (('AR', 'time', 'long', 300), 'AR-long')
    ]
    header = [x[1] for x in eval_keys]
    values = [f'{100*tao_eval.results[x[0]]:.02f}' for x in eval_keys]
    header += ['path']
    values += [log_path]
    logger.info('\n%s\n%s', ','.join(header), ','.join(values))

    if output_dir:
        # This import seems to be slow, only import when we need it.
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(output_dir / 'tensorboard') as w:
            hparams = cfg_to_flat_dict(cfg)
            for k, v in hparams.items():
                if not isinstance(v, (int, float, str, bool)):
                    hparams[k] = str(v)
            if log_path:
                hparams['log_path'] = log_path
            metrics = {x[1]: 100*tao_eval.results[x[0]] for x in eval_keys}
            if 'mota_eval' in eval_info:
                metrics.update(eval_info['mota_eval']['overall'])
            w.add_hparams(hparams, metrics)
        # HParams has weird issues that I'm tired of dealing with; for example,
        # if I report MOTA in some hparams, and not in others, then MOTA simply
        # won't show up on tensorboard. Here, I output CSVs to make my life a
        # little easier.
        with open(output_dir / 'params_metrics.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'type', 'value'])
            writer.writeheader()
            for key, value in hparams.items():
                writer.writerow({'name': key, 'type': 'param', 'value': value})
            for key, value in metrics.items():
                writer.writerow({
                    'name': key,
                    'type': 'metric',
                    'value': value
                })

        # Compute AP per category
        # The 'precision' array has shape
        #   (num_thrs, num_recalls, num_cats, num_area_rngs, num_time_rngs)
        # We want to average over all thresholds and recalls, for the "all"
        # area range and "all" time range.
        area_index = tao_eval.params.area_rng_lbl.index('all')
        time_index = tao_eval.params.time_rng_lbl.index('all')
        category_aps = np.mean(
            tao_eval.eval['precision'][:, :, :, area_index, time_index],
            axis=(0, 1))
        with open(output_dir / 'category_aps.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['category_id', 'ap'])
            writer.writeheader()
            for i, cat in enumerate(tao_eval.params.cat_ids):
                writer.writerow({'category_id': cat, 'ap': category_aps[i]})

    def copy_tao_eval(tao_eval):
        eval_copy = TaoEval(tao_eval.tao_gt, tao_eval.tao_dt)
        eval_copy.params = copy.deepcopy(tao_eval.params)
        return eval_copy

    if cfg.SUPERCATEGORY_MAP:
        cat_name_to_id = {x['name']: x['id'] for x in tao.cats.values()}
        with open(cfg.SUPERCATEGORY_MAP, 'r') as f:
            supercategory_mapping = yaml.safe_load(f)

        inv_mapping = {}
        for k in supercategory_mapping.keys():
            if supercategory_mapping[k] not in inv_mapping.keys():
                inv_mapping[supercategory_mapping[k]] = []
            inv_mapping[supercategory_mapping[k]].append(k)

        supercategories = list(inv_mapping.keys())
        import random
        random.shuffle(supercategories)
        supercategory_evals = []
        sc_eval = copy_tao_eval(tao_eval)
        for sc in supercategories:
            try:
                supercatIds = {
                    cat_name_to_id[x] for x in inv_mapping[sc]
                    if x in cat_name_to_id.keys()
                }
                sc_eval.params.cat_ids = list(supercatIds)
                sc_eval.run()
                interesting_numbers = [
                    sc_eval.results[k[0]] for k in eval_keys
                ]
                num_tracks = len([
                    x for x in tao.tracks.values()
                    if x['category_id'] in supercatIds
                ])
                logging.info(f'Evaluating supercategory: {sc}')
                sc_eval.print_results()
                supercategory_evals.append([sc] + interesting_numbers +
                                           [num_tracks])
            except KeyError:
                logging.exception(
                    f'Key error when evaluating supercategory {sc}')
                pass
            except:  # noqa: E722
                logger.exception(f'Could not evaluate on supercategory: {sc}')
        header = ['Supercategory'] + [k[1] for k in eval_keys] + ['Num tracks']
        df = pd.DataFrame(supercategory_evals, columns=header)
        df = df.set_index('Supercategory')
        logger.info(f'\n' + df.to_csv())
        if output_dir:
            with open(output_dir / 'supercategory_ap.csv', 'w') as f:
                f.write(df.to_csv())

    if cfg.EVAL_BY_DATASET:
        dataset_evals = []
        dataset_videos = defaultdict(list)
        dataset_eval = copy_tao_eval(tao_eval)
        for video in tao.vids.values():
            dataset_videos[video['metadata']['dataset']].append(video)
        for dataset, videos in dataset_videos.items():
            dataset_eval.params.vid_ids = [v['id'] for v in videos]
            dataset_eval.run()
            interesting_numbers = [
                dataset_eval.results[k[0]] for k in eval_keys
            ]
            num_tracks = len([
                x for x in tao.tracks.values()
                if x['video_id'] in dataset_eval.params.vid_ids
                and x['category_id'] in dataset_eval.params.cat_ids
            ])
            logging.info(f'Evaluating dataset: {dataset}')
            dataset_eval.print_results()
            dataset_evals.append([dataset] + interesting_numbers +
                                 [num_tracks])
        header = ['Dataset'] + [k[1] for k in eval_keys] + ['Num tracks']
        df = pd.DataFrame(dataset_evals, columns=header)
        df = df.set_index('Dataset')
        logger.info(f'\n' + df.to_csv())
        if output_dir:
            with open(output_dir / 'dataset_ap.csv', 'w') as f:
                f.write(df.to_csv())

    if cfg.MOTA.ENABLED:
        from .evaluation_mota import log_mota
        log_mota(eval_info, logger, output_dir, log_path)
