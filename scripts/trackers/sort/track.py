"""Link detections using the SORT tracker."""

import argparse
import itertools
import json
import logging
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from script_utils.common import common_setup
from torchvision.ops import nms
from tqdm import tqdm

from tao.utils.fs import dir_path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sort_with_detection_id import SortWithDetectionId
from create_json_for_eval import create_json


def sort_track_vid(video_predictions, sort_kwargs):
    """
    Args:
        video_predictions (List): video_predictions[i] is a numpy array of
            shape (n, 4), containing [x0, y0, x1, y1].

    Returns:
        tracked_predictions (List): tracked_predictions[i] is a numpy array
            of shape (n', 6), containing
                [x0, y0, x1, y1, box_index, track_id],
            where box_index is -1 or indexes into video_predictions[i].
    """
    tracker = SortWithDetectionId(**sort_kwargs)
    return [
        tracker.update(boxes) for boxes in tqdm(
            video_predictions, desc='Running tracker', disable=True)
    ]


def track_and_save(pickle_paths, output, score_threshold,
                   nms_thresh, sort_kwargs):
    paths = natsorted(pickle_paths)
    all_instances = []
    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)['instances']
        all_instances.append({
            'scores': np.array(data['scores']),
            'pred_classes': np.array(data['pred_classes']),
            'pred_boxes': np.array(data['pred_boxes'])
        })

    if score_threshold > -float('inf'):
        for i, data in enumerate(all_instances):
            valid = data['scores'] > score_threshold
            for x in ('scores', 'pred_boxes', 'pred_classes'):
                data[x] = data[x][valid]

    categories = sorted({
        x
        for data in all_instances for x in data['pred_classes']
    })

    frame_infos = defaultdict(list)
    id_gen = itertools.count(1)
    unique_track_ids = defaultdict(lambda: next(id_gen))
    for category in categories:
        class_instances = []
        for data in all_instances:
            in_class = data['pred_classes'] == category
            class_instances.append({
                k: data[k][in_class]
                for k in ('scores', 'pred_boxes', 'pred_classes')
            })

        if nms_thresh >= 0:
            for i, instances in enumerate(class_instances):
                nms_keep = nms(torch.from_numpy(instances['pred_boxes']),
                               torch.from_numpy(instances['scores']),
                               iou_threshold=nms_thresh).numpy()
                class_instances[i] = {
                    'scores': instances['scores'][nms_keep],
                    'pred_boxes': instances['pred_boxes'][nms_keep],
                    'pred_classes': instances['pred_classes'][nms_keep]
                }

        tracked_boxes = sort_track_vid(
            [x['pred_boxes'] for x in class_instances], sort_kwargs)

        for frame, frame_tracks in enumerate(tracked_boxes):
            # Each row is of the form (x0, y0, x1, y1, box_index, track_id)
            frame_boxes = frame_tracks[:, :4]
            box_indices = frame_tracks[:, 4].astype(int)
            track_ids = np.array([
                unique_track_ids[(x, category)] for x in frame_tracks[:, 5]
            ])

            frame_instances = class_instances[frame]
            frame_scores = np.zeros((len(box_indices), 1))
            frame_classes = np.zeros((len(box_indices), 1))
            for i, idx in enumerate(box_indices.astype(int)):
                if idx == -1:
                    frame_classes[i] = -1
                    frame_scores[i] = -1
                else:
                    frame_classes[i] = frame_instances['pred_classes'][idx]
                    frame_scores[i] = frame_instances['scores'][idx]
            frame_infos[paths[frame].stem].append(
                np.hstack(
                    (frame_boxes, frame_classes + 1, frame_scores,
                     box_indices[:, np.newaxis], track_ids[:, np.newaxis])))

    frame_infos = {k: np.vstack(lst) for k, lst in frame_infos.items()}
    output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(output,
                        **frame_infos,
                        field_order=[
                            'x0', 'y0', 'x1', 'y1', 'class', 'score',
                            'box_index', 'track_id'
                        ])


def track_and_save_star(args):
    track_and_save(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='Results directory with pickle or mat files')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory, where a results.json will be output, as well '
              'as a .npz file for each video, containing a boxes array of '
              'size (num_boxes, 6), of the format [x0, y0, x1, y1, class, '
              'score, box_index, track_id], where box_index maps into the '
              'pickle files'))
    parser.add_argument('--max-age', default=100, type=int)
    parser.add_argument('--min-hits', default=1, type=float)
    parser.add_argument('--min-iou', default=0.1, type=float)
    parser.add_argument('--score-threshold',
                        default=0.0001,
                        help='Float or "none".')
    parser.add_argument('--nms-thresh', type=float, default=-1)
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    args.score_threshold = (-float('inf') if args.score_threshold == 'none'
                            else float(args.score_threshold))

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    npz_dir = dir_path(args.output_dir / 'npz_files')

    def get_output_path(video):
        return npz_dir / (video + '.npz')

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    videos = [x['name'] for x in groundtruth['videos']]
    video_paths = {}
    for video in tqdm(videos, desc='Collecting paths'):
        output = get_output_path(video)
        if output.exists():
            logging.debug(f'{output} already exists, skipping...')
            continue
        vid_detections = args.detections_dir / video
        assert vid_detections.exists(), (
            f'No detections dir at {vid_detections}!')
        detection_paths = natsorted(
            (args.detections_dir / video).rglob(f'*.pkl'))
        assert detection_paths, (
            f'No detections pickles at {vid_detections}!')
        video_paths[video] = detection_paths

    if not video_paths:
        logging.info(f'Nothing to do! Exiting.')
        return
    logging.info(f'Found {len(video_paths)} videos to track.')

    tasks = []
    for video, paths in tqdm(video_paths.items()):
        output = get_output_path(video)
        tasks.append((paths, output, args.score_threshold,
                      args.nms_thresh, {
                          'iou_threshold': args.min_iou,
                          'min_hits': args.min_hits,
                          'max_age': args.max_age
                      }))

    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(track_and_save_star, tasks),
                 total=len(tasks),
                 desc='Tracking'))
    else:
        for task in tqdm(tasks):
            track_and_save(*task)
    logging.info(f'Finished')

    create_json(npz_dir, groundtruth, args.output_dir)


if __name__ == "__main__":
    main()
