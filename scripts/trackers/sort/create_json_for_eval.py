import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from natsort import natsorted
from script_utils.common import common_setup
from tqdm import tqdm


def create_json(track_result, groundtruth, output_dir):
    # Image without extension -> image id
    image_stem_to_info = {
        x['file_name'].rsplit('.', 1)[0]: x for x in groundtruth['images']
    }
    valid_videos = {x['name'] for x in groundtruth['videos']}

    all_annotations = []
    found_predictions = {}
    for video in tqdm(valid_videos):
        video_npz = track_result / f'{video}.npz'
        if not video_npz.exists():
            logging.error(f'Could not find video {video} at {video_npz}')
            continue
        video_result = np.load(video_npz)
        frame_names = [x for x in video_result.keys() if x != 'field_order']
        video_found = {}
        for frame in natsorted(frame_names):
            # (x0, y0, x1, y1, class, score, box_index, track_id)
            frame_name = f'{video}/{frame}'
            if frame_name not in image_stem_to_info:
                continue
            video_found[frame_name] = True
            image_info = image_stem_to_info[frame_name]
            all_annotations.extend([{
                # (x1, y1) -> (w, h)
                'image_id': image_info['id'],
                'video_id':  image_info['video_id'],
                'track_id': int(x[7]),
                'bbox': [x[0], x[1], x[2] - x[0], x[3] - x[1]],
                'category_id': x[4],
                'score': x[5],
            } for x in video_result[frame]])
        if not video_found:
            raise ValueError(f'Found no valid predictions for video {video}')
        found_predictions.update(video_found)
    if not found_predictions:
        raise ValueError('Found no valid predictions!')

    with_predictions = set(found_predictions.keys())
    with_labels = set(image_stem_to_info.keys())
    if with_predictions != with_labels:
        missing_videos = {
            x.rsplit('/', 1)[0]
            for x in with_labels - with_predictions
        }
        logging.warn(
            f'{len(with_labels - with_predictions)} images from '
            f'{len(missing_videos)} videos did not have predictions!')

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_annotations, f)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--track-result', required=True, type=Path)
    parser.add_argument('--annotations-json',
                        type=Path,
                        help='Annotations json')
    parser.add_argument('--output-dir', required=True, type=Path)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    with open(args.annotations_json, 'r') as f:
        groundtruth = json.load(f)

    create_json(args.track_result, groundtruth, args.output_dir)


if __name__ == "__main__":
    main()
