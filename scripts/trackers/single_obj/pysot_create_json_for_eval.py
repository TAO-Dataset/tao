import argparse
import json
import logging
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from natsort import natsorted

from script_utils.common import common_setup
from tqdm import tqdm

from tao.toolkit.tao import Tao
from tao.utils import fs


def create_json(pickle_dir, tao, frames_dir, output_dir, oracle_category):
    if not isinstance(pickle_dir, (list, tuple)):
        pickle_dir = [pickle_dir]
    image_to_id = {x['file_name']: x['id'] for x in tao.imgs.values()}

    paths = [
        (root, root / f'{video["name"]}.pkl')
        for video in tao.vids.values()
        for root in pickle_dir
    ]

    annotations = []
    # Map video to list of track ids seen so far. Used to check for duplicates
    # when multiple pickle directories are specified.
    seen_track_ids = defaultdict(set)
    for root, p in tqdm(paths):
        if not p.exists():
            logging.warn(f'Could not find tracks for video {p}')
            continue
        video_name = str(p.relative_to(root)).split('.pkl')[0]
        video_frames_dir = frames_dir / video_name
        frames = [
            str(x.relative_to(frames_dir))
            for x in natsorted(fs.glob_ext(video_frames_dir, fs.IMG_EXTENSIONS))
        ]
        if not frames:
            raise ValueError(f'Found no frames at {video_frames_dir}')
        frame_indices = {x: i for i, x in enumerate(frames)}
        with open(p, 'rb') as f:
            # Map object_id to {'boxes': np.array}
            tracks = pickle.load(f)

        for object_id, outputs in tracks.items():
            if object_id in seen_track_ids[video_name]:
                raise ValueError(
                    f'Object id {object_id} in video {video_name} seen '
                    f'multiple times!')
            if object_id not in tao.tracks:
                logging.warn(
                    f'Object id {object_id} for video {video_name} not found '
                    f'in annotations, skipping.')
                continue
            seen_track_ids[video_name].add(object_id)
            init = tao.get_kth_annotation(object_id, 0)
            init_frame = frame_indices[tao.imgs[init['image_id']]['file_name']]
            boxes = outputs['boxes']
            for i, frame in enumerate(frames[init_frame:]):
                if frame not in image_to_id:
                    continue
                x0, y0, x1, y1, score = boxes[i]
                w, h = x1 - x0 + 1, y1 - y0 + 1
                is_init = np.isinf(score)
                annotations.append({
                    'id': len(annotations),
                    'image_id': image_to_id[frame],
                    'track_id': object_id,
                    'bbox': [x0, y0, w, h],
                    'video_id': tao.imgs[image_to_id[frame]]['video_id'],
                    'category_id': (tao.tracks[object_id]['category_id']
                                    if oracle_category else 1),
                    'score': score,
                    # Numpy -> python boolean for serialization
                    '_single_object_init': bool(is_init)
                })

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(annotations, f)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations', type=Path, required=True)
    # We need the frames dir because the pickles contain boxes into the ordered
    # list frames.
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--pickle-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--oracle-category', action='store_true')
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    tao = Tao(args.annotations)
    create_json(args.pickle_dir, tao, args.frames_dir, args.output_dir,
                args.oracle_category)


if __name__ == "__main__":
    main()
