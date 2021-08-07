import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from multiprocessing import Pool
from natsort import natsorted
from PIL import Image
from script_utils.common import common_setup
from tqdm import tqdm

from tao.utils import fs
from tao.utils import vis
from tao.utils.colormap import colormap
from tao.utils.video import video_writer


def visualize_star(kwargs):
    return visualize(**kwargs)


def visualize(video_npz,
              frames_dir,
              categories,
              threshold,
              track_threshold,
              output,
              vis_categories=None,
              progress=False):
    try:
        video_result = np.load(video_npz)
    except ValueError:
        print(video_npz)
        raise
    frame_names = [
        x for x in video_result.keys()
        if x != 'field_order' and not x.startswith('__')
    ]
    output.parent.mkdir(exist_ok=True, parents=True)
    first_frame = fs.find_file_extensions(frames_dir, frame_names[0],
                                          fs.IMG_EXTENSIONS)
    if first_frame is None:
        raise ValueError(f'Could not find frame with name {frame_names[0]} in '
                         f'{frames_dir}')
    ext = first_frame.suffix
    w, h = Image.open(first_frame).size
    color_generator = itertools.cycle(colormap(rgb=True).tolist())
    colors = defaultdict(lambda: next(color_generator))

    track_scores = defaultdict(list)
    for frame in frame_names:
        for x in video_result[frame]:
            track_scores[x[-1]].append(x[5])
    track_scores = {t: np.mean(vs) for t, vs in track_scores.items()}

    with video_writer(str(output), (w, h)) as writer:
        for frame in tqdm(natsorted(frame_names), disable=not progress):
            # Format is one of
            #   (x0, y0, x1, y1, class, score, box_index, track_id)
            #   (x0, y0, x1, y1, class, score, track_id)
            frame_result = video_result[frame]
            annotations = [
                {
                    # (x1, y1) -> (w, h)
                    'bbox': [x[0], x[1], x[2] - x[0], x[3] - x[1]],
                    'category_id': x[4],
                    'score': x[5],
                    'track_id': int(x[-1])
                } for x in frame_result
                if x[5] > threshold and track_scores[x[-1]] > track_threshold
            ]
            if vis_categories is not None:
                annotations = [
                    x for x in annotations
                    if categories[x['category_id']]['name'] in vis_categories
                ]
            annotations = sorted(annotations, key=lambda x: x['score'])
            box_colors = [colors[x['track_id']] for x in annotations]
            image = np.array(Image.open(frames_dir / (frame + ext)))
            image = vis.overlay_boxes_coco(image,
                                           annotations,
                                           colors=box_colors)
            image = vis.overlay_class_coco(image,
                                           annotations,
                                           categories,
                                           font_scale=1,
                                           font_thickness=2,
                                           show_track_id=True)
            writer.write_frame(image)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--track-result', required=True, type=Path)
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--track-threshold', default=0, type=float)
    parser.add_argument(
        '--annotations-json',
        type=Path,
        help='Annotations json; we only care about the "categories" field.')
    parser.add_argument('--frames-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--vis-cats', nargs='*', type=str)
    parser.add_argument('--num-videos', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    with open(args.annotations_json, 'r') as f:
        categories = {x['id']: x for x in json.load(f)['categories']}

    all_npz = args.track_result.rglob('*.npz')
    if args.num_videos > 0:
        all_npz = list(all_npz)
        random.seed(args.seed)
        random.shuffle(all_npz)
        all_npz = all_npz[:args.num_videos]

    tasks = []
    for video_npz in all_npz:
        video = video_npz.relative_to(args.track_result).with_suffix('')
        if args.videos and str(video) not in args.videos:
            continue
        if args.videos:
            print(video)
        frames_dir = args.frames_dir / video
        output = args.output_dir / (str(video) + '.mp4')
        if output.exists():
            continue
        tasks.append({
            'video_npz': video_npz,
            'frames_dir': frames_dir,
            'categories': categories,
            'vis_categories': args.vis_cats,
            'threshold': args.threshold,
            'track_threshold': args.track_threshold,
            'output': output
        })

    if args.workers > 0:
        pool = Pool(args.workers)
        list(tqdm(pool.imap_unordered(visualize_star, tasks),
                  total=len(tasks)))
    else:
        for task in tqdm(tasks):
            task['progress'] = True
            visualize(**task)


if __name__ == "__main__":
    main()
