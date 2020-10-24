import argparse
import itertools
import logging
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image
from script_utils.common import common_setup
from tqdm import tqdm

from tao.toolkit.tao import Tao
from tao.utils import fs
from tao.utils import video as video_utils
from tao.utils import vis
from tao.utils.colormap import colormap


def visualize(pickle_path, video_name, frames_root, cats, vis_cats,
              annotations_json, threshold, output_video):
    logging.getLogger('tao.toolkit.tao.tao').setLevel(logging.WARN)
    tao = Tao(annotations_json)
    frames_dir = frames_root / video_name
    frame_paths = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))
    frames = [str(x.relative_to(frames_root)) for x in frame_paths]
    frame_indices = {x: i for i, x in enumerate(frames)}
    with open(pickle_path, 'rb') as f:
        # Map object_id to {'boxes': np.array}
        tracks = pickle.load(f)
    init_type = tracks.pop('_init_type', 'first')
    if init_type != 'first':
        raise NotImplementedError(
            'init type "{init_type}" not yet implemented.')

    frame_annotations = defaultdict(list)
    init_frames = {}
    annotation_id_generator = itertools.count()
    for object_id, outputs in tracks.items():
        init = tao.get_kth_annotation(object_id, k=0)
        init_frame = frame_indices[tao.imgs[init['image_id']]['file_name']]
        init_frames[object_id] = init_frame
        boxes = outputs['boxes']
        for i, frame in enumerate(frames[init_frame:]):
            if len(boxes) <= i:
                logging.warn(
                    f'Could not find box for object {object_id} for '
                    f'frame (index: {i}, {frame})')
                continue
            box = boxes[i].tolist()
            if len(box) == 4:
                box.append(1)
            x0, y0, x1, y1, score = box
            if score < threshold:
                continue
            w, h = x1 - x0 + 1, y1 - y0 + 1
            category = tao.tracks[object_id]['category_id']
            if (vis_cats is not None
                    and tao.cats[category]['name'] not in vis_cats):
                continue
            frame_annotations[frame].append({
                'id': next(annotation_id_generator),
                'track_id': object_id,
                'bbox': [x0, y0, w, h],
                'category_id': category,
                'score': score
            })
    size = Image.open(frame_paths[0]).size
    output_video.parent.mkdir(exist_ok=True, parents=True)
    with video_utils.video_writer(output_video, size=size) as writer:
        color_generator = itertools.cycle(colormap(as_int=True).tolist())
        colors = defaultdict(lambda: next(color_generator))
        for frame in frame_paths:
            image = np.array(Image.open(frame))
            frame_key = str(frame.relative_to(frames_root))
            tracks = frame_annotations[frame_key]
            image = vis.overlay_boxes_coco(
                image,
                tracks,
                colors=[colors[x['track_id']] for x in tracks])
            image = vis.overlay_class_coco(
                image,
                tracks,
                categories=cats,
                font_scale=1,
                font_thickness=2)
            writer.write_frame(image)


def visualize_star(kwargs):
    return visualize(**kwargs)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations', type=Path, required=True)
    # We need the frames dir because the pickles contain boxes into the ordered
    # list frames.
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--pickle-dir', type=Path, required=True)
    parser.add_argument('--oracle-category', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--vis-cats', nargs='*', type=str)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    paths = list(args.pickle_dir.rglob('*.pkl'))

    tao = Tao(args.annotations)
    cats = tao.cats.copy()
    for cat in cats.values():
        if cat['name'] == 'baby':
            cat['name'] = 'person'

    tasks = []
    for p in paths:
        video_name = str(p.relative_to(args.pickle_dir)).split('.pkl')[0]
        if args.videos is not None and video_name not in args.videos:
            continue
        output_video = args.output_dir / f'{video_name}.mp4'
        if output_video.exists():
            continue
        tasks.append({
            'pickle_path': p,
            'video_name': video_name,
            'frames_root': args.frames_dir,
            'cats': cats,
            'vis_cats': args.vis_cats,
            'annotations_json': args.annotations,
            'threshold': args.threshold,
            'output_video': output_video
        })

    if args.workers == 0:
        for task in tqdm(tasks):
            visualize(**task)
    else:
        pool = Pool(args.workers)
        list(tqdm(pool.imap_unordered(visualize_star, tasks),
                  total=len(tasks)))


if __name__ == "__main__":
    main()
