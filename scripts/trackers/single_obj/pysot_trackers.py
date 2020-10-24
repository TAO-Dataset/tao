import argparse
import logging
import pickle
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from script_utils.common import common_setup
from tqdm import tqdm

from tao.toolkit.tao import Tao
from tao.trackers.sot.pysot import PysotTracker
from tao.trackers.sot.pytracking import PytrackingTracker
from tao.trackers.sot.staple import StapleTracker
from tao.trackers.sot.srdcf import SrdcfTracker
from tao.utils.parallel.fixed_gpu_pool import FixedGpuPool
from tao.utils import fs
from tao.utils import misc


def init_tracker(args, context):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(context['gpu'])
    if args['tracker_type'] == 'pysot':
        context['tracker'] = PysotTracker(**args['tracker_init'])
    elif args['tracker_type'] == 'pytrack':
        context['tracker'] = PytrackingTracker(**args['tracker_init'])
    elif args['tracker_type'] == 'staple':
        context['tracker'] = StapleTracker(**args['tracker_init'])
    elif args['tracker_type'] == 'srdcf':
        context['tracker'] = SrdcfTracker(**args['tracker_init'])


def track_video(tracker,
                objects,
                frames_dir,
                output_pickle,
                show_progress=False,
                visualize=False):
    frames = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))
    output_pickle.parent.mkdir(exist_ok=True, parents=True)
    all_output = {}
    for object_id, info in tqdm(objects.items(),
                                'Tracking objects',
                                disable=not show_progress):
        x0, y0, w, h = info['init']
        box = [x0, y0, x0 + w, y0 + h]

        first_annotated_frame = info['first_annotated_frame']
        sot_init_frame = info['sot_init_frame']

        do_track_backward = sot_init_frame != first_annotated_frame
        if visualize:
            # only add suffix if we also have to track backward
            forward_suffix = '_forward' if do_track_backward else ''
            output_forward_video = output_pickle.with_name(
                f'{output_pickle.stem}_{object_id}{forward_suffix}.mp4')
            output_backward_video = output_pickle.with_name(
                f'{output_pickle.stem}_{object_id}_backward.mp4')
        else:
            output_forward_video = output_backward_video = None

        def do_track(object_frames, output_video=None):
            try:
                boxes, _, _ = tracker.track(object_frames,
                                            box,
                                            output_video=output_video,
                                            show_progress=show_progress)
            except (ValueError, subprocess.CalledProcessError):
                # ValueError: Sometimes SiamRPN++-LT errors with a ValueError
                # due to some error (I believe) in their context window
                # calculation. We just have to skip these videos until we can
                # debug this further.
                # CalledProcessError: Staple sometimes fails with a segfault.
                logging.exception(
                    f'Failed to track object {object_id} in video '
                    f'{frames[0].parent}, skipping... Exception:')
                raise
            return boxes

        # Track forward
        try:
            if show_progress:
                logging.info('Tracking forward')
            boxes = do_track(frames[sot_init_frame:], output_forward_video)
        except (ValueError, subprocess.CalledProcessError):
            continue

        # Track backward
        if do_track_backward:
            if show_progress:
                logging.info('Tracking backward')
            track_frames = frames[first_annotated_frame:sot_init_frame+1][::-1]
            try:
                backward_boxes = do_track(track_frames, output_backward_video)
            except ValueError:
                continue
            backward_boxes = backward_boxes[::-1]
            assert np.all(backward_boxes[-1] == boxes[0])
            boxes = np.vstack((backward_boxes[:-1], boxes))
        all_output[object_id] = {'boxes': boxes}
    with open(output_pickle, 'wb') as f:
        pickle.dump(all_output, f)


def track_video_helper(kwargs, context):
    tracker = context['tracker']
    return track_video(tracker, **kwargs)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--init',
                        choices=['first', 'biggest'],
                        default='first')
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True)

    parser.add_argument('--tracker',
                        choices=['pysot', 'pytrack', 'staple', 'srdcf'],
                        default='pysot')
    pysot_args = parser.add_argument_group('pysot_params')
    pysot_args.add_argument('--pysot-config-file', '--config-file', type=Path)
    pysot_args.add_argument('--pysot-model-path', '--model-path', type=Path)

    pytracking_args = parser.add_argument_group('pytracking_params')
    pytracking_args.add_argument('--pytrack-name')
    pytracking_args.add_argument('--pytrack-param')
    pytracking_args.add_argument(
        '--pytrack-model-path',
        help=('Specify path to model, if different from the one implied by '
              '--pytrack-param.'))

    parser.add_argument('--gpus', default=[0, 1, 2, 3], nargs='*', type=int)
    parser.add_argument('--tasks-per-gpu', default=1, type=int)
    parser.add_argument('--visualize', default=False, type=misc.parse_bool)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.init == 'first':
        common_setup(__file__, args.output_dir, args)
    else:
        common_setup(f'{Path(__file__).stem}_{args.init}', args.output_dir,
                     args)

    _num_threads = 4
    torch.set_num_threads(_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(_num_threads)

    if args.tracker == 'pysot':
        assert args.pysot_config_file is not None
        assert args.pysot_model_path is not None
    elif args.tracker == 'pytrack':
        assert args.pytrack_name is not None
        assert args.pytrack_param is not None
    elif args.tracker in ('staple', 'srdcf'):
        pass

    tao = Tao(args.annotations)

    video_tracks = defaultdict(list)
    for track_id, track in tao.tracks.items():
        video_tracks[track['video_id']].append(track)

    # List of kwargs passed to track_video().
    track_video_tasks = []
    for video_id, tracks in tqdm(video_tracks.items(),
                                 desc='Collecting tasks'):
        video_name = tao.vids[video_id]['name']
        frames_dir = args.frames_dir / video_name
        output = (args.output_dir / video_name).with_suffix('.pkl')
        if output.exists():
            logging.info(f'{output} already exists, skipping.')
            continue
        # Map track id to
        # {'frame': name, 'init': [x0, y0, w, h]}
        frames = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))
        if not frames[0].exists():
            # Just check the first frame for efficiency; usually, either all
            # frames will be missing, or all will be available.
            logging.info(
                f'Frame link {frames[0]} broken for {video_name} in '
                f'{frames_dir}, skipping.')
            continue
        objects = {}
        for track in tracks:
            annotation = tao.get_single_object_init(track['id'], args.init)
            frame_name = tao.imgs[annotation['image_id']]['file_name']
            frame_indices = {
                str(x.relative_to(args.frames_dir)): i
                for i, x in enumerate(frames)
            }
            init_frame_index = frame_indices[frame_name]

            if args.init == 'first':
                first_frame_index = init_frame_index
            else:
                first_ann = tao.get_kth_annotation(track['id'], 0)
                first_frame_name = tao.imgs[first_ann['image_id']]['file_name']
                first_frame_index = frame_indices[first_frame_name]

            objects[track['id']] = {
                'first_annotated_frame': first_frame_index,
                'sot_init_frame': init_frame_index,
                'init': annotation['bbox'],
            }
        task = {
            'objects': objects,
            'output_pickle': output,
            'frames_dir': frames_dir,
            'visualize': args.visualize
        }
        track_video_tasks.append(task)

    gpus = args.gpus * args.tasks_per_gpu
    if args.tracker == 'pysot':
        tracker_init = {
            'config_file': args.pysot_config_file,
            'model_path': args.pysot_model_path,
        }
    elif args.tracker == 'pytrack':
        tracker_init = {
            'tracker_name': args.pytrack_name,
            'tracker_param': args.pytrack_param,
            'model_path': args.pytrack_model_path
        }
    elif args.tracker in ('staple', 'srdcf'):
        tracker_init = {}

    if not track_video_tasks:
        logging.warning('No tasks found!')
        return

    gpus = gpus[:len(track_video_tasks)]
    print(gpus)
    if len(gpus) == 1:
        context = {'gpu': gpus[0]}
        init_tracker(
            {
                'tracker_init': tracker_init,
                'tracker_type': args.tracker
            }, context)
        for task in tqdm(track_video_tasks):
            task['show_progress'] = True
            track_video_helper(task, context)
    else:
        pool = FixedGpuPool(gpus,
                            initializer=init_tracker,
                            initargs={
                                'tracker_init': tracker_init,
                                'tracker_type': args.tracker
                            })
        list(
            tqdm(pool.imap_unordered(track_video_helper, track_video_tasks),
                 total=len(track_video_tasks)))


if __name__ == "__main__":
    main()
