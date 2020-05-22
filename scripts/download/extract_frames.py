import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from script_utils.common import common_setup

from tao.utils.download import (
    are_tao_frames_dumped, dump_tao_frames, remove_non_tao_frames)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--split',
                        required=True,
                        choices=['train', 'val', 'test'])
    parser.add_argument('--sources',
                        default=['BDD', 'Charades', 'YFCC100M'],
                        choices=['BDD', 'Charades', 'YFCC100M'])
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    log_dir = args.root / 'logs'
    log_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, log_dir, args)

    ann_path = args.root / f'annotations/{args.split}.json'
    with open(ann_path, 'r') as f:
        tao = json.load(f)

    checksums_path = (
        args.root / f'annotations/checksums/{args.split}_checksums.json')
    with open(checksums_path, 'r') as f:
        checksums = json.load(f)

    # checksums = {}
    # for image in tao['images']:
    #     video = image['video']
    #     if video not in checksums:
    #         checksums[video] = {}
    #     name = image['file_name'].split('/')[-1].replace('.jpeg', '.jpg')
    #     checksums[video][name] = ''

    videos_by_dataset = defaultdict(list)
    for video in tao['videos']:
        videos_by_dataset[video['metadata']['dataset']].append(video)

    videos_dir = args.root / 'videos'
    frames_dir = args.root / 'frames'
    for dataset in args.sources:
        # Collect list of videos
        ext = '.mov' if dataset == 'BDD' else '.mp4'
        videos = videos_by_dataset[dataset]
        video_paths = [
            videos_dir / f"{video['name']}{ext}" for video in videos
        ]
        output_frame_dirs = [frames_dir / video['name'] for video in videos]

        # List of (video, video path, frame directory) tuples
        to_dump = []
        for video, video_path, frame_dir in zip(videos, video_paths,
                                                output_frame_dirs):
            if not video_path.exists():
                raise ValueError(f'Could not find video at {video_path}')
            video_checksums = checksums[video['name']]
            if frame_dir.exists() and are_tao_frames_dumped(
                    frame_dir, video_checksums, warn=False):
                continue
            to_dump.append((video, video_path, frame_dir))

        # Dump frames from each video
        logging.info(f'{dataset}: Extracting frames')
        dump_tao_frames([x[1] for x in to_dump], [x[2] for x in to_dump],
                        workers=args.workers)

        to_dump = []
        for video, video_path, frame_dir in zip(videos, video_paths,
                                                output_frame_dirs):
            video_checksums = checksums[video['name']]
            # Remove frames not used for TAO.
            remove_non_tao_frames(frame_dir, set(video_checksums.keys()))
            # Compare checksums for frames
            assert are_tao_frames_dumped(frame_dir, video_checksums), (
                f'Not all TAO frames for {video} were extracted.')

        logging.info(
            f'{dataset}: Removing non-TAO frames, verifying extraction')
        logging.info(f'Successfully extracted {dataset}!')


if __name__ == "__main__":
    main()
