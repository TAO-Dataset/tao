import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from textwrap import fill

from moviepy.video.io.VideoFileClip import VideoFileClip
from script_utils.common import common_setup
from tqdm import tqdm

from tao.utils.download import (
    are_tao_frames_dumped, dump_tao_frames, remove_non_tao_frames)
from tao.utils.fs import dir_path, file_path
from tao.utils.ytdl import download_to_bytes

META_DIR = Path(__file__).resolve().parent / 'meta'


def close_clip(video):
    video.reader.close()
    if video.audio and video.audio.reader:
        video.audio.reader.close_proc()


def download_hacs(root, annotations, checksums, workers=8, debug=False):
    logging.info(f'Downloading HACS videos.')
    videos = [
        v for v in annotations['videos'] if v['metadata']['dataset'] == 'HACS'
    ]

    if debug:
        # Take 5 of each type of video.
        _scene_videos = [
            v for v in videos if v['metadata']['scene'] is not None
        ]
        _noscene_videos = [v for v in videos if v['metadata']['scene'] is None]
        videos = _scene_videos[:5] + _noscene_videos[:5]

    videos_dir = root / 'videos'
    frames_dir = root / 'frames'
    tmp_dir = dir_path(root / 'cache' / 'hacs_videos')
    missing_dir = Path(root / 'hacs_missing')

    # List of (video, video_path, frame_path)
    videos_to_dump = []
    unavailable_videos = []
    for video in tqdm(videos, desc='Downloading HACS'):
        video_path = file_path(videos_dir / f"{video['name']}.mp4")
        frame_output = dir_path(frames_dir / video['name'])
        if are_tao_frames_dumped(frame_output,
                                 checksums[video['name']],
                                 warn=False):
            continue
        if not video_path.exists():
            ytid = video['metadata']['youtube_id']
            full_video = tmp_dir / f"v_{ytid}.mp4"
            missing_downloaded = missing_dir / f"{ytid}.mp4"
            if missing_downloaded.exists():
                logging.info(
                    f'Found video downloaded by user at {missing_downloaded}.')
                shutil.copy2(missing_downloaded, full_video)
            if not full_video.exists():
                url = 'http://youtu.be/' + ytid
                try:
                    vid_bytes = download_to_bytes(url)
                except BaseException:
                    vid_bytes = None
                if isinstance(vid_bytes, int) or vid_bytes is None:
                    unavailable_videos.append(
                        (ytid, video['metadata']['action']))
                    continue
                else:
                    vid_bytes = vid_bytes.getvalue()
                    if len(vid_bytes) == 0:
                        unavailable_videos.append(
                            (ytid, video['metadata']['action']))
                        continue
                with open(full_video, 'wb') as f:
                    f.write(vid_bytes)

            if video['metadata']['scene'] is not None:
                shot_endpoints = video['metadata']['scene'].rsplit('_', 1)[1]
                start, end = shot_endpoints.split('-')
                clip = VideoFileClip(str(full_video))
                subclip = clip.subclip(
                    int(start) / clip.fps,
                    int(end) / clip.fps)
                subclip.write_videofile(str(video_path),
                                        audio=False,
                                        verbose=False,
                                        progress_bar=False)
            else:
                shutil.copy2(full_video, video_path)
        videos_to_dump.append((video['name'], video_path, frame_output))

    dump_tao_frames([x[1] for x in videos_to_dump],
                    [x[2] for x in videos_to_dump], workers)
    for video, video_path, frame_dir in videos_to_dump:
        remove_non_tao_frames(frame_dir, set(checksums[video].keys()))
        assert are_tao_frames_dumped(frame_dir, checksums[video]), (
            f'Not all TAO frames for {video} were extracted.')

    if unavailable_videos:
        missing_path = file_path(missing_dir / 'missing.txt')
        logging.error('\n'.join([
            '',
            f'{len(unavailable_videos)} video(s) could not be downloaded; '
            'please request them from the HACS website by uploading ',
            f'\t{missing_path}',
            'to the following form',
            '\thttps://goo.gl/forms/0STStcLndI32oke22',
            'See the following README for details:',
            '\thttps://github.com/hangzhaomit/HACS-dataset#request-testing-videos-and-missing-videos-new',
        ]))

        with open(missing_path, 'w') as f:
            csv.writer(f).writerows(unavailable_videos)

    if len(unavailable_videos) > 20:
        logging.error(
            fill('NOTE: Over 20 HACS videos were unavailable. This may mean '
                 'that YouTube is rate-limiting your download; please try '
                 'running this script again after a few hours, or on a '
                 'different machine.'))


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--split',
                        required=True,
                        choices=['train', 'val', 'test'])

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--workers', default=8, type=int)
    optional.add_argument(
        '--movies-dir',
        type=Path,
        help=('Directory to save AVA movies to. If you have a copy '
              'AVA locally, you can point to that directory to skip '
              'downloading. NOTE: Any movies downloaded by this script will '
              'be deleted after the script completes. Any movies that already '
              'existed on disk will not be deleted.'))

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

    download_hacs(args.root, tao, checksums, workers=args.workers)


if __name__ == "__main__":
    main()
