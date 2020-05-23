import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from script_utils.common import common_setup
from tqdm import tqdm

from tao.utils.download import (
    are_tao_frames_dumped, dump_tao_frames, remove_non_tao_frames)
from tao.utils.fs import dir_path, file_path
from tao.utils.s3 import S3Wrapper

META_DIR = Path(__file__).resolve().parent / 'meta'


def close_clip(video):
    video.reader.close()
    if video.audio and video.audio.reader:
        video.audio.reader.close_proc()


def ava_load_meta():
    info = {}
    for split in ('trainval', 'test'):
        with open(META_DIR / f'ava_file_names_{split}_v2.1.txt', 'r') as f:
            for line in f:
                stem, ext = line.strip().rsplit('.', 1)
                info[stem] = {'ext': ext, 'split': split}
    return info


def download_ava(root,
                 annotations,
                 checksums,
                 aws_bucket,
                 workers=8,
                 movies_dir=None):
    if movies_dir is None:
        movies_dir = root / 'cache' / 'ava_movies'
        movies_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f'Downloading AVA videos.')
    videos = [
        v for v in annotations['videos'] if v['metadata']['dataset'] == 'AVA'
    ]

    movie_clips = defaultdict(list)
    for v in videos:
        movie_clips[v['metadata']['movie']].append(v)

    movie_info = ava_load_meta()

    # Only construct client if necessary (in case AVA movies are saved locally,
    # we don't want to create an S3 client).
    client = None

    def download_movie(url, path):
        nonlocal client
        if client is None:
            client = S3Wrapper(aws_bucket)
        return client.download_file(url, str(path), verbose=False)

    videos_dir = root / 'videos'
    frames_root = root / 'frames'
    for movie_stem, clips in tqdm(movie_clips.items(),
                                  desc='Processing AVA movies'):
        movie = f"{movie_stem}.{movie_info[movie_stem]['ext']}"

        # List of (clip, output clip path, output frames directory) for clips
        # whose frames have not already been extracted.
        to_process = []
        for clip in clips:
            name = clip['name']
            output_clip = file_path(videos_dir / f"{name}.mp4")
            output_frames = dir_path(frames_root / name)
            if are_tao_frames_dumped(output_frames,
                                     checksums[name],
                                     warn=False):
                logging.debug(f'Skipping extracted clip: {name}')
                continue
            to_process.append((clip, output_clip, output_frames))

        # Download movie if necessary.
        if all(x[1].exists() for x in to_process):
            movie_vfc = None
        else:
            if movies_dir and (movies_dir / movie).exists():
                downloaded_movie_this_run = False
                movie_path = movies_dir / movie
                logging.debug(f'Found AVA movie {movie} at {movie_path}')
            else:
                downloaded_movie_this_run = True
                movie_path = movies_dir / movie
                if not movie_path.exists():
                    logging.debug(f'Downloading AVA movie: {movie}.')
                    url = f"{movie_info[movie_stem]['split']}/{movie}"
                    download_movie(url, movie_path)
            movie_vfc = VideoFileClip(str(movie_path))

        for clip_info, clip_path, frames_dir in tqdm(to_process,
                                                     desc='Extracting shots',
                                                     leave=False):
            if clip_path.exists():
                continue
            shot_endpoints = clip_info['metadata']['scene'].rsplit('_', 1)[1]
            start, end = shot_endpoints.split('-')
            subclip = movie_vfc.subclip(
                int(start) / movie_vfc.fps,
                int(end) / movie_vfc.fps)
            subclip.write_videofile(str(clip_path),
                                    audio=False,
                                    verbose=False,
                                    progress_bar=False)
            close_clip(subclip)

        if movie_vfc:
            close_clip(movie_vfc)
            if downloaded_movie_this_run:
                movie_path.unlink()

        logging.debug(
            f'AVA: Dumping TAO frames:\n{[x[1:] for x in to_process]}')
        dump_tao_frames([x[1] for x in to_process], [x[2] for x in to_process],
                        workers)
        for clip, clip_path, frame_dir in to_process:
            if not are_tao_frames_dumped(frame_dir, checksums[clip['name']]):
                raise ValueError(
                    f'Not all TAO frames for {clip["name"]} were extracted. '
                    f'Try deleting the clip at {clip_path} and running this '
                    f'script again.')
            remove_non_tao_frames(frame_dir,
                                  set(checksums[clip['name']].keys()))
            assert are_tao_frames_dumped(frame_dir, checksums[clip['name']]), (
                f'ERROR: TAO frames were dumped properly for {clip["name"]}, '
                f'but were deleted by `remove_non_tao_frames`! This is a bug, '
                f'please report it.')


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

    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)

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

    download_ava(args.root,
                 tao,
                 checksums,
                 aws_bucket='ava-dataset',
                 workers=args.workers,
                 movies_dir=args.movies_dir)


if __name__ == "__main__":
    main()
