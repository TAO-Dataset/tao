"""Download videos for TAO.

TODOs:
- Create list of frames, along with their checksums to be read by this script.
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from hashlib import md5
from pathlib import Path

from multiprocessing import Pool
from moviepy.video.io.VideoFileClip import VideoFileClip
from script_utils.common import common_setup
from tqdm import tqdm
from yacs.config import CfgNode as CN

from vid.utils.split_frames import dump_frames_star
from tao.toolkit.tao import Tao
from tao.utils.s3 import S3Wrapper
from tao.utils.ytdl import download_to_bytes


META_DIR = Path(__file__).resolve().parent / 'meta'
DATA_SOURCES = ['ava', 'lasot', 'argoverse', 'charades', 'hacs', 'yfcc', 'bdd']


def dir_path(path):
    """Wrapper around Path that ensures this directory is created."""
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def file_path(path):
    """Wrapper around Path that ensures parent directories are created.

        x = mkdir_parents(dir / video_with_dir_prefix)
    is short-hand for
        x = Path(dir / video_with_dir_prefix)
        x.parent.mkdir(exist_ok=True, parents=True)
    """
    if not isinstance(path, Path):
        path = Path(path)
    path.resolve().parent.mkdir(exist_ok=True, parents=True)
    return path


def get_cfg_defaults():
    _C = CN()
    _C.KEEP_VIDEOS = False
    _C.WORKERS = 8

    _C.CHECKSUMS = CN()
    _C.CHECKSUMS.VERIFY = False
    # Checksums need to be specified even if VERIFY is False, as the json
    # contains the list of all TAO images (including unlabeled ones).
    _C.CHECKSUMS.PATH = ''

    _C.VIDEOS_DIRNAME = 'videos'
    _C.FRAMES_DIRNAME = 'frames'

    # Path to Tao annotations file.
    _C.TAO_ANNOTATIONS = CN()
    _C.TAO_ANNOTATIONS.TRAIN = ''
    _C.TAO_ANNOTATIONS.VAL = ''
    _C.TAO_ANNOTATIONS.TEST = ''

    _C.YFCC = CN()
    _C.YFCC.AWS_BUCKET = 'multimedia-commons'
    _C.YFCC.AWS_PREFIX = 'data/videos/mp4/'

    _C.AVA = CN()
    _C.AVA.MOVIES = CN()
    # Directory containing original AVA movies. If any movie is missing, this
    # script will download it.
    _C.AVA.MOVIES.DIR = ''
    # Any movies that are not in MOVIES_DIR will be downloaded to a
    # subdirectory of output_directory with this name.
    _C.AVA.MOVIES.TMP_DIR = 'ava_movies'
    # Whether to keep movies downloaded to tmp_dir.
    _C.AVA.MOVIES.KEEP_TMP = True
    _C.AVA.AWS_BUCKET = 'ava-dataset'

    _C.LASOT = CN()
    # Path to LaSOT dataset, downloaded from
    # https://cis.temple.edu/lasot/download.html
    # TODO: Support downloading LaSOT automatically. Ideally, we can download
    # only the relevant videos directly from LaSOT, but it's unclear how to do
    # this right now.
    _C.LASOT.DATASET_ROOT = ''
    # If true, create symlinks to LaSOT dataset root frames instead of copying
    # them.
    _C.LASOT.CREATE_SYMLINKS = False
    # Currently unused.
    _C.LASOT.TMP_DIR = 'lasot_videos'
    _C.LASOT.KEEP_TMP = False

    _C.CHARADES = CN()
    # Path to directory containing original Charades videos, downloaded from
    # http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip
    # These are _not_ the downscaled, 480p versions.
    _C.CHARADES.VIDEOS_DIR = ''

    _C.HACS = CN()
    _C.HACS.TMP_DIR = 'hacs_videos'
    _C.HACS.KEEP_TMP = False

    _C.BDD = CN()
    # Path to directory containing original BDD videos, downloaded from
    # http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_val_00.zip
    _C.BDD.VIDEOS_DIR = ''


    # Whether to keep frames that are not used for TAO. This option is provided
    # in case it is helpful, but is not extensively tested.
    # Note: If this is True but KEEP_VIDEOS is False, the script will _always_
    # re-download all videos, as it cannot verify whether the non-tao frames
    # are properly extracted (we do not keep a list of all the non-tao frames,
    # nor their checksums).
    _C._KEEP_NON_TAO_FRAMES = False

    _C.DEBUG = False
    return _C


def close_clip(video):
    video.reader.close()
    if video.audio and video.audio.reader:
        video.audio.reader.close_proc()


def are_tao_frames_dumped(frames_dir, checksums):
    for frame, cksum in checksums.items():
        path = frames_dir / frame
        if not path.exists():
            logging.warning(f'Could not find frame at {path}!')
            return False
        if cksum != '':
            with open(path, 'rb') as f:
                md5_digest = md5(f.read()).hexdigest()
            if md5_digest != cksum:
                logging.warning(
                    f'Checksum for {path} did not match! Expected: {cksum}, '
                    f'saw: {md5_digest}')
                return False
    return True


def remove_non_tao_frames(frames_dir, keep_frames):
    frames = set(keep_frames)
    for frame in frames_dir.glob('*.jpg'):
        if frame.name not in frames:
            frame.unlink()


def dump_frames(videos, output_dirs, cfg, tqdm_desc='Converting to frames'):
    fps = None
    extension = '.jpg'
    jpeg_qscale = 2
    workers = cfg.WORKERS

    for output_dir in output_dirs:
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    dump_frames_tasks = []
    for video_path, output_dir in zip(videos, output_dirs):
        dump_frames_tasks.append(
            (video_path, output_dir, fps, extension, jpeg_qscale))

    # dump_frames code logs when, e.g., the expected number of frames does not
    # match the number of dumped frames. But these logs can have false
    # positives that are confusing, so we check that frames are correctly
    # dumped ourselves separately based on frames in TAO annotations.
    _log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    if workers > 1:
        pool = Pool(workers)
        try:
            list(
                tqdm(pool.imap_unordered(dump_frames_star, dump_frames_tasks),
                     total=len(dump_frames_tasks),
                     leave=False,
                     desc=tqdm_desc))
        except KeyboardInterrupt:
            print('Parent received control-c, exiting.')
            pool.terminate()
    else:
        for task in tqdm(dump_frames_tasks):
            dump_frames_star(dump_frames_tasks)
    logging.root.setLevel(_log_level)


def download_yfcc(cfg, annotations, checksums, output_dir):
    logging.info(f'Downloading YFCC videos.')
    client = S3Wrapper(cfg.YFCC.AWS_BUCKET)
    videos = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'YFCC100M'
    ]
    prefix = cfg.YFCC.AWS_PREFIX
    if not prefix.endswith('/'):
        prefix = prefix + '/'

    video_dir = dir_path(output_dir / cfg.VIDEOS_DIRNAME)
    frames_dir = dir_path(output_dir / cfg.FRAMES_DIRNAME)

    # List of (video, video_path, frame_path)
    videos_to_dump = []
    for video in tqdm(videos, desc='Download'):
        vidname = video['name']
        output = video_dir / f'{vidname}.mp4'
        frame_output = frames_dir / vidname
        # If we only want TAO frames, we can check the frames against the list
        # in the checksums files. Otherwise, we have to download the video,
        # since we don't have a list of non-tao frames.
        if (not cfg._KEEP_NON_TAO_FRAMES
                and frame_output.exists() and are_tao_frames_dumped(
                    frame_output, checksums[vidname])):
            continue

        videos_to_dump.append((vidname, output, frame_output))
        if output.exists():
            logging.debug(f'{output} already exists, skipping.')
            continue
        yfcc_id = vidname.split('v_', 1)[1]
        aws_key = f'{prefix}{yfcc_id[:3]}/{yfcc_id[3:6]}/{yfcc_id}.mp4'
        client.download_file(aws_key, str(output), verbose=False)

    dump_frames([x[1] for x in videos_to_dump],
                [x[2] for x in videos_to_dump],
                cfg)

    for video, video_path, frame_dir in videos_to_dump:
        assert are_tao_frames_dumped(frame_dir, checksums[video]), (
            f'Not all TAO frames for {video} were extracted.')
        if not cfg._KEEP_NON_TAO_FRAMES:
            remove_non_tao_frames(frame_dir, set(checksums[video].keys()))
        if not cfg.KEEP_VIDEOS:
            video_path.unlink()


def ava_load_meta():
    info = {}
    for split in ('trainval', 'test'):
        with open(META_DIR / f'ava_file_names_{split}_v2.1.txt', 'r') as f:
            for line in f:
                stem, ext = line.strip().rsplit('.', 1)
                info[stem] = {'ext': ext, 'split': split}
    return info


def download_ava(cfg, annotations, checksums, output_dir):
    logging.info(f'Downloading AVA videos.')
    videos = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'AVA'
    ]

    movie_clips = defaultdict(list)
    for v in videos:
        movie_clips[v['metadata']['movie']].append(v)

    movie_info = ava_load_meta()
    client = None
    movie_dir = dir_path(cfg.AVA.MOVIES.DIR) if cfg.AVA.MOVIES.DIR else None
    tmp_dir = dir_path(output_dir / cfg.AVA.MOVIES.TMP_DIR)

    keep_tmp = cfg.AVA.MOVIES.KEEP_TMP
    if not keep_tmp and movie_dir and (
            movie_dir.resolve() == tmp_dir.resolve()):
        logging.warning(
            f'AVA.MOVIES.DIR and AVA.MOVIES.TMP_DIR point to the same \n'
            f'directory, but AVA.MOVIES.KEEP_TMP is False. This would \n'
            f'result in deleting movies from AVA.MOVIES.DIR! Enabling \n'
            f'KEEP_TMP.')
        keep_tmp = True

    videos_dir = output_dir / cfg.VIDEOS_DIRNAME
    frames_dir = output_dir / cfg.FRAMES_DIRNAME
    for movie_stem, clips in tqdm(movie_clips.items(),
                                  desc='Processing AVA movies'):
        movie = f"{movie_stem}.{movie_info[movie_stem]['ext']}"

        # List of (clip, output clip path, output frames directory) for clips
        # whose frames have not already been extracted.
        to_process = []
        for clip in clips:
            name = clip['name']
            output_clip = file_path(videos_dir / f"{name}.mp4")
            output_frames = dir_path(frames_dir / name)
            if (are_tao_frames_dumped(output_frames, checksums[name])
                    and not cfg._KEEP_NON_TAO_FRAMES):
                continue
            to_process.append((clip, output_clip, output_frames))

        # Download movie if necessary.
        if all(x[1].exists() for x in to_process):
            movie_vfc = None
        else:
            if movie_dir and (movie_dir / movie).exists():
                remove_movie = False
                movie_path = movie_dir / movie
                logging.debug(f'Found AVA movie {movie} at {movie_path}')
            else:
                remove_movie = not keep_tmp
                movie_path = tmp_dir / movie
                if not movie_path.exists():
                    if not client:
                        client = S3Wrapper(cfg.AVA.AWS_BUCKET)
                    logging.info(f'Downloading AVA movie: {movie}.')
                    url = f"{movie_info[movie_stem]['split']}/{movie}"
                    client.download_file(url, str(movie_path), verbose=False)
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
            if remove_movie:
                movie_path.unlink()

        dump_frames([x[1] for x in to_process], [x[2] for x in to_process],
                    cfg)
        for clip, clip_path, frame_dir in to_process:
            assert are_tao_frames_dumped(frame_dir, checksums[clip['name']]), (
                f'Not all TAO frames for {clip["name"]} were extracted.')
            if not cfg._KEEP_NON_TAO_FRAMES:
                remove_non_tao_frames(frame_dir,
                                      set(checksums[clip['name']].keys()))
            if not cfg.KEEP_VIDEOS:
                clip_path.unlink()


def download_lasot(cfg, annotations, checksums, output_dir):
    logging.info(f'Extracting LaSOT videos.')
    videos_flat = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'LaSOT'
    ]
    category_videos = defaultdict(list)
    for v in videos_flat:
        category_videos[v['name'].rsplit('-', 1)[0]].append(v)

    for category, videos in category_videos.items():
        for video in videos:
            name = video['name']
            path = Path(cfg.LASOT.DATASET_ROOT) / category / name / 'img'
            # TODO: Download if it does not exist.
            assert path.exists(), (
                f'Could not find LaSOT frames for video {name} at {path}')
            frames_dir = dir_path(output_dir / cfg.FRAMES_DIRNAME / name)
            for frame in checksums[name]:
                frame_output = frames_dir / frame
                if not frame_output.exists():
                    if cfg.LASOT.CREATE_SYMLINKS:
                        frame_output.symlink_to((path / frame).resolve())
                    else:
                        shutil.copy2(path / frame, frame_output)
            assert are_tao_frames_dumped(frames_dir, checksums[name]), (
                f'Frames for video {name} do not match expected checksums.')


def download_charades(cfg, annotations, checksums, output_dir):
    logging.info(f'Extracting Charades videos.')
    videos = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'Charades'
    ]
    charades_dir = Path(cfg.CHARADES.VIDEOS_DIR)
    video_paths = [charades_dir / f"{v['name']}.mp4" for v in videos]
    frame_dirs = [
        dir_path(output_dir / cfg.FRAMES_DIRNAME / v['name']) for v in videos
    ]

    dump_frames(video_paths, frame_dirs, cfg)
    for video, video_path, frame_dir in zip(videos, video_paths, frame_dirs):
        name = video['name']
        assert are_tao_frames_dumped(frame_dir, checksums[name]), (
            f'Not all TAO frames for {name} were extracted.')
        if not cfg._KEEP_NON_TAO_FRAMES:
            remove_non_tao_frames(frame_dir, set(checksums[name].keys()))


def download_bdd(cfg, annotations, checksums, output_dir):
    logging.info(f'Extracting BDD videos.')
    videos = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'BDD'
    ]
    bdd_dir = Path(cfg.BDD.VIDEOS_DIR)
    video_paths = [bdd_dir / f"{v['name']}.mov" for v in videos]
    for v in video_paths:
        if not v.exists():
            raise ValueError(f'Could not find BDD video at {v}')
    frame_dirs = [
        dir_path(output_dir / cfg.FRAMES_DIRNAME / v['name']) for v in videos
    ]

    dump_frames(video_paths, frame_dirs, cfg)
    for video, video_path, frame_dir in zip(videos, video_paths, frame_dirs):
        name = video['name']
        assert are_tao_frames_dumped(frame_dir, checksums[name]), (
            f'Not all TAO frames for {name} were extracted.')
        if not cfg._KEEP_NON_TAO_FRAMES:
            remove_non_tao_frames(frame_dir, set(checksums[name].keys()))


def download_hacs(cfg, annotations, checksums, output_dir):
    logging.info(f'Downloading HACS videos.')
    videos = [
        v for anns in annotations.values() for v in anns.vids.values()
        if v['metadata']['dataset'] == 'HACS'
    ]

    if cfg.DEBUG:
        # Take 5 of each type of video.
        _scene_videos = [
            v for v in videos if v['metadata']['scene'] is not None
        ]
        _noscene_videos = [v for v in videos if v['metadata']['scene'] is None]
        videos = _scene_videos[:5] + _noscene_videos[:5]

    unavailable_videos = []

    videos_dir = dir_path(output_dir / cfg.VIDEOS_DIRNAME)
    frames_dir = dir_path(output_dir / cfg.FRAMES_DIRNAME)
    tmp_dir = dir_path(output_dir / cfg.HACS.TMP_DIR)

    # List of (video, video_path, frame_path)
    videos_to_dump = []
    for video in tqdm(videos[:10], desc='Downloading HACS'):
        video_path = dir_path(videos_dir / f"{video['name']}.mp4")
        frame_output = dir_path(frames_dir / video['name'])
        if not video_path.exists():
            ytid = video['metadata']['youtube_id']
            full_video = tmp_dir / f"v_{ytid}"
            if not full_video.exists():
                url = 'http://youtu.be/' + ytid
                try:
                    vid_bytes = download_to_bytes(url)
                except BaseException as e:
                    vid_bytes = None
                if isinstance(vid_bytes, int) or vid_bytes is None:
                    unavailable_videos.append(url)
                    continue
                else:
                    vid_bytes = vid_bytes.getvalue()
                    if len(vid_bytes) == 0:
                        unavailable_videos.append(url)
                        continue
                with open(full_video, 'wb') as f:
                    f.write(vid_bytes)

        if video['metadata']['scene'] is not None:
            shot_endpoints = video['metadata']['scene'].rsplit('_', 1)[1]
            start, end = shot_endpoints.split('-')
            clip = VideoFileClip(str(full_video))
            subclip = clip.subclip(int(start) / clip.fps, int(end) / clip.fps)
            subclip.write_videofile(str(video_path),
                                    audio=False,
                                    verbose=False,
                                    progress_bar=False)
        else:
            shutil.copy2(full_video, video_path)

        videos_to_dump.append((video['name'], video_path, frame_output))

        if not cfg.HACS.KEEP_TMP:
            full_video.unlink()

    dump_frames([x[1] for x in videos_to_dump],
                [x[2] for x in videos_to_dump],
                cfg)

    for video, video_path, frame_dir in videos_to_dump:
        assert are_tao_frames_dumped(frame_dir, checksums[video]), (
            f'Not all TAO frames for {video} were extracted.')
        if not cfg._KEEP_NON_TAO_FRAMES:
            remove_non_tao_frames(frame_dir, set(checksums[video].keys()))
        if not cfg.KEEP_VIDEOS:
            video_path.unlink()

    if unavailable_videos:
        logging.error(
            f'Some HACS videos could not be downloaded; please download them\n'
            f'manually from the HACS dataset website:\n'
            + ('\n'.join(unavailable_videos)))


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument('--config-updates', nargs='*')
    parser.add_argument('--splits',
                        default=['train'],
                        nargs='*',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--sources',
                        default=DATA_SOURCES,
                        nargs='*',
                        choices=DATA_SOURCES)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    for name in ['boto3', 'botocore', 's3transfer']:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    if args.config_updates:
        cfg.merge_from_list(args.config_updates)
    cfg.freeze()

    paths = {
        'train': cfg.TAO_ANNOTATIONS.TRAIN,
        'val': cfg.TAO_ANNOTATIONS.VAL,
        'test': cfg.TAO_ANNOTATIONS.TEST
    }
    annotations = {split: Tao(paths[split]) for split in args.splits}

    with open(cfg.CHECKSUMS.PATH, 'r') as f:
        checksums = json.load(f)
    if not cfg.CHECKSUMS.VERIFY:
        checksums = {
            v: {f: ''
                for f in frames}
            for v, frames in checksums.items()
        }

    for source in args.sources:
        if source == 'yfcc':
            download_yfcc(cfg, annotations, checksums, args.output_dir)
        elif source == 'ava':
            download_ava(cfg, annotations, checksums, args.output_dir)
        elif source == 'charades':
            download_charades(cfg, annotations, checksums, args.output_dir)
        elif source == 'lasot':
            download_lasot(cfg, annotations, checksums, args.output_dir)
        elif source == 'argoverse':
            raise NotImplementedError
        elif source == 'hacs':
            download_hacs(cfg, annotations, checksums, args.output_dir)
        elif source == 'bdd':
            download_bdd(cfg, annotations, checksums, args.output_dir)


if __name__ == "__main__":
    main()
