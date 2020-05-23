import logging
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm
from tao.utils.video import dump_frames


def dump_frames_star(task):
    return dump_frames(*task)


def dump_tao_frames(videos,
                    output_dirs,
                    workers,
                    tqdm_desc='Converting to frames'):
    fps = None
    extension = '.jpg'
    jpeg_qscale = 2

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
    logging.root.setLevel(logging.ERROR)
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
            dump_frames_star(task)
    logging.root.setLevel(_log_level)


def are_tao_frames_dumped(frames_dir, checksums, warn=True):
    for frame, cksum in checksums.items():
        if frame.endswith('.jpeg'):
            frame = frame.replace('.jpeg', '.jpg')
        path = frames_dir / frame
        if not path.exists():
            if warn:
                logging.warning(f'Could not find frame at {path}!')
            return False
        if cksum:
            with open(path, 'rb') as f:
                md5_digest = md5(f.read()).hexdigest()
            if md5_digest != cksum:
                if warn:
                    logging.warning(
                        f'Checksum for {path} did not match! '
                        f'Expected: {cksum}, saw: {md5_digest}')
                return False
    return True


def remove_non_tao_frames(frames_dir, keep_frames):
    frames = {x.split('.')[0] for x in keep_frames}
    extracted_frames = list(frames_dir.glob('*.jpg'))
    to_remove = [x for x in extracted_frames if x.stem not in frames]
    assert len(to_remove) != len(extracted_frames)
    for frame in to_remove:
        frame.unlink()
