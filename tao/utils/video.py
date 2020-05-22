import functools
import json
import logging
import os
import subprocess
from contextlib import contextmanager

from pathlib import Path

from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.tools import extensions_dict


@contextmanager
def video_writer(output,
                 size,
                 fps=30,
                 codec=None,
                 ffmpeg_params=None):
    """
    Args:
        size (tuple): (width, height) tuple
    """
    if isinstance(output, Path):
        output = str(output)

    if codec is None:
        extension = Path(output).suffix[1:]
        try:
            codec = extensions_dict[extension]['codec'][0]
        except KeyError:
            raise ValueError(f"Couldn't find the codec associated with the "
                             f"filename ({output}). Please specify codec")

    if ffmpeg_params is None:
        ffmpeg_params = [
            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-pix_fmt', 'yuv420p'
        ]
    with FFMPEG_VideoWriter(output,
                            size=size,
                            fps=fps,
                            codec=codec,
                            ffmpeg_params=ffmpeg_params) as writer:
        yield writer


def video_info(video):
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    if isinstance(video, Path):
        video = str(video)

    info = ffmpeg_parse_infos(video)
    return {
        'duration': info['duration'],
        'fps': info['video_fps'],
        'size': info['video_size']  # (width, height)
    }


def are_frames_dumped(video_path,
                      output_dir,
                      expected_fps,
                      expected_info_path,
                      expected_name_format,
                      log_reason=False):
    """Check if the output directory exists and has already been processed.

        1) Check the info.json file to see if the parameters match.
        2) Ensure that all the frames exist.

    Params:
        video_path (str)
        output_dir (str)
        expected_fps (num)
        expected_info_path (str)
        expected_name_format (str)
    """
    # Ensure that info file exists.
    if not os.path.isfile(expected_info_path):
        if log_reason:
            logging.info("Info path doesn't exist at %s" % expected_info_path)
        return False

    # Ensure that info file is valid.
    with open(expected_info_path, 'r') as info_file:
        info = json.load(info_file)
    info_valid = info['frames_per_second'] == expected_fps \
        and info['input_video_path'] == os.path.abspath(video_path)
    if not info_valid:
        if log_reason:
            logging.info("Info file (%s) is invalid" % expected_info_path)
        return False

    # Check that all frame paths exist.
    offset_if_one_indexed = 0
    if not os.path.exists(expected_name_format % 0):
        # If the 0th frame doesn't exist, either we haven't dumped the frames,
        # or the frames start with index 1 (this changed between versions of
        # moviepy, so we have to explicitly check). We can assume they start
        # with index 1, and continue.
        offset_if_one_indexed = 1

    # https://stackoverflow.com/a/28376817/1291812
    num_frames_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    expected_num_frames = subprocess.check_output(num_frames_cmd,
                                                  stderr=subprocess.STDOUT)
    expected_num_frames = int(expected_num_frames.decode().strip())
    expected_frame_paths = [
        expected_name_format % (i + offset_if_one_indexed)
        for i in range(expected_num_frames)
    ]
    missing_frames = [x for x in expected_frame_paths if not os.path.exists(x)]
    if missing_frames:
        if log_reason:
            logging.info("Missing frames:\n%s" % ('\n'.join(missing_frames)))
        return False

    # All checks passed
    return True


def dump_frames(video_path,
                output_dir,
                fps,
                extension='.jpg',
                jpeg_qscale=2):
    """Dump frames at frames_per_second from a video to output_dir.

    If frames_per_second is None, the clip's fps attribute is used instead."""
    output_dir.mkdir(exist_ok=True, parents=True)

    if extension[0] != '.':
        extension = f'.{extension}'

    try:
        video_info = ffmpeg_parse_infos(str(video_path))
        video_fps = video_info['video_fps']
    except OSError:
        logging.exception('Unable to open video (%s), skipping.' % video_path)
        raise
    except KeyError:
        logging.error('Unable to extract metadata about video (%s), skipping.'
                      % video_path)
        logging.exception('Exception:')
        return
    info_path = '{}/info.json'.format(output_dir)
    name_format = '{}/frame%04d{}'.format(output_dir, extension)

    if fps is None or fps == 0:
        fps = video_fps  # Extract all frames

    are_frames_dumped_wrapper = functools.partial(
        are_frames_dumped,
        video_path=video_path,
        output_dir=output_dir,
        expected_fps=fps,
        expected_info_path=info_path,
        expected_name_format=name_format)

    if extension.lower() in ('.jpg', '.jpeg'):
        qscale = ['-qscale:v', str(jpeg_qscale)]
    else:
        qscale = []

    if are_frames_dumped_wrapper(log_reason=False):
        return

    successfully_wrote_images = False
    try:
        if fps == video_fps:
            cmd = ['ffmpeg', '-i', str(video_path)] + qscale + [name_format]
        else:
            cmd = ['ffmpeg', '-i', str(video_path)
                   ] + qscale + ['-vf', 'fps={}'.format(fps), name_format]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        successfully_wrote_images = True
    except subprocess.CalledProcessError as e:
        logging.exception("Failed to dump images for %s", video_path)
        logging.error(e.output.decode('utf-8'))
        raise

    if successfully_wrote_images:
        info = {'frames_per_second': fps,
                'input_video_path': os.path.abspath(video_path)}
        with open(info_path, 'w') as info_file:
            json.dump(info, info_file)

        if not are_frames_dumped_wrapper(log_reason=True):
            logging.warning(
                "Images for {} don't seem to be dumped properly!".format(
                    video_path))
