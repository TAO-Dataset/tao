import json
import logging
import sys
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

import boto3
import youtube_dl
from tqdm import tqdm

# In case we use pywren, we can't import from the tao module directly here
# for some reason. Just import what we need.
sys.path.insert(0, str(Path(__file__).parent))
import s3 as s3_utils


class VideoUnavailableError(youtube_dl.DownloadError):
    pass


def get_metadata(url):
    ydl_opts = {
        'outtmpl': '-',
        'skip_download': True,
        'forcejson': True,
        'quiet': True,
        'nocheckcertificate': True,
        'cachedir': False
    }
    info = StringIO()
    try:
        with redirect_stdout(info):
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
    except youtube_dl.DownloadError as e:
        message = str(e)
        if 'This video is no longer available' in message:
            raise VideoUnavailableError(message)
        else:
            raise e
    except BaseException as e:
        print('Exception', e)
        import traceback
        traceback.print_exc()
        raise e
    return json.loads(info.getvalue())


def download_to_bytes(url, extra_opts={}):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': '-',
        'logger': logging.getLogger('youtube-dl'),
        'nocheckcertificate': True,
        'cachedir': False
    }
    ydl_opts.update(extra_opts)

    video = BytesIO()
    try:
        with redirect_stdout(video):
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
    except youtube_dl.DownloadError as e:
        message = str(e)
        if 'This video is no longer available' in message:
            raise VideoUnavailableError(message)
        else:
            raise e
    except BaseException as e:
        print('Exception', e)
        import traceback
        traceback.print_exc()
        raise e
    return video


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def pytube_download_bytes(url, extra_opts={}):
    buffer_obj = None
    try:
        buffer_obj = download_to_bytes(url, extra_opts=extra_opts)
    except VideoUnavailableError:
        return -1
    except:
        return -2
    return buffer_obj.getvalue()


def store_bytes_to_s3(object_bytes, key, bucket):
    client = boto3.client('s3')
    client.put_object(Body=object_bytes,
                      Key=key,
                      Bucket=bucket)
    return len(object_bytes)


def vid_id_to_name(vid_id):
    return f'v_{vid_id}'


def download_and_store_vids(urls,
                            ids,
                            keys,
                            bucket,
                            ytdl_params={},
                            progress=False):
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')
    logging.info('Downloading videos: %s', urls)
    downloaded = 0
    unavail = 0
    other_error = 0
    empty_bytes = 0
    blacklist = []
    for url, vid_id, key in zip(tqdm(urls, disable=not progress), ids, keys):
        logging.info(f"Downloading Video {vid_id}")
        vid_bytes = pytube_download_bytes(url, ytdl_params)

        problem = None
        if isinstance(vid_bytes, int):
            if vid_bytes == -1:
                logging.info(f'unavailable: {vid_id}')
                problem = 'unavailable'
                unavail += 1
                blacklist.append(vid_id)
            elif vid_bytes == -2:
                problem = 'error'
                other_error += 2
        elif vid_bytes is None:
            problem = 'other_download_error'
        elif len(vid_bytes) == 0:
            logging.info(f'empty bytes: {vid_id}')
            problem = 'empty_bytes'
            empty_bytes += 1

        if problem is not None:
            problem_key = f'{key}.{problem}'
            store_bytes_to_s3('', problem_key, bucket)
            continue

        logging.debug(f"Uploading Video {vid_id} to S3")
        store_bytes_to_s3(vid_bytes, key, bucket)
        downloaded += 1
        import time
        import random
        time.sleep(random.random() * 2)
    return downloaded, unavail, other_error, blacklist


def download_vids(videos,
                  s3_bucket,
                  s3_prefix,
                  cache_dir=None,
                  parallel=False,
                  video_keys=None,
                  ytdl_params={},
                  skip_exists=True,
                  chunk_size=2,
                  subset=None,
                  verbose=False):
    urls = ['http://youtu.be/'+vid for vid in videos]
    ids = videos
    client = boto3.client('s3')
    exist_ids = set(s3_utils.list_all_keys(client, s3_bucket, s3_prefix))

    if verbose:
        log = logging.info
    else:
        def log(*args, **kwargs):
            return

    if video_keys is None:
        video_keys = [
            f"{vid_id_to_name(vid_id)}.mp4" for vid_id in videos
        ]
    if s3_prefix[-1] != '/':
        s3_prefix = s3_prefix + '/'
    video_keys = [f'{s3_prefix}{key}' for key in video_keys]

    # if processed_prefix is not None:
    #     processed_key = f'{s3_prefix}{processed_prefix}'
    #     if s3_utils.key_exists(s3_bucket, processed_key):
    #         obj = client.Object(s3_bucket, processed_key)
    #         data = obj.get()['Body'].read().decode('utf-8')
    #         processed = set(data.split('\n'))
    #         valid_videos = [(url, vid, k)
    #                         for (url, vid, k) in zip(urls, ids, video_keys)
    #                         if k not in processed]
    #         if not valid_videos:
    #             return {'num_downloaded': 0, 'unavailable': 0, 'num_errors': 0}
    #         urls, ids, video_keys = zip(*valid_videos)

    if skip_exists:
        if len(exist_ids) > 0:
            valid_videos = []
            for url, vid, key in zip(urls, ids, video_keys):
                error_keys = [
                    f'{key}.{x}' for x in ('unavailable', 'empty_bytes')
                ]
                check_keys = error_keys + [key]
                if not any(x in exist_ids for x in check_keys):
                    valid_videos.append((url, vid, key))
            if not valid_videos:
                return {'num_downloaded': 0, 'unavailable': 0, 'num_errors': 0}
            urls, ids, video_keys = zip(*valid_videos)
            log(f'{len(urls)}/{len(videos)} to download.')

    if subset is not None:
        videos = videos[:subset]
        urls = urls[:subset]
    if not parallel:
        results = [
            download_and_store_vids(urls, ids, video_keys, s3_bucket,
                                    ytdl_params, progress=True)
        ]
    else:
        import pywren
        chunked_lst = list(chunks(list(zip(urls, ids, video_keys)), chunk_size))
        log(f"{len(chunked_lst)} Pywren jobs total")
        pwex = pywren.default_executor()

        def pywren_f(elem):
            urls, ids, keys = zip(*elem)
            return download_and_store_vids(urls, ids, keys, s3_bucket,
                                           ytdl_params)

        try:
            log("Mapping...")
            futures = pwex.map(pywren_f, chunked_lst)
            pywren.wait(futures)
            results = [f.result() for f in futures]
            print('len(results)', len(results))
        finally:
            for f in futures:
                f.cancel()

    unavailable = [y for x in results for y in x[3]]
    return {
        'num_downloaded': sum([x[0] for x in results]),
        'unavailable': unavailable,
        'num_errors': sum([x[2] for x in results]),
    }
