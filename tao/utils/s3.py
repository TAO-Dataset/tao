import concurrent.futures
import datetime
import math
import pathlib
import random
import shutil
import subprocess
import threading
import time
from timeit import default_timer as timer
import os
import io

import boto3
import botocore
from botocore.client import Config


default_profile = os.environ.get('AWS_PROFILE', 'default')
default_cache_root_path = (pathlib.Path(__file__).parent /
                           '../data/s3_cache').resolve()


def tar_directory(dir_name, target_filename):
    subprocess.run(
        ['tar', '-cf', str(target_filename),
         str(dir_name)], check=True)


def untar_directory(tar_filename, target_dir, strip=None, one_top_level=False):
    cmd = ['tar', '-xf', str(tar_filename)]
    if strip:
        cmd += [f'--strip={strip}']
    cmd += ['-C', str(target_dir)]
    if one_top_level:
        cmd += ['--one-top-level']
    subprocess.run(cmd, check=True)


def get_s3_client():
    if default_profile in boto3.Session()._session.available_profiles:
        session = boto3.Session(profile_name=default_profile)
    else:
        session = boto3.Session()
    config = Config(connect_timeout=250, read_timeout=250)
    return session.client('s3', config=config)


def get_s3_resource():
    if default_profile in boto3.Session()._session.available_profiles:
        session = boto3.Session(profile_name=default_profile)
    else:
        session = boto3.Session()
    config = Config(connect_timeout=250, read_timeout=250)
    return session.client('s3', config=config)


def key_exists(bucket, key):
    # Return true if a key exists in s3 bucket
    # TODO: return None from the get functions if the key doesn't exist?
    #       (this would avoid one round-trip to S3)
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise
        return False
    except:
        raise


def get_s3_object_bytes_parallel(keys, *,
                                 bucket,
                                 cache_on_local_disk=True,
                                 cache_root_path=None,
                                 verbose=True,
                                 max_num_threads=90,
                                 num_tries=5,
                                 initial_delay=1.0,
                                 delay_factor=math.sqrt(2.0),
                                 download_callback=None,
                                 skip_modification_time_check=False):
    if cache_on_local_disk:
        assert cache_root_path is not None
        cache_root_path = pathlib.Path(cache_root_path).resolve()

        missing_keys = []
        existing_keys = []
        for key in keys:
            local_filepath = cache_root_path / key
            if not local_filepath.is_file():
                missing_keys.append(key)
                local_filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                existing_keys.append(key)

        keys_to_download = missing_keys.copy()
        if skip_modification_time_check:
            if verbose:
                print(f'Skipping the file modification time check for {len(existing_keys)} keys that have local copies.')
            for key in existing_keys:
                if download_callback:
                    download_callback(1)
        else:
            if verbose:
                print(f'Getting metadata for {len(existing_keys)} keys that have local copies ... ', end='')
            metadata_start = timer()
            metadata = get_s3_object_metadata_parallel(existing_keys,
                                                       bucket=bucket,
                                                       verbose=False,
                                                       max_num_threads=max_num_threads,
                                                       num_tries=num_tries,
                                                       initial_delay=initial_delay,
                                                       delay_factor=delay_factor,
                                                       download_callback=None)
            metadata_end = timer()
            if verbose:
                print(f'took {metadata_end - metadata_start:.3f} seconds')
            for key in existing_keys:
                local_filepath = cache_root_path / key
                assert local_filepath.is_file
                local_time = datetime.datetime.fromtimestamp(local_filepath.stat().st_mtime,
                                                             datetime.timezone.utc)
                remote_time = metadata[key]['LastModified']
                if local_time <= remote_time:
                    if verbose:
                        print(f'Local copy of key "{key}" is outdated')
                    keys_to_download.append(key)
                elif download_callback:
                    download_callback(1)

        tl = threading.local()
        def cur_download_file(key):
            local_filepath = cache_root_path / key
            if verbose:
                print('{} not available locally our outdated, downloading from S3 ... '.format(key))
            download_s3_file_with_backoff(key,
                                          str(local_filepath),
                                          bucket=bucket,
                                          num_tries=num_tries,
                                          initial_delay=initial_delay,
                                          delay_factor=delay_factor,
                                          thread_local=tl)
            return local_filepath.is_file()

        if len(keys_to_download) > 0:
            download_start = timer()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
                future_to_key = {executor.submit(cur_download_file, key): key for key in keys_to_download}
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        success = future.result()
                        assert success
                        if download_callback:
                            download_callback(1)
                    except Exception as exc:
                        print('Key {} generated an exception: {}'.format(key, exc))
                        raise exc
            download_end = timer()
            if verbose:
                print('Downloading took {:.3f} seconds'.format(download_end - download_start))

        result = {}
        # TODO: parallelize this as well?
        for key in keys:
            local_filepath = cache_root_path / key
            if verbose:
                print('Reading from local file {} ... '.format(local_filepath), end='')
            with open(local_filepath, 'rb') as f:
                result[key] = f.read()
            if verbose:
                print('done')
    else:
        tl = threading.local()
        def cur_get_object_bytes(key):
            if verbose:
                print('Loading {} from S3 ... '.format(key))
            return get_s3_object_bytes_with_backoff(key,
                                                    bucket=bucket,
                                                    num_tries=num_tries,
                                                    initial_delay=initial_delay,
                                                    delay_factor=delay_factor,
                                                    thread_local=tl)[0]
        download_start = timer()
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
            future_to_key = {executor.submit(cur_get_object_bytes, key): key for key in keys}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result[key] = future.result()
                    if download_callback:
                        download_callback(1)
                except Exception as exc:
                    print('Key {} generated an exception: {}'.format(key, exc))
                    raise exc
        download_end = timer()
        if verbose:
            print('Getting object bytes took {} seconds'.format(download_end - download_start))
    return result


def get_s3_object_bytes_with_backoff(key, *,
                                     bucket,
                                     num_tries=5,
                                     initial_delay=1.0,
                                     delay_factor=math.sqrt(2.0),
                                     num_replicas=1,
                                     thread_local=None):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 'get_object_client'):
            thread_local.get_object_client = get_s3_client()
        client = thread_local.get_object_client
    delay = initial_delay
    num_tries_left = num_tries

    if num_replicas > 1:
        replicas_counter_len = len(str(num_replicas))
        format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            read_bytes = client.get_object(Key=cur_key, Bucket=bucket)["Body"].read()
            return read_bytes, cur_key
        except:
            if num_tries_left == 1:
                raise Exception('get backoff failed ' + key + ' ' + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def get_s3_object_metadata_with_backoff(key, *,
                                        bucket,
                                        num_tries=5,
                                        initial_delay=1.0,
                                        delay_factor=math.sqrt(2.0),
                                        thread_local=None):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 'get_object_client'):
            thread_local.get_object_client = get_s3_client()
        client = thread_local.get_object_client
    delay = initial_delay
    num_tries_left = num_tries
    while num_tries_left >= 1:
        try:
            metadata = client.head_object(Key=key, Bucket=bucket)
            return metadata
        except:
            if num_tries_left == 1:
                raise Exception('get backoff failed ' + key + ' ' + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def get_s3_object_metadata_parallel(keys,
                                    bucket,
                                    verbose=True,
                                    max_num_threads=20,
                                    num_tries=5,
                                    initial_delay=1.0,
                                    delay_factor=math.sqrt(2.0),
                                    download_callback=None):
    tl = threading.local()
    def cur_get_object_metadata(key):
        if verbose:
            print('Loading metadata for {} from S3 ... '.format(key))
        return get_s3_object_metadata_with_backoff(key,
                                                   bucket=bucket,
                                                   num_tries=num_tries,
                                                   initial_delay=initial_delay,
                                                   delay_factor=delay_factor,
                                                   thread_local=tl)
    download_start = timer()
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        future_to_key = {executor.submit(cur_get_object_metadata, key): key for key in keys}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result[key] = future.result()
                if download_callback:
                    download_callback(1)
            except Exception as exc:
                print('Key {} generated an exception: {}'.format(key, exc))
                raise exc
    download_end = timer()
    if verbose:
        print('Getting object metadata took {} seconds'.format(download_end - download_start))
    return result


def put_s3_object_bytes_with_backoff(file_bytes, key, bucket, num_tries=10, initial_delay=1.0, delay_factor=2.0):
    client = get_s3_client()
    delay = initial_delay
    num_tries_left = num_tries
    while num_tries_left >= 1:
        try:
            bio = io.BytesIO(file_bytes)
            client.upload_fileobj(bio, Key=key, Bucket=bucket, ExtraArgs={'ACL': 'bucket-owner-full-control'})
            return
        except:
            if num_tries_left == 1:
                print('put backoff failed' + key)
                raise Exception('put backoff failed ' + key + ' ' + str(len(file_bytes))+ ' ' + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def list_all_keys(client, bucket, prefix, max_keys=None):
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    if (objects.get('Contents') == None):
        return []
    keys = list(map(lambda x: x['Key'], objects.get('Contents', [] )))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
        if (max_keys is not None and len(keys) >= max_keys):
            break
    return list(filter(lambda x: len(x) > 0, keys))


def download_s3_file_with_caching(key, local_filename, *,
                                  bucket,
                                  cache_on_local_disk=True,
                                  cache_root_path=None,
                                  verbose=True,
                                  num_tries=5,
                                  initial_delay=1.0,
                                  delay_factor=math.sqrt(2.0),
                                  num_replicas=1,
                                  skip_modification_time_check=False):
    if cache_on_local_disk:
        assert cache_root_path is not None
        cache_root_path = pathlib.Path(cache_root_path).resolve()
        currently_cached = False

        cache_filepath = cache_root_path / key
        if not cache_filepath.is_file():
            cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            if skip_modification_time_check:
                if verbose:
                    print(f'Skipping the file modification time check the local copy in the cache.')
                currently_cached = True
            else:
                if verbose:
                    print(f'Getting metadata to check the modification time compared to the local copy ... ', end='')
                metadata_start = timer()
                metadata = get_s3_object_metadata_with_backoff(key,
                                                            bucket=bucket,
                                                            num_tries=num_tries,
                                                            initial_delay=initial_delay,
                                                            delay_factor=delay_factor)
                metadata_end = timer()
                if verbose:
                    print(f'took {metadata_end - metadata_start:.3f} seconds')
                local_time = datetime.datetime.fromtimestamp(cache_filepath.stat().st_mtime,
                                                             datetime.timezone.utc)
                remote_time = metadata['LastModified']
                if local_time <= remote_time:
                    if verbose:
                        print(f'Local copy of key "{key}" is outdated')
                else:
                    currently_cached = True
        if not currently_cached:
            if verbose:
                print('{} not available locally our outdated, downloading from S3 ... '.format(key))
            download_start = timer()
            download_s3_file_with_backoff(key, str(cache_filepath),
                                          bucket=bucket,
                                          initial_delay=initial_delay,
                                          delay_factor=delay_factor,
                                          num_replicas=num_replicas)
            download_end = timer()
            if verbose:
                print('Downloading took {:.3f} seconds'.format(download_end - download_start))
        assert cache_filepath.is_file()
        if verbose:
            print(f'Copying to the target from the cache file {cache_filepath} ...')
        shutil.copy(cache_filepath, local_filename)
    else:
        if verbose:
            print('Loading {} from S3 ... '.format(key))
        download_start = timer()
        download_s3_file_with_backoff(key, local_filename,
                                      bucket=bucket,
                                      initial_delay=initial_delay,
                                      delay_factor=delay_factor,
                                      num_replicas=num_replicas)
        download_end = timer()
        if verbose:
            print('Downloading took {:.3f} seconds'.format(download_end - download_start))


def download_s3_file_with_backoff(key, local_filename, *,
                                  bucket,
                                  num_tries=5,
                                  initial_delay=1.0,
                                  delay_factor=math.sqrt(2.0),
                                  num_replicas=1,
                                  thread_local=None):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 's3_client'):
            thread_local.s3_client = get_s3_client()
        client = thread_local.s3_client
    delay = initial_delay
    num_tries_left = num_tries

    if num_replicas > 1:
        replicas_counter_len = len(str(num_replicas))
        format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            client.download_file(bucket, cur_key, local_filename)
            return cur_key
        except:
            if num_tries_left == 1:
                raise Exception('download backoff failed ' + ' ' + str(key) + ' ' + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def upload_file_to_s3_with_backoff(local_filename, key, *,
                                   bucket,
                                   num_tries=5,
                                   initial_delay=1.0,
                                   delay_factor=math.sqrt(2.0),
                                   thread_local=None):
    assert pathlib.Path(local_filename).is_file()
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 's3_client'):
            thread_local.s3_client = get_s3_client()
        client = thread_local.s3_client
    delay = initial_delay
    num_tries_left = num_tries
    while num_tries_left >= 1:
        try:
            client.upload_file(local_filename, bucket, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
            return
        except:
            if num_tries_left == 1:
                raise Exception('upload backoff failed ' + ' ' + str(key) + ' ' + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def default_option_if_needed(*, user_option, default):
    if user_option is None:
        return default
    else:
        return user_option


class S3Wrapper:
    def __init__(self,
                 bucket,
                 cache_on_local_disk=True,
                 cache_root_path=default_cache_root_path,
                 verbose=True,
                 max_num_threads=90,
                 num_tries=5,
                 initial_delay=1.0,
                 delay_factor=math.sqrt(2.0),
                 skip_modification_time_check=False):
        self.bucket = bucket
        self.cache_on_local_disk = cache_on_local_disk
        self.client = get_s3_client()

        if self.cache_on_local_disk:
            assert cache_root_path is not None
            self.cache_root_path = pathlib.Path(cache_root_path).resolve()
            self.cache_root_path.mkdir(parents=True, exist_ok=True)
            assert self.cache_root_path.is_dir()
        else:
            self.cache_root_path = None
        self.verbose = verbose
        self.max_num_threads = max_num_threads
        self.num_tries = num_tries
        self.initial_delay = initial_delay
        self.delay_factor = delay_factor
        self.skip_modification_time_check = skip_modification_time_check

    def list_keys(self, prefix, max_keys=None):
        return list_all_keys(self.client, self.bucket, prefix, max_keys)

    def put(self, bytes_to_store, key, verbose=None):
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        put_s3_object_bytes_with_backoff(bytes_to_store,
                                         key,
                                         bucket=self.bucket,
                                         num_tries=self.num_tries,
                                         initial_delay=self.initial_delay,
                                         delay_factor=self.delay_factor)
        if cur_verbose:
            print('Stored {} bytes under key {}'.format(len(bytes_to_store), key))

    def put_multiple(self, data, verbose=None, callback=None):
        # TODO: add a new function for parallel uploading
        for key, bytes_to_store in data.items():
            self.put(bytes_to_store, key, verbose)

    def upload_dir(self, directory, prefix, verbose=None):
        directory = pathlib.Path(directory)
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        local_paths = [x for x in directory.glob('**/*') if x.is_file()]
        all_data = {}
        copied = set()
        for path in local_paths:
            path_resolved = path.resolve()
            with open(path_resolved, 'rb') as f:
                data = f.read()
            relative_path = str(path.relative_to(directory))
            assert relative_path not in copied
            copied.add(relative_path)
            cur_key = prefix + relative_path
            all_data[cur_key] = data
        self.put_multiple(all_data, verbose=verbose)

    def get_url(self, key):
        return f'https://{self.bucket}.s3.amazonaws.com/{key}'

    def upload_file(self, filename, key, verbose=None):
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        upload_file_to_s3_with_backoff(filename,
                                       key,
                                       bucket=self.bucket,
                                       num_tries=self.num_tries,
                                       initial_delay=self.initial_delay,
                                       delay_factor=self.delay_factor,
                                       thread_local=None)

    def download_file(self, key, filename, verbose=None, skip_modification_time_check=None):
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        cur_skip_time_check = default_option_if_needed(user_option=skip_modification_time_check,
                                                       default=self.skip_modification_time_check)
        download_s3_file_with_caching(key,
                                      filename,
                                      bucket=self.bucket,
                                      cache_on_local_disk=self.cache_on_local_disk,
                                      cache_root_path=self.cache_root_path,
                                      num_tries=self.num_tries,
                                      initial_delay=self.initial_delay,
                                      delay_factor=self.delay_factor,
                                      skip_modification_time_check=cur_skip_time_check,
                                      verbose=cur_verbose)

    def get(self, key, verbose=None, skip_modification_time_check=None):
        return self.get_multiple([key], verbose=verbose, skip_modification_time_check=skip_modification_time_check)[key]

    def get_multiple(self, keys, verbose=None, callback=None, skip_modification_time_check=None):
        if verbose is None:
            cur_verbose = self.verbose
        else:
            cur_verbose = verbose
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        cur_skip_time_check = default_option_if_needed(user_option=skip_modification_time_check,
                                                       default=self.skip_modification_time_check)
        return get_s3_object_bytes_parallel(keys,
                                            bucket=self.bucket,
                                            cache_on_local_disk=self.cache_on_local_disk,
                                            cache_root_path=self.cache_root_path,
                                            verbose=cur_verbose,
                                            max_num_threads=self.max_num_threads,
                                            num_tries=self.num_tries,
                                            initial_delay=self.initial_delay,
                                            delay_factor=self.delay_factor,
                                            download_callback=callback,
                                            skip_modification_time_check=cur_skip_time_check)

    def exists(self, key):
        return key_exists(self.bucket, key)
