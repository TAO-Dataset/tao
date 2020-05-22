import argparse
import json
from collections import defaultdict
from hashlib import md5
from pathlib import Path

from tqdm import tqdm
from script_utils.common import common_setup

from tao.utils import fs


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--tao-annotations', type=Path, required=True)

    args = parser.parse_args()
    output_dir = args.output_json.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(args.output_json.name, output_dir, args)

    with open(args.tao_annotations, 'r') as f:
        tao = json.load(f)
        videos = [x['name'] for x in tao['videos']]

    labeled_frames = defaultdict(set)
    for frame in tao['images']:
        video, frame_name = frame['file_name'].rsplit('/', 1)
        labeled_frames[video].add(frame_name)

    # videos = videos[:10]
    hashes = {}
    for video in tqdm(videos):
        frames = fs.glob_ext(args.frames_dir / video, ('.jpg', '.jpeg'))
        hashes[video] = {}
        for i, frame in tqdm(enumerate(frames)):
            if frame.name in labeled_frames[video]:
                with open(frame, 'rb') as f:
                    hashes[video][frame.name] = md5(f.read()).hexdigest()
            else:
                hashes[video][frame.name] = ''
        if all(x == '' for x in hashes[video].values()):
            raise ValueError(f'Did not find any labeled frames for {video}')

    with open(args.output_json, 'w') as f:
        json.dump(hashes, f)


if __name__ == "__main__":
    main()
