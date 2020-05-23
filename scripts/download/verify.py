import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from script_utils.common import common_setup
from tqdm import tqdm

from tao.utils.download import are_tao_frames_dumped


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--split',
                        required=True,
                        choices=['train', 'val', 'test'])

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

    videos_by_dataset = defaultdict(list)
    for video in tao['videos']:
        videos_by_dataset[video['metadata']['dataset']].append(video)

    status = {}
    for dataset, videos in sorted(videos_by_dataset.items()):
        status[dataset] = True
        for video in tqdm(videos, desc=f'Verifying {dataset}'):
            name = video['name']
            frame_dir = args.root / 'frames' / name
            if not are_tao_frames_dumped(frame_dir, checksums[name],
                                         warn=True):
                logging.warning(
                    f'Frames for {name} are not extracted properly. '
                    f'Skipping rest of dataset.')
                status[dataset] = False
                break

    success = []
    for dataset in sorted([d for d, v in status.items() if v]):
        success.append(f'{dataset: <12}: Verified ✓✓✓')

    failure = []
    for dataset in sorted([d for d, v in status.items() if not v]):
        failure.append(f'{dataset: <12}: FAILED 𐄂𐄂𐄂')

    if success:
        logging.info('Success!\n' + ('\n'.join(success)))
    if failure:
        logging.warning('Some datasets were not properly extracted!\n' +
                        ('\n'.join(failure)))


if __name__ == "__main__":
    main()
