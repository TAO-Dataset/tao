import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from script_utils.common import common_setup
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lvis', type=Path, required=True)
    parser.add_argument('--coco', type=Path, required=True)
    parser.add_argument('--mapping',
                        type=Path,
                        default=ROOT / 'data/lvis_coco_to_synset.json')
    parser.add_argument('--output-json',
                        type=Path,
                        required=True)
    parser.add_argument(
        '--iou-thresh',
        default=0.7,
        type=float,
        help=('If a COCO annotation overlaps with an LVIS annotations with '
              'IoU over this threshold, we use only the LVIS annotation.'))

    args = parser.parse_args()
    args.output_json.parent.mkdir(exist_ok=True, parents=True)
    common_setup(args.output_json.name + '.log', args.output_json.parent, args)

    coco = COCO(args.coco)
    lvis = COCO(args.lvis)

    synset_to_lvis_id = {x['synset']: x['id'] for x in lvis.cats.values()}
    coco_to_lvis_category = {}
    with open(args.mapping, 'r') as f:
        name_mapping = json.load(f)
    for category in coco.cats.values():
        mapped = name_mapping[category['name']]
        assert mapped['coco_cat_id'] == category['id']
        synset = mapped['synset']
        if synset not in synset_to_lvis_id:
            logging.debug(
                f'Found no LVIS category for "{category["name"]}" from COCO')
            continue
        coco_to_lvis_category[category['id']] = synset_to_lvis_id[synset]

    for image_id, image in coco.imgs.items():
        if image_id in lvis.imgs:
            coco_name = coco.imgs[image_id]['file_name']
            lvis_name = lvis.imgs[image_id]['file_name']
            assert coco_name in lvis_name
        else:
            logging.info(
                f'Image {image_id} in COCO, but not annotated in LVIS')

    lvis_highest_id = max(x['id'] for x in lvis.anns.values())
    ann_id_generator = itertools.count(lvis_highest_id + 1)
    new_annotations = []
    for image_id, lvis_anns in tqdm(lvis.imgToAnns.items()):
        if image_id not in coco.imgToAnns:
            logging.info(
                f'Image {image_id} in LVIS, but not annotated in COCO')
            continue

        coco_anns = coco.imgToAnns[image_id]
        # Compute IoU between coco_anns and lvis_anns
        # Shape (num_coco_anns, num_lvis_anns)
        mask_iou = mask_util.iou([coco.annToRLE(x) for x in coco_anns],
                                 [lvis.annToRLE(x) for x in lvis_anns],
                                 pyiscrowd=np.zeros(len(lvis_anns)))
        does_overlap = mask_iou.max(axis=1) > args.iou_thresh
        to_add = []
        for i, ann in enumerate(coco_anns):
            if does_overlap[i]:
                continue
            if ann['category_id'] not in coco_to_lvis_category:
                continue
            ann['category_id'] = coco_to_lvis_category[ann['category_id']]
            ann['id'] = next(ann_id_generator)
            to_add.append(ann)
        new_annotations.extend(to_add)

    with open(args.lvis, 'r') as f:
        merged = json.load(f)
    merged['annotations'].extend(new_annotations)
    with open(args.output_json, 'w') as f:
        json.dump(merged, f)


if __name__ == "__main__":
    main()
